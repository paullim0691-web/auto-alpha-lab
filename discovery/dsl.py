import copy
import numpy as np
import pandas as pd

INVERSE_LEV_ETFS = {"QLD", "SSO", "SH", "PSQ", "TBF", "SDS", "QID"}
CORE_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "MDY", "MTUM", "VLUE", "QUAL", "USMV", "ARKK",
    "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "GDX", "USO", "UNG", "DBC", "DBA", "UUP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "SMH", "XBI", "KRE", "ITA",
}
ALL_ETFS = CORE_ETFS | INVERSE_LEV_ETFS

ALLOWED_KINDS = {
    "seed_benchmark", "cs_momentum", "ts_momentum", "mean_reversion", "vol_breakout",
    "macro_rotator", "regime_cs_momentum", "dual_momentum", "sharpe_momentum",
    "pairs_reversion", "sec_event_drift", "composite",
}

def _normalize_weights(w: pd.DataFrame, max_pos: float = 0.40) -> pd.DataFrame:
    clipped = w.clip(lower=-max_pos, upper=max_pos).fillna(0.0)
    gross = clipped.abs().sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / gross.replace(0.0, np.nan), 1.0)
    return clipped.mul(scale, axis=0).fillna(0.0)

def _filter_cols(closes: pd.DataFrame, pool: str) -> list[str]:
    cols = list(closes.columns)
    if pool == "etfs":
        sub = [c for c in cols if c in CORE_ETFS]
        return sub if sub else cols
    if pool == "stocks":
        sub = [c for c in cols if c not in ALL_ETFS]
        return sub if sub else cols
    return [c for c in cols if c not in INVERSE_LEV_ETFS]

def validate_spec(spec: dict) -> bool:
    if not isinstance(spec, dict):
        return False
    kind = spec.get("kind") or spec.get("type")
    if kind not in ALLOWED_KINDS:
        return False
    if kind == "composite":
        legs = spec.get("legs", [])
        return isinstance(legs, list) and 2 <= len(legs) <= 3 and all(validate_spec(l.get("spec", {})) for l in legs)
    return True

def perturb_spec(spec: dict, factor: float) -> dict:
    s = copy.deepcopy(spec)
    for k in ("lookback", "skip", "vol_window", "trend_window", "mom_window", "hedge_mom_window"):
        if k in s and isinstance(s[k], (int, float)):
            base = int(s[k])
            s[k] = max(0, int(round(base * factor))) if k == "skip" else max(3, min(252, int(round(base * factor))))
    if s.get("kind") == "composite":
        for leg in s.get("legs", []):
            leg["spec"] = perturb_spec(leg["spec"], factor)
    return s

def _pick_defensive_series(closes: pd.DataFrame, pool: list[str], mom_w: int, pure_shy: bool = False) -> np.ndarray:
    fallback = "SHY" if "SHY" in closes.columns else ""
    if pure_shy:
        return np.full(len(closes), fallback, dtype=object)
    avail = [a for a in pool if a in closes.columns]
    if not avail:
        return np.full(len(closes), fallback, dtype=object)
    arr = closes[avail].pct_change(mom_w, fill_method=None).fillna(-np.inf).values
    best_idx = np.argmax(arr, axis=1)
    best_vals = arr[np.arange(len(arr)), best_idx]
    avail_arr = np.array(avail, dtype=object)
    return np.where(best_vals > 0.0, avail_arr[best_idx], fallback)

def dsl_to_weight_fn(spec: dict, macro_df: pd.DataFrame = None, sec_df: pd.DataFrame = None):
    def weight_fn(closes: pd.DataFrame, dollar_vol: pd.DataFrame) -> pd.DataFrame:
        kind = spec.get("kind") or spec.get("type")
        max_pos = float(spec.get("max_pos_weight", 0.40))
        cols = _filter_cols(closes, spec.get("pool", "all"))
        sub_c = closes[cols]
        w_arr = np.zeros((len(closes), len(closes.columns)), dtype=float)
        col_idx = {c: i for i, c in enumerate(closes.columns)}

        if kind == "seed_benchmark":
            name = spec.get("name")
            if name == "bench_spy_hold" and "SPY" in col_idx:
                w_arr[:, col_idx["SPY"]] = 1.0
                return pd.DataFrame(w_arr, index=closes.index, columns=closes.columns)
            if name == "bench_cs_momentum":
                return dsl_to_weight_fn({"kind": "cs_momentum", "lookback": 126, "skip": 0, "top_pct": 0.2, "max_pos_weight": 0.25})(closes, dollar_vol)
            if name == "bench_st_reversal":
                return dsl_to_weight_fn({"kind": "mean_reversion", "lookback": 5, "trend_window": 200, "top_pct": 0.15, "max_pos_weight": 0.25})(closes, dollar_vol)

        elif kind in ("cs_momentum", "regime_cs_momentum"):
            lb = int(spec.get("lookback", 63))
            sk = int(spec.get("skip", 5))
            top_p = float(spec.get("top_pct", 0.20))
            allow_short = bool(spec.get("allow_short", False))
            sig = sub_c.shift(sk).pct_change(lb, fill_method=None) if sk > 0 else sub_c.pct_change(lb, fill_method=None)
            vol = sub_c.pct_change(fill_method=None).rolling(max(15, lb // 2)).std().replace(0.0, np.nan)
            rk = sig.rank(axis=1, pct=True)
            raw = ((rk >= (1.0 - top_p)).astype(float) - ((rk <= top_p).astype(float) if allow_short else 0.0)) / vol
            sub_w = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).values
            for j, c in enumerate(cols):
                w_arr[:, col_idx[c]] = sub_w[:, j]

            if kind == "regime_cs_momentum":
                tw = int(spec.get("trend_window", 40))
                vix_th = float(spec.get("vix_thresh", 23.0))
                vol_ratio_th = float(spec.get("vol_spike_ratio", 1.35))
                hr = float(spec.get("hedge_ratio", 1.0))
                hmw = int(spec.get("hedge_mom_window", 15))
                pure_shy = bool(spec.get("pure_shy", False))
                def_pool = spec.get("defensive_assets", ["SHY", "GLD", "UUP", "SH", "PSQ"])

                spy = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
                spy_r = spy.pct_change(fill_method=None)
                vol_spike = (spy_r.rolling(10).std() / spy_r.rolling(60).std().replace(0.0, np.nan)) > vol_ratio_th
                risk_off = (spy < spy.rolling(tw).mean()) | vol_spike
                if macro_df is not None and "VIXCLS" in macro_df.columns:
                    risk_off = risk_off | (macro_df["VIXCLS"].reindex(closes.index).ffill().fillna(0.0) > vix_th)

                ro_mask = risk_off.fillna(False).values
                chosen_def = _pick_defensive_series(closes, def_pool, hmw, pure_shy=pure_shy)
                w_arr[ro_mask] *= (1.0 - hr)
                for i in np.where(ro_mask)[0]:
                    da = chosen_def[i]
                    if da and da in col_idx:
                        w_arr[i, col_idx[da]] += hr

        elif kind == "dual_momentum":
            lb = int(spec.get("lookback", 21))
            top_k = int(spec.get("top_k", 3))
            use_inv = bool(spec.get("use_inverse", False))
            risk_pool = [a for a in spec.get("assets", ["SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "DBC", "UUP", "XLE", "XLV", "XLK", "XLF", "XLU", "SMH"]) if a in col_idx]
            mom_arr = closes[risk_pool].pct_change(lb, fill_method=None).fillna(-np.inf).values
            shy_arr = closes["SHY"].pct_change(lb, fill_method=None).fillna(0.0).values if "SHY" in col_idx else np.zeros(len(closes))
            def_chosen = _pick_defensive_series(closes, ["SHY", "SH", "PSQ", "UUP", "GLD"], max(10, lb), pure_shy=not use_inv)
            rp_idx = [col_idx[a] for a in risk_pool]

            for i in range(len(closes)):
                row_m = mom_arr[i]
                order = np.argsort(row_m)[::-1]
                winners = [rp_idx[j] for j in order if row_m[j] > shy_arr[i]][:top_k]
                for ci in winners:
                    w_arr[i, ci] = 1.0 / top_k
                rem = 1.0 - len(winners) / top_k
                if rem > 1e-6 and def_chosen[i] and def_chosen[i] in col_idx:
                    w_arr[i, col_idx[def_chosen[i]]] += rem

        elif kind == "sharpe_momentum":
            lb = int(spec.get("lookback", 42))
            tw = int(spec.get("trend_window", 60))
            top_p = float(spec.get("top_pct", 0.20))
            ret = sub_c.pct_change(lb, fill_method=None)
            vol = (sub_c.pct_change(fill_method=None).rolling(lb).std() * np.sqrt(lb)).replace(0.0, np.nan)
            sr = ret / vol
            uptrend = (sub_c > sub_c.rolling(tw).mean()) & (ret > 0)
            rk = sr.where(uptrend).rank(axis=1, pct=True)
            mask = (rk >= (1.0 - top_p)).astype(float)
            target_n = max(3, int(len(cols) * top_p))
            eq_w = mask.div(mask.sum(axis=1).clip(lower=target_n), axis=0).fillna(0.0)
            for j, c in enumerate(cols):
                w_arr[:, col_idx[c]] = eq_w.values[:, j]
            if "SHY" in col_idx:
                w_arr[:, col_idx["SHY"]] += np.clip(1.0 - eq_w.sum(axis=1).values, 0.0, 1.0)

        elif kind == "macro_rotator":
            vix_th = float(spec.get("vix_thresh", 22.0))
            tw = int(spec.get("trend_window", 40))
            mw = int(spec.get("mom_window", 21))
            pure_shy = bool(spec.get("pure_shy", False))
            on_assets = [a for a in spec.get("risk_on_assets", ["QQQ", "QLD", "SPY", "SMH"]) if a in col_idx]
            off_assets = [a for a in spec.get("risk_off_assets", ["SHY", "GLD", "UUP", "SH", "PSQ"]) if a in col_idx]
            spy = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
            risk_off = spy < spy.rolling(tw).mean()
            if macro_df is not None and "VIXCLS" in macro_df.columns:
                risk_off = risk_off | (macro_df["VIXCLS"].reindex(closes.index).ffill().fillna(0.0) > vix_th)
            ro_mask = risk_off.fillna(False).values
            def_chosen = _pick_defensive_series(closes, off_assets, mw, pure_shy=pure_shy)
            on_mom = closes[on_assets].pct_change(mw, fill_method=None).fillna(-np.inf).values if on_assets else np.empty((len(closes), 0))
            on_idx = [col_idx[a] for a in on_assets]

            for i in range(len(closes)):
                if ro_mask[i]:
                    da = def_chosen[i]
                    if da and da in col_idx:
                        w_arr[i, col_idx[da]] = 1.0
                elif len(on_idx) > 0:
                    order = np.argsort(on_mom[i])[::-1]
                    chosen = [on_idx[j] for j in order if on_mom[i][j] > 0.0][:2]
                    if chosen:
                        for ci in chosen:
                            w_arr[i, ci] = 1.0 / len(chosen)
                    elif "SHY" in col_idx:
                        w_arr[i, col_idx["SHY"]] = 1.0

        elif kind in ("ts_momentum", "mean_reversion", "vol_breakout", "pairs_reversion", "sec_event_drift"):
            w_df = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
            if kind == "ts_momentum":
                lb, vw = int(spec.get("lookback", 60)), int(spec.get("vol_window", 40))
                ret = sub_c.pct_change(lb, fill_method=None)
                vol = sub_c.pct_change(fill_method=None).rolling(vw).std().replace(0.0, np.nan)
                z = (ret / (vol * np.sqrt(max(lb, 1)))).clip(-2.0, 2.0)
                if not bool(spec.get("allow_short", False)):
                    z = z.clip(lower=0.0)
                raw = z / vol
                w_df[cols] = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            elif kind == "mean_reversion":
                lb, tw, top_p = int(spec.get("lookback", 5)), int(spec.get("trend_window", 100)), float(spec.get("top_pct", 0.15))
                spy = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
                bull = spy > spy.rolling(tw).mean()
                rk = sub_c.pct_change(lb, fill_method=None).where(sub_c > sub_c.rolling(tw).mean()).rank(axis=1, pct=True)
                lm = ((rk <= top_p) & bull.values[:, None]).astype(float)
                w_df[cols] = lm.div(lm.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
                if "SHY" in w_df.columns:
                    w_df.loc[~bull.fillna(False), "SHY"] = 1.0
            elif kind == "vol_breakout":
                lb = int(spec.get("lookback", 20))
                vol = sub_c.pct_change(fill_method=None).rolling(lb).std().replace(0.0, np.nan)
                sig = pd.DataFrame(np.where(sub_c > sub_c.shift(1).rolling(lb).max(), 1.0, 0.0), index=sub_c.index, columns=cols) / vol
                w_df[cols] = sig.div(sig.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            elif kind == "pairs_reversion":
                lb, ze = int(spec.get("lookback", 30)), float(spec.get("z_entry", 1.5))
                pairs = [p for p in spec.get("pairs", [["SMH", "QQQ"], ["GLD", "SLV"]]) if p[0] in closes.columns and p[1] in closes.columns]
                for a, b in pairs:
                    sp = np.log(closes[a]) - np.log(closes[b])
                    z = (sp - sp.rolling(lb).mean()) / sp.rolling(lb).std().replace(0.0, np.nan)
                    pos = np.where(z < -ze, 1.0, np.where(z > ze, -1.0, 0.0))
                    w_df[a] += pos / (2.0 * max(len(pairs), 1))
                    w_df[b] -= pos / (2.0 * max(len(pairs), 1))
            elif kind == "sec_event_drift":
                mw, top_p = int(spec.get("mom_window", 10)), float(spec.get("top_pct", 0.20))
                ev = sec_df.reindex(index=closes.index, columns=cols).fillna(0.0) if sec_df is not None else 0.0
                sc = sub_c.pct_change(mw, fill_method=None).rank(axis=1, pct=True) + 0.25 * (ev > 0).astype(float)
                m = (sc.rank(axis=1, pct=True) >= (1.0 - top_p)).astype(float)
                w_df[cols] = m.div(m.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            return _normalize_weights(w_df, max_pos=max_pos)

        elif kind == "composite":
            w_df = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
            for leg in spec.get("legs", []):
                w_df = w_df + float(leg.get("weight", 0.5)) * dsl_to_weight_fn(leg["spec"], macro_df=macro_df, sec_df=sec_df)(closes, dollar_vol)
            return _normalize_weights(w_df, max_pos=max_pos)

        return _normalize_weights(pd.DataFrame(w_arr, index=closes.index, columns=closes.columns), max_pos=max_pos)
    return weight_fn
