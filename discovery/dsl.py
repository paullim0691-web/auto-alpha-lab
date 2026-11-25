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

def _normalize_weights(w: pd.DataFrame, max_pos: float = 0.35) -> pd.DataFrame:
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

def _pick_defensive(closes: pd.DataFrame, dt, pool: list[str], mom_w: int) -> str:
    avail = [a for a in pool if a in closes.columns]
    if not avail:
        return "SHY" if "SHY" in closes.columns else closes.columns[0]
    mom = closes[avail].pct_change(mom_w, fill_method=None).loc[dt].dropna()
    pos = mom[mom > 0]
    if len(pos) > 0:
        return str(pos.idxmax())
    return "SHY" if "SHY" in closes.columns else avail[0]

def dsl_to_weight_fn(spec: dict, macro_df: pd.DataFrame = None, sec_df: pd.DataFrame = None):
    def weight_fn(closes: pd.DataFrame, dollar_vol: pd.DataFrame) -> pd.DataFrame:
        kind = spec.get("kind") or spec.get("type")
        max_pos = float(spec.get("max_pos_weight", 0.35))
        cols = _filter_cols(closes, spec.get("pool", "all"))
        sub_c = closes[cols]
        w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

        if kind == "seed_benchmark":
            name = spec.get("name")
            if name == "bench_spy_hold" and "SPY" in closes.columns:
                w["SPY"] = 1.0
                return w
            if name == "bench_cs_momentum":
                return dsl_to_weight_fn({"kind": "cs_momentum", "lookback": 126, "skip": 0, "top_pct": 0.2, "allow_short": False, "max_pos_weight": 0.25})(closes, dollar_vol)
            if name == "bench_st_reversal":
                return dsl_to_weight_fn({"kind": "mean_reversion", "lookback": 5, "trend_window": 200, "top_pct": 0.15, "allow_short": False, "max_pos_weight": 0.25})(closes, dollar_vol)

        elif kind in ("cs_momentum", "regime_cs_momentum"):
            lb = int(spec.get("lookback", 63))
            sk = int(spec.get("skip", 5))
            top_p = float(spec.get("top_pct", 0.20))
            allow_short = bool(spec.get("allow_short", False))
            sig = sub_c.shift(sk).pct_change(lb, fill_method=None) if sk > 0 else sub_c.pct_change(lb, fill_method=None)
            vol = sub_c.pct_change(fill_method=None).rolling(max(20, lb // 2)).std().replace(0.0, np.nan)
            rk = sig.rank(axis=1, pct=True)
            raw = ((rk >= (1.0 - top_p)).astype(float) - ((rk <= top_p).astype(float) if allow_short else 0.0)) / vol
            w[cols] = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

            if kind == "regime_cs_momentum":
                tw = int(spec.get("trend_window", 100))
                vix_th = float(spec.get("vix_thresh", 24.0))
                hr = float(spec.get("hedge_ratio", 1.0))
                hmw = int(spec.get("hedge_mom_window", 21))
                def_pool = spec.get("defensive_assets", ["SHY", "GLD", "UUP", "SH", "PSQ"])
                spy = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
                spy_bad = spy < spy.rolling(tw).mean()
                if macro_df is not None and "VIXCLS" in macro_df.columns:
                    vix_bad = macro_df["VIXCLS"].reindex(closes.index).ffill().fillna(0.0) > vix_th
                    risk_off = spy_bad | vix_bad
                else:
                    risk_off = spy_bad
                for dt in closes.index[risk_off.fillna(False)]:
                    w.loc[dt] = w.loc[dt] * (1.0 - hr)
                    da = _pick_defensive(closes, dt, def_pool, hmw)
                    w.loc[dt, da] = w.loc[dt, da] + hr

        elif kind == "dual_momentum":
            lb = int(spec.get("lookback", 63))
            top_k = int(spec.get("top_k", 3))
            use_inv = bool(spec.get("use_inverse", True))
            risk_pool = [a for a in spec.get("assets", ["SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "DBC", "UUP", "XLE", "XLV", "XLK", "XLF", "XLU", "SMH"]) if a in closes.columns]
            mom = closes[risk_pool].pct_change(lb, fill_method=None)
            shy_ret = closes["SHY"].pct_change(lb, fill_method=None) if "SHY" in closes.columns else 0.0
            def_pool = ["SHY", "SH", "PSQ", "UUP", "GLD"] if use_inv else ["SHY"]
            for dt in closes.index:
                s = mom.loc[dt].dropna()
                th = float(shy_ret.loc[dt]) if isinstance(shy_ret, pd.Series) and pd.notna(shy_ret.loc[dt]) else 0.0
                winners = s[s > th].nlargest(top_k).index.tolist()
                for c in winners:
                    w.loc[dt, c] = 1.0 / top_k
                rem = 1.0 - len(winners) / top_k
                if rem > 1e-6:
                    da = _pick_defensive(closes, dt, def_pool, max(10, lb // 2))
                    w.loc[dt, da] = w.loc[dt, da] + rem

        elif kind == "sharpe_momentum":
            lb = int(spec.get("lookback", 63))
            tw = int(spec.get("trend_window", 100))
            top_p = float(spec.get("top_pct", 0.20))
            ret = sub_c.pct_change(lb, fill_method=None)
            vol = (sub_c.pct_change(fill_method=None).rolling(lb).std() * np.sqrt(lb)).replace(0.0, np.nan)
            sr = ret / vol
            uptrend = (sub_c > sub_c.rolling(tw).mean()) & (ret > 0)
            rk = sr.where(uptrend).rank(axis=1, pct=True)
            mask = (rk >= (1.0 - top_p)).astype(float)
            cnt = mask.sum(axis=1)
            target_n = max(3, int(len(cols) * top_p))
            eq_w = mask.div(cnt.clip(lower=target_n), axis=0).fillna(0.0)
            w[cols] = eq_w
            if "SHY" in closes.columns:
                w["SHY"] = w["SHY"] + (1.0 - eq_w.sum(axis=1)).clip(lower=0.0)

        elif kind == "ts_momentum":
            lb = int(spec.get("lookback", 60))
            vw = int(spec.get("vol_window", 40))
            allow_short = bool(spec.get("allow_short", False))
            ret = sub_c.pct_change(lb, fill_method=None)
            vol = sub_c.pct_change(fill_method=None).rolling(vw).std().replace(0.0, np.nan)
            z = (ret / (vol * np.sqrt(max(lb, 1)))).clip(-2.0, 2.0)
            if not allow_short:
                z = z.clip(lower=0.0)
            raw = z / vol
            w[cols] = raw.div(raw.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        elif kind == "mean_reversion":
            lb = int(spec.get("lookback", 5))
            tw = int(spec.get("trend_window", 150))
            top_p = float(spec.get("top_pct", 0.15))
            r_short = sub_c.pct_change(lb, fill_method=None)
            sma = sub_c.rolling(tw).mean()
            spy = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
            bull_mkt = spy > spy.rolling(tw).mean()
            long_rk = r_short.where(sub_c > sma).rank(axis=1, pct=True)
            long_m = ((long_rk <= top_p) & bull_mkt.values[:, None]).astype(float)
            w[cols] = long_m.div(long_m.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            if "SHY" in closes.columns:
                w.loc[~bull_mkt.fillna(False), "SHY"] = 1.0

        elif kind == "vol_breakout":
            lb = int(spec.get("lookback", 20))
            roll_hi = sub_c.shift(1).rolling(lb).max()
            vol = sub_c.pct_change(fill_method=None).rolling(lb).std().replace(0.0, np.nan)
            sig = pd.DataFrame(np.where(sub_c > roll_hi, 1.0, 0.0), index=sub_c.index, columns=cols) / vol
            w[cols] = sig.div(sig.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        elif kind == "macro_rotator":
            vix_th = float(spec.get("vix_thresh", 23.0))
            tw = int(spec.get("trend_window", 100))
            mw = int(spec.get("mom_window", 30))
            on_assets = [a for a in spec.get("risk_on_assets", ["QQQ", "QLD", "SPY", "SMH"]) if a in closes.columns]
            off_assets = [a for a in spec.get("risk_off_assets", ["SHY", "GLD", "UUP", "SH", "PSQ"]) if a in closes.columns]
            spy = closes["SPY"] if "SPY" in closes.columns else closes.iloc[:, 0]
            risk_off = spy < spy.rolling(tw).mean()
            if macro_df is not None and "VIXCLS" in macro_df.columns:
                risk_off = risk_off | (macro_df["VIXCLS"].reindex(closes.index).ffill().fillna(0.0) > vix_th)
            mom = closes.pct_change(mw, fill_method=None)
            for dt in closes.index:
                if bool(risk_off.loc[dt]):
                    da = _pick_defensive(closes, dt, off_assets, mw)
                    w.loc[dt, da] = 1.0
                elif on_assets:
                    sc = mom.loc[dt, on_assets].dropna()
                    pos = sc[sc > 0].nlargest(min(2, len(sc)))
                    chosen = pos.index.tolist() if len(pos) > 0 else ["SHY" if "SHY" in closes.columns else on_assets[0]]
                    for c in chosen:
                        w.loc[dt, c] = 1.0 / len(chosen)

        elif kind == "pairs_reversion":
            lb = int(spec.get("lookback", 30))
            ze = float(spec.get("z_entry", 1.5))
            pairs = [p for p in spec.get("pairs", [["SMH", "QQQ"], ["GLD", "SLV"]]) if p[0] in closes.columns and p[1] in closes.columns]
            for a, b in pairs:
                spread = np.log(closes[a]) - np.log(closes[b])
                z = (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std().replace(0.0, np.nan)
                pos = np.where(z < -ze, 1.0, np.where(z > ze, -1.0, 0.0))
                w[a] += pos / (2.0 * max(len(pairs), 1))
                w[b] -= pos / (2.0 * max(len(pairs), 1))

        elif kind == "sec_event_drift":
            mw = int(spec.get("mom_window", 10))
            top_p = float(spec.get("top_pct", 0.20))
            ev = sec_df.reindex(index=closes.index, columns=cols).fillna(0.0) if sec_df is not None else 0.0
            score = sub_c.pct_change(mw, fill_method=None).rank(axis=1, pct=True) + 0.25 * (ev > 0).astype(float)
            mask = (score.rank(axis=1, pct=True) >= (1.0 - top_p)).astype(float)
            w[cols] = mask.div(mask.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        elif kind == "composite":
            for leg in spec.get("legs", []):
                w = w + float(leg.get("weight", 0.5)) * dsl_to_weight_fn(leg["spec"], macro_df=macro_df, sec_df=sec_df)(closes, dollar_vol)

        return _normalize_weights(w, max_pos=max_pos)
    return weight_fn
