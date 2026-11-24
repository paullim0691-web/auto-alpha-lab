import copy
import numpy as np
import pandas as pd

ETF_SET = {
    "SPY", "QQQ", "IWM", "DIA", "MDY", "MTUM", "VLUE", "QUAL", "USMV", "ARKK",
    "QLD", "SSO", "SH", "PSQ", "TBF", "SDS", "QID", "TLT", "IEF", "SHY", "HYG", "LQD",
    "GLD", "SLV", "GDX", "USO", "UNG", "DBC", "DBA", "UUP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "SMH", "XBI", "KRE", "ITA",
}

ALLOWED_KINDS = {
    "seed_benchmark", "cs_momentum", "ts_momentum", "mean_reversion",
    "vol_breakout", "macro_rotator", "pairs_reversion", "sec_event_drift", "composite",
}

def _normalize_weights(w: pd.DataFrame, max_pos: float = 0.25) -> pd.DataFrame:
    clipped = w.clip(lower=-max_pos, upper=max_pos).fillna(0.0)
    gross = clipped.abs().sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / gross.replace(0.0, np.nan), 1.0)
    return clipped.mul(scale, axis=0).fillna(0.0)

def _filter_cols(closes: pd.DataFrame, pool: str) -> list[str]:
    cols = list(closes.columns)
    if pool == "etfs":
        sub = [c for c in cols if c in ETF_SET]
        return sub if sub else cols
    if pool == "stocks":
        sub = [c for c in cols if c not in ETF_SET]
        return sub if sub else cols
    return cols

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
    for k in ("lookback", "skip", "vol_window", "trend_window", "atr_window", "mom_window"):
        if k in s and isinstance(s[k], (int, float)):
            base = int(s[k])
            if k == "skip":
                s[k] = max(0, int(round(base * factor)))
            else:
                s[k] = max(3, min(252, int(round(base * factor))))
    if s.get("kind") == "composite":
        for leg in s.get("legs", []):
            leg["spec"] = perturb_spec(leg["spec"], factor)
    return s

def dsl_to_weight_fn(spec: dict, macro_df: pd.DataFrame = None, sec_df: pd.DataFrame = None):
    def weight_fn(closes: pd.DataFrame, dollar_vol: pd.DataFrame) -> pd.DataFrame:
        kind = spec.get("kind") or spec.get("type")
        max_pos = float(spec.get("max_pos_weight", 0.25))
        pool = spec.get("pool", "all")
        cols = _filter_cols(closes, pool)
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

        elif kind == "cs_momentum":
            lb = int(spec.get("lookback", 63))
            sk = int(spec.get("skip", 5))
            top_p = float(spec.get("top_pct", 0.20))
            allow_short = bool(spec.get("allow_short", False))
            sig = sub_c.shift(sk).pct_change(lb, fill_method=None) if sk > 0 else sub_c.pct_change(lb, fill_method=None)
            vol = sub_c.pct_change(fill_method=None).rolling(max(20, lb // 2)).std().replace(0.0, np.nan)
            inv_v = (1.0 / vol).clip(upper=200.0)
            rk = sig.rank(axis=1, pct=True)
            long_m = (rk >= (1.0 - top_p)).astype(float)
            short_m = -(rk <= top_p).astype(float) if allow_short else 0.0
            raw = (long_m + short_m) * inv_v
            row_abs = raw.abs().sum(axis=1).replace(0.0, np.nan)
            w[cols] = raw.div(row_abs, axis=0).fillna(0.0)

        elif kind == "ts_momentum":
            lb = int(spec.get("lookback", 60))
            vw = int(spec.get("vol_window", 40))
            allow_short = bool(spec.get("allow_short", True))
            ret = sub_c.pct_change(lb, fill_method=None)
            vol = sub_c.pct_change(fill_method=None).rolling(vw).std().replace(0.0, np.nan)
            z = (ret / (vol * np.sqrt(max(lb, 1)))).clip(-2.0, 2.0)
            if not allow_short:
                z = z.clip(lower=0.0)
            raw = z / vol
            row_abs = raw.abs().sum(axis=1).replace(0.0, np.nan)
            w[cols] = raw.div(row_abs, axis=0).fillna(0.0)

        elif kind == "mean_reversion":
            lb = int(spec.get("lookback", 5))
            tw = int(spec.get("trend_window", 150))
            top_p = float(spec.get("top_pct", 0.15))
            allow_short = bool(spec.get("allow_short", False))
            r_short = sub_c.pct_change(lb, fill_method=None)
            sma = sub_c.rolling(tw).mean()
            long_rk = r_short.where(sub_c > sma).rank(axis=1, pct=True)
            long_m = (long_rk <= top_p).astype(float)
            short_m = 0.0
            if allow_short:
                short_rk = r_short.where(sub_c < sma).rank(axis=1, pct=True)
                short_m = -(short_rk >= (1.0 - top_p)).astype(float)
            raw = long_m + short_m
            row_abs = raw.abs().sum(axis=1).replace(0.0, np.nan)
            w[cols] = raw.div(row_abs, axis=0).fillna(0.0)

        elif kind == "vol_breakout":
            lb = int(spec.get("lookback", 20))
            allow_short = bool(spec.get("allow_short", False))
            roll_hi = sub_c.shift(1).rolling(lb).max()
            roll_lo = sub_c.shift(1).rolling(lb).min()
            vol = sub_c.pct_change(fill_method=None).rolling(lb).std().replace(0.0, np.nan)
            sig = np.where(sub_c > roll_hi, 1.0, np.where((sub_c < roll_lo) & allow_short, -1.0, 0.0))
            raw = pd.DataFrame(sig, index=sub_c.index, columns=cols) / vol
            row_abs = raw.abs().sum(axis=1).replace(0.0, np.nan)
            w[cols] = raw.div(row_abs, axis=0).fillna(0.0)

        elif kind == "macro_rotator":
            vix_th = float(spec.get("vix_thresh", 24.0))
            curve_th = float(spec.get("curve_thresh", 0.0))
            mw = int(spec.get("mom_window", 42))
            on_assets = [a for a in spec.get("risk_on_assets", ["QQQ", "QLD", "SPY", "SMH"]) if a in closes.columns]
            off_assets = [a for a in spec.get("risk_off_assets", ["TLT", "GLD", "UUP", "SH"]) if a in closes.columns]
            if macro_df is not None and not macro_df.empty and "VIXCLS" in macro_df.columns:
                m = macro_df.reindex(closes.index).ffill().fillna(0.0)
                vix = m["VIXCLS"]
                curve = m.get("DGS10", 0.0) - m.get("DGS2", 0.0)
                risk_off = (vix > vix_th) | (curve < curve_th)
            else:
                spy_vol = closes.get("SPY", closes.iloc[:, 0]).pct_change(fill_method=None).rolling(20).std() * np.sqrt(252) * 100
                risk_off = spy_vol > vix_th
            mom = closes.pct_change(mw, fill_method=None)
            for dt in closes.index:
                pool_a = off_assets if bool(risk_off.loc[dt]) else on_assets
                if not pool_a:
                    continue
                scores = mom.loc[dt, pool_a].dropna()
                pos = scores[scores > 0]
                chosen = pos.nlargest(min(2, len(pos))).index.tolist() if len(pos) > 0 else pool_a[:1]
                for c in chosen:
                    w.loc[dt, c] = 1.0 / len(chosen)

        elif kind == "pairs_reversion":
            lb = int(spec.get("lookback", 30))
            ze = float(spec.get("z_entry", 1.5))
            pairs = spec.get("pairs", [["SMH", "QQQ"], ["XLE", "USO"], ["GLD", "SLV"], ["HYG", "LQD"]])
            valid_pairs = [(a, b) for a, b in pairs if a in closes.columns and b in closes.columns]
            for a, b in valid_pairs:
                spread = np.log(closes[a]) - np.log(closes[b])
                z = (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std().replace(0.0, np.nan)
                pos = np.where(z < -ze, 1.0, np.where(z > ze, -1.0, 0.0))
                w[a] += pos / (2.0 * max(len(valid_pairs), 1))
                w[b] -= pos / (2.0 * max(len(valid_pairs), 1))

        elif kind == "sec_event_drift":
            mw = int(spec.get("mom_window", 10))
            top_p = float(spec.get("top_pct", 0.20))
            r_m = sub_c.pct_change(mw, fill_method=None)
            ev = sec_df.reindex(index=closes.index, columns=cols).fillna(0.0) if sec_df is not None else 0.0
            score = r_m.rank(axis=1, pct=True) + 0.25 * (ev > 0).astype(float)
            mask = (score.rank(axis=1, pct=True) >= (1.0 - top_p)).astype(float)
            row_s = mask.sum(axis=1).replace(0.0, np.nan)
            w[cols] = mask.div(row_s, axis=0).fillna(0.0)

        elif kind == "composite":
            for leg in spec.get("legs", []):
                lw = float(leg.get("weight", 0.5))
                sub_fn = dsl_to_weight_fn(leg["spec"], macro_df=macro_df, sec_df=sec_df)
                w = w + lw * sub_fn(closes, dollar_vol)

        return _normalize_weights(w, max_pos=max_pos)
    return weight_fn
