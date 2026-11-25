import hashlib
import json
import numpy as np
from core_kernel.db import get_active_strategies, get_total_trials_count, record_failure, save_trial
from core_kernel.engine import lockbox_walk_forward_eval, verify_no_lookahead
from discovery.dsl import dsl_to_weight_fn, perturb_spec, validate_spec

def _spec_id(spec: dict, prefix: str = "evo") -> str:
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"{prefix}_{spec.get('kind', 'strat')[:8]}_{h}"

def _random_candidate(rng: np.random.Generator) -> dict:
    kind = rng.choice([
        "regime_cs_momentum", "dual_momentum", "sharpe_momentum", "macro_rotator",
        "ts_momentum", "mean_reversion", "vol_breakout",
    ])
    if kind == "regime_cs_momentum":
        return {"kind": "regime_cs_momentum", "pool": str(rng.choice(["all", "stocks", "etfs"])),
                "lookback": int(rng.choice([21, 42, 63, 90, 126])), "skip": int(rng.choice([0, 5])),
                "top_pct": float(rng.choice([0.15, 0.20, 0.25])), "trend_window": int(rng.choice([60, 100, 150])),
                "vix_thresh": float(rng.choice([21.0, 24.0, 28.0])), "hedge_ratio": float(rng.choice([0.75, 1.0])),
                "hedge_mom_window": int(rng.choice([10, 21, 42])),
                "defensive_assets": ["SHY", "GLD", "UUP", "SH", "PSQ", "QID"], "max_pos_weight": 0.45}
    if kind == "dual_momentum":
        return {"kind": "dual_momentum", "lookback": int(rng.choice([21, 42, 63, 90, 126])),
                "top_k": int(rng.choice([2, 3, 4])), "use_inverse": bool(rng.choice([True, False])),
                "max_pos_weight": 0.50}
    if kind == "sharpe_momentum":
        return {"kind": "sharpe_momentum", "pool": str(rng.choice(["all", "stocks", "etfs"])),
                "lookback": int(rng.choice([21, 42, 63, 90])), "trend_window": int(rng.choice([50, 100, 150])),
                "top_pct": float(rng.choice([0.15, 0.20, 0.25])), "max_pos_weight": 0.45}
    if kind == "macro_rotator":
        return {"kind": "macro_rotator", "vix_thresh": float(rng.choice([20.0, 23.0, 26.0])),
                "trend_window": int(rng.choice([60, 100, 150])), "mom_window": int(rng.choice([21, 35, 50])),
                "risk_on_assets": ["QQQ", "QLD", "SSO", "SMH", "NVDA", "SPY"],
                "risk_off_assets": ["SHY", "GLD", "UUP", "SH", "PSQ", "QID", "SDS"], "max_pos_weight": 0.50}
    if kind == "ts_momentum":
        return {"kind": "ts_momentum", "pool": "etfs",
                "lookback": int(rng.choice([15, 30, 60, 90])), "vol_window": int(rng.choice([20, 40])),
                "allow_short": bool(rng.choice([True, False])), "max_pos_weight": 0.30}
    if kind == "mean_reversion":
        return {"kind": "mean_reversion", "pool": str(rng.choice(["all", "stocks"])),
                "lookback": int(rng.choice([3, 5, 8])), "trend_window": int(rng.choice([80, 120, 160])),
                "top_pct": float(rng.choice([0.15, 0.20])), "max_pos_weight": 0.40}
    return {"kind": "vol_breakout", "pool": str(rng.choice(["all", "stocks"])),
            "lookback": int(rng.choice([15, 25, 45])), "max_pos_weight": 0.25}

def evaluate_and_register_spec(
    spec: dict, origin: str, closes, dollar_vol, macro_df, sec_df, n_trials: int, regime_tag: str
) -> tuple[bool, dict]:
    if not validate_spec(spec):
        return False, {"error": "invalid_dsl"}
    sid = _spec_id(spec, prefix=origin[:4])
    fn = dsl_to_weight_fn(spec, macro_df=macro_df, sec_df=sec_df)
    verify_no_lookahead(closes, dollar_vol, fn)
    _, metrics, _ = lockbox_walk_forward_eval(closes, dollar_vol, fn, n_trials=n_trials)

    plateau_lws = [metrics["pre_log_wealth"]]
    for pf in (0.8, 1.2):
        p_fn = dsl_to_weight_fn(perturb_spec(spec, pf), macro_df=macro_df, sec_df=sec_df)
        _, p_m, _ = lockbox_walk_forward_eval(closes, dollar_vol, p_fn, n_trials=n_trials)
        plateau_lws.append(p_m["pre_log_wealth"])
    plateau_ok = sum(1 for x in plateau_lws if x > 0.0) >= 2 and float(np.median(plateau_lws)) > 0.01

    passed = bool(
        metrics["pre_log_wealth"] > 0.01
        and metrics["deflated_sharpe"] > 0.50
        and metrics["lockbox_log_wealth"] > 0.0
        and plateau_ok
    )
    metrics["plateau_ok"] = plateau_ok
    save_trial(sid, origin, spec, metrics, n_trials, passed)
    if not passed:
        reason = f"pre_lw={metrics['pre_log_wealth']:.3f}, lock_lw={metrics['lockbox_log_wealth']:.3f}, dsr={metrics['deflated_sharpe']:.2f}, plateau={plateau_ok}"
        record_failure(sid, origin, regime_tag, reason, f"Failed {spec.get('kind')} under {regime_tag}: {reason}")
    return passed, {"strategy_id": sid, **metrics, "passed": passed}

def run_mathematical_evolution(ohlcv: dict, macro_df, sec_df, regime_tag: str, n_candidates: int = 80) -> list[dict]:
    closes, dollar_vol = ohlcv["close"], ohlcv["dollar_vol"]
    n_candidates = max(int(n_candidates), 80)
    rng = np.random.default_rng()
    active = [a["spec"] for a in get_active_strategies() if validate_spec(a["spec"]) and a["spec"].get("kind")]

    candidates = []
    for _ in range(n_candidates):
        if active and rng.random() < 0.35:
            parent = active[int(rng.integers(0, len(active)))]
            candidates.append(perturb_spec(parent, float(rng.choice([0.80, 0.90, 1.10, 1.25]))))
        else:
            candidates.append(_random_candidate(rng))

    if len(active) >= 2:
        c1, c2 = active[0], active[-1]
        if c1.get("kind") != "composite" and c2.get("kind") != "composite":
            candidates.append({"kind": "composite", "legs": [{"weight": 0.5, "spec": c1}, {"weight": 0.5, "spec": c2}], "max_pos_weight": 0.35})

    n_trials = get_total_trials_count()
    results = []
    for spec in candidates:
        n_trials += 1
        _, res = evaluate_and_register_spec(spec, "evo_math", closes, dollar_vol, macro_df, sec_df, n_trials, regime_tag)
        results.append(res)
    return results
