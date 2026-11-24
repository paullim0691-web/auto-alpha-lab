import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

def simulate_weights(
    closes: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    target_weights: pd.DataFrame,
    base_cost_bps: float = 5.0,
    borrow_annual_bps: float = 60.0,
    portfolio_notional: float = 100_000.0,
) -> pd.Series:
    asset_rets = closes.pct_change(fill_method=None).fillna(0.0)
    aligned_w = target_weights.reindex_like(closes).fillna(0.0)
    exec_w = aligned_w.shift(1).fillna(0.0)
    gross_ret = (exec_w * asset_rets).sum(axis=1)

    turnover = (exec_w - exec_w.shift(1).fillna(0.0)).abs()
    trade_dollars = turnover * portfolio_notional
    dv_safe = dollar_vol.reindex_like(closes).replace(0.0, np.nan)
    impact_bps = (10.0 * np.sqrt((trade_dollars / dv_safe).fillna(0.0).clip(upper=0.05))) * 100.0
    total_cost = (turnover * ((base_cost_bps + impact_bps) * 1e-4)).sum(axis=1)

    short_exposure = exec_w.clip(upper=0.0).abs().sum(axis=1)
    borrow_cost = short_exposure * (borrow_annual_bps * 1e-4 / 252.0)
    net_ret = gross_ret - total_cost - borrow_cost
    return net_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def verify_no_lookahead(closes: pd.DataFrame, dollar_vol: pd.DataFrame, weight_fn) -> bool:
    w_orig = weight_fn(closes, dollar_vol)
    shocked = closes.copy()
    cutoff = len(shocked) - 10
    shocked.iloc[cutoff:] = shocked.iloc[cutoff:] * 1.50
    w_shocked = weight_fn(shocked, dollar_vol)
    diff = (w_orig.iloc[:cutoff] - w_shocked.iloc[:cutoff]).abs().max().max()
    if diff > 1e-12:
        raise AssertionError(f"LOOKAHEAD BIAS DETECTED! Pre-cutoff weights changed by {diff}")
    return True

def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> float:
    r = returns.dropna()
    n = len(r)
    if n < 60 or r.std() < 1e-12:
        return 0.0
    sr = (r.mean() / r.std()) * math.sqrt(252.0)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0
    n_eff = max(int(n_trials), 1)
    euler_gamma = 0.5772156649
    if n_eff == 1:
        sr_benchmark = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_eff)
        z2 = norm.ppf(1.0 - 1.0 / (n_eff * math.e))
        sr_benchmark = ((1.0 - euler_gamma) * z1 + euler_gamma * z2) * math.sqrt(1.0 / max(n / 252.0, 1.0))
    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom_sq <= 1e-9:
        return 0.0
    stat = ((sr - sr_benchmark) * math.sqrt(n - 1)) / math.sqrt(denom_sq)
    return float(norm.cdf(stat))

def compute_metrics(returns: pd.Series, n_trials: int) -> dict:
    r = returns.dropna()
    if len(r) < 10:
        return {"log_wealth": 0.0, "cagr": 0.0, "sharpe": 0.0, "deflated_sharpe": 0.0, "max_drawdown": 0.0}
    safe_r = r.clip(lower=-0.99)
    equity = (1.0 + safe_r).cumprod()
    log_wealth = float(np.log(equity.iloc[-1]))
    years = max(len(r) / 252.0, 1e-6)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    sharpe = float((r.mean() / r.std()) * math.sqrt(252.0)) if r.std() > 1e-12 else 0.0
    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min())
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials)
    return {"log_wealth": log_wealth, "cagr": cagr, "sharpe": sharpe, "deflated_sharpe": dsr, "max_drawdown": mdd}

def lockbox_walk_forward_eval(
    closes: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    weight_fn,
    n_trials: int,
    lockbox_days: int = 63,
    train_days: int = 252,
    test_days: int = 63,
    embargo_days: int = 5,
) -> tuple[pd.Series, dict, dict]:
    """
    데이터 마지막 63거래일(Lockbox)을 탐색 루프에서 완전 격리.
    앞구간 Walk-Forward + 최종 맹검 Lockbox를 모두 계산.
    """
    weights = weight_fn(closes, dollar_vol)
    full_net_ret = simulate_weights(closes, dollar_vol, weights)
    avg_gross = float(weights.iloc[train_days:].abs().sum(axis=1).mean())

    pre_end = len(full_net_ret) - lockbox_days
    oos_chunks = []
    idx = train_days + embargo_days
    while idx < pre_end:
        end_idx = min(idx + test_days, pre_end)
        oos_chunks.append(full_net_ret.iloc[idx:end_idx])
        idx += test_days

    pre_oos = pd.concat(oos_chunks) if oos_chunks else full_net_ret.iloc[train_days:pre_end]
    lockbox_ret = full_net_ret.iloc[pre_end:]
    combined_oos = pd.concat([pre_oos, lockbox_ret])

    pre_metrics = compute_metrics(pre_oos, n_trials=n_trials)
    lock_metrics = compute_metrics(lockbox_ret, n_trials=1)
    comb_metrics = compute_metrics(combined_oos, n_trials=n_trials)
    comb_metrics["avg_gross_exposure"] = avg_gross
    comb_metrics["pre_log_wealth"] = pre_metrics["log_wealth"]
    comb_metrics["lockbox_log_wealth"] = lock_metrics["log_wealth"]
    comb_metrics["lockbox_cagr"] = lock_metrics["cagr"]
    return combined_oos, comb_metrics, lock_metrics

def bootstrapped_kelly_allocate(
    strategy_oos_returns: pd.DataFrame,
    strategy_gross_map: dict,
    max_gross_leverage: float = 1.80,
    n_boot: int = 40,
    block_len: int = 15,
) -> dict:
    """
    단일 전략 몰빵(2.0x) 오류를 제거한 블록 부트스트랩 하한 로그-부(Log-Wealth) 켈리 최적화.
    실제 포트폴리오 총 노출(Underlying Gross Exposure)이 Alpaca 마진 한도(1.80x) 이하가 되도록 제약.
    """
    cols = list(strategy_oos_returns.columns)
    R = strategy_oos_returns.dropna().values
    t_len, n_strat = R.shape
    if t_len < 60 or n_strat == 0:
        return {c: 0.0 for c in cols}

    rng = np.random.default_rng(42)
    n_blocks = max(1, t_len // block_len)
    boot_samples = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, t_len - block_len), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:t_len]
        boot_samples.append(R[idx])

    gross_vec = np.array([max(0.25, float(strategy_gross_map.get(c, 1.0))) for c in cols])

    def obj_fn(w: np.ndarray) -> float:
        lws = []
        for B in boot_samples:
            pr = B @ w
            if np.any(1.0 + pr <= 0.05):
                return 1e6 + float(np.sum(np.maximum(0.05 - (1.0 + pr), 0.0)) * 1e4)
            lws.append(float(np.mean(np.log(1.0 + pr))))
        arr = np.array(lws)
        # 기대 복리성장률 - 0.5 * 부트스트랩 표준편차 (추정 불확실성 방어 + 분산 유도)
        return -(float(arr.mean()) - 0.50 * float(arr.std()))

    w0 = np.full(n_strat, min(0.8, max_gross_leverage) / float(np.sum(gross_vec)))
    bounds = [(0.0, max_gross_leverage / max(g, 1.0)) for g in gross_vec]
    constraints = [{"type": "ineq", "fun": lambda w: max_gross_leverage - float(np.sum(w * gross_vec))}]
    res = minimize(obj_fn, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 250, "ftol": 1e-8})
    if not res.success:
        return {c: 0.0 for c in cols}
    raw_w = {c: (round(float(val), 4) if float(val) > 0.02 else 0.0) for c, val in zip(cols, res.x)}
    return raw_w

def cusum_alpha_status(returns: pd.Series, expected_daily_mean: float) -> tuple[float, bool]:
    """교정된 하방 CUSUM 알파 사망 감지기 (k=0.50). 임계치 6.0 초과 AND 최근 60일 평균 음수일 때 퇴출 판정"""
    r = returns.dropna().tail(60)
    if len(r) < 20 or r.std() < 1e-12:
        return 0.0, False
    z = (r - max(expected_daily_mean, 0.0)) / r.std()
    s = 0.0
    for val in z:
        s = min(0.0, s + float(val) + 0.50)
    stat = round(abs(s), 4)
    decayed = bool(stat > 6.0 and float(r.mean()) < 0.0)
    return stat, decayed
