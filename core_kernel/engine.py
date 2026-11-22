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
    """
    엄격한 인과적(Causal) 시뮬레이터:
    t일 장 마감 기준 산출된 비중 target_weights[t]는 t+1일 수익률에만 적용됨(.shift(1)).
    거래비용 + 공매도 차입비 + 거래대금 대비 슬리피지 충격 비용을 차감한 순수익률 반환.
    """
    asset_rets = closes.pct_change(fill_method=None).fillna(0.0)
    aligned_w = target_weights.reindex_like(closes).fillna(0.0)

    # 미래 참조 차단: 반드시 1영업일 시프트 후 수익률 곱셈
    exec_w = aligned_w.shift(1).fillna(0.0)
    gross_ret = (exec_w * asset_rets).sum(axis=1)

    # 회전율 및 변동 슬리피지 계산
    turnover = (exec_w - exec_w.shift(1).fillna(0.0)).abs()
    trade_dollars = turnover * portfolio_notional
    dv_safe = dollar_vol.reindex_like(closes).replace(0.0, np.nan)
    impact_bps = (10.0 * np.sqrt((trade_dollars / dv_safe).fillna(0.0).clip(upper=0.05))) * 100.0
    total_cost = (turnover * ((base_cost_bps + impact_bps) * 1e-4)).sum(axis=1)

    # 공매도 포지션 차입 비용 차감
    short_exposure = exec_w.clip(upper=0.0).abs().sum(axis=1)
    borrow_cost = short_exposure * (borrow_annual_bps * 1e-4 / 252.0)

    net_ret = gross_ret - total_cost - borrow_cost
    return net_ret.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def verify_no_lookahead(closes: pd.DataFrame, dollar_vol: pd.DataFrame, weight_fn) -> bool:
    """
    미래 데이터 차단 자체 감사:
    마지막 10거래일에 인위적 폭등(+50%)을 주입했을 때,
    그 직전 날짜까지의 시그널 비중이 1e-12 이상 변하면 미래 참조 버그로 판정하고 중단.
    """
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
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio"""
    r = returns.dropna()
    n = len(r)
    if n < 60 or r.std() == 0:
        return 0.0

    sr = (r.mean() / r.std()) * math.sqrt(252.0)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # excess -> normal kurtosis

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
    if len(r) == 0:
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

    return {
        "log_wealth": log_wealth,
        "cagr": cagr,
        "sharpe": sharpe,
        "deflated_sharpe": dsr,
        "max_drawdown": mdd,
    }


def purged_walk_forward_eval(
    closes: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    weight_fn,
    n_trials: int,
    train_days: int = 252,
    test_days: int = 63,
    embargo_days: int = 5,
) -> tuple[pd.Series, dict]:
    """
    표본 외(OOS) 전진 검증 구간만 이어 붙여 임의 과최적화 없는 실제 성적 계산.
    """
    weights = weight_fn(closes, dollar_vol)
    full_net_ret = simulate_weights(closes, dollar_vol, weights)

    oos_chunks = []
    idx = train_days + embargo_days
    while idx < len(full_net_ret):
        end_idx = min(idx + test_days, len(full_net_ret))
        oos_chunks.append(full_net_ret.iloc[idx:end_idx])
        idx += test_days

    oos_ret = pd.concat(oos_chunks) if oos_chunks else full_net_ret.iloc[train_days:]
    metrics = compute_metrics(oos_ret, n_trials=n_trials)
    return oos_ret, metrics


def shrinkage_kelly_allocate(
    strategy_oos_returns: pd.DataFrame,
    max_gross_leverage: float = 2.0,
) -> dict:
    """
    무(無)임의상수 기대 로그 부(Log-Wealth) 극대화 비중 최적화:
    과거 추정 불확실성(표준오차)만큼 기대수익률을 축소한 뒤
    표본 외 기하평균 성장률 E[ln(1 + w^T r)]을 최대화하는 전략 가중치 도출.
    """
    cols = list(strategy_oos_returns.columns)
    R = strategy_oos_returns.dropna().values
    t_len, n_strat = R.shape
    if t_len < 60 or n_strat == 0:
        return {c: 0.0 for c in cols}

    # Bayesian Shrinkage: 표본 오차만큼 평균을 0 방향으로 수축
    means = R.mean(axis=0)
    stds = R.std(axis=0, ddof=1) + 1e-8
    se = stds / math.sqrt(t_len)
    shrink_factor = np.maximum(0.0, 1.0 - (se / (np.abs(means) + 1e-9)))
    R_shrunk = R - means * (1.0 - shrink_factor)

    def neg_expected_log_wealth(w: np.ndarray) -> float:
        port_r = R_shrunk @ w
        if np.any(1.0 + port_r <= 0.05):
            return 1e6 + float(np.sum(np.maximum(0.05 - (1.0 + port_r), 0.0)) * 1e4)
        return -float(np.mean(np.log(1.0 + port_r)))

    w0 = np.full(n_strat, min(1.0, max_gross_leverage) / n_strat)
    bounds = [(0.0, max_gross_leverage) for _ in range(n_strat)]
    constraints = [{"type": "ineq", "fun": lambda w: max_gross_leverage - float(np.sum(w))}]

    res = minimize(
        neg_expected_log_wealth,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not res.success:
        return {c: 0.0 for c in cols}
    return {c: round(float(val), 4) for c, val in zip(cols, res.x)}


def cusum_alpha_status(returns: pd.Series, expected_daily_mean: float) -> float:
    """전략 성과 붕괴 감지용 하방 CUSUM 통계량"""
    r = returns.dropna().tail(60)
    if len(r) == 0 or r.std() < 1e-12:
        return 0.0
    z = (r - expected_daily_mean) / r.std()
    s = 0.0
    for val in z:
        s = min(0.0, s + float(val) + 0.25)
    return round(abs(s), 4)
