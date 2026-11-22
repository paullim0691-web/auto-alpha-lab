import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from core_kernel.data_gate import SEED_UNIVERSE, check_alpaca_paper_account, fetch_clean_daily_bars, fetch_fred_macro
from core_kernel.db import get_total_trials_count, init_schema, record_quota, save_trial
from core_kernel.engine import (
    cusum_alpha_status,
    purged_walk_forward_eval,
    shrinkage_kelly_allocate,
    verify_no_lookahead,
)


def check_llm_providers_zero_token() -> dict:
    """토큰을 1개도 소모하지 않고 무료 LLM 키 정상 여부와 사용 가능 모델 확인"""
    status = {}

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            r = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}",
                timeout=15,
            )
            if r.status_code == 200:
                models = [m["name"].split("/")[-1] for m in r.json().get("models", []) if "flash" in m["name"]]
                status["gemini"] = {"ok": True, "sample_models": models[:5]}
                record_quota("gemini_health", calls=1, tokens=0, status="OK")
            else:
                status["gemini"] = {"ok": False, "code": r.status_code}
                record_quota("gemini_health", calls=1, tokens=0, status=f"ERR_{r.status_code}")
        except Exception as e:
            status["gemini"] = {"ok": False, "error": str(e)}

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {groq_key}"},
                timeout=15,
            )
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])][:5]
                status["groq"] = {"ok": True, "sample_models": models}
                record_quota("groq_health", calls=1, tokens=0, status="OK")
            else:
                status["groq"] = {"ok": False, "code": r.status_code}
                record_quota("groq_health", calls=1, tokens=0, status=f"ERR_{r.status_code}")
        except Exception as e:
            status["groq"] = {"ok": False, "error": str(e)}

    return status


# --- Phase 1 기준선(Benchmark) 전략들: 심판 엔진 검증용 ---
def bench_spy_hold(closes: pd.DataFrame, dollar_vol: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    if "SPY" in w.columns:
        w["SPY"] = 1.0
    return w


def bench_cross_sectional_momentum(closes: pd.DataFrame, dollar_vol: pd.DataFrame) -> pd.DataFrame:
    """과거 126일(6개월) 수익률 상위 20% 롱, 변동성 역가중"""
    ret_126 = closes.pct_change(126, fill_method=None)
    vol_63 = closes.pct_change(fill_method=None).rolling(63).std().replace(0.0, np.nan)
    inv_vol = 1.0 / vol_63

    ranks = ret_126.rank(axis=1, pct=True)
    selected = (ranks >= 0.80).astype(float) * inv_vol
    row_sum = selected.sum(axis=1).replace(0.0, np.nan)
    return selected.div(row_sum, axis=0).fillna(0.0)


def bench_short_term_reversal(closes: pd.DataFrame, dollar_vol: pd.DataFrame) -> pd.DataFrame:
    """5일 단기 과매도 반등(하위 15% 중 200일 추세 상승 종목) vs 단기 과열 매도"""
    ret_5 = closes.pct_change(5, fill_method=None)
    sma_200 = closes.rolling(200).mean()
    uptrend = closes > sma_200
    ranks = ret_5.where(uptrend).rank(axis=1, pct=True)
    long_mask = (ranks <= 0.15).astype(float)
    row_sum = long_mask.sum(axis=1).replace(0.0, np.nan)
    return long_mask.div(row_sum, axis=0).fillna(0.0)


def main():
    print("[1/5] Initializing Neon Postgres schema...")
    init_schema()

    print("[2/5] Checking Alpaca Paper Account & Macro/LLM connectivity...")
    alpaca_info = check_alpaca_paper_account()
    fred_info = fetch_fred_macro()
    llm_info = check_llm_providers_zero_token()

    print("[3/5] Fetching and sanity-checking market bars...")
    closes, dollar_vol, data_meta = fetch_clean_daily_bars(SEED_UNIVERSE, lookback_days=1200)

    benchmarks = {
        "bench_spy_hold": bench_spy_hold,
        "bench_cs_momentum": bench_cross_sectional_momentum,
        "bench_st_reversal": bench_short_term_reversal,
    }

    print("[4/5] Running Lookahead-Bias Self-Test & Purged Walk-Forward evaluation...")
    n_trials = get_total_trials_count()
    oos_returns_dict = {}
    eval_summary = {}

    for name, fn in benchmarks.items():
        verify_no_lookahead(closes, dollar_vol, fn)
        n_trials += 1
        oos_ret, metrics = purged_walk_forward_eval(closes, dollar_vol, fn, n_trials=n_trials)
        passed = bool(metrics["log_wealth"] > 0.0 and metrics["deflated_sharpe"] > 0.50)
        save_trial(
            strategy_id=name,
            origin="phase1_benchmark",
            spec={"type": "seed_benchmark", "name": name},
            metrics=metrics,
            n_trials=n_trials,
            passed=passed,
        )
        oos_returns_dict[name] = oos_ret
        eval_summary[name] = {
            **metrics,
            "passed": passed,
            "cusum_decay": cusum_alpha_status(oos_ret, float(oos_ret.mean())),
        }

    print("[5/5] Computing Shrinkage Log-Wealth Kelly allocation...")
    oos_df = pd.DataFrame(oos_returns_dict).dropna()
    kelly_weights = shrinkage_kelly_allocate(oos_df, max_gross_leverage=2.0)

    output = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE_1_ACTIVE",
        "alpaca_paper": {
            "status": alpaca_info["account_status"],
            "equity": alpaca_info["equity"],
            "buying_power": alpaca_info["buying_power"],
            "tradable_us_equities": alpaca_info["tradable_us_equities"],
        },
        "data_meta": data_meta,
        "fred_latest": fred_info,
        "llm_health": llm_info,
        "benchmark_oos_eval": eval_summary,
        "shrinkage_kelly_weights": kelly_weights,
    }

    Path("docs").mkdir(exist_ok=True)
    Path("docs/status.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print("PHASE 1 COMPLETE:", json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
