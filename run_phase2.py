import json
from datetime import datetime, timezone
from pathlib import Path

from core_kernel.data_gate import UNIVERSE, check_alpaca_paper_account, fetch_clean_ohlcv, fetch_fred_history, fetch_sec_event_scores
from core_kernel.db import get_total_trials_count, init_schema
from discovery.llm_researcher import run_llm_research_cycle
from discovery.quant_search import run_mathematical_evolution
from execution.paper_trader import run_paper_execution_and_dashboard, write_readme_dashboard

def main():
    print("[1/6] Initializing DB schema...")
    init_schema()

    print("[2/6] Fetching Alpaca account, OHLCV, FRED macro & SEC EDGAR...")
    alpaca_info = check_alpaca_paper_account()
    ohlcv, data_meta = fetch_clean_ohlcv(UNIVERSE, lookback_days=1200)
    closes = ohlcv["close"]
    macro_df, fred_latest = fetch_fred_history(closes.index)
    sec_df = fetch_sec_event_scores(list(closes.columns), closes.index)

    vix = fred_latest.get("VIXCLS", {}).get("value", 20.0)
    spread = fred_latest.get("DGS10", {}).get("value", 4.0) - fred_latest.get("DGS2", {}).get("value", 4.0)
    regime_tag = f"{'HIGH_VOL' if vix > 22 else 'LOW_VOL'}_{'POS_CURVE' if spread >= 0 else 'INV_CURVE'}"
    print(f"Detected Regime: {regime_tag} (VIX={vix}, 10Y-2Y={spread:.2f})")

    print("[3/6] Running Mathematical Evolutionary Rule Search...")
    evo_results = run_mathematical_evolution(ohlcv, macro_df, sec_df, regime_tag, n_candidates=24)
    evo_passed = [r for r in evo_results if r.get("passed")]

    print("[4/6] Running Multi-LLM Researcher + Adversarial Critic...")
    n_trials_now = get_total_trials_count()
    llm_results = run_llm_research_cycle(ohlcv, macro_df, sec_df, fred_latest, regime_tag, n_trials_now)

    print("[5/6] Executing Shadow Ledger, Kelly Allocation & Alpaca Paper Orders...")
    paper_exec = run_paper_execution_and_dashboard(ohlcv, macro_df, sec_df, alpaca_info, regime_tag)

    print("[6/6] Writing docs/status.json and README.md Live Dashboard...")
    output = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "PHASE_2_AUTONOMOUS_LIVE",
        "regime": regime_tag,
        "data_meta": data_meta,
        "fred_latest": fred_latest,
        "evolution_summary": {"tested": len(evo_results), "passed": len(evo_passed), "winners": evo_passed[:5]},
        "llm_research": llm_results,
        "paper_execution": paper_exec,
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/status.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    write_readme_dashboard(output)
    print(f"PHASE 2 COMPLETE | Equity=${paper_exec['account_equity']:,.2f} | Orders={len(paper_exec['orders_submitted'])}")

if __name__ == "__main__":
    main()
