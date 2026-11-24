import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from core_kernel.db import get_active_strategies, log_paper_orders, record_failure, retire_strategy, upsert_shadow_equity
from core_kernel.engine import bootstrapped_kelly_allocate, cusum_alpha_status, lockbox_walk_forward_eval
from discovery.dsl import dsl_to_weight_fn

def run_paper_execution_and_dashboard(
    ohlcv: dict, macro_df, sec_df, alpaca_info: dict, regime_tag: str
) -> dict:
    closes, dollar_vol = ohlcv["close"], ohlcv["dollar_vol"]
    active = get_active_strategies()

    oos_returns, gross_map, latest_weights = {}, {}, {}
    surviving_meta = []
    retired_now = []
    shadow_rows = []
    today_dt = closes.index[-1]

    for st in active:
        sid = st["strategy_id"]
        spec = st["spec"]
        fn = dsl_to_weight_fn(spec, macro_df=macro_df, sec_df=sec_df)
        w_df = fn(closes, dollar_vol)
        oos_ret, metrics, _ = lockbox_walk_forward_eval(closes, dollar_vol, fn, n_trials=1)
        cusum_val, decayed = cusum_alpha_status(oos_ret, float(oos_ret.mean()))

        # Shadow Ledger에 최근 5영업일 기록 및 실시간 생존 판정
        eq = (1.0 + oos_ret.clip(lower=-0.99)).cumprod()
        for dt_idx in oos_ret.index[-5:]:
            shadow_rows.append((
                dt_idx, sid, float(eq.loc[dt_idx]), float(oos_ret.loc[dt_idx]),
                float(w_df.loc[dt_idx].abs().sum()), cusum_val,
            ))

        if decayed and len(active) > 2:
            retire_strategy(sid)
            record_failure(sid, st["origin"], regime_tag, f"CUSUM_DECAY={cusum_val}", f"Live alpha died (cusum={cusum_val}, lockbox={metrics['lockbox_log_wealth']:.3f})")
            retired_now.append(sid)
            continue

        oos_returns[sid] = oos_ret
        gross_map[sid] = metrics.get("avg_gross_exposure", 1.0)
        latest_weights[sid] = w_df.iloc[-1]
        surviving_meta.append({"strategy_id": sid, "origin": st["origin"], "cusum": cusum_val, **metrics})

    upsert_shadow_equity(shadow_rows)
    kelly_w = bootstrapped_kelly_allocate(pd.DataFrame(oos_returns), gross_map, max_gross_leverage=1.80)

    # 최종 목표 포트폴리오 비중 합성
    target_port = pd.Series(0.0, index=closes.columns)
    for sid, kw in kelly_w.items():
        if kw > 0 and sid in latest_weights:
            target_port = target_port + kw * latest_weights[sid]

    # Alpaca 공매도 가능 종목 검증 & 안전 마진(1.80x)
    shortable_map = alpaca_info["shortable_map"]
    for sym in list(target_port.index):
        if target_port[sym] < 0 and not shortable_map.get(sym, False):
            target_port[sym] = 0.0
    gross_tot = float(target_port.abs().sum())
    if gross_tot > 1.80:
        target_port = target_port * (1.80 / gross_tot)

    # Alpaca Paper 계좌 실제 리밸런싱 주문 전송
    client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    client.cancel_orders()
    acct = client.get_account()
    equity = float(acct.equity)
    positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}
    last_px = closes.iloc[-1].to_dict()
    frac_map = alpaca_info["fractionable_map"]

    submitted_orders = []
    all_syms = sorted(set(positions.keys()) | set(target_port[target_port.abs() > 0.005].index))

    def _submit(sym: str, side: OrderSide, qty: float):
        try:
            o = client.submit_order(MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY))
            submitted_orders.append({"symbol": sym, "side": str(side.value), "qty": float(qty), "status": str(o.status.value), "order_id": str(o.id)})
        except Exception as e:
            submitted_orders.append({"symbol": sym, "side": str(side.value), "qty": float(qty), "status": "ERR", "error": str(e)[:180]})

    # 1단계: 기존 포지션 청산·축소(방향 전환 시 안전 청산), 2단계: 신규 진입·증액
    pending_opens = []
    for sym in all_syms:
        px = float(last_px.get(sym, 0.0))
        if px <= 0:
            continue
        cur_q = positions.get(sym, 0.0)
        tgt_dollars = float(target_port.get(sym, 0.0)) * equity

        if tgt_dollars < 0:
            tgt_q = -float(int(abs(tgt_dollars) // px))  # 공매도는 무조건 정수 주식만 가능
        elif abs(tgt_dollars) < 75.0:
            tgt_q = 0.0
        else:
            tgt_q = float(int(tgt_dollars // px))
            if tgt_q == 0 and frac_map.get(sym, False):
                tgt_q = round(tgt_dollars / px, 4)

        if cur_q * tgt_q < 0:  # 롱↔숏 반전 시 오늘은 전량 청산만 수행
            _submit(sym, OrderSide.SELL if cur_q > 0 else OrderSide.BUY, abs(cur_q))
            continue

        delta = tgt_q - cur_q
        if abs(delta * px) < 75.0 and tgt_q != 0:
            continue
        if (cur_q > 0 and delta < 0) or (cur_q < 0 and delta > 0):
            _submit(sym, OrderSide.SELL if delta < 0 else OrderSide.BUY, round(abs(delta), 4))
        elif abs(delta) > 0:
            pending_opens.append((sym, OrderSide.BUY if delta > 0 else OrderSide.SELL, round(abs(delta), 4)))

    for sym, side, qty in pending_opens:
        _submit(sym, side, qty)

    log_paper_orders(submitted_orders)
    return {
        "trade_date": str(today_dt),
        "account_equity": equity,
        "target_gross_leverage": round(float(target_port.abs().sum()), 4),
        "target_net_exposure": round(float(target_port.sum()), 4),
        "kelly_weights": kelly_w,
        "surviving_strategies": surviving_meta,
        "retired_now": retired_now,
        "orders_submitted": submitted_orders,
        "current_positions": positions,
        "top_targets": {k: round(float(v), 4) for k, v in target_port.items() if abs(v) > 0.01},
    }

def write_readme_dashboard(status: dict):
    pa = status["paper_execution"]
    eq = pa["account_equity"]
    pnl_pct = (eq / 100_000.0 - 1.0) * 100.0
    lines = [
        "# AutoAlphaLab — Autonomous Quant & Paper Trading Fund",
        f"- **Updated (UTC):** `{status['updated_at_utc']}` | **Regime:** `{status['regime']}`",
        f"- **Alpaca Paper Equity:** **`${eq:,.2f}`** (`{pnl_pct:+.2f}%` vs $100k start) | **Gross Leverage:** `{pa['target_gross_leverage']}x` | **Net Exposure:** `{pa['target_net_exposure']}x`",
        "",
        "## Active Portfolio Strategies (Bootstrapped Kelly Allocation)",
        "| Strategy ID | Origin | Kelly Weight | OOS CAGR | Sharpe | Lockbox CAGR | MDD | CUSUM |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    kw = pa["kelly_weights"]
    for s in sorted(pa["surviving_strategies"], key=lambda x: kw.get(x["strategy_id"], 0.0), reverse=True):
        sid = s["strategy_id"]
        lines.append(
            f"| `{sid}` | {s['origin']} | **{kw.get(sid, 0.0):.2f}** | {s['cagr']*100:.1f}% | {s['sharpe']:.2f} | {s['lockbox_cagr']*100:.1f}% | {s['max_drawdown']*100:.1f}% | {s['cusum']:.2f} |"
        )
    lines += ["", "## Target Portfolio Weights", f"`{pa['top_targets']}`", "", "## Recent Submitted Orders", f"Total orders this run: **{len(pa['orders_submitted'])}**"]
    for o in pa["orders_submitted"][:15]:
        lines.append(f"- `{o['symbol']}` {o['side']} `{o['qty']}` -> `{o['status']}`")
    Path("README.md").write_text("\n".join(lines), encoding="utf-8")
