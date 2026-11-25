import json
import os
from datetime import datetime, timezone
import psycopg

def get_conn():
    dsn = os.environ.get("NEON_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError("NEON_DATABASE_URL is missing")
    return psycopg.connect(dsn)

def init_schema():
    ddl = """
    CREATE TABLE IF NOT EXISTS quota_ledger (
        id BIGSERIAL PRIMARY KEY,
        run_date DATE NOT NULL,
        provider TEXT NOT NULL,
        calls_used INTEGER NOT NULL DEFAULT 0,
        tokens_used INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'OK',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (run_date, provider)
    );
    CREATE TABLE IF NOT EXISTS strategies (
        strategy_id TEXT PRIMARY KEY,
        origin TEXT NOT NULL,
        spec_json JSONB NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS trials (
        trial_id BIGSERIAL PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        run_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        oos_log_wealth DOUBLE PRECISION NOT NULL,
        oos_cagr DOUBLE PRECISION NOT NULL,
        oos_sharpe DOUBLE PRECISION NOT NULL,
        deflated_sharpe DOUBLE PRECISION NOT NULL,
        max_drawdown DOUBLE PRECISION NOT NULL,
        n_trials_at_eval INTEGER NOT NULL,
        passed BOOLEAN NOT NULL,
        details JSONB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS shadow_equity (
        id BIGSERIAL PRIMARY KEY,
        trade_date DATE NOT NULL,
        strategy_id TEXT NOT NULL,
        nav DOUBLE PRECISION NOT NULL,
        daily_return DOUBLE PRECISION NOT NULL,
        gross_exposure DOUBLE PRECISION NOT NULL,
        cusum_stat DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        UNIQUE (trade_date, strategy_id)
    );
    CREATE TABLE IF NOT EXISTS failure_memory (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        strategy_id TEXT NOT NULL,
        origin TEXT NOT NULL,
        regime TEXT NOT NULL,
        reason TEXT NOT NULL,
        lesson TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS paper_orders (
        id BIGSERIAL PRIMARY KEY,
        run_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty DOUBLE PRECISION NOT NULL,
        status TEXT NOT NULL,
        order_id TEXT,
        details JSONB NOT NULL
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

def record_quota(provider: str, calls: int = 0, tokens: int = 0, status: str = "OK"):
    today = datetime.now(timezone.utc).date()
    sql = """
    INSERT INTO quota_ledger (run_date, provider, calls_used, tokens_used, status)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (run_date, provider) DO UPDATE SET
        calls_used = quota_ledger.calls_used + EXCLUDED.calls_used,
        tokens_used = quota_ledger.tokens_used + EXCLUDED.tokens_used,
        status = EXCLUDED.status,
        updated_at = NOW();
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (today, provider, calls, tokens, status))
        conn.commit()

def get_today_quota(provider: str) -> int:
    today = datetime.now(timezone.utc).date()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT calls_used FROM quota_ledger WHERE run_date=%s AND provider=%s;", (today, provider))
            row = cur.fetchone()
            return int(row[0]) if row else 0

def get_total_trials_count() -> int:
    # 최근 30일 탐색 횟수 기준 다중검정 페널티 유지 (오래된 실패가 영구적으로 연구소를 마비시키지 않도록 방지)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trials WHERE run_ts > NOW() - INTERVAL '30 days';")
            row = cur.fetchone()
            return int(row[0]) if row else 0

def save_trial(strategy_id: str, origin: str, spec: dict, metrics: dict, n_trials: int, passed: bool):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO strategies (strategy_id, origin, spec_json, status)
                VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (strategy_id) DO UPDATE SET status = CASE
                    WHEN strategies.status = 'RETIRED' THEN 'RETIRED'
                    ELSE EXCLUDED.status END;
                """,
                (strategy_id, origin, json.dumps(spec), "ACTIVE" if passed else "REJECTED"),
            )
            cur.execute(
                """
                INSERT INTO trials (
                    strategy_id, oos_log_wealth, oos_cagr, oos_sharpe,
                    deflated_sharpe, max_drawdown, n_trials_at_eval, passed, details
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                """,
                (
                    strategy_id, float(metrics["log_wealth"]), float(metrics["cagr"]),
                    float(metrics["sharpe"]), float(metrics["deflated_sharpe"]),
                    float(metrics["max_drawdown"]), int(n_trials), bool(passed), json.dumps(metrics),
                ),
            )
        conn.commit()

def get_active_strategies() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT strategy_id, origin, spec_json, created_at FROM strategies WHERE status='ACTIVE';")
            return [{"strategy_id": r[0], "origin": r[1], "spec": r[2], "created_at": r[3]} for r in cur.fetchall()]

def retire_strategy(strategy_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE strategies SET status='RETIRED' WHERE strategy_id=%s;", (strategy_id,))
        conn.commit()

def record_failure(strategy_id: str, origin: str, regime: str, reason: str, lesson: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO failure_memory (strategy_id, origin, regime, reason, lesson) VALUES (%s, %s, %s, %s, %s);",
                (strategy_id, origin, regime, reason[:300], lesson[:500]),
            )
        conn.commit()

def get_recent_failures(limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT strategy_id, regime, reason, lesson FROM failure_memory ORDER BY id DESC LIMIT %s;", (limit,))
            return [{"strategy_id": r[0], "regime": r[1], "reason": r[2], "lesson": r[3]} for r in cur.fetchall()]

def upsert_shadow_equity(rows: list[tuple]):
    if not rows:
        return
    sql = """
    INSERT INTO shadow_equity (trade_date, strategy_id, nav, daily_return, gross_exposure, cusum_stat)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (trade_date, strategy_id) DO UPDATE SET
        nav = EXCLUDED.nav, daily_return = EXCLUDED.daily_return,
        gross_exposure = EXCLUDED.gross_exposure, cusum_stat = EXCLUDED.cusum_stat;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()

def log_paper_orders(orders: list[dict]):
    if not orders:
        return
    sql = "INSERT INTO paper_orders (symbol, side, qty, status, order_id, details) VALUES (%s, %s, %s, %s, %s, %s::jsonb);"
    with get_conn() as conn:
        with conn.cursor() as cur:
            for o in orders:
                cur.execute(sql, (o["symbol"], o["side"], float(o["qty"]), o["status"], o.get("order_id"), json.dumps(o)))
        conn.commit()
