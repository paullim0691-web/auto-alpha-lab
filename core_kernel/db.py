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

    CREATE TABLE IF NOT EXISTS macro_series (
        obs_date DATE NOT NULL,
        series_id TEXT NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (obs_date, series_id)
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


def get_total_trials_count() -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM trials;")
            row = cur.fetchone()
            return int(row[0]) if row else 0


def save_trial(
    strategy_id: str,
    origin: str,
    spec: dict,
    metrics: dict,
    n_trials: int,
    passed: bool,
):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO strategies (strategy_id, origin, spec_json, status)
                VALUES (%s, %s, %s::jsonb, %s)
                ON CONFLICT (strategy_id) DO NOTHING;
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
                    strategy_id,
                    float(metrics["log_wealth"]),
                    float(metrics["cagr"]),
                    float(metrics["sharpe"]),
                    float(metrics["deflated_sharpe"]),
                    float(metrics["max_drawdown"]),
                    int(n_trials),
                    bool(passed),
                    json.dumps(metrics),
                ),
            )
        conn.commit()
