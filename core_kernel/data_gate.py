import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

# Phase 1 초기 다중자산·섹터·인버스/레버리지 + 고변동성 유동성 대표 풀
# Phase 2에서 Alpaca 전체 상장 종목 중 유동성 필터 통과 종목으로 자동 확장됨
SEED_UNIVERSE = [
    # Equity Indices & Factors
    "SPY", "QQQ", "IWM", "DIA", "MTUM", "VLUE", "QUAL", "USMV",
    # Inverse & Leveraged Tools (for dynamic log-wealth optimizer)
    "SH", "PSQ", "TBF", "QLD", "SSO",
    # Rates, Credit, Commodities, FX
    "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "USO", "DBC", "UUP",
    # Sectors
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "SMH",
    # High-liquidity single names across sectors (small/mid/large mix)
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO",
    "JPM", "GS", "XOM", "CVX", "LLY", "UNH", "COST", "WMT", "NFLX", "PLTR",
    "COIN", "MSTR", "SOFI", "RKLB", "APP", "SMCI", "ARM", "CRWD", "PANW",
]


def check_alpaca_paper_account() -> dict:
    client = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )
    acct = client.get_account()
    assets = client.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    shortable_map = {a.symbol: bool(a.shortable and a.easy_to_borrow) for a in assets if a.tradable}
    return {
        "account_status": str(acct.status),
        "equity": float(acct.equity),
        "buying_power": float(acct.buying_power),
        "tradable_us_equities": len(shortable_map),
        "shortable_map": shortable_map,
    }


def fetch_clean_daily_bars(symbols: list[str], lookback_days: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    배당·액면분할이 반영된(Adjustment.ALL) 일봉 종가 & 거래대금 수집.
    무료 티어 제약에 따라 20분 지연 SIP를 우선 시도하고, 계정 제한 시 IEX로 자동 폴백.
    """
    data_client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
    )
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_dt = end_dt - timedelta(days=lookback_days)

    unique_syms = sorted(set(symbols))
    all_frames = []
    feed_used = "SIP_DELAYED"

    for i in range(0, len(unique_syms), 40):
        batch = unique_syms[i : i + 40]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment=Adjustment.ALL,
                feed=DataFeed.SIP,
            )
            bars = data_client.get_stock_bars(req)
        except APIError:
            feed_used = "IEX"
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)

        df = bars.df
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No market bars returned from Alpaca")

    raw = pd.concat(all_frames).reset_index()
    raw["date"] = pd.to_datetime(raw["timestamp"]).dt.tz_convert("America/New_York").dt.date

    closes = raw.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    volumes = raw.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last").sort_index()
    dollar_vol = closes * volumes

    # 데이터 오염 방지 게이트(Sanity Gate)
    quarantined = []
    valid_cols = []
    for col in closes.columns:
        s = closes[col].dropna()
        if len(s) < 260 or (s <= 0).any():
            quarantined.append((col, "insufficient_or_nonpositive"))
            continue
        rets = s.pct_change(fill_method=None).dropna()
        if (rets.abs() > 0.85).any():
            quarantined.append((col, "extreme_unadjusted_jump"))
            continue
        # 최근 60일 중위 일거래대금 $2,000,000 미만 제외 (슬리피지 폭증 방어)
        med_dv = dollar_vol[col].tail(60).median()
        if pd.isna(med_dv) or med_dv < 2_000_000:
            quarantined.append((col, "low_dollar_volume"))
            continue
        valid_cols.append(col)

    closes = closes[valid_cols].ffill(limit=2)
    dollar_vol = dollar_vol[valid_cols].fillna(0.0)

    meta = {
        "feed_used": feed_used,
        "valid_symbols": len(valid_cols),
        "quarantined": quarantined,
        "start_date": str(closes.index.min()),
        "end_date": str(closes.index.max()),
        "trading_days": len(closes),
    }
    return closes, dollar_vol, meta


def fetch_fred_macro(series_ids=("DFF", "DGS10", "DGS2", "VIXCLS")) -> dict:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        return {"status": "MISSING_KEY"}
    latest = {}
    for sid in series_ids:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        valid = [o for o in obs if o.get("value") not in (None, ".")]
        if valid:
            latest[sid] = {"date": valid[0]["date"], "value": float(valid[0]["value"])}
    return latest
