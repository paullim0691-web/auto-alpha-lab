import os
import time
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

UNIVERSE = [
    # Equity Indices, Style & Small/Mid Factors
    "SPY", "QQQ", "IWM", "DIA", "MDY", "MTUM", "VLUE", "QUAL", "USMV", "ARKK",
    # Leveraged & Inverse ETFs (2x/1x tools for asymmetric regime return)
    "QLD", "SSO", "SH", "PSQ", "TBF", "SDS", "QID",
    # Rates, Credit, Commodities, FX
    "TLT", "IEF", "SHY", "HYG", "LQD", "GLD", "SLV", "GDX", "USO", "UNG", "DBC", "DBA", "UUP",
    # Sectors & Industries
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "SMH", "XBI", "KRE", "ITA",
    # Liquid Large/Mid/Small Alpha Single Stocks
    "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "MU", "MRVL", "ANET",
    "JPM", "GS", "MS", "SCHW", "XOM", "CVX", "OXY", "SLB", "LLY", "UNH", "VRTX", "REGN",
    "COST", "WMT", "NFLX", "DIS", "PLTR", "COIN", "MSTR", "HOOD", "SOFI", "RKLB", "ASTS",
    "APP", "SMCI", "ARM", "CRWD", "PANW", "NET", "SNOW", "DDOG", "UBER", "ABNB", "TSM", "ASML",
    "CAT", "DE", "GE", "LMT", "RTX", "CEG", "VST", "FCX", "NEM",
]

def check_alpaca_paper_account() -> dict:
    client = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    acct = client.get_account()
    assets = client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    shortable_map = {a.symbol: bool(a.shortable and a.easy_to_borrow) for a in assets if a.tradable}
    fractionable_map = {a.symbol: bool(a.fractionable) for a in assets if a.tradable}
    return {
        "account_status": str(acct.status),
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "tradable_us_equities": len(shortable_map),
        "shortable_map": shortable_map,
        "fractionable_map": fractionable_map,
    }

def fetch_clean_ohlcv(symbols: list[str], lookback_days: int = 1200) -> tuple[dict, dict]:
    data_client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    end_dt = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_dt = end_dt - timedelta(days=lookback_days)
    unique_syms = sorted(set(symbols))
    all_frames = []
    feed_used = "SIP_DELAYED"

    for i in range(0, len(unique_syms), 40):
        batch = unique_syms[i : i + 40]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                start=start_dt, end=end_dt, adjustment=Adjustment.ALL, feed=DataFeed.SIP,
            )
            bars = data_client.get_stock_bars(req)
        except APIError:
            feed_used = "IEX"
            req = StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                start=start_dt, end=end_dt, adjustment=Adjustment.ALL, feed=DataFeed.IEX,
            )
            bars = data_client.get_stock_bars(req)
        if not bars.df.empty:
            all_frames.append(bars.df)

    if not all_frames:
        raise RuntimeError("No market bars returned from Alpaca")

    raw = pd.concat(all_frames).reset_index()
    raw["date"] = pd.to_datetime(raw["timestamp"]).dt.tz_convert("America/New_York").dt.date
    closes = raw.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    opens = raw.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
    highs = raw.pivot_table(index="date", columns="symbol", values="high", aggfunc="last").sort_index()
    lows = raw.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
    volumes = raw.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last").sort_index()
    dollar_vol = closes * volumes

    quarantined, valid_cols = [], []
    for col in closes.columns:
        s = closes[col].dropna()
        if len(s) < 260 or (s <= 0).any():
            quarantined.append((col, "insufficient_or_nonpositive"))
            continue
        rets = s.pct_change(fill_method=None).dropna()
        if (rets.abs() > 0.90).any():
            quarantined.append((col, "extreme_unadjusted_jump"))
            continue
        med_dv = dollar_vol[col].tail(60).median()
        if pd.isna(med_dv) or med_dv < 2_000_000:
            quarantined.append((col, "low_dollar_volume"))
            continue
        valid_cols.append(col)

    closes = closes[valid_cols].ffill(limit=2)
    ohlcv = {
        "close": closes,
        "open": opens.reindex(index=closes.index, columns=valid_cols).ffill(limit=2),
        "high": highs.reindex(index=closes.index, columns=valid_cols).ffill(limit=2),
        "low": lows.reindex(index=closes.index, columns=valid_cols).ffill(limit=2),
        "dollar_vol": dollar_vol.reindex(index=closes.index, columns=valid_cols).fillna(0.0),
    }
    meta = {
        "feed_used": feed_used,
        "valid_symbols": len(valid_cols),
        "quarantined": quarantined,
        "start_date": str(closes.index.min()),
        "end_date": str(closes.index.max()),
        "trading_days": len(closes),
    }
    return ohlcv, meta

def fetch_fred_history(date_index, series_ids=("DFF", "DGS10", "DGS2", "VIXCLS")) -> tuple[pd.DataFrame, dict]:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame(index=date_index), {}
    series_map = {}
    latest = {}
    for sid in series_ids:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": sid, "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": 1500},
            timeout=15,
        )
        r.raise_for_status()
        rows = [(datetime.strptime(o["date"], "%Y-%m-%d").date(), float(o["value"]))
                for o in r.json().get("observations", []) if o.get("value") not in (None, ".")]
        if rows:
            s = pd.Series(dict(rows)).sort_index()
            series_map[sid] = s
            latest[sid] = {"date": str(s.index[-1]), "value": float(s.iloc[-1])}
    macro_df = pd.DataFrame(series_map).reindex(sorted(set(date_index) | set(pd.DataFrame(series_map).index))).sort_index().ffill()
    # 1영업일 지연(.shift(1))로 거시지표 발표 시차 미래참조 원천 차단
    macro_df = macro_df.reindex(date_index).ffill().shift(1).bfill().fillna(0.0)
    return macro_df, latest

def fetch_sec_event_scores(symbols: list[str], date_index, max_tickers: int = 25) -> pd.DataFrame:
    """SEC EDGAR 공식 API(무료)에서 최근 8-K/Form 4 공시 밀도 수집 (1일 지연 적용)"""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    df_out = pd.DataFrame(0.0, index=date_index, columns=symbols)
    if not ua:
        return df_out
    headers = {"User-Agent": ua}
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=15)
        r.raise_for_status()
        ticker_cik = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}
        targets = [s for s in symbols if s in ticker_cik][-max_tickers:]
        for sym in targets:
            time.sleep(0.12)
            resp = requests.get(f"https://data.sec.gov/submissions/CIK{ticker_cik[sym]}.json", headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            recent = resp.json().get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            fdates = recent.get("filingDate", [])
            counts = {}
            for fm, fd in zip(forms, fdates):
                if fm in ("8-K", "4"):
                    dt = datetime.strptime(fd, "%Y-%m-%d").date()
                    counts[dt] = counts.get(dt, 0.0) + (1.5 if fm == "8-K" else 1.0)
            if counts:
                s = pd.Series(counts).reindex(date_index).fillna(0.0).rolling(5, min_periods=1).sum().shift(1).fillna(0.0)
                df_out[sym] = s
    except Exception:
        pass
    return df_out
