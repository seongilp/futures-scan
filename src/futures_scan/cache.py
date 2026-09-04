"""Parquet-backed OHLCV cache with incremental-append support."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path("data/candles")
TICKERS_CACHE = Path("data/tickers")
ONE_DAY_SECONDS = 86_400

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def candle_path(exchange: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-").replace(":", "-")
    return CACHE_ROOT / exchange / f"{safe_symbol}_{timeframe}.parquet"


def load_candles(exchange: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Return cached candles for a symbol, or None if no cache exists."""
    path = candle_path(exchange, symbol, timeframe)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def save_candles(exchange: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Persist candles, deduplicated by timestamp, sorted ascending."""
    path = candle_path(exchange, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = (
        df[OHLCV_COLUMNS]
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    deduped.to_parquet(path, index=False)


def merge_candles(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new candles into an existing cache frame (new data wins on overlap)."""
    if existing is None or existing.empty:
        return new[OHLCV_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    combined = pd.concat([existing[OHLCV_COLUMNS], new[OHLCV_COLUMNS]], ignore_index=True)
    return (
        combined.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class TickerCacheEntry:
    exchange: str
    symbols: list[str]
    fetched_at: float


def ticker_cache_path(exchange: str) -> Path:
    return TICKERS_CACHE / f"{exchange}.json"


def load_ticker_cache(exchange: str, max_age_seconds: float = ONE_DAY_SECONDS) -> list[str] | None:
    path = ticker_cache_path(exchange)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if time.time() - payload["fetched_at"] > max_age_seconds:
        return None
    return payload["symbols"]


def save_ticker_cache(exchange: str, symbols: list[str]) -> None:
    path = ticker_cache_path(exchange)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"exchange": exchange, "symbols": symbols, "fetched_at": time.time()}
    path.write_text(json.dumps(payload))
