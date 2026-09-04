"""ccxt-based market data fetching with exchange fallback and incremental caching."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import pandas as pd

from futures_scan import cache

logger = logging.getLogger("futures_scan.fetch")

# Ordered fallback chain: Binance USDT-M perpetuals first, then Bybit, then OKX.
EXCHANGE_FALLBACK_CHAIN = ["binanceusdm", "bybit", "okx"]

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

MAX_CONCURRENT_REQUESTS = 8


def _build_exchange(exchange_id: str):
    import ccxt.async_support as ccxt_async

    klass = getattr(ccxt_async, exchange_id)
    return klass({"enableRateLimit": True})


async def _exchange_reachable(exchange) -> bool:
    try:
        await exchange.load_markets()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "try next exchange"
        logger.warning("exchange %s unreachable: %s", exchange.id, exc)
        return False


async def select_exchange(preferred_chain: list[str] | None = None):
    """Return a connected ccxt exchange, falling back through the chain and logging fallbacks."""
    chain = preferred_chain or EXCHANGE_FALLBACK_CHAIN
    last_error: Exception | None = None
    for i, exchange_id in enumerate(chain):
        exchange = _build_exchange(exchange_id)
        if await _exchange_reachable(exchange):
            if i > 0:
                logger.warning(
                    "FALLBACK: %s unreachable, using %s instead", chain[i - 1], exchange_id
                )
            return exchange
        try:
            await exchange.close()
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError(f"No reachable exchange in fallback chain: {chain}") from last_error


def usdt_perpetual_symbols(exchange) -> list[str]:
    """All USDT-margined perpetual swap symbols on a loaded ccxt exchange."""
    symbols = []
    for symbol, market in exchange.markets.items():
        if not market.get("swap"):
            continue
        if market.get("quote") != "USDT":
            continue
        if market.get("settle") not in ("USDT", None):
            continue
        if not market.get("active", True):
            continue
        symbols.append(symbol)
    return sorted(symbols)


async def get_symbols(exchange, use_cache: bool = True) -> list[str]:
    cached = cache.load_ticker_cache(exchange.id) if use_cache else None
    if cached is not None:
        logger.info("using cached symbol list for %s (%d symbols)", exchange.id, len(cached))
        return cached
    symbols = usdt_perpetual_symbols(exchange)
    cache.save_ticker_cache(exchange.id, symbols)
    return symbols


def _ohlcv_to_df(raw: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=cache.OHLCV_COLUMNS)
    return df.astype(
        {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        }
    )


async def fetch_candles_incremental(
    exchange, symbol: str, timeframe: str, limit: int, semaphore: asyncio.Semaphore
) -> pd.DataFrame:
    """Fetch OHLCV for a symbol, reusing the cache and only pulling new bars."""
    existing = cache.load_candles(exchange.id, symbol, timeframe)
    tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)

    since = None
    fetch_limit = limit
    if existing is not None and not existing.empty:
        last_ts = int(existing["timestamp"].max())
        since = last_ts + tf_ms
        now_ms = int(time.time() * 1000)
        bars_needed = max(1, (now_ms - since) // tf_ms + 2)
        fetch_limit = min(limit, bars_needed)
        if since >= now_ms:
            return existing.tail(limit).reset_index(drop=True)

    async with semaphore:
        raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=fetch_limit)

    if not raw:
        return (existing if existing is not None else pd.DataFrame(columns=cache.OHLCV_COLUMNS)).tail(
            limit
        ).reset_index(drop=True)

    new_df = _ohlcv_to_df(raw)
    merged = cache.merge_candles(existing, new_df)
    cache.save_candles(exchange.id, symbol, timeframe, merged)
    return merged.tail(limit).reset_index(drop=True)


@dataclass
class FetchResult:
    symbol: str
    df: pd.DataFrame | None
    error: str | None = None


async def fetch_many(
    exchange,
    symbols: list[str],
    timeframe: str,
    limit: int,
    on_progress=None,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
) -> list[FetchResult]:
    """Fetch OHLCV for many symbols concurrently, bounded by a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[FetchResult] = []
    done_count = 0
    lock = asyncio.Lock()

    async def _one(symbol: str) -> FetchResult:
        nonlocal done_count
        try:
            df = await fetch_candles_incremental(exchange, symbol, timeframe, limit, semaphore)
            result = FetchResult(symbol=symbol, df=df)
        except Exception as exc:  # noqa: BLE001
            result = FetchResult(symbol=symbol, df=None, error=str(exc))
        async with lock:
            done_count += 1
            if on_progress:
                on_progress(done_count, len(symbols), symbol)
        return result

    results = await asyncio.gather(*(_one(s) for s in symbols))
    return list(results)
