"""RSI + volume-spike scanning logic."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures_scan.indicators import rsi, volume_spike_ratio

DEFAULT_RSI_THRESHOLD = 30.0
DEFAULT_VOL_MULT = 3.0  # 3x = +200%
VOL_LOOKBACK = 20


@dataclass(frozen=True)
class ScanHit:
    symbol: str
    rsi: float
    vol_ratio: float
    close: float
    change_24h_pct: float | None


def evaluate_symbol(
    df: pd.DataFrame,
    symbol: str,
    rsi_threshold: float = DEFAULT_RSI_THRESHOLD,
    vol_mult: float = DEFAULT_VOL_MULT,
) -> ScanHit | None:
    """Evaluate one symbol's OHLCV frame against the scan condition.

    Condition: RSI(14) < threshold on the last CLOSED bar, AND the current or
    previous bar's volume is >= vol_mult times the trailing 20-bar average.
    """
    min_bars = VOL_LOOKBACK + 15
    if df is None or len(df) < min_bars:
        return None

    working = df.reset_index(drop=True)
    rsi_series = rsi(working["close"], period=14)
    vol_ratio_series = volume_spike_ratio(working["volume"], lookback=VOL_LOOKBACK)

    last_idx = len(working) - 1
    last_rsi = rsi_series.iloc[last_idx]
    if pd.isna(last_rsi) or last_rsi >= rsi_threshold:
        return None

    ratio_last = vol_ratio_series.iloc[last_idx]
    ratio_prev = vol_ratio_series.iloc[last_idx - 1] if last_idx - 1 >= 0 else float("nan")
    best_ratio = max(
        ratio_last if pd.notna(ratio_last) else -1.0,
        ratio_prev if pd.notna(ratio_prev) else -1.0,
    )
    if best_ratio < vol_mult:
        return None

    close = float(working["close"].iloc[last_idx])
    change_24h = None
    bars_per_day = None
    # Best-effort 24h change using timestamp deltas rather than assuming timeframe.
    ts = working["timestamp"]
    if len(ts) > 1:
        step_ms = int(ts.iloc[-1] - ts.iloc[-2])
        if step_ms > 0:
            bars_per_day = round(86_400_000 / step_ms)
    if bars_per_day and len(working) > bars_per_day:
        prev_close = float(working["close"].iloc[-1 - bars_per_day])
        if prev_close:
            change_24h = (close - prev_close) / prev_close * 100.0

    return ScanHit(
        symbol=symbol,
        rsi=round(float(last_rsi), 2),
        vol_ratio=round(float(best_ratio), 2),
        close=close,
        change_24h_pct=round(change_24h, 2) if change_24h is not None else None,
    )
