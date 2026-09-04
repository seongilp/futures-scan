"""Technical indicators. All functions take/return new pandas objects (no mutation)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses Wilder's smoothing (equivalent to an EMA with alpha=1/period) applied
    to the average gain/loss series, matching the classic TradingView RSI.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 and avg_gain > 0 -> RSI = 100. When both 0 -> RSI = 50 (flat).
    out = out.where(avg_loss != 0.0, other=100.0)
    out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), other=50.0)
    return out.rename("rsi")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().rename("atr")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean().rename(f"ema_{period}")


def volume_spike_ratio(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Ratio of current volume to the trailing `lookback`-bar average (excluding current bar)."""
    trailing_avg = volume.shift(1).rolling(window=lookback, min_periods=lookback).mean()
    return (volume / trailing_avg).rename("vol_ratio")


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of a series.

    When the rolling window has zero variance (a perfectly flat window), every value in
    it equals the mean by definition, so the z-score is 0 rather than NaN/inf.
    """
    roll_mean = series.rolling(window=window, min_periods=window).mean()
    roll_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    result = (series - roll_mean) / roll_std.replace(0.0, np.nan)
    return result.where(roll_std != 0.0, other=0.0).rename("zscore")


def whale_momentum(df: pd.DataFrame) -> pd.Series:
    """Custom oscillator: whale-activity proxy x price-trend, EMA-smoothed to -100..100.

    Definition
    ----------
    whale proxy   = zscore(volume, 20) * (|close - open| / ATR(14))
    trend         = normalized EMA(9) - EMA(21) spread, i.e. (ema9 - ema21) / close
    raw           = whale_proxy * trend
    oscillator    = EMA(raw, 5), then rescaled so that +/-3 std maps to +/-100
                    (clipped to [-100, 100]) for a stable, comparable scale.

    Requires columns: open, high, low, close, volume.
    """
    vol_z = zscore(df["volume"], window=20)
    body = (df["close"] - df["open"]).abs()
    atr14 = atr(df["high"], df["low"], df["close"], period=14)
    whale_proxy = vol_z * (body / atr14.replace(0.0, np.nan))

    ema9 = ema(df["close"], 9)
    ema21 = ema(df["close"], 21)
    trend = (ema9 - ema21) / df["close"]

    raw = (whale_proxy * trend).rename("raw")
    smoothed = raw.ewm(span=5, min_periods=5, adjust=False).mean()

    std = smoothed.rolling(window=100, min_periods=20).std(ddof=0)
    std = std.replace(0.0, np.nan)
    scaled = (smoothed / std) * (100.0 / 3.0)
    scaled = scaled.clip(lower=-100.0, upper=100.0)
    return scaled.fillna(0.0).rename("whale_momentum")
