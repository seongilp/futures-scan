from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futures_scan.indicators import atr, ema, rsi, volume_spike_ratio, whale_momentum

from .fixtures import flat_then_crash_and_spike, make_ohlcv, rsi_known_series


def _reference_wilder_rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Independent reference implementation: classic Wilder RSI with an SMA-seeded average.

    Used only in tests, deliberately NOT sharing code with futures_scan.indicators.rsi,
    so the comparison is meaningful.
    """
    n = len(closes)
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    out: list[float | None] = [None] * n
    if len(gains) < period:
        return out

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi_from_avgs(avg_gain, avg_loss)
    return out


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def test_rsi_matches_known_wilder_values_closely():
    df = rsi_known_series()
    closes = df["close"].tolist()
    reference = _reference_wilder_rsi(closes, period=14)

    computed = rsi(df["close"], period=14)

    # Both methods converge after a few smoothing steps; compare the last available point.
    last_idx = len(closes) - 1
    assert reference[last_idx] is not None
    assert computed.iloc[last_idx] == pytest.approx(reference[last_idx], abs=2.5)


def test_rsi_all_gains_saturates_near_100():
    closes = [100 + i for i in range(30)]
    df = make_ohlcv(closes)
    result = rsi(df["close"], period=14)
    assert result.iloc[-1] > 95


def test_rsi_all_losses_saturates_near_0():
    closes = [130 - i for i in range(30)]
    df = make_ohlcv(closes)
    result = rsi(df["close"], period=14)
    assert result.iloc[-1] < 5


def test_rsi_flat_series_is_50():
    closes = [100.0] * 30
    df = make_ohlcv(closes)
    result = rsi(df["close"], period=14)
    assert result.iloc[-1] == pytest.approx(50.0)


def test_rsi_bounded_0_100():
    df = flat_then_crash_and_spike()
    result = rsi(df["close"], period=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_volume_spike_ratio_detects_spike():
    df = flat_then_crash_and_spike()
    ratios = volume_spike_ratio(df["volume"], lookback=20)
    assert ratios.iloc[-1] >= 3.0


def test_volume_spike_ratio_normal_bar_is_near_1():
    df = flat_then_crash_and_spike()
    ratios = volume_spike_ratio(df["volume"], lookback=20)
    # a bar well before the spike, in the flat region, should sit near 1x
    assert ratios.iloc[30] == pytest.approx(1.0, abs=0.2)


def test_ema_converges_to_constant_value():
    closes = [50.0] * 40
    df = make_ohlcv(closes)
    result = ema(df["close"], period=9)
    assert result.iloc[-1] == pytest.approx(50.0)


def test_atr_zero_for_flat_series():
    closes = [50.0] * 30
    df = make_ohlcv(closes)
    result = atr(df["high"], df["low"], df["close"], period=14)
    # our synthetic wicks are +/-0.1% around close, so ATR should be small and stable
    assert result.iloc[-1] < 0.5


def test_whale_momentum_output_range_and_shape():
    df = flat_then_crash_and_spike()
    result = whale_momentum(df)
    assert len(result) == len(df)
    valid = result.dropna()
    assert (valid >= -100).all() and (valid <= 100).all()


def test_whale_momentum_spikes_negative_on_volume_and_price_crash():
    df = flat_then_crash_and_spike()
    result = whale_momentum(df)
    # the crash+volume-spike bar should register a strongly negative reading
    assert result.iloc[-1] < 0
