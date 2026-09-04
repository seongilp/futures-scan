from __future__ import annotations

from futures_scan.scan import evaluate_symbol

from .fixtures import flat_then_crash_and_spike, make_ohlcv


def test_evaluate_symbol_detects_hit():
    df = flat_then_crash_and_spike()
    hit = evaluate_symbol(df, "TEST/USDT:USDT")
    assert hit is not None
    assert hit.symbol == "TEST/USDT:USDT"
    assert hit.rsi < 30
    assert hit.vol_ratio >= 3.0


def test_evaluate_symbol_no_hit_on_flat_series():
    closes = [100.0 + (i % 3) * 0.01 for i in range(60)]
    df = make_ohlcv(closes)
    hit = evaluate_symbol(df, "FLAT/USDT:USDT")
    assert hit is None


def test_evaluate_symbol_too_few_bars_returns_none():
    df = make_ohlcv([100.0] * 10)
    hit = evaluate_symbol(df, "SHORT/USDT:USDT")
    assert hit is None


def test_evaluate_symbol_respects_custom_thresholds():
    df = flat_then_crash_and_spike()
    # crank the volume requirement absurdly high -> no hit even though RSI condition holds
    hit = evaluate_symbol(df, "TEST/USDT:USDT", vol_mult=100.0)
    assert hit is None


def test_evaluate_symbol_vol_spike_on_previous_bar_still_counts():
    df = flat_then_crash_and_spike()
    # append one more normal-volume bar after the spike; the spike is now the "previous" bar
    import pandas as pd

    extra = df.iloc[[-1]].copy()
    extra["timestamp"] = df["timestamp"].iloc[-1] + 3_600_000
    extra["volume"] = 100.0
    extra["close"] = df["close"].iloc[-1] * 0.995
    extra["open"] = df["close"].iloc[-1]
    extra["high"] = max(extra["open"].iloc[0], extra["close"].iloc[0]) * 1.001
    extra["low"] = min(extra["open"].iloc[0], extra["close"].iloc[0]) * 0.999
    extended = pd.concat([df, extra], ignore_index=True)

    hit = evaluate_symbol(extended, "TEST/USDT:USDT")
    assert hit is not None
