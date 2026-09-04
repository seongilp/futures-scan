"""Synthetic OHLCV fixtures used across the test suite (no network required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000


def make_ohlcv(closes: list[float], volumes: list[float] | None = None, start_ts: int = 0) -> pd.DataFrame:
    """Build a minimal OHLCV frame from a close-price series (open=prev close, small wicks)."""
    n = len(closes)
    volumes = volumes or [100.0] * n
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * 1.001 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.999 for o, c in zip(opens, closes)]
    return pd.DataFrame(
        {
            "timestamp": [start_ts + i * HOUR_MS for i in range(n)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def rsi_known_series() -> pd.DataFrame:
    """The classic Wilder RSI textbook opening sequence, extended with a mild oscillation
    so that an EWM-based RSI (which decays the SMA-seed influence exponentially rather than
    dropping it abruptly) has enough bars to converge with a classic SMA-seeded Wilder RSI.
    """
    closes = [
        44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    ]
    # gentle, deterministic continuation (small mean-reverting wiggle) to let both
    # smoothing methods converge past the initial seeding difference.
    extra = []
    price = closes[-1]
    for i in range(40):
        price = price + (0.15 if i % 2 == 0 else -0.12)
        extra.append(round(price, 4))
    return make_ohlcv(closes + extra)


def flat_then_crash_and_spike(n_flat: int = 40) -> pd.DataFrame:
    """A flat series, then a sharp multi-bar decline (drives RSI < 30) with a volume spike on the last bar."""
    closes = [100.0 + np.sin(i / 3) * 0.1 for i in range(n_flat)]
    volumes = [100.0] * n_flat
    # sharp decline over the last 8 bars
    decline = [closes[-1] * (1 - 0.02 * k) for k in range(1, 9)]
    closes = closes + decline
    volumes = volumes + [100.0] * 7 + [900.0]  # last bar: 9x the trailing average
    return make_ohlcv(closes, volumes)


def pivot_low_series() -> pd.DataFrame:
    """A series with two clear pivot-low clusters near 95.0 and 110.0.

    Builds the `low` column explicitly (rather than deriving it from open/close) so each
    trough is a strict, unique local minimum: with a derived low = min(open, close) * 0.999,
    a simple V-shaped close sequence produces a *tied* minimum on both sides of the trough
    (since low[t] and low[t+1] both reduce to close[t] * 0.999), which never registers as a
    strict pivot. Highs are kept comfortably above so they never interfere with the low scan.
    """
    closes = []
    closes += [101, 101, 101, 101, 101]  # flat lead-in so the first trough isn't too close to bar 0
    closes += [100, 99, 97, 95.1, 97, 99, 101, 103, 105]
    closes += [107, 109, 110.2, 108, 106, 104]
    closes += [102, 100, 98, 96, 95.0, 97, 99, 101, 103]
    closes += [105, 107, 109, 110.1, 108, 106]
    closes += [107, 107, 107, 107, 107]  # flat lead-out so the last trough isn't too close to the end
    n = len(closes)
    opens = [closes[0]] + closes[:-1]

    trough_indices = {8: 95.05, 24: 94.95}
    peak_indices = {16: 110.25, 31: 110.15}
    lows = []
    highs = []
    for i in range(n):
        base = min(opens[i], closes[i])
        top = max(opens[i], closes[i])
        lows.append(trough_indices.get(i, base * 0.999))
        highs.append(peak_indices.get(i, top * 1.001))

    return pd.DataFrame(
        {
            "timestamp": [i * HOUR_MS for i in range(n)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * n,
        }
    )


def synthetic_trade_setup() -> pd.DataFrame:
    """A series engineered to trigger one signal bar, then hit take-profit within the hold window."""
    closes = [100.0] * 25
    closes += [99, 97, 94, 90, 86]  # decline drives RSI well under 30
    volumes = [100.0] * 25 + [100.0] * 4 + [900.0]  # spike on the signal (last) bar
    # after signal bar (index 29), price rallies to +3% within a few bars
    closes += [86.5, 87.5, 88.6, 89.7, 90.9]  # entry ~86 -> +3% target ~88.6
    volumes += [100.0] * 5
    return make_ohlcv(closes, volumes)


def make_backtest_hit_series(outcome: str = "tp") -> pd.DataFrame:
    """A signal-bar setup whose *next*-bar entry deterministically resolves to a given outcome.

    outcome:
      "tp"   -> price rallies through +3% on the very next bar after entry.
      "sl"   -> price drops through -1.5% on the very next bar after entry.
      "time" -> price hovers between SL and TP for the full 24-bar hold, exits on close.
    """
    closes = [100.0] * 25
    closes += [99, 97, 94, 90, 86]  # decline drives RSI under 30 by bar 29 (the signal/close bar)
    volumes = [100.0] * 25 + [100.0] * 4 + [900.0]  # spike on bar 29
    entry_price = closes[-1]  # ~86, matches the open of the bar right after the signal

    if outcome == "tp":
        # entry bar (30) opens at entry_price then rallies straight through +3%
        closes += [entry_price * 1.05]
        volumes += [100.0]
    elif outcome == "sl":
        closes += [entry_price * 0.90]
        volumes += [100.0]
    elif outcome == "time":
        # entry bar + 24 more bars (bars_held counts from the entry bar itself), all staying
        # strictly between SL (-1.5%) and TP (+3%), so the position only exits on the time limit.
        hold = [entry_price * (1 + 0.001 * ((-1) ** i)) for i in range(25)]
        closes += hold
        volumes += [100.0] * 25
    else:
        raise ValueError(f"unknown outcome: {outcome}")

    return make_ohlcv(closes, volumes)
