"""Pivot-low support level detection via clustering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PIVOT_LOOKAROUND = 5
CLUSTER_TOLERANCE_PCT = 0.005  # 0.5%
MIN_TOUCHES = 2


@dataclass(frozen=True)
class SupportLevel:
    price: float
    touches: int


def find_pivot_lows(low: pd.Series, lookaround: int = PIVOT_LOOKAROUND) -> list[tuple[int, float]]:
    """Indices/prices where `low` is the minimum within +/- lookaround bars."""
    pivots: list[tuple[int, float]] = []
    n = len(low)
    values = low.to_numpy()
    for i in range(lookaround, n - lookaround):
        window = values[i - lookaround : i + lookaround + 1]
        if values[i] == window.min() and (window == values[i]).sum() == 1:
            pivots.append((i, float(values[i])))
    return pivots


def cluster_levels(
    pivot_prices: list[float],
    tolerance_pct: float = CLUSTER_TOLERANCE_PCT,
    min_touches: int = MIN_TOUCHES,
) -> list[SupportLevel]:
    """Cluster pivot prices within `tolerance_pct` of each other; keep clusters with >= min_touches."""
    if not pivot_prices:
        return []

    ordered = sorted(pivot_prices)
    clusters: list[list[float]] = [[ordered[0]]]
    for price in ordered[1:]:
        cluster_mean = sum(clusters[-1]) / len(clusters[-1])
        if abs(price - cluster_mean) / cluster_mean <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    levels = [
        SupportLevel(price=round(sum(c) / len(c), 8), touches=len(c))
        for c in clusters
        if len(c) >= min_touches
    ]
    return sorted(levels, key=lambda lv: lv.touches, reverse=True)


def find_support_levels(
    df: pd.DataFrame,
    lookaround: int = PIVOT_LOOKAROUND,
    tolerance_pct: float = CLUSTER_TOLERANCE_PCT,
    min_touches: int = MIN_TOUCHES,
) -> list[SupportLevel]:
    """Full pipeline: pivot lows -> clustered support levels, from an OHLCV frame."""
    pivots = find_pivot_lows(df["low"], lookaround=lookaround)
    prices = [p for _, p in pivots]
    return cluster_levels(prices, tolerance_pct=tolerance_pct, min_touches=min_touches)
