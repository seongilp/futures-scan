from __future__ import annotations

from futures_scan.support import cluster_levels, find_pivot_lows, find_support_levels

from .fixtures import pivot_low_series


def test_cluster_levels_groups_nearby_prices():
    levels = cluster_levels([100.0, 100.2, 100.1, 150.0, 150.3], tolerance_pct=0.005, min_touches=2)
    assert len(levels) == 2
    prices = sorted(lv.price for lv in levels)
    assert prices[0] == pytest_approx_range(99.5, 100.5)
    assert prices[1] == pytest_approx_range(149.5, 150.5)


def pytest_approx_range(lo, hi):
    class _InRange:
        def __eq__(self, other):
            return lo <= other <= hi

    return _InRange()


def test_cluster_levels_drops_singleton_touches():
    levels = cluster_levels([100.0, 200.0, 300.0], tolerance_pct=0.005, min_touches=2)
    assert levels == []


def test_find_pivot_lows_identifies_local_minima():
    df = pivot_low_series()
    pivots = find_pivot_lows(df["low"], lookaround=5)
    assert len(pivots) >= 2
    prices = [p for _, p in pivots]
    assert any(abs(p - 95.0) / 95.0 < 0.02 for p in prices)


def test_find_support_levels_end_to_end():
    df = pivot_low_series()
    levels = find_support_levels(df)
    assert len(levels) >= 1
    for lvl in levels:
        assert lvl.touches >= 2


def test_find_support_levels_empty_for_monotonic_series():
    from .fixtures import make_ohlcv

    df = make_ohlcv([100.0 + i for i in range(30)])
    levels = find_support_levels(df)
    assert levels == []
