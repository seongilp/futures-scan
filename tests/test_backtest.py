from __future__ import annotations

import pytest

from futures_scan.backtest import run_backtest, summarize
from futures_scan.strategy import generate_trades

from .fixtures import make_backtest_hit_series


def test_generate_trades_take_profit_outcome():
    df = make_backtest_hit_series(outcome="tp")
    trades = generate_trades(df, "TP/USDT:USDT")
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "tp"
    assert trade.pnl_pct == pytest.approx(3.0, abs=0.05)


def test_generate_trades_stop_loss_outcome():
    df = make_backtest_hit_series(outcome="sl")
    trades = generate_trades(df, "SL/USDT:USDT")
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "sl"
    assert trade.pnl_pct == pytest.approx(-1.5, abs=0.05)


def test_generate_trades_time_based_exit():
    df = make_backtest_hit_series(outcome="time")
    trades = generate_trades(df, "TIME/USDT:USDT")
    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "time"
    assert trade.bars_held == 24


def test_fees_are_applied():
    df = make_backtest_hit_series(outcome="tp")
    trades = generate_trades(df, "FEE/USDT:USDT")
    trade = trades[0]
    gross_pnl = 0.03 * 1000
    fees = 1000 * 0.0004 * 2
    assert trade.pnl_usdt == pytest.approx(gross_pnl - fees, abs=1.0)


def test_no_trades_on_flat_series():
    from .fixtures import make_ohlcv

    df = make_ohlcv([100.0 + (i % 3) * 0.01 for i in range(60)])
    trades = generate_trades(df, "FLAT/USDT:USDT")
    assert trades == []


def test_run_backtest_aggregates_across_symbols():
    frames = {
        "A/USDT:USDT": make_backtest_hit_series(outcome="tp"),
        "B/USDT:USDT": make_backtest_hit_series(outcome="sl"),
    }
    trades_by_symbol, overall = run_backtest(frames)
    assert overall.total_trades == 2
    assert overall.win_rate_pct == 50.0
    assert set(trades_by_symbol.keys()) == {"A/USDT:USDT", "B/USDT:USDT"}


def test_summarize_empty_trades():
    overall = summarize([])
    assert overall.total_trades == 0
    assert overall.win_rate_pct == 0.0
    assert "실제가 아님" in overall.disclaimer
