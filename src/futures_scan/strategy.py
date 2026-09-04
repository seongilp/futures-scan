"""Shared strategy rules: RSI+volume-spike entry, TP/SL/time-based exit.

These are the canonical Python rules. The replay HTML re-implements the same
rules in JavaScript (see render/templates/replay.html.j2); tests assert trade
counts match between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures_scan.indicators import rsi
from futures_scan.scan import DEFAULT_RSI_THRESHOLD, DEFAULT_VOL_MULT, VOL_LOOKBACK
from futures_scan.indicators import volume_spike_ratio

TAKE_PROFIT_PCT = 0.03
STOP_LOSS_PCT = 0.015
MAX_HOLD_BARS = 24
FEE_PCT_ONE_WAY = 0.0004
NOTIONAL_USDT = 1_000.0


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_index: int
    entry_time: int
    entry_price: float
    exit_index: int
    exit_time: int
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "time"
    bars_held: int
    pnl_pct: float
    pnl_usdt: float


def compute_signals(
    df: pd.DataFrame,
    rsi_threshold: float = DEFAULT_RSI_THRESHOLD,
    vol_mult: float = DEFAULT_VOL_MULT,
) -> pd.Series:
    """Boolean series: True at bar i means bar i's CLOSE triggers an entry at bar i+1's open."""
    rsi_series = rsi(df["close"], period=14)
    vol_ratio_series = volume_spike_ratio(df["volume"], lookback=VOL_LOOKBACK)
    vol_ratio_prev = vol_ratio_series.shift(1)

    rsi_ok = rsi_series < rsi_threshold
    vol_ok = (vol_ratio_series >= vol_mult) | (vol_ratio_prev >= vol_mult)
    return (rsi_ok & vol_ok).fillna(False).rename("signal")


def simulate_trade(
    df: pd.DataFrame,
    entry_index: int,
    symbol: str,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    max_hold_bars: int = MAX_HOLD_BARS,
    fee_pct_one_way: float = FEE_PCT_ONE_WAY,
    notional_usdt: float = NOTIONAL_USDT,
) -> Trade | None:
    """Simulate a single long trade entered at df[entry_index]'s open.

    Exit priority within a bar: stop-loss checked before take-profit (conservative
    assumption when both could theoretically trigger intrabar).
    """
    n = len(df)
    if entry_index >= n:
        return None

    entry_price = float(df["open"].iloc[entry_index])
    tp_price = entry_price * (1 + take_profit_pct)
    sl_price = entry_price * (1 - stop_loss_pct)

    last_bar = min(entry_index + max_hold_bars, n - 1)
    for i in range(entry_index, last_bar + 1):
        bar_low = float(df["low"].iloc[i])
        bar_high = float(df["high"].iloc[i])
        if bar_low <= sl_price:
            exit_price, reason = sl_price, "sl"
        elif bar_high >= tp_price:
            exit_price, reason = tp_price, "tp"
        elif i == last_bar:
            exit_price, reason = float(df["close"].iloc[i]), "time"
        else:
            continue

        pnl_pct = (exit_price - entry_price) / entry_price
        gross_pnl_usdt = pnl_pct * notional_usdt
        fees_usdt = notional_usdt * fee_pct_one_way * 2
        pnl_usdt = gross_pnl_usdt - fees_usdt

        return Trade(
            symbol=symbol,
            entry_index=entry_index,
            entry_time=int(df["timestamp"].iloc[entry_index]),
            entry_price=entry_price,
            exit_index=i,
            exit_time=int(df["timestamp"].iloc[i]),
            exit_price=exit_price,
            exit_reason=reason,
            bars_held=i - entry_index,
            pnl_pct=round(pnl_pct * 100, 4),
            pnl_usdt=round(pnl_usdt, 4),
        )
    return None


def generate_trades(
    df: pd.DataFrame,
    symbol: str,
    rsi_threshold: float = DEFAULT_RSI_THRESHOLD,
    vol_mult: float = DEFAULT_VOL_MULT,
) -> list[Trade]:
    """Walk the series, taking one trade per signal; skip overlapping entries while a trade is open."""
    signals = compute_signals(df, rsi_threshold=rsi_threshold, vol_mult=vol_mult)
    trades: list[Trade] = []
    n = len(df)
    i = 0
    while i < n - 1:
        if bool(signals.iloc[i]):
            entry_index = i + 1
            trade = simulate_trade(df, entry_index, symbol)
            if trade is not None:
                trades.append(trade)
                i = trade.exit_index + 1
                continue
        i += 1
    return trades
