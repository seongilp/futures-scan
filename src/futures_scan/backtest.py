"""Backtest aggregation: per-symbol trade generation -> summary stats."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from futures_scan.strategy import NOTIONAL_USDT, Trade, generate_trades

DISCLAIMER = (
    "백테스트는 실제가 아님, 슬리피지·펀딩 미반영 "
    "(This backtest is NOT real trading; slippage and funding fees are NOT modeled.)"
)


@dataclass(frozen=True)
class BacktestSummary:
    total_trades: int
    win_rate_pct: float
    total_pnl_usdt: float
    max_drawdown_usdt: float
    avg_pnl_pct: float
    disclaimer: str = DISCLAIMER


def max_drawdown(cumulative_pnl: list[float]) -> float:
    """Max drawdown (in USDT) of a cumulative-PnL curve."""
    peak = 0.0
    max_dd = 0.0
    for value in cumulative_pnl:
        peak = max(peak, value)
        max_dd = min(max_dd, value - peak)
    return round(max_dd, 4)


def summarize(trades: list[Trade]) -> BacktestSummary:
    if not trades:
        return BacktestSummary(
            total_trades=0, win_rate_pct=0.0, total_pnl_usdt=0.0, max_drawdown_usdt=0.0, avg_pnl_pct=0.0
        )

    ordered = sorted(trades, key=lambda t: t.exit_time)
    wins = sum(1 for t in ordered if t.pnl_usdt > 0)
    cumulative: list[float] = []
    running = 0.0
    for t in ordered:
        running += t.pnl_usdt
        cumulative.append(running)

    return BacktestSummary(
        total_trades=len(ordered),
        win_rate_pct=round(100 * wins / len(ordered), 2),
        total_pnl_usdt=round(running, 2),
        max_drawdown_usdt=max_drawdown(cumulative),
        avg_pnl_pct=round(sum(t.pnl_pct for t in ordered) / len(ordered), 4),
    )


def run_backtest(candles_by_symbol: dict[str, pd.DataFrame]) -> tuple[dict[str, list[Trade]], BacktestSummary]:
    """Run the strategy across many symbols' candle frames. Returns (trades_by_symbol, overall summary)."""
    trades_by_symbol: dict[str, list[Trade]] = {}
    all_trades: list[Trade] = []
    for symbol, df in candles_by_symbol.items():
        trades = generate_trades(df, symbol)
        trades_by_symbol[symbol] = trades
        all_trades.extend(trades)
    return trades_by_symbol, summarize(all_trades)


def trades_to_json(trades_by_symbol: dict[str, list[Trade]]) -> dict:
    return {symbol: [asdict(t) for t in trades] for symbol, trades in trades_by_symbol.items()}


def per_symbol_summaries(trades_by_symbol: dict[str, list[Trade]]) -> dict[str, BacktestSummary]:
    return {symbol: summarize(trades) for symbol, trades in trades_by_symbol.items()}


def summary_to_markdown(
    overall: BacktestSummary,
    per_symbol: dict[str, BacktestSummary],
    notional_usdt: float = NOTIONAL_USDT,
) -> str:
    lines = [
        "# Backtest Summary",
        "",
        f"> **{DISCLAIMER}**",
        "",
        f"- Notional per trade: {notional_usdt:.0f} USDT",
        f"- Total trades: {overall.total_trades}",
        f"- Win rate: {overall.win_rate_pct}%",
        f"- Total PnL: {overall.total_pnl_usdt} USDT",
        f"- Max drawdown: {overall.max_drawdown_usdt} USDT",
        f"- Avg PnL per trade: {overall.avg_pnl_pct}%",
        "",
        "## Per-symbol",
        "",
        "| Symbol | Trades | Win Rate | Total PnL (USDT) | Max DD (USDT) |",
        "|---|---:|---:|---:|---:|",
    ]
    for symbol, s in sorted(per_symbol.items(), key=lambda kv: kv[1].total_pnl_usdt, reverse=True):
        if s.total_trades == 0:
            continue
        lines.append(
            f"| {symbol} | {s.total_trades} | {s.win_rate_pct}% | {s.total_pnl_usdt} | {s.max_drawdown_usdt} |"
        )
    lines.append("")
    return "\n".join(lines)
