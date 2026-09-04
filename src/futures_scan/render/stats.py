"""Derived statistics for the dashboard.

Every value here is computed from real scan/backtest output or cached candles.
Nothing is fabricated. The definitions are documented in README.md
("Dashboard metric definitions").
"""

from __future__ import annotations

import math
from statistics import median

from futures_scan.strategy import STOP_LOSS_PCT, TAKE_PROFIT_PCT, Trade

STRIKE_PCT = TAKE_PROFIT_PCT * 100.0
STOP_PCT = STOP_LOSS_PCT * 100.0
#: Win rate at which a TP/SL pair breaks even, ignoring fees.
BREAKEVEN_WR_PCT = 100.0 * STOP_LOSS_PCT / (TAKE_PROFIT_PCT + STOP_LOSS_PCT)


def sharpe(returns_pct: list[float]) -> float:
    """Per-trade Sharpe ratio (mean / stdev of trade returns), 0.0 if undefined."""
    n = len(returns_pct)
    if n < 2:
        return 0.0
    mean = sum(returns_pct) / n
    var = sum((r - mean) ** 2 for r in returns_pct) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return 0.0
    return round(mean / sd, 3)


def tail_stats(returns_pct: list[float]) -> dict:
    """Tail mass at the take-profit strike and the payout multiple it implies."""
    n = len(returns_pct)
    if n == 0:
        return {"tail_count": 0, "tail_mass_pct": 0.0, "implied_multiple": None,
                "stop_mass_pct": 0.0}
    tail = sum(1 for r in returns_pct if r >= STRIKE_PCT - 1e-9)
    stops = sum(1 for r in returns_pct if r <= -STOP_PCT + 1e-9)
    mass = tail / n
    return {
        "tail_count": tail,
        "tail_mass_pct": round(100 * mass, 2),
        "implied_multiple": round(1 / mass, 1) if mass > 0 else None,
        "stop_mass_pct": round(100 * stops / n, 2),
    }


def stat_row(label: str, value: str, tone: str = "") -> dict:
    return {"label": label, "value": value, "tone": tone}


def _money(v: float) -> str:
    return f"{v:+,.2f}"


def _tone(v: float) -> str:
    return "pos" if v >= 0 else "neg"


def lattice_rows(trades: list[Trade], overall, curve: list[float]) -> list[dict]:
    """Left stat column of the Probability Lattice section."""
    if not trades:
        return [stat_row("balls dropped", "0")]
    rets = [t.pnl_pct for t in trades]
    wins = [t for t in trades if t.pnl_usdt > 0]
    losses = [t for t in trades if t.pnl_usdt <= 0]
    ev = overall.total_pnl_usdt / len(trades)
    best = max(rets)
    worst = min(rets)
    peak = max(curve) if curve else 0.0
    return [
        stat_row("balls dropped", f"{len(trades):,}"),
        stat_row("landed green", f"{len(wins):,}", "pos"),
        stat_row("landed red", f"{len(losses):,}", "neg"),
        stat_row("green rate", f"{overall.win_rate_pct:.1f}%"),
        stat_row("ev / trade", _money(ev), _tone(ev)),
        stat_row("session pnl", _money(overall.total_pnl_usdt), _tone(overall.total_pnl_usdt)),
        stat_row("realised", _money(sum(t.pnl_usdt for t in trades)), _tone(overall.total_pnl_usdt)),
        stat_row("equity peak", _money(peak), _tone(peak)),
        stat_row("max drawdown", f"{overall.max_drawdown_usdt:,.2f}", "neg"),
        stat_row("best hit", f"{best:+.2f}%", "pos" if best >= 0 else "neg"),
        stat_row("worst hit", f"{worst:+.2f}%", "pos" if worst >= 0 else "neg"),
        stat_row("sharpe / trade", f"{sharpe(rets):.2f}"),
        stat_row("breakeven wr", f"{BREAKEVEN_WR_PCT:.1f}%"),
        stat_row("edge vs gate", f"{overall.win_rate_pct - BREAKEVEN_WR_PCT:+.1f}pp",
                 _tone(overall.win_rate_pct - BREAKEVEN_WR_PCT)),
    ]


def ridge_rows(trades: list[Trade], overall, symbol_count: int) -> list[dict]:
    """Left stat column of the Tail Probability Ridge section."""
    if not trades:
        return [stat_row("sessions", "0")]
    rets = [t.pnl_pct for t in trades]
    tail = tail_stats(rets)
    holds = [t.bars_held for t in trades]
    entries = [t.entry_price for t in trades]
    tp_exits = sum(1 for t in trades if t.exit_reason == "tp")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")
    return [
        stat_row("symbols", f"{symbol_count}"),
        stat_row("sessions", f"{len(trades):,}"),
        stat_row("strike", f"+{STRIKE_PCT:.0f}%"),
        stat_row("stop", f"-{STOP_PCT:.1f}%"),
        stat_row("tail mass", f"{tail['tail_mass_pct']:.2f}%"),
        stat_row("implied mult",
                 f"x{tail['implied_multiple']:.1f}" if tail["implied_multiple"] else "n/a"),
        stat_row("stop mass", f"{tail['stop_mass_pct']:.2f}%", "neg"),
        stat_row("tp exits", f"{tp_exits:,}", "pos"),
        stat_row("time exits", f"{time_exits:,}"),
        stat_row("avg hold", f"{sum(holds)/len(holds):.1f} bars"),
        stat_row("avg entry", f"{sum(entries)/len(entries):,.4g}"),
        stat_row("best hit", f"{max(rets):+.2f}%", "pos"),
        stat_row("avg return", f"{overall.avg_pnl_pct:+.3f}%", _tone(overall.avg_pnl_pct)),
        stat_row("realised", _money(overall.total_pnl_usdt), _tone(overall.total_pnl_usdt)),
    ]


def hero_rows(trades: list[Trade], overall, hits, symbols_count: int, scan_seconds: float) -> list[dict]:
    """Left stat column of the hero block."""
    rsis = [h.rsi for h in hits]
    vols = [h.vol_ratio for h in hits]
    changes = [h.change_24h_pct for h in hits if h.change_24h_pct is not None]
    rows = [
        stat_row("symbols scanned", f"{symbols_count:,}"),
        stat_row("scan hits", f"{len(hits)}"),
        stat_row("scan time", f"{scan_seconds:.1f}s"),
        stat_row("median rsi", f"{median(rsis):.1f}" if rsis else "n/a"),
        stat_row("median vol x", f"{median(vols):.2f}" if vols else "n/a"),
        stat_row("median 24h", f"{median(changes):+.2f}%" if changes else "n/a",
                 _tone(median(changes)) if changes else ""),
    ]
    if trades:
        rets = [t.pnl_pct for t in trades]
        rows += [
            stat_row("trades", f"{len(trades):,}"),
            stat_row("win rate", f"{overall.win_rate_pct:.1f}%"),
            stat_row("ev / trade", _money(overall.total_pnl_usdt / len(trades)),
                     _tone(overall.total_pnl_usdt)),
            stat_row("sharpe", f"{sharpe(rets):.2f}"),
            stat_row("max drawdown", f"{overall.max_drawdown_usdt:,.2f}", "neg"),
            stat_row("realised", _money(overall.total_pnl_usdt), _tone(overall.total_pnl_usdt)),
        ]
    return rows


def median_fair(hits, trades: list[Trade]) -> float | None:
    """Median scan-hit close projected forward by the median realised trade return."""
    closes = [h.close for h in hits]
    if not closes:
        return None
    if not trades:
        return round(median(closes), 6)
    m = median([t.pnl_pct for t in trades]) / 100.0
    return round(median(closes) * (1 + m), 6)
