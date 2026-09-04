"""Build the out/index.html dashboard: scan hits, backtest results, and the
probability-lattice / tail-ridge / relationship-graph visual sections.

All numbers passed into the template come from real scan/backtest output or the
local candle cache — never fabricated. See README.md for metric definitions.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from pathlib import Path
from statistics import median

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from futures_scan.backtest import BacktestSummary
from futures_scan.render import stats as st
from futures_scan.render.graph import build_graph
from futures_scan.scan import ScanHit
from futures_scan.strategy import Trade

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=True),
    )


def _iso(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M UTC")


def _short(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%b %d %H:%M")


def _trade_card(t: Trade) -> dict:
    return {
        "symbol": t.symbol,
        "short": t.symbol.split("/")[0],
        "multiple": round(1 + t.pnl_pct / 100, 4),
        "pnl_pct": t.pnl_pct,
        "pnl_usdt": t.pnl_usdt,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "entry_time_iso": _iso(t.entry_time),
        "entry_time_short": _short(t.entry_time),
        "exit_time_short": _short(t.exit_time),
        "exit_reason": t.exit_reason,
        "bars_held": t.bars_held,
    }


def build_index_context(
    exchange: str,
    timeframe: str,
    symbols_count: int,
    scan_seconds: float,
    hits: list[ScanHit],
    overall: BacktestSummary,
    trades_by_symbol: dict[str, list[Trade]],
    candles_by_symbol: dict[str, pd.DataFrame],
    scan_round: int = 1,
) -> dict:
    all_trades: list[Trade] = [t for trades in trades_by_symbol.values() for t in trades]
    all_trades_sorted = sorted(all_trades, key=lambda t: t.exit_time)

    per_symbol_pnl = {s: sum(t.pnl_usdt for t in ts) for s, ts in trades_by_symbol.items() if ts}
    top_symbols = [
        {
            "symbol": s,
            "short": s.split("/")[0],
            "total_pnl_usdt": round(pnl, 2),
            "total_trades": len(trades_by_symbol[s]),
            "win_rate_pct": round(
                100 * sum(1 for t in trades_by_symbol[s] if t.pnl_usdt > 0) / len(trades_by_symbol[s]), 1
            ),
        }
        for s, pnl in sorted(per_symbol_pnl.items(), key=lambda kv: kv[1], reverse=True)[:4]
    ]

    top_wins = [_trade_card(t) for t in sorted(all_trades, key=lambda t: t.pnl_pct, reverse=True)[:5]]
    best_trade = top_wins[0] if top_wins else None

    running = 0.0
    pnl_curve: list[float] = []
    for t in all_trades_sorted:
        running += t.pnl_usdt
        pnl_curve.append(round(running, 2))

    trades_flat = [
        {
            "symbol": t.symbol,
            "short": t.symbol.split("/")[0],
            "pnl_usdt": t.pnl_usdt,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "win": t.pnl_usdt > 0,
        }
        for t in all_trades_sorted
    ]

    trades_by_symbol_pct = {s: [t.pnl_pct for t in ts] for s, ts in trades_by_symbol.items() if ts}

    change_by_symbol = {h.symbol: h.change_24h_pct for h in hits}
    graph = build_graph(
        exchange=exchange,
        timeframe=timeframe,
        hit_symbols=[h.symbol for h in hits],
        change_by_symbol=change_by_symbol,
        pnl_by_symbol=per_symbol_pnl,
    )

    rets = [t.pnl_pct for t in all_trades]
    tail = st.tail_stats(rets)
    win_rate = overall.win_rate_pct
    edge_pp = round(win_rate - st.BREAKEVEN_WR_PCT, 1) if all_trades else 0.0

    change_24h_list = [h.change_24h_pct for h in hits if h.change_24h_pct is not None]

    graph_rows = [
        st.stat_row("bear paths", f"{graph.get('bear_paths', 0):,}", "neg"),
        st.stat_row("bull paths", f"{graph.get('bull_paths', 0):,}", "pos"),
        st.stat_row("paths sim", f"{len(graph.get('edges', [])):,}"),
        st.stat_row("bear nodes", f"{graph.get('bear_nodes', 0)}", "neg"),
        st.stat_row("bull nodes", f"{graph.get('bull_nodes', 0)}", "pos"),
        st.stat_row("neighbours", f"{graph.get('neighbour_count', 0)}"),
        st.stat_row("mean |rho|", f"{graph.get('signal_pct', 0.0):.1f}%"),
    ]

    fair = st.median_fair(hits, all_trades)

    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "symbols_count": symbols_count,
        "scan_seconds": round(scan_seconds, 1),
        "scan_round": scan_round,
        "generated_at_utc": dt.datetime.utcnow().strftime("%H:%M:%S"),
        "generated_date": dt.datetime.utcnow().strftime("%Y-%m-%d"),
        "hits": [asdict(h) for h in hits],
        "overall": asdict(overall),
        "top_symbols": top_symbols,
        "top_wins": top_wins,
        "best_trade": best_trade,
        "pnl_curve": pnl_curve,
        "trades": trades_flat,
        "trades_by_symbol_pct": trades_by_symbol_pct,
        # graph
        "edges": graph.get("edges", []),
        "nodes": graph.get("nodes", []),
        "hubs": graph.get("hubs", []),
        "p_up": graph.get("p_up", 0.0),
        "p_down": graph.get("p_down", 0.0),
        "signal_pct": graph.get("signal_pct", 0.0),
        "graph_rows": graph_rows,
        "predicted_dir": "UP" if graph.get("p_up", 0.0) >= 50 else "DOWN",
        "median_fair": fair,
        "confidence_pct": max(graph.get("p_up", 0.0), graph.get("p_down", 0.0)),
        # stat columns
        "hero_rows": st.hero_rows(all_trades, overall, hits, symbols_count, scan_seconds),
        "lattice_rows": st.lattice_rows(all_trades, overall, pnl_curve),
        "ridge_rows": st.ridge_rows(all_trades, overall, len(trades_by_symbol_pct)),
        # derived scalars
        "sharpe": st.sharpe(rets),
        "strike_pct": st.STRIKE_PCT,
        "stop_pct": st.STOP_PCT,
        "breakeven_wr": round(st.BREAKEVEN_WR_PCT, 1),
        "edge_pp": edge_pp,
        "tail_mass_pct": tail["tail_mass_pct"],
        "tail_count": tail["tail_count"],
        "implied_multiple": tail["implied_multiple"],
        "median_return_pct": round(median(rets), 3) if rets else 0.0,
        "loss_share_pct": round(100 - win_rate, 1) if all_trades else 0.0,
        "profit_share_pct": win_rate,
        "change_24h_list": change_24h_list,
        "chart_links": [
            {"symbol": h.symbol, "path": f"charts/{h.symbol.replace('/', '-').replace(':', '-')}.html",
             "replay": f"replay/{h.symbol.replace('/', '-').replace(':', '-')}.html"}
            for h in hits
        ],
    }


def render_index(context: dict, out_path: Path) -> Path:
    env = _env()
    template = env.get_template("index.html.j2")
    html = template.render(**context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
