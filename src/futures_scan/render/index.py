"""Build the out/index.html dashboard: scan hits, backtest results, and the
probability-lattice / tail-ridge / relationship-graph visual sections.

All numbers passed into the template come from real scan/backtest output —
never fabricated.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from futures_scan.backtest import BacktestSummary
from futures_scan.scan import ScanHit
from futures_scan.strategy import Trade

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=True),
    )


def _node_kind(symbol: str, trades: list[Trade]) -> str:
    if symbol.startswith("BTC/"):
        return "hub"
    pnl = sum(t.pnl_usdt for t in trades)
    if not trades:
        return "catalyst"
    return "bull" if pnl >= 0 else "bear"


def _correlation_edges(returns_by_symbol: dict[str, pd.Series], threshold: float = 0.5) -> list[dict]:
    symbols = list(returns_by_symbol.keys())
    edges: list[dict] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            joined = pd.concat([returns_by_symbol[a], returns_by_symbol[b]], axis=1, join="inner")
            joined.columns = ["a", "b"]
            joined = joined.dropna()
            if len(joined) < 20:
                continue
            rho = float(joined["a"].corr(joined["b"]))
            if pd.isna(rho) or abs(rho) < threshold:
                continue
            edges.append({"a": a, "b": b, "rho": round(rho, 3)})
    return edges


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

    per_symbol_pnl = {
        s: sum(t.pnl_usdt for t in ts) for s, ts in trades_by_symbol.items() if ts
    }
    top_symbols = [
        {
            "symbol": s,
            "total_pnl_usdt": round(pnl, 2),
            "total_trades": len(trades_by_symbol[s]),
            "win_rate_pct": round(
                100 * sum(1 for t in trades_by_symbol[s] if t.pnl_usdt > 0) / len(trades_by_symbol[s]), 1
            ),
        }
        for s, pnl in sorted(per_symbol_pnl.items(), key=lambda kv: kv[1], reverse=True)[:4]
    ]

    best_trade = None
    if all_trades:
        bt = max(all_trades, key=lambda t: t.pnl_pct)
        best_trade = {
            "symbol": bt.symbol,
            "pnl_pct": bt.pnl_pct,
            "pnl_usdt": bt.pnl_usdt,
            "entry_price": bt.entry_price,
            "exit_price": bt.exit_price,
            "entry_time_iso": dt.datetime.utcfromtimestamp(bt.entry_time / 1000).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        }

    running = 0.0
    pnl_curve = []
    for t in all_trades_sorted:
        running += t.pnl_usdt
        pnl_curve.append(round(running, 2))

    trades_flat = [
        {
            "symbol": t.symbol,
            "pnl_usdt": t.pnl_usdt,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "win": t.pnl_usdt > 0,
        }
        for t in all_trades_sorted
    ]

    trades_by_symbol_pct = {
        s: [t.pnl_pct for t in ts] for s, ts in trades_by_symbol.items() if ts
    }

    returns_by_symbol = {
        s: df["close"].pct_change().dropna() for s, df in candles_by_symbol.items() if len(df) > 20
    }
    edges = _correlation_edges(returns_by_symbol)

    nodes = [
        {"symbol": s, "kind": _node_kind(s, trades_by_symbol.get(s, []))}
        for s in candles_by_symbol.keys()
    ]

    win_trades = sum(1 for t in all_trades if t.pnl_usdt > 0)
    p_up = round(100 * win_trades / len(all_trades), 1) if all_trades else 0.0

    change_24h_list = [h.change_24h_pct for h in hits if h.change_24h_pct is not None]

    return {
        "exchange": exchange,
        "timeframe": timeframe,
        "symbols_count": symbols_count,
        "scan_seconds": round(scan_seconds, 1),
        "scan_round": scan_round,
        "generated_at_utc": dt.datetime.utcnow().strftime("%H:%M:%S"),
        "hits": [asdict(h) for h in hits],
        "overall": asdict(overall),
        "top_symbols": top_symbols,
        "best_trade": best_trade,
        "pnl_curve": pnl_curve,
        "trades": trades_flat,
        "trades_by_symbol_pct": trades_by_symbol_pct,
        "edges": edges,
        "nodes": nodes,
        "p_up": p_up,
        "p_down": round(100 - p_up, 1) if all_trades else 0.0,
        "change_24h_list": change_24h_list,
        "chart_links": [{"symbol": h.symbol, "path": f"charts/{h.symbol.replace('/', '-').replace(':', '-')}.html"} for h in hits],
    }


def render_index(context: dict, out_path: Path) -> Path:
    env = _env()
    template = env.get_template("index.html.j2")
    html = template.render(**context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
