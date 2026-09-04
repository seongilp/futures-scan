"""Render a candle-by-candle replay HTML page for a symbol."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from futures_scan.strategy import generate_trades

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=True),
    )


def build_replay_context(df: pd.DataFrame, symbol: str, exchange: str, timeframe: str, days: int) -> dict:
    working = df.reset_index(drop=True)
    candles = [
        {
            "timestamp": int(row.timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in working.itertuples()
    ]
    python_trades = generate_trades(working, symbol)
    wins = [t for t in python_trades if t.pnl_usdt > 0]
    pnl = sum(t.pnl_usdt for t in python_trades)
    closes = working["close"].astype(float)
    stat_rows = [
        {"label": "bars", "value": f"{len(working):,}", "tone": ""},
        {"label": "window", "value": f"{days}d", "tone": ""},
        {"label": "timeframe", "value": timeframe, "tone": ""},
        {"label": "last close", "value": f"{float(closes.iloc[-1]):.6g}", "tone": ""},
        {"label": "window return",
         "value": f"{100*(float(closes.iloc[-1])-float(closes.iloc[0]))/float(closes.iloc[0]):+.2f}%",
         "tone": "pos" if closes.iloc[-1] >= closes.iloc[0] else "neg"},
        {"label": "reference fills", "value": f"{len(python_trades)}", "tone": ""},
        {"label": "reference wins", "value": f"{len(wins)}", "tone": "pos"},
        {"label": "reference wr",
         "value": f"{100*len(wins)/len(python_trades):.1f}%" if python_trades else "n/a", "tone": ""},
        {"label": "reference pnl", "value": f"{pnl:+.2f}", "tone": "pos" if pnl >= 0 else "neg"},
        {"label": "take profit", "value": "+3.0%", "tone": "pos"},
        {"label": "stop loss", "value": "-1.5%", "tone": "neg"},
        {"label": "max hold", "value": "24 bars", "tone": ""},
        {"label": "fee / side", "value": "0.04%", "tone": ""},
        {"label": "notional", "value": "1,000", "tone": ""},
    ]
    return {
        "stat_rows": stat_rows,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "days": days,
        "candles": candles,
        "python_trades": [asdict(t) for t in python_trades],
        "python_trade_count": len(python_trades),
    }


def render_replay(context: dict, out_path: Path) -> Path:
    env = _env()
    template = env.get_template("replay.html.j2")
    html = template.render(**context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def bars_for_days(timeframe: str, days: int) -> int:
    tf_hours = {"1m": 1 / 60, "5m": 5 / 60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}.get(timeframe, 1)
    return max(30, int((days * 24) / tf_hours))
