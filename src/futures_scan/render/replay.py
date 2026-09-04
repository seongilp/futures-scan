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
    return {
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
