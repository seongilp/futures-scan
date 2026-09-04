"""Render a per-symbol chart HTML page: candles + volume + RSI + whale-momentum + support + entry zones."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from futures_scan.indicators import rsi, whale_momentum
from futures_scan.scan import ScanHit
from futures_scan.strategy import compute_signals
from futures_scan.support import SupportLevel, find_support_levels

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=True),
    )


def symbol_to_filename(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-") + ".html"


def build_chart_context(
    df: pd.DataFrame, symbol: str, hit: ScanHit | None, exchange: str, timeframe: str
) -> dict:
    working = df.reset_index(drop=True).copy()
    working["rsi"] = rsi(working["close"], period=14)
    whale = whale_momentum(working)
    signals = compute_signals(working)
    support_levels = find_support_levels(working)

    candles = [
        {
            "timestamp": int(row.timestamp),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "rsi": None if pd.isna(row.rsi) else round(float(row.rsi), 2),
        }
        for row in working.itertuples()
    ]

    signal_bars = [
        {"timestamp": int(working["timestamp"].iloc[i])}
        for i in range(len(working))
        if bool(signals.iloc[i])
    ]

    whale_series = [
        {"timestamp": int(ts), "value": round(float(v), 2)}
        for ts, v in zip(working["timestamp"], whale)
    ]

    return {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "candles": candles,
        "candle_count": len(candles),
        "support_levels": [asdict(lv) for lv in support_levels],
        "signal_bars": signal_bars,
        "whale_momentum": whale_series,
        "rsi": hit.rsi if hit else (None if candles[-1]["rsi"] is None else candles[-1]["rsi"]),
        "vol_ratio": hit.vol_ratio if hit else 0.0,
        "change_24h_pct": hit.change_24h_pct if hit else None,
    }


def render_chart(context: dict, out_path: Path) -> Path:
    env = _env()
    template = env.get_template("chart.html.j2")
    html = template.render(**context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
