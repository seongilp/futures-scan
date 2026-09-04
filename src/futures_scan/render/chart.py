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

    closes = working["close"].astype(float)
    vols = working["volume"].astype(float)
    last_whale = float(whale.iloc[-1]) if len(whale) else 0.0
    window_high = float(working["high"].max())
    window_low = float(working["low"].min())
    last_close = float(closes.iloc[-1])
    first_close = float(closes.iloc[0])
    stat_rows = [
        {"label": "bars", "value": f"{len(working):,}", "tone": ""},
        {"label": "last close", "value": f"{last_close:.6g}", "tone": ""},
        {"label": "window high", "value": f"{window_high:.6g}", "tone": ""},
        {"label": "window low", "value": f"{window_low:.6g}", "tone": ""},
        {"label": "window range", "value": f"{100*(window_high-window_low)/window_low:.1f}%", "tone": ""},
        {"label": "window return", "value": f"{100*(last_close-first_close)/first_close:+.2f}%",
         "tone": "pos" if last_close >= first_close else "neg"},
        {"label": "rsi(14)", "value": f"{float(working['rsi'].iloc[-1]):.1f}" if not pd.isna(working["rsi"].iloc[-1]) else "n/a", "tone": ""},
        {"label": "vol x avg", "value": f"{float(vols.iloc[-1]) / float(vols.tail(20).mean()):.2f}", "tone": ""},
        {"label": "avg volume", "value": f"{float(vols.mean()):,.0f}", "tone": ""},
        {"label": "whale mom", "value": f"{last_whale:+.1f}",
         "tone": "pos" if last_whale >= 0 else "neg"},
        {"label": "entry bars", "value": f"{sum(1 for i in range(len(working)) if bool(signals.iloc[i]))}", "tone": ""},
        {"label": "support lv", "value": f"{len(support_levels)}", "tone": ""},
    ]

    return {
        "stat_rows": stat_rows,
        "symbol": symbol,
        "short": symbol.split("/")[0],
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
