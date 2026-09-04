"""Verify the JS strategy engine embedded in replay.html.j2 produces the same trade
count as the Python backtest engine, for the exact same candle data.

This runs the JS with Node (no DOM needed — we only exercise the pure strategy
functions, extracted from the template between the "Strategy engine" / "Replay UI"
markers) so it works fully offline.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from futures_scan.render.replay import build_replay_context, render_replay
from futures_scan.strategy import generate_trades

from .fixtures import flat_then_crash_and_spike, make_ohlcv

TEMPLATE_PATH = Path(__file__).parent.parent / "src/futures_scan/render/templates/replay.html.j2"

NODE_AVAILABLE = shutil.which("node") is not None


def _extract_engine_js(rendered_html: str) -> str:
    match = re.search(
        r"// ---- Strategy engine.*?\n(.*?)// ---- Replay UI ----",
        rendered_html,
        re.DOTALL,
    )
    assert match, "could not find strategy-engine JS block in rendered replay HTML"
    return match.group(1)


def _extract_json_blob(rendered_html: str, element_id: str) -> str:
    pattern = rf'<script type="application/json" id="{element_id}">(.*?)</script>'
    match = re.search(pattern, rendered_html, re.DOTALL)
    assert match, f"could not find #{element_id} json blob"
    return match.group(1)


def _multi_episode_candles():
    """Chain several crash+spike episodes so the strategy fires more than once."""
    frames = [flat_then_crash_and_spike(n_flat=30) for _ in range(3)]
    closes: list[float] = []
    volumes: list[float] = []
    for f in frames:
        closes.extend(f["close"].tolist())
        volumes.extend(f["volume"].tolist())
    return make_ohlcv(closes, volumes)


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is required to execute the replay JS engine")
def test_js_and_python_trade_counts_match(tmp_path):
    df = _multi_episode_candles()
    symbol = "PARITY/USDT:USDT"
    ctx = build_replay_context(df, symbol, "binanceusdm", "1h", days=30)
    out_path = render_replay(ctx, tmp_path / "replay.html")
    html = out_path.read_text()

    engine_js = _extract_engine_js(html)
    candles_json = _extract_json_blob(html, "candles-data")
    python_trades_json = _extract_json_blob(html, "python-trades-data")

    python_trade_count = len(json.loads(python_trades_json))
    assert python_trade_count == len(generate_trades(df.reset_index(drop=True), symbol))
    assert python_trade_count > 0, "fixture should produce at least one trade"

    node_script = f"""
{engine_js}
const candles = {candles_json};
const trades = generateAllTrades(candles);
console.log(JSON.stringify({{ count: trades.length }}));
"""
    result = subprocess.run(
        ["node", "-e", node_script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"node execution failed: {result.stderr}"
    js_result = json.loads(result.stdout.strip().splitlines()[-1])

    assert js_result["count"] == python_trade_count
