"""Typer CLI: scan, chart, backtest, replay, indicator, all."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from futures_scan import fetch
from futures_scan.backtest import per_symbol_summaries, run_backtest, summarize, summary_to_markdown
from futures_scan.indicators import whale_momentum
from futures_scan.render.chart import build_chart_context, render_chart, symbol_to_filename
from futures_scan.render.index import build_index_context, render_index
from futures_scan.render.replay import bars_for_days, build_replay_context, render_replay
from futures_scan.scan import DEFAULT_RSI_THRESHOLD, DEFAULT_VOL_MULT, ScanHit, evaluate_symbol

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(show_path=False)])
logger = logging.getLogger("futures_scan")

app = typer.Typer(help="RSI + volume-spike scanner for crypto perpetual futures.")
console = Console()

OUT_DIR = Path("out")
CHARTS_DIR = OUT_DIR / "charts"
REPLAY_DIR = OUT_DIR / "replay"


def _default_symbol_limit(limit: int | None) -> int | None:
    return limit


async def _fetch_universe(timeframe: str, limit: int, symbol_limit: int | None):
    exchange = await fetch.select_exchange()
    try:
        symbols = await fetch.get_symbols(exchange)
        if symbol_limit:
            symbols = symbols[:symbol_limit]

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"fetching {timeframe} candles from {exchange.id}", total=len(symbols))

            def on_progress(done, total, symbol):
                progress.update(task, completed=done)

            results = await fetch.fetch_many(exchange, symbols, timeframe, limit, on_progress=on_progress)
        return exchange.id, results
    finally:
        await exchange.close()


def _candles_by_symbol(results) -> dict[str, pd.DataFrame]:
    return {r.symbol: r.df for r in results if r.df is not None and not r.df.empty}


@app.command()
def scan(
    tf: str = typer.Option("1h", help="Timeframe, e.g. 1h, 15m, 4h"),
    limit: int = typer.Option(500, help="Candles per symbol"),
    rsi: float = typer.Option(DEFAULT_RSI_THRESHOLD, help="RSI threshold (below = hit)"),
    vol_mult: float = typer.Option(DEFAULT_VOL_MULT, help="Volume multiple vs 20-bar avg (3.0 = +200%)"),
    symbol_limit: int = typer.Option(None, help="Limit number of symbols (debug)"),
):
    """Scan all USDT perpetual futures for RSI<30 + volume-spike hits."""
    start = time.time()
    exchange_id, results = asyncio.run(_fetch_universe(tf, limit, symbol_limit))
    errors = [r for r in results if r.error]
    if errors:
        logger.warning("%d symbols failed to fetch (e.g. %s: %s)", len(errors), errors[0].symbol, errors[0].error)

    hits: list[ScanHit] = []
    for r in results:
        if r.df is None:
            continue
        hit = evaluate_symbol(r.df, r.symbol, rsi_threshold=rsi, vol_mult=vol_mult)
        if hit:
            hits.append(hit)

    elapsed = time.time() - start
    table = Table(title=f"Scan hits: RSI<{rsi} & vol>={vol_mult}x  ({exchange_id}, {tf})")
    table.add_column("Symbol")
    table.add_column("RSI", justify="right")
    table.add_column("Vol x", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("24h %", justify="right")
    for h in sorted(hits, key=lambda x: x.rsi):
        table.add_row(h.symbol, f"{h.rsi:.1f}", f"{h.vol_ratio:.2f}", f"{h.close:.6g}", f"{h.change_24h_pct}")
    console.print(table)
    console.print(f"[bold]{len(hits)}[/bold] hits out of {len(results)} symbols in {elapsed:.1f}s on {exchange_id}")

    OUT_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    out_path = OUT_DIR / f"scan_{ts}.json"
    payload = {
        "exchange": exchange_id,
        "timeframe": tf,
        "rsi_threshold": rsi,
        "vol_mult": vol_mult,
        "symbols_scanned": len(results),
        "elapsed_seconds": elapsed,
        "hits": [asdict(h) for h in hits],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    console.print(f"saved [cyan]{out_path}[/cyan]")
    return payload


def _load_latest_scan() -> dict:
    files = sorted(OUT_DIR.glob("scan_*.json"))
    if not files:
        raise typer.BadParameter("No scan_*.json found. Run `futures-scan scan` first.")
    return json.loads(files[-1].read_text())


@app.command()
def chart(
    scan_file: str = typer.Option(None, help="Path to a scan_*.json; defaults to the latest"),
    tf: str = typer.Option(None, help="Override timeframe for chart candle fetch"),
    limit: int = typer.Option(500, help="Candles per symbol"),
):
    """Render per-symbol chart HTML pages + out/index.html for the latest (or given) scan."""
    payload = json.loads(Path(scan_file).read_text()) if scan_file else _load_latest_scan()
    exchange_id = payload["exchange"]
    timeframe = tf or payload["timeframe"]
    hits = [ScanHit(**h) for h in payload["hits"]]

    symbols = [h.symbol for h in hits]
    if not symbols:
        console.print("[yellow]No scan hits to chart.[/yellow]")
        _write_index(exchange_id, timeframe, payload["symbols_scanned"], payload["elapsed_seconds"], [], {}, {})
        return

    exchange = asyncio.run(fetch.select_exchange([exchange_id, *fetch.EXCHANGE_FALLBACK_CHAIN]))
    try:
        results = asyncio.run(fetch.fetch_many(exchange, symbols, timeframe, limit))
    finally:
        asyncio.run(exchange.close())

    candles_by_symbol = _candles_by_symbol(results)
    hits_by_symbol = {h.symbol: h for h in hits}

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for symbol, df in candles_by_symbol.items():
        ctx = build_chart_context(df, symbol, hits_by_symbol.get(symbol), exchange_id, timeframe)
        render_chart(ctx, CHARTS_DIR / symbol_to_filename(symbol))
    console.print(f"rendered [bold]{len(candles_by_symbol)}[/bold] chart pages to [cyan]{CHARTS_DIR}[/cyan]")

    # The dashboard shows realised strategy results, so derive them from the
    # candles we already have (no extra network calls, same rules as `backtest`).
    trades_by_symbol, _ = run_backtest(candles_by_symbol)
    _write_index(exchange_id, timeframe, payload["symbols_scanned"], payload["elapsed_seconds"], hits, trades_by_symbol, candles_by_symbol)


def _write_index(exchange_id, timeframe, symbols_count, scan_seconds, hits, trades_by_symbol, candles_by_symbol):
    overall = summarize([t for ts in trades_by_symbol.values() for t in ts])
    ctx = build_index_context(
        exchange=exchange_id,
        timeframe=timeframe,
        symbols_count=symbols_count,
        scan_seconds=scan_seconds,
        hits=hits,
        overall=overall,
        trades_by_symbol=trades_by_symbol,
        candles_by_symbol=candles_by_symbol,
    )
    render_index(ctx, OUT_DIR / "index.html")
    console.print(f"wrote [cyan]{OUT_DIR / 'index.html'}[/cyan]")


@app.command()
def backtest(
    scan_file: str = typer.Option(None, help="Path to a scan_*.json; defaults to the latest"),
    tf: str = typer.Option(None, help="Override timeframe"),
    limit: int = typer.Option(500, help="Candles per symbol"),
):
    """Backtest the RSI+volume-spike strategy over the latest scan's hit symbols."""
    payload = json.loads(Path(scan_file).read_text()) if scan_file else _load_latest_scan()
    exchange_id = payload["exchange"]
    timeframe = tf or payload["timeframe"]
    hits = [ScanHit(**h) for h in payload["hits"]]
    symbols = [h.symbol for h in hits]

    if not symbols:
        console.print("[yellow]No scan hits to backtest.[/yellow]")
        return

    exchange = asyncio.run(fetch.select_exchange([exchange_id, *fetch.EXCHANGE_FALLBACK_CHAIN]))
    try:
        results = asyncio.run(fetch.fetch_many(exchange, symbols, timeframe, limit))
    finally:
        asyncio.run(exchange.close())

    candles_by_symbol = _candles_by_symbol(results)
    trades_by_symbol, overall = run_backtest(candles_by_symbol)
    per_symbol = per_symbol_summaries(trades_by_symbol)

    console.print(f"[bold]{overall.total_trades}[/bold] trades, win rate {overall.win_rate_pct}%, "
                  f"total PnL {overall.total_pnl_usdt} USDT, MDD {overall.max_drawdown_usdt} USDT")
    console.print("[bold red]백테스트는 실제가 아님, 슬리피지·펀딩 미반영[/bold red]")

    OUT_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    out_json = OUT_DIR / f"backtest_{ts}.json"
    out_json.write_text(json.dumps({
        "overall": asdict(overall),
        "per_symbol": {s: asdict(v) for s, v in per_symbol.items()},
        "trades": {s: [asdict(t) for t in ts_] for s, ts_ in trades_by_symbol.items()},
    }, indent=2))
    out_md = OUT_DIR / f"backtest_{ts}.md"
    out_md.write_text(summary_to_markdown(overall, per_symbol))
    console.print(f"saved [cyan]{out_json}[/cyan] and [cyan]{out_md}[/cyan]")

    _write_index(exchange_id, timeframe, payload["symbols_scanned"], payload["elapsed_seconds"], hits, trades_by_symbol, candles_by_symbol)
    return trades_by_symbol, overall


@app.command()
def replay(
    days: int = typer.Option(7, help="How many days back to replay"),
    scan_file: str = typer.Option(None, help="Path to a scan_*.json; defaults to the latest"),
    tf: str = typer.Option(None, help="Override timeframe"),
    symbol: str = typer.Option(None, help="Replay a single symbol only"),
):
    """Render a candle-by-candle replay HTML page per scan-hit symbol."""
    payload = json.loads(Path(scan_file).read_text()) if scan_file else _load_latest_scan()
    exchange_id = payload["exchange"]
    timeframe = tf or payload["timeframe"]
    hits = [ScanHit(**h) for h in payload["hits"]]
    symbols = [symbol] if symbol else [h.symbol for h in hits]

    if not symbols:
        console.print("[yellow]No scan hits to replay.[/yellow]")
        return

    limit = bars_for_days(timeframe, days)
    exchange = asyncio.run(fetch.select_exchange([exchange_id, *fetch.EXCHANGE_FALLBACK_CHAIN]))
    try:
        results = asyncio.run(fetch.fetch_many(exchange, symbols, timeframe, limit))
    finally:
        asyncio.run(exchange.close())

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for r in results:
        if r.df is None or r.df.empty:
            continue
        ctx = build_replay_context(r.df, r.symbol, exchange_id, timeframe, days)
        render_replay(ctx, REPLAY_DIR / symbol_to_filename(r.symbol))
        rendered += 1
    console.print(f"rendered [bold]{rendered}[/bold] replay pages to [cyan]{REPLAY_DIR}[/cyan]")


indicator_app = typer.Typer(help="Standalone indicator computations")
app.add_typer(indicator_app, name="indicator")


@indicator_app.command("whale-momentum")
def indicator_whale_momentum(
    symbol: str = typer.Argument(..., help="e.g. BTC/USDT:USDT"),
    tf: str = typer.Option("1h"),
    limit: int = typer.Option(500),
    exchange_id: str = typer.Option(None, "--exchange"),
):
    """Compute the custom whale-momentum oscillator for one symbol and print recent values."""
    chain = [exchange_id, *fetch.EXCHANGE_FALLBACK_CHAIN] if exchange_id else None
    exchange = asyncio.run(fetch.select_exchange(chain))
    try:
        semaphore = asyncio.Semaphore(1)
        df = asyncio.run(fetch.fetch_candles_incremental(exchange, symbol, tf, limit, semaphore))
    finally:
        asyncio.run(exchange.close())

    series = whale_momentum(df)
    table = Table(title=f"whale-momentum: {symbol} ({tf})")
    table.add_column("timestamp")
    table.add_column("value", justify="right")
    for ts, val in list(zip(df["timestamp"], series))[-20:]:
        table.add_row(str(int(ts)), f"{val:.2f}")
    console.print(table)


@app.command(name="all")
def run_all(
    tf: str = typer.Option("1h", help="Timeframe"),
    limit: int = typer.Option(500, help="Candles per symbol"),
    rsi: float = typer.Option(DEFAULT_RSI_THRESHOLD),
    vol_mult: float = typer.Option(DEFAULT_VOL_MULT),
    days: int = typer.Option(7, help="Replay window in days"),
    symbol_limit: int = typer.Option(None, help="Limit number of symbols (debug)"),
):
    """Run scan -> chart -> backtest -> replay end to end."""
    scan_payload = scan(tf=tf, limit=limit, rsi=rsi, vol_mult=vol_mult, symbol_limit=symbol_limit)
    scan_files = sorted(OUT_DIR.glob("scan_*.json"))
    scan_file = str(scan_files[-1])

    chart(scan_file=scan_file, tf=tf, limit=limit)
    if scan_payload["hits"]:
        backtest(scan_file=scan_file, tf=tf, limit=limit)
        replay(days=days, scan_file=scan_file, tf=tf, symbol=None)

    console.print(f"[bold green]done.[/bold green] open [cyan]{(OUT_DIR / 'index.html').resolve()}[/cyan]")


if __name__ == "__main__":
    app()
