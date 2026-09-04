# futures-scan

CLI for crypto perpetual-futures RSI + volume-spike scanning, chart generation, support
levels, entry zones, backtesting, candle-by-candle replay, and a custom oscillator. No
TradingView, no Pine Script — pure Python + HTML (lightweight-charts, d3-force).

Read-only. No API keys. No orders are ever placed.

## Install

```bash
uv sync
```

## Usage

```bash
uv run futures-scan scan --tf 1h --rsi 30 --vol-mult 3.0
uv run futures-scan chart                 # renders out/charts/*.html + out/index.html
uv run futures-scan backtest              # runs the strategy over the scan hits
uv run futures-scan replay --days 7       # candle-by-candle replay per hit symbol
uv run futures-scan indicator whale-momentum BTC/USDT:USDT
uv run futures-scan all --tf 1h           # scan -> chart -> backtest -> replay
```

Open `out/index.html` when `all` (or `chart`) finishes.

## Rule definitions

### Scan condition
- RSI(14), Wilder-smoothed, on the **last closed bar** is `< --rsi` (default 30).
- **AND** the current bar's or the **previous** bar's volume is `>= --vol-mult` times
  (default 3.0x = +200%) the trailing 20-bar average volume (average excludes the bar
  being compared, i.e. it looks strictly backward).

### Support levels
- Pivot lows: a bar's `low` is the minimum within a +/-5 bar window (and strictly unique
  in that window).
- Pivot prices within 0.5% of each other are clustered into one level.
- A cluster becomes a support line only if it has >= 2 touches.

### Entry zone
- Any bar that satisfies the scan condition is marked with an "ENTRY" marker.
- Each support level gets a +/-0.5% dotted band around it on the chart.

### Backtest strategy
- Entry: on the **open of the bar after** a scan-condition close.
- Take-profit: entry price `+3%`.
- Stop-loss: entry price `-1.5%` (checked before take-profit if both would trigger on the
  same bar — the conservative assumption).
- Max hold: 24 bars, exits at that bar's close if neither TP nor SL is hit.
- Fees: 0.04% **one-way**, charged on both entry and exit (0.08% round-trip total).
- Notional: 1,000 USDT per trade (fixed, no compounding, no leverage math).

> **This backtest is NOT real trading. Slippage and funding fees are NOT modeled.**
> (백테스트는 실제가 아님, 슬리피지·펀딩 미반영 — this disclaimer is also printed by the CLI
> and embedded in every backtest report.)

### Custom oscillator: whale-momentum (`indicators.py::whale_momentum`)
```
whale_proxy = zscore(volume, 20) * (|close - open| / ATR(14))
trend       = (EMA(9) - EMA(21)) / close
raw         = whale_proxy * trend
oscillator  = EMA(raw, 5), rescaled so +/-3 std -> +/-100, clipped to [-100, 100]
```
"Whale activity" proxy (volume anomaly x candle body size relative to ATR) multiplied by
a normalized short/long EMA trend, smoothed. When a rolling window has zero variance
(e.g. perfectly flat volume), its z-score is defined as 0, not NaN/undefined.

## Exchange fallback

Tries `binanceusdm` (Binance USDT-M perpetuals) first. If unreachable (e.g. HTTP 451 /
geo-block), falls back to `bybit`, then `okx`, logging a `FALLBACK:` warning each time.
All USDT-margined perpetual swap symbols on the selected exchange are scanned.

## Caching

- Candles: `data/candles/{exchange}/{symbol}_{tf}.parquet`. Re-fetches only fetch bars
  after the last cached timestamp (incremental).
- Symbol list: `data/tickers/{exchange}.json`, refreshed once per day.
- Fetches run concurrently (`asyncio` + a semaphore) with `enableRateLimit` on, so a
  200+-symbol scan finishes in roughly a minute.

## Known gotchas

- ccxt's `binanceusdm` occasionally lists tokenized-stock perpetuals (e.g. `AAPL/USDT`,
  `ADBE/USDT`) alongside crypto — they pass the same USDT-perpetual-swap filter and will
  show up in scans like any other symbol.
- RSI here is an EWM-based Wilder approximation (`ewm(alpha=1/period, adjust=False)`).
  It converges to the classic SMA-seeded Wilder RSI after a handful of bars but can
  differ by a few points immediately after the warm-up period on short series.
- The replay page's strategy engine is a hand-written JS port of `strategy.py` — kept in
  sync by `tests/test_replay_parity.py`, which runs the embedded JS with Node and checks
  the trade count against the Python backtest on identical candle data.
- lightweight-charts is loaded from jsdelivr (`lightweight-charts@4.1.3` standalone
  build); the cdnjs mirror does not host this package.
- `out/index.html`'s "Relationship Graph" reads the local parquet cache
  (`data/candles/{exchange}/`); with an empty cache it has no neighbour symbols to add
  and falls back to the scan hits alone.

## Dashboard metric definitions

Every number on `out/index.html` is derived from the scan JSON, the backtest trades, or
the cached candles. Nothing is invented. The non-obvious ones:

| Metric | Definition |
| --- | --- |
| `sharpe` | mean / stdev of per-trade returns (`pnl_pct`), sample stdev, not annualised |
| `breakeven wr` | `SL / (TP + SL)` = 1.5 / 4.5 = 33.3% — the win rate at which the TP/SL pair breaks even before fees |
| `edge vs gate` | realised win rate minus `breakeven wr`, in percentage points |
| `tail mass` | share of trades whose return reached the take-profit strike (`pnl_pct >= +3%`) |
| `implied mult` | `1 / tail mass` — the payout a binary on "reaches the strike" would need to be fair |
| `stop mass` | share of trades that exited at the stop (`pnl_pct <= -1.5%`) |
| `equity peak` | maximum of the cumulative-PnL curve |
| `median fair` | median scan-hit close projected forward by the median realised trade return |
| lattice bin | `clamp(round(pnl_pct), -3, +3)` — one 1%-wide bin per ball; the ball's path through the pegs is chosen so it lands in its real bin |
| `P(UP)` | share of scan-hit symbols whose latest `whale_momentum` value is positive |
| `P(DOWN)` | `100 - P(UP)` |
| `confidence` | `max(P(UP), P(DOWN))` — how one-sided the momentum vote is |
| graph nodes | scan-hit symbols plus up to 40 cached symbols whose return correlation with any hit is `|rho| >= 0.3` (last 300 bars) |
| graph edges | node pairs with `|rho| >= 0.3`; only the 60 strongest are drawn, the counters report all of them |
| `bull/bear paths` | edges with positive / negative `rho` |
| node colour | green if the symbol's latest `whale_momentum > 0`, red if `< 0`, grey if undefined |
| cluster hub | `HUB_PRIME` = correlated with BTC at `|rho| >= 0.6`; otherwise `BEAR_CLUSTER` (negative momentum) or `CATALYST_RING` (positive) |
| `iter` / `iter per sec` | real d3-force simulation ticks, measured in the browser |
| `convergence` | `1 - alpha` of the running force simulation |

## Tests

```bash
uv run pytest -q
```

All tests run against fixed, offline fixtures (`tests/fixtures.py`) — no network calls.
`test_replay_parity.py` needs `node` on PATH; it's skipped automatically if missing.
