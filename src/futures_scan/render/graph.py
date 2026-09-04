"""Relationship-graph construction for the dashboard.

Nodes are the scan-hit symbols plus the most strongly correlated symbols found
in the local parquet candle cache. Edges are real |Pearson rho| of hourly
returns. Nothing is synthesised except the three cluster hubs, which are
labels for groups of real nodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from futures_scan.cache import CACHE_ROOT
from futures_scan.indicators import whale_momentum

HUB_PRIME = "HUB_PRIME"
BEAR_CLUSTER = "BEAR_CLUSTER"
CATALYST_RING = "CATALYST_RING"

EDGE_THRESHOLD = 0.3
BTC_CORR_THRESHOLD = 0.6
MAX_NEIGHBOURS = 40
RETURN_BARS = 300


def _cached_symbols(exchange: str, timeframe: str) -> dict[str, Path]:
    root = CACHE_ROOT / exchange
    if not root.exists():
        return {}
    suffix = f"_{timeframe}.parquet"
    out: dict[str, Path] = {}
    for p in sorted(root.glob(f"*{suffix}")):
        stem = p.name[: -len(suffix)]
        parts = stem.split("-")
        if len(parts) >= 3:
            symbol = f"{'-'.join(parts[:-2])}/{parts[-2]}:{parts[-1]}"
        else:
            symbol = stem
        out[symbol] = p
    return out


def _returns_from_cache(exchange: str, timeframe: str, symbols: list[str] | None = None) -> dict[str, pd.Series]:
    paths = _cached_symbols(exchange, timeframe)
    wanted = paths if symbols is None else {s: paths[s] for s in symbols if s in paths}
    series: dict[str, pd.Series] = {}
    for symbol, path in wanted.items():
        try:
            df = pd.read_parquet(path, columns=["timestamp", "close"])
        except Exception:
            continue
        if len(df) < 60:
            continue
        df = df.sort_values("timestamp").tail(RETURN_BARS)
        s = pd.Series(df["close"].pct_change().values, index=df["timestamp"].values).dropna()
        if len(s) >= 50:
            series[symbol] = s
    return series


def _momentum_from_cache(exchange: str, timeframe: str, symbol: str) -> float | None:
    path = CACHE_ROOT / exchange / (symbol.replace("/", "-").replace(":", "-") + f"_{timeframe}.parquet")
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if len(df) < 60:
        return None
    return float(whale_momentum(df.sort_values("timestamp").reset_index(drop=True)).iloc[-1])


def build_graph(
    exchange: str,
    timeframe: str,
    hit_symbols: list[str],
    change_by_symbol: dict[str, float | None],
    pnl_by_symbol: dict[str, float],
) -> dict:
    """Build nodes/edges/hub assignments plus the derived direction metrics."""
    all_returns = _returns_from_cache(exchange, timeframe)
    hit_present = [s for s in hit_symbols if s in all_returns]

    # --- correlated neighbours of the scan hits -------------------------------
    frame = pd.DataFrame(all_returns).dropna(how="all")
    # Keep only bars every column shares, so a plain numpy correlation is valid.
    frame = frame.dropna(axis=1, thresh=int(0.8 * len(frame))).dropna(axis=0)
    hit_present = [s for s in hit_present if s in frame.columns]
    neighbours: dict[str, float] = {}
    corr_full = None
    if hit_present and len(frame) >= 50 and frame.shape[1] >= 2:
        corr_full = pd.DataFrame(
            np.corrcoef(frame.to_numpy().T), index=frame.columns, columns=frame.columns
        )
        for other in corr_full.index:
            if other in hit_symbols:
                continue
            vals = corr_full.loc[other, hit_present].dropna()
            if vals.empty:
                continue
            best = float(vals.abs().max())
            if best >= EDGE_THRESHOLD:
                neighbours[other] = best
    top_neighbours = sorted(neighbours.items(), key=lambda kv: kv[1], reverse=True)[:MAX_NEIGHBOURS]
    neighbour_symbols = [s for s, _ in top_neighbours]

    node_symbols = list(dict.fromkeys(hit_present + neighbour_symbols))
    if not node_symbols:
        return {
            "nodes": [], "edges": [], "hubs": [], "p_up": 0.0, "p_down": 0.0,
            "bear_paths": 0, "bull_paths": 0, "edge_pp": 0.0, "confidence_pct": 0.0,
            "signal_pct": 0.0, "btc_symbol": None,
        }

    node_symbols = [s for s in node_symbols if corr_full is not None and s in corr_full.index]
    if not node_symbols:
        node_symbols = hit_present
    corr_sub = corr_full.loc[node_symbols, node_symbols] if corr_full is not None else pd.DataFrame()

    momentum = {s: _momentum_from_cache(exchange, timeframe, s) for s in node_symbols}

    btc = next((s for s in frame.columns if s.startswith("BTC/")), None)
    btc_corr: dict[str, float] = {}
    if btc is not None and corr_full is not None and btc in corr_full.index:
        for s in node_symbols:
            if s in corr_full.columns:
                btc_corr[s] = float(corr_full.loc[btc, s])

    def hub_for(sym: str) -> str:
        if btc is not None and abs(btc_corr.get(sym, 0.0)) >= BTC_CORR_THRESHOLD:
            return HUB_PRIME
        mom = momentum.get(sym)
        if mom is not None and mom < 0:
            return BEAR_CLUSTER
        return CATALYST_RING

    nodes = []
    for s in node_symbols:
        mom = momentum.get(s)
        nodes.append({
            "symbol": s,
            "short": s.split("/")[0].lower(),
            "hub": hub_for(s),
            "hit": s in hit_symbols,
            "momentum": None if mom is None else round(mom, 2),
            "change_24h": change_by_symbol.get(s),
            "pnl": round(pnl_by_symbol.get(s, 0.0), 2),
            "btc_rho": round(btc_corr.get(s, 0.0), 3),
        })

    edges = []
    seen = set()
    for i, a in enumerate(node_symbols):
        for b in node_symbols[i + 1:]:
            if a not in corr_sub.index or b not in corr_sub.columns:
                continue
            rho = corr_sub.loc[a, b]
            if pd.isna(rho) or abs(float(rho)) < EDGE_THRESHOLD:
                continue
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"a": a, "b": b, "rho": round(float(rho), 3)})

    scored = [n for n in nodes if n["momentum"] is not None]
    bulls = sum(1 for n in scored if n["momentum"] > 0)
    bears = len(scored) - bulls
    p_up = round(100 * bulls / len(scored), 1) if scored else 0.0

    hit_scored = [n for n in nodes if n["hit"] and n["momentum"] is not None]
    hit_bulls = sum(1 for n in hit_scored if n["momentum"] > 0)
    hit_p_up = round(100 * hit_bulls / len(hit_scored), 1) if hit_scored else p_up

    bull_paths = sum(1 for e in edges if e["rho"] > 0)
    bear_paths = len(edges) - bull_paths

    hubs = [
        {"id": BEAR_CLUSTER, "label": "bear_cluster", "kind": "bear",
         "count": sum(1 for n in nodes if n["hub"] == BEAR_CLUSTER)},
        {"id": HUB_PRIME, "label": "hub_prime", "kind": "hub",
         "count": sum(1 for n in nodes if n["hub"] == HUB_PRIME)},
        {"id": CATALYST_RING, "label": "catalyst_ring", "kind": "catalyst",
         "count": sum(1 for n in nodes if n["hub"] == CATALYST_RING)},
    ]

    rhos = [abs(e["rho"]) for e in edges]
    signal_pct = round(100 * float(np.mean(rhos)), 1) if rhos else 0.0

    return {
        "nodes": nodes,
        "edges": edges,
        "hubs": hubs,
        "p_up": hit_p_up,
        "p_down": round(100 - hit_p_up, 1),
        "graph_p_up": p_up,
        "bull_paths": bull_paths,
        "bear_paths": bear_paths,
        "bull_nodes": bulls,
        "bear_nodes": bears,
        "signal_pct": signal_pct,
        "btc_symbol": btc,
        "neighbour_count": len(neighbour_symbols),
    }
