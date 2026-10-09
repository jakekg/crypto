"""Forecast fan chart: trailing price history plus the 80% interval and mean at each horizon."""
from __future__ import annotations

import json

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .dataset import build_panel  # noqa: E402

SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"


def fan_chart(symbol: str, report_json: str, out_png: str) -> None:
    js = json.load(open(report_json))
    panel, _ = build_panel(symbol)
    hist = panel["price__close"].iloc[-180:]
    t0, p0 = hist.index[-1], hist.iloc[-1]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(hist.index, hist.values, color=SERIES, lw=2)
    for h, H in js["horizons"].items():
        f = H["final"]
        t = t0 + pd.Timedelta(days=int(h))
        lo, hi = p0 * np.exp(f["z_lo"] * f["sigma"]), p0 * np.exp(f["z_hi"] * f["sigma"])
        mid = p0 * np.exp(f["mu"])
        ax.plot([t, t], [lo, hi], color=SERIES, lw=6, alpha=0.25, solid_capstyle="round")
        ax.plot([t], [mid], "o", ms=8, color=SERIES, mec=SURFACE, mew=2)
        ax.annotate(f"{h}d", (t, hi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.axvline(t0, color=GRID, lw=1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 100 else f"{v:,.3g}"))
    pup = "   ".join(f"{h}d {H['final']['p_up'] * 100:.0f}%" for h, H in js["horizons"].items())
    ax.text(0, 1.01, f"P(up): {pup}", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(f"{symbol}-USD: 180d history and forecast (dot = expected value, bar = 80% interval)",
                 loc="left", fontsize=11, color=INK, pad=18)
    edges = [h for h, H in js["horizons"].items() if H["verdict"].startswith("EDGE")]
    edge_txt = f"out-of-sample edge after costs at: {', '.join(edges)}d" if edges else \
        "no horizon shows a tradable out-of-sample edge after costs"
    ax.text(0, -0.12, f"As of {js['asof']} close. Blockworks MCP data. 28d/90d are regime tilts; {edge_txt}.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
