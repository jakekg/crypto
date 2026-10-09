"""Run the full pipeline for one asset and render the per-horizon output blocks."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import model as M
from .dataset import build_panel
from .features import build_features

warnings.filterwarnings("ignore")

# Requested inputs the Blockworks MCP does not serve (discovered, not assumed; see README).
MISSING = [
    "Taker buy/sell volume and bid/ask spreads (no trade-level or order-book data for CEX markets)",
    "Funding rates (no funding field anywhere) -> proxied by Binance perp/spot close premium",
    "Dated-futures basis (perpetual-markets excludes dated futures) -> same perp-premium proxy",
    "CEX liquidations and any long/short liquidation split (only Hyperliquid total liquidations, 2025-05+)",
    "CEX open interest history (only ~300 days, 2025-12+) -> Hyperliquid OI (2023-05+) used in ablation",
    "Options implied vol / skew",
    "Native-asset exchange inflows/outflows and whale/labeled-wallet flows (only USD-stablecoin CEX balances; "
    "treasury-company purchases for BTC)",
    "DXY, nominal/real yields with usable history (tokenized TIP/IEF/TLT only from 2025)",
    "Equities with usable history (tokenized SPY close only from 2025-01) -> ablation only",
    "Social sentiment (2025-01+) and mindshare (2024-07+): too short for 28d/90d; not used",
]


def _px(v: float) -> str:
    """Price with ~4 significant digits, readable for both BTC and sub-dollar alts."""
    if v >= 1000:
        return f"${v:,.0f}"
    return f"${v:,.{max(0, 3 - int(np.floor(np.log10(abs(v)))))}f}"


def _fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{max(nd - 2, 1)}f}%" if pct else f"{x:+.{nd}f}"


# source -> (column prefix in the panel, staleness key, publication-lag key)
_SRC = {
    "price": ("price__", "price", "price"), "btc": ("btc__", "btc", "price"), "eth": ("eth__", "eth", "price"),
    "perp": ("perp__", "perp", "perp"), "spot": ("spot__", "spot", "perp"),
    "dominance": ("dom__", "dom", "dominance"), "stables": ("stab__", "stab", "stables"),
    "gold": ("gold__", "gold", "gold"), "onchain": ("chain__", "chain", "onchain"),
    "hyperliquid": ("hl__", "hl", "hyperliquid"), "etf": ("etf__", "etf", "etf"), "dat": ("dat__", "dat", "dat"),
    "supply": ("supply__", "supply", "supply"), "equities": ("eq__", "eq", "equities"),
    "unlocks": ("unlock_fwd_", None, None),
}


def _source_table(panel, meta, sources, D):
    rows = []
    for s in sorted(sources):
        prefix, stale_key, lag_key = _SRC[s]
        m = meta.get(s, {})
        cols = [c for c in panel.columns if c.startswith(prefix)]
        sub = panel.loc[D.index.min():, cols]
        miss = float(sub.isna().all(axis=1).mean()) if cols else float("nan")
        st = panel[f"stale__{stale_key}"].iloc[-1] if stale_key and f"stale__{stale_key}" in panel else 0
        lag = f"{C.LAG[lag_key]}d" if lag_key else "schedule (forward-known)"
        rows.append(f"| {s} | `{m.get('model')}` / `{m.get('key')}` | {m.get('first')} -> {m.get('last')} "
                    f"| {miss * 100:.1f}% | {int(st) if np.isfinite(st) else 'n/a'}d | {lag} |")
    hdr = ("| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |\n"
           "|---|---|---|---|---|---|")
    return hdr + "\n" + "\n".join(rows)


def _ablation(panel, fs, h, core):
    ext = [f for f, t in fs.tiers.items() if t == "extended"]
    keep = [f for f in ext if fs.X[f].notna().sum() >= 700]
    dropped = [f for f in ext if f not in keep]
    if not keep:
        return dict(kept=[], dropped=dropped, note="no extended feature has >= 700 days of history")
    D = M.make_design(panel, fs.X[core + keep], h)
    a = M.walk_forward(D, core, h, min_train=365, retrain=91, with_gbm=False)
    b = M.walk_forward(D, core + keep, h, min_train=365, retrain=91, with_gbm=False)
    if a.oos.empty:
        return dict(kept=keep, dropped=dropped, note="window too short for a walk-forward test")
    ea, eb = M.evaluate(a.oos, h), M.evaluate(b.oos, h)
    if "nonoverlap" not in ea or ea["nonoverlap"]["n"] < 4:
        return dict(kept=keep, dropped=dropped,
                    note=f"window {D.index.min().date()} -> {D.index.max().date()} leaves fewer than 4 non-overlapping "
                         f"{h}d test windows; no meaningful comparison possible")
    return dict(kept=keep, dropped=dropped, window=f"{D.index.min().date()} -> {D.index.max().date()}",
                test=f"{a.oos.index.min().date()} -> {a.oos.index.max().date()}",
                n_nonoverlap=ea["nonoverlap"]["n"],
                core=dict(ic=ea["nonoverlap"]["ic"], hit=ea["nonoverlap"]["hit"], brier=ea["nonoverlap"]["brier"],
                          ic_ovl=ea["overlapping"]["ic"]),
                ext=dict(ic=eb["nonoverlap"]["ic"], hit=eb["nonoverlap"]["hit"], brier=eb["nonoverlap"]["brier"],
                         ic_ovl=eb["overlapping"]["ic"]),
                brier_clim=ea["nonoverlap"]["brier_clim"])


def _verdict(ev, h):
    n = ev["nonoverlap"]
    beats_ic = np.isfinite(n["ic"]) and n["ic"] > max(n["ic_mom"], 0) and n.get("ic_t", 0) > 2
    beats_net = n["sharpe_model"] > max(n["sharpe_mom"], n["sharpe_bh"], 0) and n["net_model_t"] > 2
    calib = n["brier"] < n["brier_clim"]
    if beats_ic and beats_net and calib:
        return "EDGE (out-of-sample beats zero, momentum and buy-and-hold after costs, IC t>2, calibrated)"
    reasons = []
    if not beats_ic:
        reasons.append(f"non-overlapping IC {n['ic']:+.3f} (t={n.get('ic_t', float('nan')):.2f}) does not clear "
                       f"momentum ({n['ic_mom']:+.3f}) with t>2")
    if not beats_net:
        reasons.append(f"net Sharpe {n['sharpe_model']:.2f} vs momentum {n['sharpe_mom']:.2f} / buy-and-hold "
                       f"{n['sharpe_bh']:.2f} (t={n['net_model_t']:.2f})")
    if not calib:
        reasons.append(f"Brier {n['brier']:.4f} not below climatology {n['brier_clim']:.4f}")
    return "NO TRADABLE EDGE CLAIMED - " + "; ".join(reasons)


def run_asset(symbol: str, horizons=C.HORIZONS) -> tuple[str, dict]:
    panel, meta = build_panel(symbol)
    last = panel.index.max()
    close = float(panel["price__close"].iloc[-1])
    md = [f"# {symbol}-USD probabilistic forward-return forecast",
          "",
          f"Decision time: **{(last + pd.Timedelta(days=1)).strftime('%Y-%m-%d 00:00 UTC')}** "
          f"(last completed daily bar {last.date()}, VWAP close {_px(close)}). "
          f"Snapshot of Blockworks MCP data; nothing after the last completed bar is used.",
          "",
          "Primary model: ridge regression on vol-scaled forward returns (shrinkage picked by an inner, purged "
          "time split), with the predictive distribution taken from empirical residuals. Comparators: zero return, "
          "prevailing mean (drift), trailing same-horizon momentum, and a depth-2 gradient-boosted model. "
          f"Walk-forward is purged with an embargo of one full horizon. Costs (round trip per non-overlapping trade): "
          + ", ".join(f"{h}d {C.COST_BPS[h]}bps" for h in horizons) + ".",
          ""]
    js = dict(asset=symbol, asof=str(last.date()), close=close, horizons={})
    summary_rows = []
    for h in horizons:
        fs = build_features(panel, symbol, h)
        core = [f for f, t in fs.tiers.items() if t == "core"]
        D = M.make_design(panel, fs.X[core], h)
        wf = M.walk_forward(D, core, h)
        ev = M.evaluate(wf.oos, h)
        fin = M.final_fit(D, core, h)
        stab = M.coef_stability(wf.coefs, fin["coef"])
        abl = _ablation(panel, fs, h, core)
        sources = sorted({s for f in core for s in fs.sources[f]})
        o, n = ev["overlapping"], ev["nonoverlap"]
        verdict = _verdict(ev, h)
        indicative = h >= 28 and n["n"] < 8 or (h == 90 and n["n"] < 8)
        # Prices for interval.
        lo, hi = fin["z_lo"] * fin["sigma"], fin["z_hi"] * fin["sigma"]
        js["horizons"][h] = dict(n_features=len(core), features=core, sample_start=str(D.index.min().date()),
                                 folds=wf.folds, eval=ev, final={k: v for k, v in fin.items() if k != "coef"},
                                 coef=stab.to_dict(orient="index"), ablation=abl, verdict=verdict)
        summary_rows.append(f"| {h}d | {fin['p_up'] * 100:.1f}% | {fin['p3_dn'] * 100:.0f}/{fin['p3_fl'] * 100:.0f}/{fin['p3_up'] * 100:.0f}% "
                            f"| {np.expm1(fin['mu']) * 100:+.2f}% | {np.expm1(lo) * 100:+.1f}% / {np.expm1(hi) * 100:+.1f}% "
                            f"| {_px(close * np.exp(lo))} - {_px(close * np.exp(hi))} | {n['ic']:+.3f} ({n['n']}) "
                            f"| {'indicative only' if h >= 28 and n['n'] < 8 else ('regime tilt' if h >= 28 else 'no edge claimed' if verdict.startswith('NO') else 'edge')} |")

        b = []
        b.append(f"## Horizon {h}d" + (" - regime tilt, not a price path" if h >= 28 else ""))
        b.append("")
        b.append(f"**Current forecast** (as of {fin['asof']}, model trained on {fin['n_train']} labelled days through "
                 f"{fin['train_end']}, shrinkage kappa={fin['kappa']}):")
        b.append("")
        b.append(f"- P(up) = **{fin['p_up'] * 100:.1f}%**; ternary P(down / flat / up) = "
                 f"{fin['p3_dn'] * 100:.1f}% / {fin['p3_fl'] * 100:.1f}% / {fin['p3_up'] * 100:.1f}% "
                 f"(flat band = +-{C.FLAT_BAND_K} x {C.vol_window(h)}d trailing vol scaled to {h}d = +-{np.expm1(C.FLAT_BAND_K * fin['sigma']) * 100:.2f}%)")
        b.append(f"- Expected log return {fin['mu'] * 100:+.2f}% (simple {np.expm1(fin['mu']) * 100:+.2f}%) = "
                 f"shrunk drift {fin['drift_mu'] * 100:+.2f}% (sample drift {fin['drift_only_mu'] * 100:+.2f}% x prior weight "
                 f"{fin['drift_w']:.2f}) + feature signal {fin['signal_mu'] * 100:+.2f}%; GBM comparator {fin['mu_gbm'] * 100:+.2f}%")
        b.append(f"- {int(C.PI_LEVEL * 100)}% prediction interval: {np.expm1(lo) * 100:+.1f}% to {np.expm1(hi) * 100:+.1f}% "
                 f"-> {_px(close * np.exp(lo))} to {_px(close * np.exp(hi))} (horizon vol {fin['sigma'] * 100:.1f}%)")
        if (fin["p_up"] - 0.5) * fin["mu"] < 0:
            b.append("- Note: P(up) and the expected return point in different directions because the empirical "
                     "residual distribution is skewed (mean and median differ); both come from the same distribution.")
        b.append(f"- Verdict: **{verdict}**")
        if h >= 28:
            b.append(f"- Non-overlapping test windows: **{n['n']}**" +
                     (" - fewer than ~8, so this horizon is **indicative only**." if n["n"] < 8 else
                      " - treat as a regime tilt; windows are few and regimes do not repeat."))
        b.append("")
        b.append(f"**Data actually used** (model sample {D.index.min().date()} -> {D.index.max().date()}, "
                 f"{len(core)} core features):")
        b.append("")
        b.append(_source_table(panel, meta, sources, D))
        b.append("")
        b.append("**Walk-forward out-of-sample results** "
                 f"({len(wf.folds)} purged folds, retrain every {C.WALK[h]['retrain']}d, embargo {h}d, "
                 f"test {wf.oos.index.min().date()} -> {wf.oos.index.max().date()}):")
        b.append("")
        b.append("| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |")
        b.append("|---|---|---|---|---|---|")
        g = ev.get("gbm_nonoverlap", {})
        b.append(f"| N | {o['n']} | {n['n']} | {n['n']} | {n['n']} | {n['n']} |")
        b.append(f"| IC, signal only (Spearman, vol-scaled) | {o['ic']:+.3f} | {n['ic']:+.3f} (t={n.get('ic_t', np.nan):.2f}) | {n['ic_mom']:+.3f} | n/a (constant) | {g.get('ic', np.nan):+.3f} |")
        b.append(f"| IC, full forecast incl. per-fold drift | {o['ic_full']:+.3f} | {n['ic_full']:+.3f} | | | |")
        b.append(f"| Hit rate | {o['hit']:.3f} | {n['hit']:.3f} | {n['hit_mom']:.3f} | drift {n.get('hit_pm', np.nan):.3f} / long {n['hit_long']:.3f} | {g.get('hit', np.nan):.3f} |")
        b.append(f"| Drift-only R2 vs zero | | {n.get('r2_pm_vs_zero', np.nan):+.4f} | | | |")
        b.append(f"| R2 OOS vs zero / vs drift | {o['r2_oos_vs_zero']:+.4f} / {o.get('r2_oos_vs_mean', np.nan):+.4f} | {n['r2_oos_vs_zero']:+.4f} / {n.get('r2_oos_vs_mean', np.nan):+.4f} | | | {g.get('r2_oos_vs_zero', np.nan):+.4f} |")
        b.append(f"| Brier P(up) (clim / 0.5) | {o['brier']:.4f} ({o['brier_clim']:.4f} / {o['brier_half']:.4f}) | {n['brier']:.4f} ({n['brier_clim']:.4f}) | | | {g.get('brier', np.nan):.4f} |")
        b.append(f"| {int(C.PI_LEVEL * 100)}% interval coverage | {o.get('pi_coverage', np.nan):.3f} | {n.get('pi_coverage', np.nan):.3f} | | | |")
        b.append(f"| Ternary log-loss (clim) | {o['logloss3']:.4f} ({o['logloss3_clim']:.4f}) | {n['logloss3']:.4f} ({n['logloss3_clim']:.4f}) | | | |")
        b.append(f"| Mean net return / trade after {C.COST_BPS[h]}bps | | {n['net_model_bps']:+.0f}bps (t={n['net_model_t']:.2f}) | {n['net_mom_bps']:+.0f}bps (t={n['net_mom_t']:.2f}) | buy-and-hold {n['net_bh_bps']:+.0f}bps | {g.get('net_model_bps', np.nan):+.0f}bps |")
        b.append(f"| Annualised Sharpe (net) | | {n['sharpe_model']:.2f} | {n['sharpe_mom']:.2f} | buy-and-hold {n['sharpe_bh']:.2f} | |")
        b.append("")
        gb_better = g and np.isfinite(g.get("ic", np.nan)) and g["ic"] > n["ic"] and g["brier"] < n["brier"] and g["net_model_bps"] > n["net_model_bps"]
        b.append("Complexity check: the depth-2 GBM " + ("beats" if gb_better else "does not beat") +
                 " ridge on non-overlapping IC, Brier and net return together, so " +
                 ("it is worth a closer look, but with this few windows it is not adopted." if gb_better else
                  "the simpler linear model is kept."))
        b.append("")
        if ev["regimes"]:
            b.append("Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):")
            b.append("")
            b.append("| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |")
            b.append("|---|---|---|---|---|---|---|")
            for r, v in ev["regimes"].items():
                b.append(f"| {r} | {v['n']} | {v['n_indep']} | {v['ic']:+.3f} | {v['hit']:.3f} | {v['hit_mom']:.3f} | {v['hit_long']:.3f} |")
            b.append("")
        b.append("IC by calendar year: " + ", ".join(f"{y}: {v:+.2f}" for y, v in ev["ic_by_year"].items()))
        b.append("")
        b.append("Calibration of P(up) (OOS quintiles): " + "; ".join(
            f"p={c['p_mean']:.2f} -> {c['realised']:.2f} (n={int(c['n'])})" for c in ev["calibration"]))
        b.append("")
        b.append("**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of "
                 "walk-forward folds with the same sign; 'stable' if >= 80%):")
        b.append("")
        b.append("| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |")
        b.append("|---|---|---|---|---|---|---|")
        latest = D[core].iloc[-1]
        for f, r in stab.head(8).iterrows():
            b.append(f"| `{f}` | {r['coef_final']:+.4f} | {r['coef_mean_folds']:+.4f} | {r['sign_agreement'] * 100:.0f}% "
                     f"| {'yes' if r['sign_agreement'] >= 0.8 else 'no'} | {latest[f]:+.2f} | {fs.desc[f]} |")
        b.append("")
        b.append("<details><summary>Latest feature snapshot (all core features)</summary>")
        b.append("")
        b.append("| feature | value | description |")
        b.append("|---|---|---|")
        for f in core:
            b.append(f"| `{f}` | {latest[f]:+.3f} | {fs.desc[f]} |")
        b.append("")
        b.append("</details>")
        b.append("")
        if abl.get("core"):
            b.append(f"**Short-history ablation** (core vs core + {', '.join('`' + k + '`' for k in abl['kept'])}; "
                     f"same rows, window {abl['window']}, test {abl['test']}, {abl['n_nonoverlap']} non-overlapping windows): "
                     f"IC {abl['core']['ic']:+.3f} -> {abl['ext']['ic']:+.3f} (overlapping {abl['core']['ic_ovl']:+.3f} -> {abl['ext']['ic_ovl']:+.3f}), "
                     f"hit {abl['core']['hit']:.3f} -> {abl['ext']['hit']:.3f}, Brier {abl['core']['brier']:.4f} -> {abl['ext']['brier']:.4f} "
                     f"(clim {abl['brier_clim']:.4f}). Not used in the primary forecast: too few independent windows to trust.")
        else:
            b.append(f"**Short-history ablation**: {abl.get('note')}")
        if abl.get("dropped"):
            b.append(f"Too short to evaluate (< 700 days): {', '.join('`' + k + '`' for k in abl['dropped'])}.")
        b.append("")
        b.append("**Failure modes for this horizon:**")
        b.append("")
        fm = []
        if h >= 28:
            fm.append(f"Sparse non-overlapping samples: {n['n']} independent test windows; one regime shift can flip the sign "
                      "(see IC by year).")
        else:
            fm.append(f"Regime shift: IC by year ranges {min(ev['ic_by_year'].values()):+.2f} to {max(ev['ic_by_year'].values()):+.2f}; "
                      "drift learned from the training window dominates the sign of the forecast.")
        fm.append(f"Thin history: model sample starts {D.index.min().date()} (binding source: "
                  f"{fs.X[core].apply(lambda s: s.first_valid_index()).idxmax()}).")
        if "onchain" in sources or "stables" in sources:
            fm.append("Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for "
                      "stablecoin balances are applied retroactively (today's label set), a look-ahead risk that "
                      "likely flatters `stab_cexflow_*`.")
        if "perp" in sources:
            fm.append("Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; "
                      "venue-specific wash or incentive volume and index-construction quirks feed straight into them.")
        if "unlocks" in sources:
            fm.append("Unlock schedules are the current vintage; past revisions are not recoverable (look-ahead risk).")
        fm.append("Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; "
                  "gold (PAXG) is the only macro proxy.")
        b += [f"- {x}" for x in fm]
        b.append("")
        md += b
    head = ["## Summary", "",
            "| horizon | P(up) | P(down/flat/up) | expected return | 80% interval | price range | non-overlap IC (N) | status |",
            "|---|---|---|---|---|---|---|---|"] + summary_rows + [""]
    md = md[:6] + head + md[6:]
    md += ["## Requested inputs not available in the Blockworks MCP", ""] + [f"- {m}" for m in MISSING] + [""]
    return "\n".join(md), js


def to_json(js: dict) -> str:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, pd.Timestamp):
            return str(o)
        return str(o)
    return json.dumps(js, default=conv, indent=1)
