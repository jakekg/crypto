"""Point-in-time daily panel for one asset.

Conventions
-----------
* Index = UTC date T of a daily bar covering [T, T+1d). Decision time for row T is
  T+1d 00:00 UTC, i.e. only the last *completed* bar is used.
* Bars whose end is after the snapshot time are dropped (no partial bars).
* Each source is shifted by its publication lag (config.LAG) and forward-filled only up
  to config.FFILL days; per-source staleness (days since the last real observation) is
  kept as `stale__<source>`.
* Resampling rules: the MCP already serves 1d bars; where we aggregate we use sum for
  volumes and flows, last for open interest / balances / AUM, and the VWAP close of
  `asset-price` as the reference price.
* Targets: y_h = log(close[T+h] / close[T]); label_h in {-1, 0, 1} using a flat band of
  FLAT_BAND_K * sigma_d(T) * sqrt(h), where sigma_d is trailing daily log-return vol over
  vol_window(h) days ending at T. Labels exist only when T+h is a completed bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .io import load, snapshot_time

DAY = pd.Timedelta(days=1)


def _align(df: pd.DataFrame | None, idx: pd.DatetimeIndex, lag: int, ffill: int,
           name: str, out: dict, stale: dict, stocks: list[str] | None = None) -> None:
    """Shift by publication lag, reindex to the panel, forward-fill stock columns only.

    Flow columns (volumes, mints, purchases, liquidations) are never forward-filled: a
    missing flow is missing, not a repeat of yesterday. `stocks=None` means every column
    is a level/stock (prices, balances, OI) and may be filled up to `ffill` days.
    """
    if df is None or df.empty:
        return
    df = df.reindex(df.index.union(idx)).sort_index()
    df = df.shift(lag, freq="D") if lag else df
    df = df.reindex(idx)
    observed = df.notna().any(axis=1)
    last_obs = pd.Series(np.where(observed, np.arange(len(idx)), np.nan), index=idx).ffill()
    stale[name] = pd.Series(np.arange(len(idx)), index=idx) - last_obs
    if ffill:
        cols = list(df.columns) if stocks is None else [c for c in stocks if c in df.columns]
        df[cols] = df[cols].ffill(limit=ffill)
    for c in df.columns:
        out[f"{name}__{c}"] = df[c]


def build_panel(symbol: str) -> tuple[pd.DataFrame, dict]:
    if symbol not in C.ASSETS:
        raise KeyError(f"{symbol} not in config.ASSETS")
    a = C.ASSETS[symbol]
    px = load("asset-price", a["asset_id"], ["high", "low", "close", "volume"])
    if px is None:
        raise FileNotFoundError(f"No asset-price snapshot for {symbol}; see `python -m bwpredict.manifest {symbol}`")

    snap = snapshot_time()
    px = px[px.index + DAY <= snap].dropna(subset=["close"])
    px = px[px["close"] > 0]
    idx = pd.date_range(px.index.min(), px.index.max(), freq="D")
    out: dict[str, pd.Series] = {}
    stale: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}

    def reg(name, model, key, df):
        if df is not None and not df.dropna(how="all").empty:
            d = df.dropna(how="all")
            meta[name] = dict(model=model, key=key, first=str(d.index.min().date()),
                              last=str(d.index.max().date()), n=int(len(d)),
                              columns=list(df.columns))
        else:
            meta[name] = dict(model=model, key=key, first=None, last=None, n=0, columns=[])

    reg("price", "asset-price", a["asset_id"], px)
    _align(px, idx, C.LAG["price"], 0, "price", out, stale)

    if symbol != "BTC":
        btc = load("asset-price", C.BTC_ID, ["close"])
        reg("btc", "asset-price", C.BTC_ID, btc)
        _align(btc, idx, C.LAG["price"], 0, "btc", out, stale)
    if symbol != "ETH":
        eth = load("asset-price", C.ETH_ID, ["close"])
        reg("eth", "asset-price", C.ETH_ID, eth)
        _align(eth, idx, C.LAG["price"], 0, "eth", out, stale)

    # Same-exchange perp/spot pair -> perp premium (funding/basis proxy) and perp activity.
    if a.get("perp"):
        perp = load("perpetual-markets", a["perp"], ["closePrice", "volumeUSD", "openInterestUSD"])
        reg("perp", "perpetual-markets", a["perp"], perp)
        _align(perp, idx, C.LAG["perp"], C.FFILL["perp"], "perp", out, stale,
               stocks=["closePrice", "openInterestUSD"])
    if a.get("spot_market"):
        spot = load("bqen", a["spot_market"], ["close-price-raw", "volume"])
        reg("spot", "bqen", a["spot_market"], spot)
        _align(spot, idx, C.LAG["perp"], C.FFILL["perp"], "spot", out, stale,
               stocks=["close-price-raw"])

    mk, key = C.MARKET["btc_mcap"]
    dom = load(mk, key, ["circulating-marketcap", "circulating-marketcap-dominance"])
    reg("dominance", mk, key, dom)
    _align(dom, idx, C.LAG["dominance"], C.FFILL["dominance"], "dom", out, stale)

    mk, key = C.MARKET["stables"]
    st = load(mk, key, ["circulatingSupplyUSD", "mintAmountUSD", "burnAmountUSD", "balanceCEXUSD"])
    reg("stables", mk, key, st)
    _align(st, idx, C.LAG["stables"], C.FFILL["stables"], "stab", out, stale,
           stocks=["circulatingSupplyUSD", "balanceCEXUSD"])

    mk, key = C.MARKET["gold"]
    gold = load(mk, key, ["close"])
    reg("gold", mk, key, gold)
    _align(gold, idx, C.LAG["gold"], C.FFILL["gold"], "gold", out, stale)

    if a.get("network"):
        ch = load("blockchains", a["network"], ["active-addresses", "total-transactions", "fee-revenue"])
        reg("onchain", "blockchains", a["network"], ch)
        _align(ch, idx, C.LAG["onchain"], C.FFILL["onchain"], "chain", out, stale, stocks=[])

    if a.get("hl_pair"):
        mk, key = C.MARKET["hyperliquid"]
        p = a["hl_pair"]
        hl = load(mk, key, [f"pair{p}USDOpenInterestUSD", f"pair{p}USDLiquidationVolumeUSD"])
        if hl is not None:
            hl.columns = ["oi", "liq"]
        reg("hyperliquid", mk, key, hl)
        _align(hl, idx, C.LAG["hyperliquid"], C.FFILL["hyperliquid"], "hl", out, stale,
               stocks=["oi"])

    if a.get("etf"):
        etf = load("etf-assets", a["etf"], ["type-spot-flow-usd", "type-spot-aum-usd"])
        reg("etf", "etf-assets", a["etf"], etf)
        if etf is not None and not etf.dropna(how="all").empty:
            # Flows exist on US trading days only: non-trading days inside the live range
            # are genuine zero-flow days (sum rule), AUM is a stock (last rule).
            first = etf["type-spot-aum-usd"].first_valid_index()
            full = pd.date_range(first, etf.index.max(), freq="D")
            etf = etf.reindex(full)
            trading = etf["type-spot-aum-usd"].notna()
            etf.loc[~trading, "type-spot-flow-usd"] = 0.0
            # The latest row is often provisional (AUM present, flow null): leave as NaN.
        _align(etf, idx, C.LAG["etf"], C.FFILL["etf"], "etf", out, stale,
               stocks=["type-spot-aum-usd"])

    if a.get("dat"):
        dat = load("dat-companies-by-asset", a["dat"], ["datPurchasesNative", "datAUMNative"])
        reg("dat", "dat-companies-by-asset", a["dat"], dat)
        _align(dat, idx, C.LAG["dat"], C.FFILL["dat"], "dat", out, stale,
               stocks=["datAUMNative"])

    if a.get("unlocks"):
        un = load("token-unlocks", a["asset_id"], ["pct-of-circulating-supply-unlocked"])
        reg("unlocks", "token-unlocks", a["asset_id"], un)
        if un is not None:
            # The schedule is known in advance (no lag), but it is today's vintage of the
            # schedule -> flagged as a look-ahead risk in the report.
            s = un.iloc[:, 0].reindex(pd.date_range(un.index.min(), un.index.max(), freq="D")).fillna(0)
            for h in C.HORIZONS:
                fwd = s[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)  # (T, T+h]
                out[f"unlock_fwd_{h}"] = fwd.reindex(idx)
        sup = load("asset-supply", a["asset_id"], ["circulating-supply"])
        reg("supply", "asset-supply", a["asset_id"], sup)
        _align(sup, idx, C.LAG["supply"], C.FFILL["supply"], "supply", out, stale)

    mk, key = C.MARKET["equities"]
    eq = load(mk, key, ["underlyingPriceUSD"])
    reg("equities", mk, key, eq)
    if eq is not None:
        eq = eq.dropna()
    _align(eq, idx, C.LAG["equities"], C.FFILL["equities"], "eq", out, stale)

    panel = pd.DataFrame(out, index=idx)
    for k, s in stale.items():
        panel[f"stale__{k}"] = s

    # Targets and labels.
    lc = np.log(panel["price__close"])
    r1 = lc.diff()
    panel["ret1"] = r1
    for h in C.HORIZONS:
        w = C.vol_window(h)
        sig = r1.rolling(w, min_periods=int(w * 0.8)).std()
        panel[f"sigma_{h}"] = sig * np.sqrt(h)
        y = lc.shift(-h) - lc
        panel[f"y_{h}"] = y
        band = C.FLAT_BAND_K * panel[f"sigma_{h}"]
        lab = np.where(y > band, 1, np.where(y < -band, -1, 0)).astype(float)
        lab[y.isna().values | band.isna().values] = np.nan
        panel[f"label_{h}"] = lab
    return panel, meta
