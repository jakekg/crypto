"""Print the exact Blockworks MCP `timeseries_data` calls that populate data/raw for an asset.

The Blockworks MCP is reached through an MCP client (e.g. Claude with the Blockworks
connector), not through a Python SDK, so refreshing data is a two-step loop:

  1. `python -m bwpredict.manifest BTC` -> JSON list of calls (each <= 5,000 points).
  2. Run each call with the MCP `timeseries_data` tool and save the verbatim JSON response
     into data/raw/<anything>.json. The loader indexes files by (model, series key), so
     names do not matter and overlapping windows are de-duplicated (latest file wins).
  3. Update data/raw/_snapshot.json with the fetch time (used to drop incomplete bars).
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from . import config as C

POINT_CAP = 5000


def _chunks(model, series, fields, start, end):
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    per = max(1, POINT_CAP // max(1, len(series)) - 10)
    out, s = [], pd.Timestamp(start)
    while s < pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=per), pd.Timestamp(end))
        out.append(dict(tool="timeseries_data", model=model, series=series, selections=fields,
                        granularity="1d", start=str(s.date()), end=str(e.date())))
        s = e
    return out if days > 0 else []


def plan(symbol: str, end: str | None = None) -> list[dict]:
    end = end or str((pd.Timestamp.now('UTC').tz_localize(None).normalize() + pd.Timedelta(days=1)).date())
    a = C.ASSETS[symbol]
    calls = []
    calls += _chunks("asset-price", [a["asset_id"]], ["high", "low", "close", "volume"], "2014-01-01", end)
    if symbol != "BTC":
        calls += _chunks("asset-price", [C.BTC_ID], ["close"], "2014-01-01", end)
    if symbol != "ETH":
        calls += _chunks("asset-price", [C.ETH_ID], ["close"], "2015-08-01", end)
    if a.get("perp"):
        calls += _chunks("perpetual-markets", [a["perp"]], ["closePrice", "volumeUSD", "openInterestUSD"], "2019-01-01", end)
    if a.get("spot_market"):
        calls += _chunks("bqen", [a["spot_market"]], ["close-price-raw", "volume"], "2017-08-01", end)
    calls += _chunks("asset-marketcap", [C.BTC_ID],
                     ["circulating-marketcap", "circulating-marketcap-dominance"], "2013-03-01", end)
    calls += _chunks("stablecoins-by-currency", ["usd"],
                     ["circulatingSupplyUSD", "mintAmountUSD", "burnAmountUSD", "balanceCEXUSD"], "2018-01-01", end)
    calls += _chunks("asset-price", [C.PAXG_ID], ["close"], "2019-10-01", end)
    if a.get("network"):
        calls += _chunks("blockchains", [a["network"]], ["active-addresses", "total-transactions", "fee-revenue"],
                         "2014-01-01", end)
    if a.get("hl_pair"):
        p = a["hl_pair"]
        calls += _chunks("perp-dex-protocols", ["hyperliquid"],
                         [f"pair{p}USDOpenInterestUSD", f"pair{p}USDLiquidationVolumeUSD"], "2023-01-01", end)
    if a.get("etf"):
        calls += _chunks("etf-assets", [a["etf"]], ["type-spot-flow-usd", "type-spot-aum-usd"], "2024-01-01", end)
    if a.get("dat"):
        calls += _chunks("dat-companies-by-asset", [a["dat"]], ["datPurchasesNative", "datAUMNative"], "2020-08-01", end)
    if a.get("unlocks"):
        far = str((pd.Timestamp(end) + pd.Timedelta(days=400)).date())
        calls += _chunks("token-unlocks", [a["asset_id"]], ["pct-of-circulating-supply-unlocked"], "2018-01-01", far)
        calls += _chunks("asset-supply", [a["asset_id"]], ["circulating-supply"], "2018-01-01", end)
    calls += _chunks("tokenized-equities-by-equity", ["spy"], ["underlyingPriceUSD"], "2025-01-01", end)
    return calls


if __name__ == "__main__":
    sym = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    print(json.dumps(plan(sym), indent=1))
