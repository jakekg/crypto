"""Asset registry, horizon settings and point-in-time publication rules.

Identifiers were discovered through the Blockworks MCP (tabular `assets`, `markets`,
`perpetual-markets`, `blockchains`, `etf-assets`, `dat-company-assets`). Nothing here is
assumed: an asset entry only lists sources that exist in the MCP for that asset.
"""
from __future__ import annotations

BTC_ID = "1e31218a-e44e-4285-820c-8282ee222035"
ETH_ID = "21c795f5-1bfd-40c3-858e-e9d7e820c6d0"
PAXG_ID = "be5685e4-c09d-4671-8e50-abbe1dd50b43"

# Per-asset sources. Keys:
#   asset_id     asset-price / asset-supply / token-unlocks series key
#   spot_market  bqen market id of the Binance USDT spot pair (same-exchange basis leg)
#   perp         perpetual-markets id of the Binance USDT-margined perpetual
#   network      blockchains network id (onchain activity), if the asset has its own chain
#   etf          etf-assets slug, if spot ETFs exist
#   hl_pair      Hyperliquid pair prefix in perp-dex-protocols (BTC/ETH/SOL/HYPE/BNB/XRP/AVAX only)
#   dat          dat-companies-by-asset id for treasury-company purchases
#   unlocks      True if a token-unlocks schedule should be used
ASSETS: dict[str, dict] = {
    "BTC": dict(asset_id=BTC_ID, spot_market="6057-140-68", perp="binance_BTCUSDT",
                network="54e8d8d0-979a-4297-887f-2ea8a989753c", etf="bitcoin",
                hl_pair="Bitcoin", dat="MSTR_btc", unlocks=False),
    "ETH": dict(asset_id=ETH_ID, spot_market="6059-140-68", perp="binance_ETHUSDT",
                network="09f15639-06cf-4460-83b8-a86de2f2db28", etf="ethereum",
                hl_pair="Ethereum", dat=None, unlocks=False),
    "SOL": dict(asset_id="b3d5d66c-26a2-404c-9325-91dc714a722b", spot_market="6079-140-68",
                perp="binance_SOLUSDT", network="b204e48e-9812-43e6-a00b-5fcfd40e04a2",
                etf="solana", hl_pair="Solana", dat=None, unlocks=True),
    # Further liquid alts: ids resolved from the MCP. Run `python -m bwpredict.manifest XRP`
    # to print the exact MCP calls that populate data/raw for them.
    "XRP": dict(asset_id="97775be0-2608-4720-b7af-f85b24c7eb2d", spot_market="6238-140-68",
                perp="binance_XRPUSDT", network=None, etf=None, hl_pair="XRP", dat=None, unlocks=True),
    "BNB": dict(asset_id="7dc551ba-cfed-4437-a027-386044415e3e", spot_market="347-140-68",
                perp="binance_BNBUSDT", network=None, etf=None, hl_pair="BNB", dat=None, unlocks=False),
    "DOGE": dict(asset_id="7d793fa7-5fc6-432a-b26b-d1b10769d42e", spot_market="91-140-68",
                 perp="binance_DOGEUSDT", network=None, etf=None, hl_pair=None, dat=None, unlocks=False),
    "ADA": dict(asset_id="362f0140-ecdd-4205-b8a0-36f0fd5d8167", spot_market="6058-140-68",
                perp="binance_ADAUSDT", network="5d04ce60-a3ba-41f2-b8e5-1a3af5514126", etf=None,
                hl_pair=None, dat=None, unlocks=True),
    "AVAX": dict(asset_id="2db6b38a-681a-4514-9d67-691e319597ee", spot_market="1265-140-68",
                 perp="binance_AVAXUSDT", network="60b82a80-8324-4d5f-bfe5-7e756b0023f9", etf=None,
                 hl_pair="Avalanche", dat=None, unlocks=True),
    "LINK": dict(asset_id="a9c04b71-33e4-4fed-8126-cf1612c7e198", spot_market="6135-140-68",
                 perp="binance_LINKUSDT", network=None, etf=None, hl_pair=None, dat=None, unlocks=True),
    "TRX": dict(asset_id="1c077d6e-99c7-491c-b24d-1d359011cd81", spot_market="609-140-68",
                perp="binance_TRXUSDT", network="b0b77938-d093-4d3e-ba5e-a27608b55e9d", etf=None,
                hl_pair=None, dat=None, unlocks=False),
    "SUI": dict(asset_id="78c4b0c5-8cbe-4f05-b639-0eb942e86dd5", spot_market="21023-140-68",
                perp="binance_SUIUSDT", network=None, etf=None, hl_pair=None, dat=None, unlocks=True),
    "HYPE": dict(asset_id="99b6a5f4-2200-4098-82fd-53d091bdee70", spot_market=None,
                 perp="hyperliquid_HYPE", network=None, etf=None, hl_pair="Hyperliquid", dat=None,
                 unlocks=True),
}

# Market-wide sources used for every asset.
MARKET = dict(
    btc_mcap=("asset-marketcap", BTC_ID),         # BTC dominance -> total crypto mcap
    stables=("stablecoins-by-currency", "usd"),    # USD stablecoin supply / mints / CEX balances
    gold=("asset-price", PAXG_ID),                 # PAXG as the only long-history macro hedge proxy
    hyperliquid=("perp-dex-protocols", "hyperliquid"),
    equities=("tokenized-equities-by-equity", "spy"),  # underlying SPY close, only from 2025-01
)

# Publication lag (days) applied before a value may be used at decision time t.
# Decision time t = 00:00 UTC right after daily bar t closes; we only use bar t and earlier.
LAG = {
    "price": 0,       # VWAP candle for [t, t+1d) is complete at decision time
    "perp": 0,        # exchange candle and end-of-day OI, same timestamp convention
    "dominance": 0,
    "gold": 0,        # PAXG trades 24/7
    "hyperliquid": 0,
    "onchain": 1,     # chain metrics for day t are finalised with a delay; last point often partial
    "stables": 1,     # stablecoin balances/mints for day t land after the day closes
    "etf": 1,         # US ETF flows are published after the US close / next morning
    "dat": 1,         # treasury purchases are dated on the 8-K announcement day
    "supply": 1,
    "equities": 1,    # US close for date t is after 00:00 UTC of t+1 only for t's session
}

# Forward-fill limits (days) for slow series; anything staler becomes NaN and is recorded.
FFILL = {"onchain": 3, "stables": 3, "etf": 4, "dat": 0, "supply": 7, "equities": 4,
         "gold": 3, "hyperliquid": 2, "perp": 1, "dominance": 2}

HORIZONS = [1, 7, 14, 28, 90]

# Round-trip cost per non-overlapping trade, in basis points. Wider for longer horizons:
# turnover is lower but gap/funding/slippage risk per position is larger.
COST_BPS = {1: 10, 7: 15, 14: 20, 28: 30, 90: 50}

# Walk-forward settings per horizon. retrain = test block length in days.
WALK = {
    1:  dict(retrain=91,  min_train=730),
    7:  dict(retrain=91,  min_train=730),
    14: dict(retrain=91,  min_train=730),
    28: dict(retrain=182, min_train=1095),
    90: dict(retrain=182, min_train=1095),
}

# Vol window (days) used to normalise the target and to set the ternary flat band.
def vol_window(h: int) -> int:
    return max(30, h)

FLAT_BAND_K = 0.5   # flat = |fwd log return| <= 0.5 * trailing vol scaled to the horizon
PI_LEVEL = 0.80     # prediction interval coverage
