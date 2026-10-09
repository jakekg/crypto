"""Horizon-specific engineered features.

Every feature at row T uses only panel values at T or earlier (the panel already applies
publication lags). Level-type features are converted to rolling z-scores over a trailing
window (past-only); model-time standardisation is refit on each training window in
model.py. Pure price technicals are compressed into two composites instead of being
passed as independent signals.

Each feature is tagged:
  tier "core"     -> long history (>= ~2019), used by the primary model
  tier "extended" -> short history (Hyperliquid OI 2023+, ETF 2024+, liquidations 2025+,
                     SPY 2025+). Evaluated only as an ablation on its own window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

Z_WIN, Z_MIN = 730, 180


def rz(x: pd.Series, win: int = Z_WIN, minp: int = Z_MIN) -> pd.Series:
    """Trailing (past-and-present only) z-score, clipped to +-4."""
    m = x.rolling(win, min_periods=minp).mean()
    s = x.rolling(win, min_periods=minp).std()
    return ((x - m) / s.replace(0, np.nan)).clip(-4, 4)


def rpct(x: pd.Series, win: int = 730, minp: int = 365) -> pd.Series:
    """Trailing percentile rank of today's value within the window, centred to [-0.5, 0.5]."""
    return x.rolling(win, min_periods=minp).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True) - 0.5


@dataclass
class FeatureSet:
    X: pd.DataFrame
    tiers: dict          # feature -> "core" | "extended"
    sources: dict        # feature -> source keys used
    desc: dict           # feature -> human description


class _Builder:
    def __init__(self, p: pd.DataFrame, symbol: str):
        self.p, self.symbol = p, symbol
        self.cols, self.tiers, self.sources, self.desc = {}, {}, {}, {}
        lc = np.log(p["price__close"])
        self.lc, self.r1 = lc, lc.diff()
        self.sd30 = self.r1.rolling(30, min_periods=24).std()

    def has(self, *cols) -> bool:
        return all(c in self.p.columns and self.p[c].notna().sum() > 200 for c in cols)

    def add(self, name, series, tier, sources, desc):
        self.cols[name] = series.replace([np.inf, -np.inf], np.nan)
        self.tiers[name], self.sources[name], self.desc[name] = tier, sources, desc

    # ---------- building blocks ----------
    def nret(self, n):
        """n-day log return scaled by trailing daily vol * sqrt(n) (unit-free momentum)."""
        return (self.lc - self.lc.shift(n)) / (self.sd30 * np.sqrt(n))

    def vol(self, n):
        return self.r1.rolling(n, min_periods=max(3, int(n * 0.8))).std()

    def premium(self):
        """Binance perp close / Binance spot close - 1 at the same 00:00 UTC timestamp."""
        return self.p["perp__closePrice"] / self.p["spot__close-price-raw"] - 1

    def technicals(self, tier="core"):
        c = self.p["price__close"]
        delta = c.diff()
        up, dn = delta.clip(lower=0).ewm(alpha=1 / 14).mean(), (-delta.clip(upper=0)).ewm(alpha=1 / 14).mean()
        rsi = 100 - 100 / (1 + up / dn)
        ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
        trend = pd.concat([rz((rsi - 50) / 50), rz(np.log(c / ma50)), rz(np.log(c / ma200))], axis=1).mean(axis=1)
        bbw = 4 * c.rolling(20).std() / c.rolling(20).mean()
        self.add("tech_trend", trend, tier, ["price"],
                 "Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores)")
        self.add("tech_bbwidth", rz(np.log(bbw)), tier, ["price"],
                 "Bollinger(20,2) band width, log, trailing z (compression vs expansion)")

    def cross_asset(self, n, tier="core"):
        if self.symbol != "BTC" and self.has("btc__close"):
            lb = np.log(self.p["btc__close"])
            self.add(f"btc_ret_{n}d", (lb - lb.shift(n)) / (lb.diff().rolling(30).std() * np.sqrt(n)),
                     tier, ["btc"], f"BTC {n}d return, vol-scaled")
            rel = self.lc - lb
            self.add(f"rel_btc_{n}d", rz(rel - rel.shift(n)), tier, ["price", "btc"],
                     f"{self.symbol}/BTC {n}d relative return, trailing z")
        if self.symbol == "BTC" and self.has("eth__close"):
            le = np.log(self.p["eth__close"])
            ratio = le - self.lc
            self.add(f"ethbtc_{n}d", rz(ratio - ratio.shift(n)), tier, ["eth", "price"],
                     f"ETH/BTC {n}d change (alt risk appetite), trailing z")
        if self.has("dom__circulating-marketcap-dominance"):
            d = self.p["dom__circulating-marketcap-dominance"]
            self.add(f"btcdom_chg_{n}d", rz(d - d.shift(n)), tier, ["dominance"],
                     f"BTC dominance {n}d change, trailing z")

    def gold(self, n, tier):
        if self.has("gold__close"):
            g = np.log(self.p["gold__close"])
            self.add(f"gold_ret_{n}d", rz(g - g.shift(n)), tier, ["gold"],
                     f"PAXG (gold) {n}d return, trailing z (only long-history macro proxy)")

    def stables(self, n, tier="core"):
        if self.has("stab__circulatingSupplyUSD", "stab__balanceCEXUSD"):
            sup = self.p["stab__circulatingSupplyUSD"]
            cex = self.p["stab__balanceCEXUSD"]
            self.add(f"stab_cexflow_{n}d", rz((cex - cex.shift(n)) / sup), tier, ["stables"],
                     f"Cumulative {n}d USD-stablecoin net flow into CEX wallets / supply, trailing z")
            self.add(f"stab_growth_{n}d", rz(np.log(sup / sup.shift(n))), tier, ["stables"],
                     f"USD-stablecoin supply growth over {n}d (mint minus burn), trailing z")

    def perp_block(self, n_lvl, n_chg, persistence=None, tier="core"):
        if not self.has("perp__closePrice", "spot__close-price-raw"):
            return
        prem = self.premium().clip(-0.01, 0.01)
        lvl = prem.rolling(n_lvl, min_periods=1).mean()
        self.add(f"prem_lvl_{n_lvl}d", rz(lvl), tier, ["perp", "spot"],
                 f"Perp premium (funding proxy) {n_lvl}d mean, trailing z")
        if n_chg:
            self.add(f"prem_chg_{n_chg}d", rz(lvl - lvl.shift(n_chg)), tier, ["perp", "spot"],
                     f"Change in perp premium over {n_chg}d, trailing z")
        if persistence:
            med = prem.rolling(365, min_periods=90).median()
            pers = (prem > med).astype(float).where(prem.notna()).rolling(persistence, min_periods=persistence // 2).mean()
            self.add(f"prem_persist_{persistence}d", pers - 0.5, tier, ["perp", "spot"],
                     f"Share of last {persistence}d with premium above its 1y median (funding persistence)")
        pv = self.p["perp__volumeUSD"].rolling(7, min_periods=5).sum()
        sv = self.p["price__volume"].rolling(7, min_periods=5).sum()
        self.add("perp_spot_vol", rz(np.log(pv / sv)), tier, ["perp", "price"],
                 "Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share)")

    def onchain(self, n_short, n_long, tier="core"):
        if self.has("chain__active-addresses"):
            aa = self.p["chain__active-addresses"].rolling(7, min_periods=4).mean()
            self.add(f"addr_growth_{n_short}_{n_long}", rz(np.log(aa.rolling(n_short, min_periods=3).mean()
                                                                  / aa.rolling(n_long, min_periods=n_long // 2).mean())),
                     tier, ["onchain"], f"Active addresses {n_short}d vs {n_long}d average, log, trailing z")
        if self.has("chain__fee-revenue"):
            fee = self.p["chain__fee-revenue"].rolling(7, min_periods=4).mean()
            self.add(f"fee_growth_{n_short}_{n_long}", rz(np.log(fee.rolling(n_short, min_periods=3).mean()
                                                                 / fee.rolling(n_long, min_periods=n_long // 2).mean())),
                     tier, ["onchain"], f"Network fee revenue {n_short}d vs {n_long}d average, log, trailing z")

    def valuation(self, tier="core"):
        if self.has("chain__total-transactions"):
            tx = self.p["chain__total-transactions"].rolling(28, min_periods=20).mean()
            v = np.log(self.p["price__close"]) - np.log(tx)
            self.add("price_to_activity", rz(v, 1460, 365), tier, ["price", "onchain"],
                     "log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity)")

    def regime(self, tier="core"):
        v30 = self.vol(30)
        self.add("vol_regime", rpct(v30), tier, ["price"], "30d realised vol percentile within trailing 2y (centred)")

    # ---------- extended (short history) ----------
    def extended(self, h):
        p = self.p
        if self.has("hl__oi"):
            oi = np.log(p["hl__oi"])
            n = {1: 1, 7: 7, 14: 14, 28: 28, 90: 28}[h]
            self.add(f"hl_oi_chg_{n}d", rz(oi - oi.shift(n), 365, 120), "extended", ["hyperliquid"],
                     f"Hyperliquid {self.symbol} open interest {n}d log change, trailing z")
            if h <= 14 and f"prem_lvl_3d" in self.cols:
                self.add("int_prem_x_oi", self.cols["prem_lvl_3d"].clip(lower=0) * self.cols[f"hl_oi_chg_{n}d"].clip(lower=0),
                         "extended", ["perp", "spot", "hyperliquid"], "High funding x rising OI (positive parts)")
        if h <= 7 and self.has("hl__liq"):
            liq = np.log1p(p["hl__liq"].clip(lower=0).rolling(3, min_periods=1).sum())
            self.add("hl_liq_3d", rz(liq, 365, 120), "extended", ["hyperliquid"],
                     "Hyperliquid liquidation volume 3d (no long/short split), log, trailing z")
        if self.has("etf__type-spot-flow-usd", "etf__type-spot-aum-usd"):
            n = {1: 1, 7: 7, 14: 14, 28: 28, 90: 90}[h]
            fl = p["etf__type-spot-flow-usd"].rolling(n, min_periods=max(1, int(n * 0.6))).sum()
            self.add(f"etf_flow_{n}d", rz(fl / p["etf__type-spot-aum-usd"], 365, 120), "extended", ["etf"],
                     f"Spot ETF {n}d net flow / AUM, trailing z")
        if self.has("eq__underlyingPriceUSD"):
            n = {1: 1, 7: 7, 14: 14, 28: 28, 90: 90}[h]
            e = np.log(p["eq__underlyingPriceUSD"])
            self.add(f"spy_ret_{n}d", (e - e.shift(n)) / (e.diff().rolling(60, min_periods=30).std() * np.sqrt(n)),
                     "extended", ["equities"], f"SPY {n}d return, vol-scaled (equity beta proxy)")
        if h >= 28 and self.has("dat__datPurchasesNative"):
            buys = p["dat__datPurchasesNative"].fillna(0).clip(lower=0).rolling(h, min_periods=1).sum()
            self.add(f"mstr_buys_{h}d", rz(np.log1p(buys), 730, 365), "extended", ["dat"],
                     f"Strategy (MSTR) BTC purchases over trailing {h}d, log, trailing z")


def build_features(panel: pd.DataFrame, symbol: str, h: int) -> FeatureSet:
    b = _Builder(panel, symbol)
    if h == 1:
        for n in (1, 3, 7):
            b.add(f"ret_{n}d", b.nret(n), "core", ["price"], f"{n}d log return, vol-scaled")
        b.add("vol_7v90", rz(np.log(b.vol(7) / b.vol(90))), "core", ["price"],
              "7d vs 90d realised vol, log, trailing z (vol spike)")
        rng = np.log(panel["price__high"] / panel["price__low"]).clip(lower=1e-5)
        b.add("range_1d", rz(np.log(rng)), "core", ["price"],
              "1d high/low range, log, trailing z (liquidation-cluster proxy)")
        b.technicals()
        b.perp_block(3, 3)
        b.stables(1)
        b.stables(7)
        b.cross_asset(1)
        b.gold(1, "core")
        # Interactions (spec): exchange inflows + negative momentum; vol spike + liquidation cluster.
        if "stab_cexflow_7d" in b.cols:
            b.add("int_cexflow_x_negmom", b.cols["stab_cexflow_7d"] * b.cols["ret_7d"].clip(upper=0), "core",
                  ["stables", "price"], "Stablecoin CEX inflow (7d) x negative 7d momentum")
        b.add("int_volspike_x_range", b.cols["vol_7v90"].clip(lower=0) * b.cols["range_1d"].clip(lower=0), "core",
              ["price"], "Vol spike x large daily range (liquidation-cluster proxy)")
        b.extended(h)
    elif h in (7, 14):
        for n in (7, 14, 28):
            b.add(f"ret_{n}d", b.nret(n), "core", ["price"], f"{n}d log return, vol-scaled")
        b.add("vol_14v90", rz(np.log(b.vol(14) / b.vol(90))), "core", ["price"], "14d vs 90d realised vol, log, trailing z")
        b.regime()
        b.technicals()
        b.perp_block(7, 7, persistence=14)
        b.stables(h)
        b.cross_asset(h)
        b.gold(h, "core")
        b.onchain(7, 90)
        b.extended(h)
    else:  # 28, 90: slow regime tilt. No order-book or single-day liquidation features.
        b.add("ret_28d", b.nret(28), "core", ["price"], "28d log return, vol-scaled")
        b.add("ret_90d", b.nret(90), "core", ["price"], "90d log return, vol-scaled")
        c = panel["price__close"]
        b.add("drawdown_365", rz(np.log(c / c.rolling(365, min_periods=200).max()), 1460, 365), "core", ["price"],
              "Drawdown from trailing 1y high, 4y trailing z")
        b.add("trend_200", rz(np.log(c / c.rolling(200, min_periods=150).mean()), 1460, 365), "core", ["price"],
              "Distance to 200d MA, 4y trailing z")
        b.regime()
        b.stables(28)
        b.stables(90)
        b.cross_asset(h)
        b.onchain(28, 365)
        b.valuation()
        if symbol != "BTC" and b.has("btc__close"):
            rb = np.log(panel["btc__close"]).diff()
            beta = b.r1.rolling(180, min_periods=120).cov(rb) / rb.rolling(180, min_periods=120).var()
            b.add("beta_btc_180d", rz(beta, 1460, 365), "core", ["price", "btc"], "Rolling 180d beta to BTC, 4y trailing z")
        # Use the unlock schedule only if it still covers the decision date (XRP's escrow
        # schedule in the MCP stops in 2022-06, which would otherwise truncate the sample).
        if f"unlock_fwd_{h}" in panel.columns and pd.notna(panel[f"unlock_fwd_{h}"].iloc[-1]):
            b.add(f"unlock_next_{h}d", panel[f"unlock_fwd_{h}"], "core", ["unlocks"],
                  f"Scheduled unlocks over the next {h}d as % of circulating supply (current schedule vintage)")
        if b.has("supply__circulating-supply"):
            s = np.log(panel["supply__circulating-supply"])
            b.add("supply_infl_90d", rz(s - s.shift(90), 1460, 365), "core", ["supply"],
                  "Trailing 90d circulating supply inflation, trailing z")
        b.gold(h, "extended")   # PAXG starts 2019-10; keep long-horizon core back to 2018
        b.extended(h)
    X = pd.DataFrame(b.cols, index=panel.index)
    return FeatureSet(X=X, tiers=b.tiers, sources=b.sources, desc=b.desc)
