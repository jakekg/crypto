# XRP-USD probabilistic forward-return forecast

Decision time: **2026-10-09 00:00 UTC** (last completed daily bar 2026-10-08, VWAP close $1.379). Snapshot of Blockworks MCP data; nothing after the last completed bar is used.

Primary model: ridge regression on vol-scaled forward returns (shrinkage picked by an inner, purged time split), with the predictive distribution taken from empirical residuals. Comparators: zero return, prevailing mean (drift), trailing same-horizon momentum, and a depth-2 gradient-boosted model. Walk-forward is purged with an embargo of one full horizon. Costs (round trip per non-overlapping trade): 1d 10bps, 7d 15bps, 14d 20bps, 28d 30bps, 90d 50bps.

## Summary

| horizon | P(up) | P(down/flat/up) | expected return | 80% interval | price range | non-overlap IC (N) | status |
|---|---|---|---|---|---|---|---|
| 1d | 48.4% | 27/50/24% | +0.08% | -4.1% / +4.4% | $1.322 - $1.439 | -0.003 (1552) | no edge claimed |
| 7d | 44.5% | 29/48/23% | +0.76% | -10.7% / +13.8% | $1.231 - $1.569 | -0.075 (218) | no edge claimed |
| 14d | 41.2% | 32/45/22% | +1.74% | -15.2% / +25.2% | $1.169 - $1.726 | -0.085 (107) | no edge claimed |
| 28d | 41.3% | 34/43/23% | +2.99% | -21.3% / +55.3% | $1.085 - $2.141 | +0.080 (45) | regime tilt |
| 90d | 49.4% | 29/45/25% | +3.10% | -30.1% / +43.1% | $0.9634 - $1.974 | +0.248 (12) | regime tilt |

## Horizon 1d

**Current forecast** (as of 2026-10-08, model trained on 2284 labelled days through 2026-10-07, shrinkage kappa=100.0):

- P(up) = **48.4%**; ternary P(down / flat / up) = 26.7% / 49.6% / 23.8% (flat band = +-0.5 x 30d trailing vol scaled to 1d = +-1.84%)
- Expected log return +0.08% (simple +0.08%) = shrunk drift +0.10% (sample drift +0.16% x prior weight 0.61) + feature signal -0.01%; GBM comparator -0.03%
- 80% prediction interval: -4.1% to +4.4% -> $1.322 to $1.439 (horizon vol 3.7%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC -0.003 (t=-0.13) does not clear momentum (-0.061) with t>2; net Sharpe -0.05 vs momentum -0.60 / buy-and-hold 0.41 (t=-0.10)**

**Data actually used** (model sample 2020-07-07 -> 2026-10-08, 20 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| perp | `perpetual-markets` / `binance_XRPUSDT` | 2020-01-06 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2015-01-26 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6238-140-68` | 2018-05-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (18 purged folds, retrain every 91d, embargo 1d, test 2022-07-09 -> 2026-10-07):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1552 | 1552 | 1552 | 1552 | 1552 |
| IC, signal only (Spearman, vol-scaled) | -0.003 | -0.003 (t=-0.13) | -0.061 | n/a (constant) | +0.015 |
| IC, full forecast incl. per-fold drift | -0.024 | -0.024 | | | |
| Hit rate | 0.484 | 0.484 | 0.461 | drift 0.484 / long 0.484 | 0.525 |
| Drift-only R2 vs zero | | -0.0025 | | | |
| R2 OOS vs zero / vs drift | -0.0007 / +0.0018 | -0.0007 / +0.0018 | | | -0.0178 |
| Brier P(up) (clim / 0.5) | 0.2498 (0.2504 / 0.2500) | 0.2498 (0.2504) | | | 0.2556 |
| 80% interval coverage | 0.796 | 0.796 | | | |
| Ternary log-loss (clim) | 1.0423 (1.0424) | 1.0423 (1.0424) | | | |
| Mean net return / trade after 10bps | | -1bps (t=-0.10) | -13bps (t=-1.24) | buy-and-hold +9bps | +6bps |
| Annualised Sharpe (net) | | -0.05 | -0.60 | buy-and-hold 0.41 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 492 | 492.0 | +0.019 | 0.480 | 0.463 | 0.480 |
| bear | 476 | 476.0 | -0.025 | 0.492 | 0.487 | 0.492 |
| chop | 584 | 584.0 | +0.012 | 0.481 | 0.437 | 0.481 |

IC by calendar year: 2022: -0.12, 2023: -0.03, 2024: +0.06, 2025: +0.01, 2026: -0.09

Calibration of P(up) (OOS quintiles): p=0.49 -> 0.47 (n=326); p=0.49 -> 0.46 (n=300); p=0.50 -> 0.51 (n=359); p=0.50 -> 0.51 (n=302); p=0.50 -> 0.47 (n=265)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_7d` | +0.0008 | +0.0009 | 100% | yes | -0.83 | 7d log return, vol-scaled |
| `prem_lvl_3d` | +0.0006 | +0.0009 | 100% | yes | -0.73 | Perp premium (funding proxy) 3d mean, trailing z |
| `ret_3d` | +0.0006 | +0.0007 | 100% | yes | -1.41 | 3d log return, vol-scaled |
| `stab_growth_7d` | +0.0006 | +0.0006 | 100% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `vol_7v90` | +0.0005 | +0.0005 | 100% | yes | -0.30 | 7d vs 90d realised vol, log, trailing z (vol spike) |
| `tech_trend` | +0.0005 | +0.0005 | 100% | yes | -0.31 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `int_volspike_x_range` | +0.0004 | +0.0002 | 72% | no | +0.00 | Vol spike x large daily range (liquidation-cluster proxy) |
| `perp_spot_vol` | +0.0004 | +0.0004 | 100% | yes | +0.46 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_1d` | -0.854 | 1d log return, vol-scaled |
| `ret_3d` | -1.407 | 3d log return, vol-scaled |
| `ret_7d` | -0.830 | 7d log return, vol-scaled |
| `vol_7v90` | -0.296 | 7d vs 90d realised vol, log, trailing z (vol spike) |
| `range_1d` | +0.885 | 1d high/low range, log, trailing z (liquidation-cluster proxy) |
| `tech_trend` | -0.308 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -0.713 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_3d` | -0.726 | Perp premium (funding proxy) 3d mean, trailing z |
| `prem_chg_3d` | -0.706 | Change in perp premium over 3d, trailing z |
| `perp_spot_vol` | +0.463 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_1d` | -0.123 | Cumulative 1d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_1d` | -0.622 | USD-stablecoin supply growth over 1d (mint minus burn), trailing z |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `btc_ret_1d` | -1.012 | BTC 1d return, vol-scaled |
| `rel_btc_1d` | -0.376 | XRP/BTC 1d relative return, trailing z |
| `btcdom_chg_1d` | +0.497 | BTC dominance 1d change, trailing z |
| `gold_ret_1d` | +0.402 | PAXG (gold) 1d return, trailing z (only long-history macro proxy) |
| `int_cexflow_x_negmom` | +0.331 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `int_volspike_x_range` | +0.000 | Vol spike x large daily range (liquidation-cluster proxy) |

</details>

**Short-history ablation**: no extended feature has >= 700 days of history
Too short to evaluate (< 700 days): `spy_ret_1d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.12 to +0.06; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-07-07 (binding source: perp_spot_vol).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 7d

**Current forecast** (as of 2026-10-08, model trained on 2275 labelled days through 2026-10-01, shrinkage kappa=100.0):

- P(up) = **44.5%**; ternary P(down / flat / up) = 29.1% / 48.1% / 22.8% (flat band = +-0.5 x 30d trailing vol scaled to 7d = +-4.95%)
- Expected log return +0.76% (simple +0.76%) = shrunk drift +0.81% (sample drift +1.32% x prior weight 0.61) + feature signal -0.05%; GBM comparator -0.57%
- 80% prediction interval: -10.7% to +13.8% -> $1.231 to $1.569 (horizon vol 9.7%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC -0.075 (t=-1.11) does not clear momentum (-0.076) with t>2; net Sharpe 0.34 vs momentum -0.13 / buy-and-hold 0.44 (t=0.70); Brier 0.2497 not below climatology 0.2491**

**Data actually used** (model sample 2020-07-10 -> 2026-10-08, 17 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| perp | `perpetual-markets` / `binance_XRPUSDT` | 2020-01-06 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2015-01-26 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6238-140-68` | 2018-05-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (17 purged folds, retrain every 91d, embargo 7d, test 2022-07-24 -> 2026-10-01):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1531 | 218 | 218 | 218 | 218 |
| IC, signal only (Spearman, vol-scaled) | -0.076 | -0.075 (t=-1.11) | -0.076 | n/a (constant) | -0.017 |
| IC, full forecast incl. per-fold drift | -0.122 | -0.122 | | | |
| Hit rate | 0.464 | 0.464 | 0.451 | drift 0.464 / long 0.464 | 0.491 |
| Drift-only R2 vs zero | | -0.0283 | | | |
| R2 OOS vs zero / vs drift | -0.0073 / +0.0204 | -0.0072 / +0.0205 | | | -0.0985 |
| Brier P(up) (clim / 0.5) | 0.2497 (0.2491 / 0.2500) | 0.2497 (0.2491) | | | 0.2791 |
| 80% interval coverage | 0.811 | 0.811 | | | |
| Ternary log-loss (clim) | 1.0424 (1.0415) | 1.0424 (1.0415) | | | |
| Mean net return / trade after 15bps | | +49bps (t=0.70) | -55bps (t=-0.26) | buy-and-hold +64bps | -44bps |
| Annualised Sharpe (net) | | 0.34 | -0.13 | buy-and-hold 0.44 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 486 | 69.4 | -0.013 | 0.430 | 0.461 | 0.430 |
| bear | 461 | 65.9 | -0.063 | 0.503 | 0.456 | 0.503 |
| chop | 584 | 83.4 | -0.093 | 0.461 | 0.438 | 0.461 |

IC by calendar year: 2022: -0.11, 2023: -0.06, 2024: -0.00, 2025: -0.19, 2026: -0.13

Calibration of P(up) (OOS quintiles): p=0.44 -> 0.55 (n=313); p=0.45 -> 0.47 (n=300); p=0.45 -> 0.44 (n=306); p=0.45 -> 0.46 (n=308); p=0.46 -> 0.39 (n=304)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_14d` | +0.0017 | +0.0034 | 100% | yes | -0.78 | 14d log return, vol-scaled |
| `stab_growth_7d` | +0.0015 | +0.0020 | 100% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `ret_7d` | +0.0015 | +0.0028 | 100% | yes | -0.83 | 7d log return, vol-scaled |
| `prem_lvl_7d` | +0.0015 | +0.0024 | 100% | yes | -0.37 | Perp premium (funding proxy) 7d mean, trailing z |
| `vol_regime` | -0.0014 | -0.0025 | 100% | yes | +0.01 | 30d realised vol percentile within trailing 2y (centred) |
| `perp_spot_vol` | +0.0012 | +0.0011 | 100% | yes | +0.46 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `prem_persist_14d` | +0.0012 | +0.0020 | 100% | yes | -0.07 | Share of last 14d with premium above its 1y median (funding persistence) |
| `btc_ret_7d` | +0.0012 | +0.0022 | 100% | yes | -0.71 | BTC 7d return, vol-scaled |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -0.830 | 7d log return, vol-scaled |
| `ret_14d` | -0.778 | 14d log return, vol-scaled |
| `ret_28d` | +0.172 | 28d log return, vol-scaled |
| `vol_14v90` | -1.090 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | +0.012 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | -0.308 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -0.713 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | -0.374 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | +0.171 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | -0.071 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.463 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `btc_ret_7d` | -0.708 | BTC 7d return, vol-scaled |
| `rel_btc_7d` | -0.538 | XRP/BTC 7d relative return, trailing z |
| `btcdom_chg_7d` | +0.178 | BTC dominance 7d change, trailing z |
| `gold_ret_7d` | -0.462 | PAXG (gold) 7d return, trailing z (only long-history macro proxy) |

</details>

**Short-history ablation**: no extended feature has >= 700 days of history
Too short to evaluate (< 700 days): `spy_ret_7d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.19 to -0.00; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-07-10 (binding source: prem_chg_7d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 14d

**Current forecast** (as of 2026-10-08, model trained on 2268 labelled days through 2026-09-24, shrinkage kappa=100.0):

- P(up) = **41.2%**; ternary P(down / flat / up) = 32.5% / 45.3% / 22.2% (flat band = +-0.5 x 30d trailing vol scaled to 14d = +-7.08%)
- Expected log return +1.73% (simple +1.74%) = shrunk drift +1.77% (sample drift +2.91% x prior weight 0.61) + feature signal -0.04%; GBM comparator +0.90%
- 80% prediction interval: -15.2% to +25.2% -> $1.169 to $1.726 (horizon vol 13.7%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC -0.085 (t=-0.87) does not clear momentum (-0.082) with t>2; net Sharpe 0.34 vs momentum -0.08 / buy-and-hold 0.40 (t=0.69); Brier 0.2499 not below climatology 0.2478**

**Data actually used** (model sample 2020-07-10 -> 2026-10-08, 17 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| perp | `perpetual-markets` / `binance_XRPUSDT` | 2020-01-06 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2015-01-26 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6238-140-68` | 2018-05-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (17 purged folds, retrain every 91d, embargo 14d, test 2022-08-07 -> 2026-09-24):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1510 | 107 | 107 | 107 | 107 |
| IC, signal only (Spearman, vol-scaled) | -0.086 | -0.085 (t=-0.87) | -0.082 | n/a (constant) | -0.016 |
| IC, full forecast incl. per-fold drift | -0.122 | -0.120 | | | |
| Hit rate | 0.446 | 0.446 | 0.446 | drift 0.446 / long 0.446 | 0.484 |
| Drift-only R2 vs zero | | -0.0675 | | | |
| R2 OOS vs zero / vs drift | -0.0165 / +0.0475 | -0.0165 / +0.0478 | | | -0.1643 |
| Brier P(up) (clim / 0.5) | 0.2499 (0.2478 / 0.2500) | 0.2499 (0.2478) | | | 0.3055 |
| 80% interval coverage | 0.838 | 0.838 | | | |
| Ternary log-loss (clim) | 1.0594 (1.0508) | 1.0594 (1.0508) | | | |
| Mean net return / trade after 20bps | | +109bps (t=0.69) | -47bps (t=-0.16) | buy-and-hold +129bps | +25bps |
| Annualised Sharpe (net) | | 0.34 | -0.08 | buy-and-hold 0.40 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 479 | 34.2 | -0.132 | 0.378 | 0.443 | 0.378 |
| bear | 447 | 31.9 | +0.096 | 0.526 | 0.459 | 0.526 |
| chop | 584 | 41.7 | -0.138 | 0.440 | 0.440 | 0.440 |

IC by calendar year: 2022: -0.43, 2023: +0.00, 2024: -0.11, 2025: -0.09, 2026: -0.09

Calibration of P(up) (OOS quintiles): p=0.39 -> 0.57 (n=302); p=0.40 -> 0.44 (n=306); p=0.40 -> 0.39 (n=300); p=0.41 -> 0.43 (n=315); p=0.41 -> 0.40 (n=287)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `vol_regime` | -0.0025 | -0.0030 | 100% | yes | +0.01 | 30d realised vol percentile within trailing 2y (centred) |
| `btc_ret_14d` | +0.0023 | +0.0027 | 100% | yes | -0.43 | BTC 14d return, vol-scaled |
| `ret_7d` | +0.0019 | +0.0026 | 100% | yes | -0.83 | 7d log return, vol-scaled |
| `prem_lvl_7d` | +0.0017 | +0.0018 | 100% | yes | -0.37 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | +0.0016 | +0.0018 | 100% | yes | +0.17 | Change in perp premium over 7d, trailing z |
| `ret_14d` | +0.0015 | +0.0024 | 100% | yes | -0.78 | 14d log return, vol-scaled |
| `perp_spot_vol` | +0.0014 | +0.0012 | 94% | yes | +0.46 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `prem_persist_14d` | +0.0012 | +0.0013 | 100% | yes | -0.07 | Share of last 14d with premium above its 1y median (funding persistence) |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -0.830 | 7d log return, vol-scaled |
| `ret_14d` | -0.778 | 14d log return, vol-scaled |
| `ret_28d` | +0.172 | 28d log return, vol-scaled |
| `vol_14v90` | -1.090 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | +0.012 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | -0.308 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -0.713 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | -0.374 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | +0.171 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | -0.071 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.463 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_14d` | -0.508 | Cumulative 14d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_14d` | -0.461 | USD-stablecoin supply growth over 14d (mint minus burn), trailing z |
| `btc_ret_14d` | -0.426 | BTC 14d return, vol-scaled |
| `rel_btc_14d` | -0.617 | XRP/BTC 14d relative return, trailing z |
| `btcdom_chg_14d` | +0.185 | BTC dominance 14d change, trailing z |
| `gold_ret_14d` | -0.826 | PAXG (gold) 14d return, trailing z (only long-history macro proxy) |

</details>

**Short-history ablation**: no extended feature has >= 700 days of history
Too short to evaluate (< 700 days): `spy_ret_14d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.43 to +0.00; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-07-10 (binding source: prem_chg_7d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 28d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 2434 labelled days through 2026-09-10, shrinkage kappa=30.0):

- P(up) = **41.3%**; ternary P(down / flat / up) = 33.8% / 43.2% / 23.0% (flat band = +-0.5 x 30d trailing vol scaled to 28d = +-10.15%)
- Expected log return +2.94% (simple +2.99%) = shrunk drift +3.21% (sample drift +5.14% x prior weight 0.63) + feature signal -0.27%; GBM comparator +7.43%
- 80% prediction interval: -21.3% to +55.3% -> $1.085 to $2.141 (horizon vol 19.3%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.080 (t=0.53) does not clear momentum (-0.196) with t>2; net Sharpe 0.34 vs momentum -0.71 / buy-and-hold 0.39 (t=0.64)**
- Non-overlapping test windows: **45** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2020-01-12 -> 2026-10-08, 14 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2015-01-26 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |
| supply | `asset-supply` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2018-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (8 purged folds, retrain every 182d, embargo 28d, test 2023-03-08 -> 2026-09-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1283 | 45 | 45 | 45 | 45 |
| IC, signal only (Spearman, vol-scaled) | +0.097 | +0.080 (t=0.53) | -0.196 | n/a (constant) | +0.168 |
| IC, full forecast incl. per-fold drift | -0.130 | -0.134 | | | |
| Hit rate | 0.451 | 0.450 | 0.433 | drift 0.450 / long 0.450 | 0.531 |
| Drift-only R2 vs zero | | -0.0782 | | | |
| R2 OOS vs zero / vs drift | -0.0211 / +0.0536 | -0.0206 / +0.0534 | | | -0.2479 |
| Brier P(up) (clim / 0.5) | 0.2468 (0.2487 / 0.2500) | 0.2468 (0.2487) | | | 0.3067 |
| 80% interval coverage | 0.813 | 0.813 | | | |
| Ternary log-loss (clim) | 1.0649 (1.0550) | 1.0648 (1.0550) | | | |
| Mean net return / trade after 30bps | | +235bps (t=0.64) | -329bps (t=-1.33) | buy-and-hold +265bps | +336bps |
| Annualised Sharpe (net) | | 0.34 | -0.71 | buy-and-hold 0.39 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 404 | 14.4 | -0.263 | 0.309 | 0.423 | 0.309 |
| bear | 391 | 14.0 | +0.383 | 0.524 | 0.448 | 0.524 |
| chop | 488 | 17.4 | -0.085 | 0.508 | 0.428 | 0.508 |

IC by calendar year: 2023: +0.17, 2024: -0.05, 2025: +0.07, 2026: +0.17

Calibration of P(up) (OOS quintiles): p=0.41 -> 0.47 (n=263); p=0.42 -> 0.34 (n=250); p=0.42 -> 0.36 (n=265); p=0.43 -> 0.51 (n=255); p=0.44 -> 0.58 (n=250)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_90d` | -0.0107 | -0.0037 | 100% | yes | +0.64 | 90d log return, vol-scaled |
| `vol_regime` | -0.0105 | -0.0036 | 100% | yes | +0.01 | 30d realised vol percentile within trailing 2y (centred) |
| `rel_btc_28d` | -0.0105 | -0.0039 | 100% | yes | -0.25 | XRP/BTC 28d relative return, trailing z |
| `beta_btc_180d` | -0.0095 | -0.0018 | 100% | yes | +0.72 | Rolling 180d beta to BTC, 4y trailing z |
| `btcdom_chg_28d` | +0.0087 | +0.0031 | 100% | yes | +0.12 | BTC dominance 28d change, trailing z |
| `btc_ret_28d` | +0.0078 | +0.0027 | 100% | yes | +0.60 | BTC 28d return, vol-scaled |
| `ret_28d` | -0.0055 | -0.0019 | 100% | yes | +0.17 | 28d log return, vol-scaled |
| `stab_growth_90d` | +0.0046 | +0.0020 | 100% | yes | -0.37 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.172 | 28d log return, vol-scaled |
| `ret_90d` | +0.641 | 90d log return, vol-scaled |
| `drawdown_365` | -0.660 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.087 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | +0.012 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `btc_ret_28d` | +0.605 | BTC 28d return, vol-scaled |
| `rel_btc_28d` | -0.255 | XRP/BTC 28d relative return, trailing z |
| `btcdom_chg_28d` | +0.123 | BTC dominance 28d change, trailing z |
| `beta_btc_180d` | +0.717 | Rolling 180d beta to BTC, 4y trailing z |
| `supply_infl_90d` | -1.308 | Trailing 90d circulating supply inflation, trailing z |

</details>

**Short-history ablation** (core vs core + `gold_ret_28d`; same rows, window 2020-04-29 -> 2026-10-08, test 2021-06-24 -> 2026-09-10, 68 non-overlapping windows): IC +0.028 -> +0.024 (overlapping +0.035 -> +0.033), hit 0.443 -> 0.443, Brier 0.2537 -> 0.2533 (clim 0.2486). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_28d`.

**Failure modes for this horizon:**

- Sparse non-overlapping samples: 45 independent test windows; one regime shift can flip the sign (see IC by year).
- Thin history: model sample starts 2020-01-12 (binding source: stab_cexflow_90d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 90d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 2372 labelled days through 2026-07-10, shrinkage kappa=100.0):

- P(up) = **49.4%**; ternary P(down / flat / up) = 29.0% / 45.5% / 25.5% (flat band = +-0.5 x 90d trailing vol scaled to 90d = +-18.25%)
- Expected log return +3.05% (simple +3.10%) = shrunk drift +3.19% (sample drift +5.16% x prior weight 0.62) + feature signal -0.14%; GBM comparator -0.95%
- 80% prediction interval: -30.1% to +43.1% -> $0.9634 to $1.974 (horizon vol 33.5%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.248 (t=0.81) does not clear momentum (-0.016) with t>2; net Sharpe 0.39 vs momentum 0.29 / buy-and-hold 0.41 (t=0.70)**
- Non-overlapping test windows: **12** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2020-01-12 -> 2026-10-08, 14 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2015-01-26 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |
| supply | `asset-supply` / `97775be0-2608-4720-b7af-f85b24c7eb2d` | 2018-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (7 purged folds, retrain every 182d, embargo 90d, test 2023-07-10 -> 2026-07-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1097 | 12 | 12 | 12 | 12 |
| IC, signal only (Spearman, vol-scaled) | +0.292 | +0.248 (t=0.81) | -0.016 | n/a (constant) | -0.164 |
| IC, full forecast incl. per-fold drift | -0.172 | -0.173 | | | |
| Hit rate | 0.464 | 0.464 | 0.467 | drift 0.464 / long 0.464 | 0.488 |
| Drift-only R2 vs zero | | -0.0463 | | | |
| R2 OOS vs zero / vs drift | -0.0488 / -0.0040 | -0.0515 / -0.0050 | | | -0.2281 |
| Brier P(up) (clim / 0.5) | 0.2607 (0.2635 / 0.2500) | 0.2607 (0.2636) | | | 0.3668 |
| 80% interval coverage | 0.715 | 0.715 | | | |
| Ternary log-loss (clim) | 1.1522 (1.1318) | 1.1515 (1.1315) | | | |
| Mean net return / trade after 50bps | | +593bps (t=0.70) | +86bps (t=0.52) | buy-and-hold +643bps | +108bps |
| Annualised Sharpe (net) | | 0.39 | 0.29 | buy-and-hold 0.41 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 330 | 3.7 | +0.051 | 0.321 | 0.321 | 0.321 |
| bear | 351 | 3.9 | +0.701 | 0.419 | 0.581 | 0.419 |
| chop | 416 | 4.6 | -0.108 | 0.615 | 0.483 | 0.615 |

IC by calendar year: 2023: +0.07, 2024: -0.38, 2025: +0.60, 2026: +0.24

Calibration of P(up) (OOS quintiles): p=0.51 -> 0.08 (n=220); p=0.51 -> 0.42 (n=231); p=0.53 -> 0.42 (n=211); p=0.54 -> 0.79 (n=217); p=0.76 -> 0.62 (n=218)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `beta_btc_180d` | -0.0033 | -0.0007 | 43% | no | +0.72 | Rolling 180d beta to BTC, 4y trailing z |
| `ret_28d` | -0.0019 | -0.0096 | 100% | yes | +0.17 | 28d log return, vol-scaled |
| `stab_growth_90d` | +0.0015 | +0.0082 | 100% | yes | -0.37 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `stab_growth_28d` | +0.0014 | +0.0097 | 100% | yes | -0.50 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `btcdom_chg_90d` | +0.0011 | +0.0002 | 71% | no | +0.18 | BTC dominance 90d change, trailing z |
| `rel_btc_90d` | -0.0010 | -0.0105 | 100% | yes | -0.24 | XRP/BTC 90d relative return, trailing z |
| `vol_regime` | -0.0010 | -0.0055 | 100% | yes | +0.01 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_90d` | -0.0009 | +0.0108 | 14% | no | -0.46 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.172 | 28d log return, vol-scaled |
| `ret_90d` | +0.641 | 90d log return, vol-scaled |
| `drawdown_365` | -0.660 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.087 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | +0.012 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `btc_ret_90d` | +1.254 | BTC 90d return, vol-scaled |
| `rel_btc_90d` | -0.241 | XRP/BTC 90d relative return, trailing z |
| `btcdom_chg_90d` | +0.181 | BTC dominance 90d change, trailing z |
| `beta_btc_180d` | +0.717 | Rolling 180d beta to BTC, 4y trailing z |
| `supply_infl_90d` | -1.308 | Trailing 90d circulating supply inflation, trailing z |

</details>

**Short-history ablation** (core vs core + `gold_ret_90d`; same rows, window 2020-06-30 -> 2026-10-08, test 2021-12-27 -> 2026-07-10, 18 non-overlapping windows): IC +0.029 -> -0.048 (overlapping +0.034 -> -0.045), hit 0.489 -> 0.474, Brier 0.2682 -> 0.2549 (clim 0.2741). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_90d`.

**Failure modes for this horizon:**

- Sparse non-overlapping samples: 12 independent test windows; one regime shift can flip the sign (see IC by year).
- Thin history: model sample starts 2020-01-12 (binding source: stab_cexflow_90d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Requested inputs not available in the Blockworks MCP

- Taker buy/sell volume and bid/ask spreads (no trade-level or order-book data for CEX markets)
- Funding rates (no funding field anywhere) -> proxied by Binance perp/spot close premium
- Dated-futures basis (perpetual-markets excludes dated futures) -> same perp-premium proxy
- CEX liquidations and any long/short liquidation split (only Hyperliquid total liquidations, 2025-05+)
- CEX open interest history (only ~300 days, 2025-12+) -> Hyperliquid OI (2023-05+) used in ablation
- Options implied vol / skew
- Native-asset exchange inflows/outflows and whale/labeled-wallet flows (only USD-stablecoin CEX balances; treasury-company purchases for BTC)
- DXY, nominal/real yields with usable history (tokenized TIP/IEF/TLT only from 2025)
- Equities with usable history (tokenized SPY close only from 2025-01) -> ablation only
- Social sentiment (2025-01+) and mindshare (2024-07+): too short for 28d/90d; not used
