# SOL-USD probabilistic forward-return forecast

Decision time: **2026-10-09 00:00 UTC** (last completed daily bar 2026-10-08, VWAP close $109.5). Snapshot of Blockworks MCP data; nothing after the last completed bar is used.

Primary model: ridge regression on vol-scaled forward returns (shrinkage picked by an inner, purged time split), with the predictive distribution taken from empirical residuals. Comparators: zero return, prevailing mean (drift), trailing same-horizon momentum, and a depth-2 gradient-boosted model. Walk-forward is purged with an embargo of one full horizon. Costs (round trip per non-overlapping trade): 1d 10bps, 7d 15bps, 14d 20bps, 28d 30bps, 90d 50bps.

## Summary

| horizon | P(up) | P(down/flat/up) | expected return | 80% interval | price range | non-overlap IC (N) | status |
|---|---|---|---|---|---|---|---|
| 1d | 49.1% | 29/43/28% | +0.03% | -3.9% / +4.5% | $105.3 - $114.4 | +0.026 (1300) | no edge claimed |
| 7d | 48.4% | 29/39/31% | +0.22% | -10.4% / +13.6% | $98.09 - $124.4 | -0.015 (165) | no edge claimed |
| 14d | 48.6% | 29/39/31% | +0.38% | -14.8% / +20.5% | $93.30 - $132.0 | -0.026 (81) | no edge claimed |
| 28d | 48.8% | 30/39/31% | +0.27% | -23.1% / +33.7% | $84.19 - $146.4 | +0.156 (21) | regime tilt |
| 90d | 48.4% | 33/38/29% | +0.78% | -33.0% / +45.1% | $73.33 - $158.9 | +0.152 (5) | indicative only |

## Horizon 1d

**Current forecast** (as of 2026-10-08, model trained on 2032 labelled days through 2026-10-07, shrinkage kappa=100.0):

- P(up) = **49.1%**; ternary P(down / flat / up) = 28.9% / 43.3% / 27.8% (flat band = +-0.5 x 30d trailing vol scaled to 1d = +-1.71%)
- Expected log return +0.03% (simple +0.03%) = shrunk drift +0.04% (sample drift +0.07% x prior weight 0.58) + feature signal -0.01%; GBM comparator -0.63%
- 80% prediction interval: -3.9% to +4.5% -> $105.3 to $114.4 (horizon vol 3.4%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.026 (t=0.95) does not clear momentum (+0.002) with t>2; net Sharpe -0.02 vs momentum -0.38 / buy-and-hold 0.57 (t=-0.04); Brier 0.2502 not below climatology 0.2501**

**Data actually used** (model sample 2021-03-16 -> 2026-10-08, 20 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| perp | `perpetual-markets` / `binance_SOLUSDT` | 2020-09-14 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-10 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6079-140-68` | 2020-08-11 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (15 purged folds, retrain every 91d, embargo 1d, test 2023-03-18 -> 2026-10-07):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1300 | 1300 | 1300 | 1300 | 1300 |
| IC, signal only (Spearman, vol-scaled) | +0.026 | +0.026 (t=0.95) | +0.002 | n/a (constant) | -0.019 |
| IC, full forecast incl. per-fold drift | -0.026 | -0.026 | | | |
| Hit rate | 0.507 | 0.507 | 0.503 | drift 0.494 / long 0.497 | 0.485 |
| Drift-only R2 vs zero | | -0.0023 | | | |
| R2 OOS vs zero / vs drift | -0.0011 / +0.0012 | -0.0011 / +0.0012 | | | -0.0334 |
| Brier P(up) (clim / 0.5) | 0.2502 (0.2501 / 0.2500) | 0.2502 (0.2501) | | | 0.2590 |
| 80% interval coverage | 0.791 | 0.791 | | | |
| Ternary log-loss (clim) | 1.0821 (1.0827) | 1.0821 (1.0827) | | | |
| Mean net return / trade after 10bps | | -0bps (t=-0.04) | -8bps (t=-0.72) | buy-and-hold +13bps | -12bps |
| Annualised Sharpe (net) | | -0.02 | -0.38 | buy-and-hold 0.57 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 537 | 537.0 | -0.002 | 0.505 | 0.497 | 0.507 |
| bear | 375 | 375.0 | +0.020 | 0.485 | 0.499 | 0.488 |
| chop | 388 | 388.0 | +0.042 | 0.531 | 0.515 | 0.492 |

IC by calendar year: 2023: +0.05, 2024: -0.03, 2025: -0.01, 2026: +0.04

Calibration of P(up) (OOS quintiles): p=0.49 -> 0.49 (n=260); p=0.49 -> 0.53 (n=275); p=0.49 -> 0.50 (n=249); p=0.50 -> 0.49 (n=257); p=0.50 -> 0.47 (n=259)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_3d` | +0.0007 | +0.0022 | 100% | yes | -1.67 | 3d log return, vol-scaled |
| `tech_trend` | +0.0007 | +0.0020 | 100% | yes | +0.25 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `int_cexflow_x_negmom` | -0.0006 | -0.0000 | 60% | no | +0.35 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `ret_7d` | +0.0006 | +0.0018 | 100% | yes | -0.87 | 7d log return, vol-scaled |
| `prem_lvl_3d` | +0.0005 | +0.0020 | 100% | yes | -0.20 | Perp premium (funding proxy) 3d mean, trailing z |
| `int_volspike_x_range` | -0.0005 | -0.0017 | 100% | yes | +0.31 | Vol spike x large daily range (liquidation-cluster proxy) |
| `stab_growth_7d` | +0.0004 | +0.0016 | 100% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `tech_bbwidth` | +0.0004 | +0.0022 | 100% | yes | -1.33 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_1d` | -1.795 | 1d log return, vol-scaled |
| `ret_3d` | -1.666 | 3d log return, vol-scaled |
| `ret_7d` | -0.871 | 7d log return, vol-scaled |
| `vol_7v90` | +0.244 | 7d vs 90d realised vol, log, trailing z (vol spike) |
| `range_1d` | +1.289 | 1d high/low range, log, trailing z (liquidation-cluster proxy) |
| `tech_trend` | +0.255 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.335 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_3d` | -0.203 | Perp premium (funding proxy) 3d mean, trailing z |
| `prem_chg_3d` | -0.157 | Change in perp premium over 3d, trailing z |
| `perp_spot_vol` | -0.439 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_1d` | -0.123 | Cumulative 1d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_1d` | -0.622 | USD-stablecoin supply growth over 1d (mint minus burn), trailing z |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `btc_ret_1d` | -1.012 | BTC 1d return, vol-scaled |
| `rel_btc_1d` | -1.542 | SOL/BTC 1d relative return, trailing z |
| `btcdom_chg_1d` | +0.497 | BTC dominance 1d change, trailing z |
| `gold_ret_1d` | +0.402 | PAXG (gold) 1d return, trailing z (only long-history macro proxy) |
| `int_cexflow_x_negmom` | +0.348 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `int_volspike_x_range` | +0.314 | Vol spike x large daily range (liquidation-cluster proxy) |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_1d`, `int_prem_x_oi`, `etf_flow_1d`; same rows, window 2024-05-01 -> 2026-10-07, test 2025-05-03 -> 2026-10-07, 523 non-overlapping windows): IC +0.020 -> +0.012 (overlapping +0.020 -> +0.012), hit 0.499 -> 0.497, Brier 0.2502 -> 0.2502 (clim 0.2503). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `hl_liq_3d`, `spy_ret_1d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.03 to +0.05; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2021-03-16 (binding source: perp_spot_vol).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 7d

**Current forecast** (as of 2026-10-08, model trained on 1899 labelled days through 2026-10-01, shrinkage kappa=100.0):

- P(up) = **48.4%**; ternary P(down / flat / up) = 29.5% / 39.1% / 31.4% (flat band = +-0.5 x 30d trailing vol scaled to 7d = +-4.59%)
- Expected log return +0.22% (simple +0.22%) = shrunk drift +0.21% (sample drift +0.37% x prior weight 0.57) + feature signal +0.02%; GBM comparator +4.52%
- 80% prediction interval: -10.4% to +13.6% -> $98.09 to $124.4 (horizon vol 9.0%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC -0.015 (t=-0.19) does not clear momentum (+0.010) with t>2; net Sharpe 0.00 vs momentum 0.64 / buy-and-hold 0.69 (t=0.01)**

**Data actually used** (model sample 2021-07-21 -> 2026-10-08, 19 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `b204e48e-9812-43e6-a00b-5fcfd40e04a2` | 2020-10-03 -> 2026-10-08 | 0.0% | 0d | 1d |
| perp | `perpetual-markets` / `binance_SOLUSDT` | 2020-09-14 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-10 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6079-140-68` | 2020-08-11 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (13 purged folds, retrain every 91d, embargo 7d, test 2023-08-04 -> 2026-10-01):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1155 | 165 | 165 | 165 | 165 |
| IC, signal only (Spearman, vol-scaled) | -0.017 | -0.015 (t=-0.19) | +0.010 | n/a (constant) | -0.027 |
| IC, full forecast incl. per-fold drift | -0.134 | -0.133 | | | |
| Hit rate | 0.492 | 0.492 | 0.481 | drift 0.481 / long 0.511 | 0.492 |
| Drift-only R2 vs zero | | -0.0159 | | | |
| R2 OOS vs zero / vs drift | -0.0033 / +0.0123 | -0.0033 / +0.0124 | | | -0.1139 |
| Brier P(up) (clim / 0.5) | 0.2515 (0.2518 / 0.2500) | 0.2515 (0.2518) | | | 0.2768 |
| 80% interval coverage | 0.809 | 0.809 | | | |
| Ternary log-loss (clim) | 1.0908 (1.0917) | 1.0908 (1.0917) | | | |
| Mean net return / trade after 15bps | | -13bps (t=0.01) | +1bps (t=1.13) | buy-and-hold +98bps | +7bps |
| Annualised Sharpe (net) | | 0.00 | 0.64 | buy-and-hold 0.69 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 500 | 71.4 | -0.018 | 0.470 | 0.482 | 0.516 |
| bear | 349 | 49.9 | +0.031 | 0.493 | 0.458 | 0.513 |
| chop | 306 | 43.7 | -0.009 | 0.526 | 0.507 | 0.500 |

IC by calendar year: 2023: +0.15, 2024: -0.07, 2025: -0.14, 2026: -0.13

Calibration of P(up) (OOS quintiles): p=0.47 -> 0.62 (n=246); p=0.48 -> 0.54 (n=242); p=0.48 -> 0.52 (n=210); p=0.49 -> 0.43 (n=228); p=0.50 -> 0.42 (n=229)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `tech_trend` | +0.0015 | +0.0027 | 100% | yes | +0.25 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `ret_28d` | +0.0012 | +0.0028 | 100% | yes | +0.58 | 28d log return, vol-scaled |
| `ret_14d` | +0.0010 | +0.0015 | 100% | yes | -0.52 | 14d log return, vol-scaled |
| `prem_lvl_7d` | +0.0010 | +0.0014 | 100% | yes | +0.01 | Perp premium (funding proxy) 7d mean, trailing z |
| `addr_growth_7_90` | +0.0010 | +0.0015 | 100% | yes | +1.27 | Active addresses 7d vs 90d average, log, trailing z |
| `btc_ret_7d` | +0.0009 | +0.0019 | 100% | yes | -0.71 | BTC 7d return, vol-scaled |
| `fee_growth_7_90` | +0.0009 | +0.0016 | 100% | yes | +1.03 | Network fee revenue 7d vs 90d average, log, trailing z |
| `stab_growth_7d` | +0.0008 | +0.0007 | 85% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -0.871 | 7d log return, vol-scaled |
| `ret_14d` | -0.521 | 14d log return, vol-scaled |
| `ret_28d` | +0.584 | 28d log return, vol-scaled |
| `vol_14v90` | -0.265 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | -0.112 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | +0.255 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.335 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | +0.006 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | +0.587 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | +0.143 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | -0.439 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `btc_ret_7d` | -0.708 | BTC 7d return, vol-scaled |
| `rel_btc_7d` | -0.563 | SOL/BTC 7d relative return, trailing z |
| `btcdom_chg_7d` | +0.178 | BTC dominance 7d change, trailing z |
| `gold_ret_7d` | -0.462 | PAXG (gold) 7d return, trailing z (only long-history macro proxy) |
| `addr_growth_7_90` | +1.265 | Active addresses 7d vs 90d average, log, trailing z |
| `fee_growth_7_90` | +1.026 | Network fee revenue 7d vs 90d average, log, trailing z |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_7d`, `etf_flow_7d`; same rows, window 2024-05-04 -> 2026-10-08, test 2025-05-18 -> 2026-10-01, 71 non-overlapping windows): IC -0.113 -> -0.091 (overlapping -0.104 -> -0.084), hit 0.478 -> 0.480, Brier 0.2506 -> 0.2507 (clim 0.2510). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `hl_liq_3d`, `spy_ret_7d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.14 to +0.15; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2021-07-21 (binding source: addr_growth_7_90).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 14d

**Current forecast** (as of 2026-10-08, model trained on 1892 labelled days through 2026-09-24, shrinkage kappa=100.0):

- P(up) = **48.6%**; ternary P(down / flat / up) = 29.1% / 39.4% / 31.4% (flat band = +-0.5 x 30d trailing vol scaled to 14d = +-6.55%)
- Expected log return +0.38% (simple +0.38%) = shrunk drift +0.37% (sample drift +0.66% x prior weight 0.56) + feature signal +0.01%; GBM comparator +0.68%
- 80% prediction interval: -14.8% to +20.5% -> $93.30 to $132.0 (horizon vol 12.7%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC -0.026 (t=-0.23) does not clear momentum (+0.127) with t>2; net Sharpe -0.38 vs momentum 1.17 / buy-and-hold 0.69 (t=-0.67)**

**Data actually used** (model sample 2021-07-21 -> 2026-10-08, 19 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `b204e48e-9812-43e6-a00b-5fcfd40e04a2` | 2020-10-03 -> 2026-10-08 | 0.0% | 0d | 1d |
| perp | `perpetual-markets` / `binance_SOLUSDT` | 2020-09-14 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-10 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6079-140-68` | 2020-08-11 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (13 purged folds, retrain every 91d, embargo 14d, test 2023-08-18 -> 2026-09-24):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1134 | 81 | 81 | 81 | 81 |
| IC, signal only (Spearman, vol-scaled) | -0.028 | -0.026 (t=-0.23) | +0.127 | n/a (constant) | +0.083 |
| IC, full forecast incl. per-fold drift | -0.306 | -0.308 | | | |
| Hit rate | 0.463 | 0.463 | 0.549 | drift 0.461 / long 0.537 | 0.549 |
| Drift-only R2 vs zero | | -0.0363 | | | |
| R2 OOS vs zero / vs drift | -0.0172 / +0.0181 | -0.0172 / +0.0184 | | | -0.1523 |
| Brier P(up) (clim / 0.5) | 0.2545 (0.2553 / 0.2500) | 0.2545 (0.2553) | | | 0.2845 |
| 80% interval coverage | 0.825 | 0.825 | | | |
| Ternary log-loss (clim) | 1.0930 (1.0957) | 1.0930 (1.0957) | | | |
| Mean net return / trade after 20bps | | -152bps (t=-0.67) | +307bps (t=2.06) | buy-and-hold +214bps | +250bps |
| Annualised Sharpe (net) | | -0.38 | 1.17 | buy-and-hold 0.69 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 488 | 34.9 | -0.179 | 0.443 | 0.559 | 0.564 |
| bear | 349 | 24.9 | +0.002 | 0.499 | 0.501 | 0.559 |
| chop | 297 | 21.2 | -0.006 | 0.455 | 0.589 | 0.468 |

IC by calendar year: 2023: +0.09, 2024: -0.28, 2025: -0.09, 2026: -0.09

Calibration of P(up) (OOS quintiles): p=0.47 -> 0.69 (n=228); p=0.48 -> 0.60 (n=230); p=0.48 -> 0.50 (n=224); p=0.48 -> 0.52 (n=226); p=0.49 -> 0.37 (n=226)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `tech_trend` | +0.0020 | +0.0036 | 100% | yes | +0.25 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `btc_ret_14d` | +0.0016 | +0.0037 | 100% | yes | -0.43 | BTC 14d return, vol-scaled |
| `ret_14d` | +0.0015 | +0.0022 | 100% | yes | -0.52 | 14d log return, vol-scaled |
| `rel_btc_14d` | +0.0013 | +0.0016 | 100% | yes | -0.29 | SOL/BTC 14d relative return, trailing z |
| `ret_28d` | +0.0013 | +0.0030 | 100% | yes | +0.58 | 28d log return, vol-scaled |
| `prem_lvl_7d` | +0.0012 | +0.0020 | 85% | yes | +0.01 | Perp premium (funding proxy) 7d mean, trailing z |
| `addr_growth_7_90` | +0.0011 | +0.0011 | 85% | yes | +1.27 | Active addresses 7d vs 90d average, log, trailing z |
| `tech_bbwidth` | +0.0011 | +0.0053 | 100% | yes | -1.33 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -0.871 | 7d log return, vol-scaled |
| `ret_14d` | -0.521 | 14d log return, vol-scaled |
| `ret_28d` | +0.584 | 28d log return, vol-scaled |
| `vol_14v90` | -0.265 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | -0.112 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | +0.255 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.335 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | +0.006 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | +0.587 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | +0.143 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | -0.439 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_14d` | -0.508 | Cumulative 14d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_14d` | -0.461 | USD-stablecoin supply growth over 14d (mint minus burn), trailing z |
| `btc_ret_14d` | -0.426 | BTC 14d return, vol-scaled |
| `rel_btc_14d` | -0.288 | SOL/BTC 14d relative return, trailing z |
| `btcdom_chg_14d` | +0.185 | BTC dominance 14d change, trailing z |
| `gold_ret_14d` | -0.826 | PAXG (gold) 14d return, trailing z (only long-history macro proxy) |
| `addr_growth_7_90` | +1.265 | Active addresses 7d vs 90d average, log, trailing z |
| `fee_growth_7_90` | +1.026 | Network fee revenue 7d vs 90d average, log, trailing z |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_14d`, `etf_flow_14d`; same rows, window 2024-05-08 -> 2026-10-08, test 2025-06-05 -> 2026-09-24, 34 non-overlapping windows): IC -0.158 -> -0.139 (overlapping -0.139 -> -0.128), hit 0.438 -> 0.438, Brier 0.2497 -> 0.2497 (clim 0.2518). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_14d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.28 to +0.09; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2021-07-21 (binding source: addr_growth_7_90).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 28d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 1741 labelled days through 2026-09-10, shrinkage kappa=100.0):

- P(up) = **48.8%**; ternary P(down / flat / up) = 29.8% / 39.0% / 31.2% (flat band = +-0.5 x 30d trailing vol scaled to 28d = +-9.39%)
- Expected log return +0.27% (simple +0.27%) = shrunk drift -0.01% (sample drift -0.02% x prior weight 0.54) + feature signal +0.28%; GBM comparator +5.87%
- 80% prediction interval: -23.1% to +33.7% -> $84.19 to $146.4 (horizon vol 17.9%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.156 (t=0.69) does not clear momentum (-0.072) with t>2; net Sharpe -1.45 vs momentum -0.88 / buy-and-hold -0.57 (t=-1.89)**
- Non-overlapping test windows: **21** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2021-12-05 -> 2026-10-08, 18 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `b204e48e-9812-43e6-a00b-5fcfd40e04a2` | 2020-10-03 -> 2026-10-08 | 0.0% | 0d | 1d |
| price | `asset-price` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-10 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |
| supply | `asset-supply` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-11 -> 2026-10-08 | 0.0% | 0d | 1d |
| unlocks | `token-unlocks` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-11 -> 2027-12-31 | 0.0% | 0d | schedule (forward-known) |

**Walk-forward out-of-sample results** (4 purged folds, retrain every 182d, embargo 28d, test 2025-01-29 -> 2026-09-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 590 | 21 | 21 | 21 | 21 |
| IC, signal only (Spearman, vol-scaled) | +0.145 | +0.156 (t=0.69) | -0.072 | n/a (constant) | +0.234 |
| IC, full forecast incl. per-fold drift | -0.051 | -0.047 | | | |
| Hit rate | 0.432 | 0.432 | 0.447 | drift 0.419 / long 0.493 | 0.637 |
| Drift-only R2 vs zero | | -0.0194 | | | |
| R2 OOS vs zero / vs drift | +0.0023 / +0.0213 | +0.0020 / +0.0210 | | | -0.6486 |
| Brier P(up) (clim / 0.5) | 0.2478 (0.2510 / 0.2500) | 0.2478 (0.2509) | | | 0.2772 |
| 80% interval coverage | 0.824 | 0.824 | | | |
| Ternary log-loss (clim) | 1.0873 (1.0936) | 1.0873 (1.0936) | | | |
| Mean net return / trade after 30bps | | -550bps (t=-1.89) | -279bps (t=-1.14) | buy-and-hold -247bps | +368bps |
| Annualised Sharpe (net) | | -1.45 | -0.88 | buy-and-hold -0.57 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 123 | 4.4 | +0.363 | 0.382 | 0.553 | 0.528 |
| bear | 297 | 10.6 | +0.028 | 0.508 | 0.397 | 0.545 |
| chop | 170 | 6.1 | +0.277 | 0.335 | 0.459 | 0.376 |

IC by calendar year: 2025: +0.41, 2026: -0.45

Calibration of P(up) (OOS quintiles): p=0.46 -> 0.36 (n=121); p=0.47 -> 0.40 (n=120); p=0.47 -> 0.30 (n=115); p=0.47 -> 0.70 (n=119); p=0.49 -> 0.72 (n=115)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `unlock_next_28d` | -0.0030 | -0.0037 | 100% | yes | +0.00 | Scheduled unlocks over the next 28d as % of circulating supply (current schedule vintage) |
| `price_to_activity` | -0.0028 | -0.0016 | 100% | yes | -1.08 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `fee_growth_28_365` | +0.0025 | +0.0025 | 100% | yes | +1.17 | Network fee revenue 28d vs 365d average, log, trailing z |
| `drawdown_365` | +0.0020 | +0.0029 | 100% | yes | +0.08 | Drawdown from trailing 1y high, 4y trailing z |
| `stab_growth_28d` | +0.0020 | +0.0024 | 100% | yes | -0.50 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_growth_90d` | +0.0019 | +0.0024 | 100% | yes | -0.37 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `addr_growth_28_365` | +0.0019 | +0.0016 | 100% | yes | +0.27 | Active addresses 28d vs 365d average, log, trailing z |
| `trend_200` | +0.0019 | +0.0022 | 100% | yes | +0.54 | Distance to 200d MA, 4y trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.584 | 28d log return, vol-scaled |
| `ret_90d` | +1.051 | 90d log return, vol-scaled |
| `drawdown_365` | +0.077 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.540 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | -0.112 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `btc_ret_28d` | +0.605 | BTC 28d return, vol-scaled |
| `rel_btc_28d` | +0.502 | SOL/BTC 28d relative return, trailing z |
| `btcdom_chg_28d` | +0.123 | BTC dominance 28d change, trailing z |
| `addr_growth_28_365` | +0.269 | Active addresses 28d vs 365d average, log, trailing z |
| `fee_growth_28_365` | +1.168 | Network fee revenue 28d vs 365d average, log, trailing z |
| `price_to_activity` | -1.077 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `beta_btc_180d` | -0.763 | Rolling 180d beta to BTC, 4y trailing z |
| `unlock_next_28d` | +0.002 | Scheduled unlocks over the next 28d as % of circulating supply (current schedule vintage) |
| `supply_infl_90d` | -1.516 | Trailing 90d circulating supply inflation, trailing z |

</details>

**Short-history ablation** (core vs core + `gold_ret_28d`, `hl_oi_chg_28d`, `etf_flow_28d`; same rows, window 2024-05-16 -> 2026-10-08, test 2025-07-11 -> 2026-09-10, 15 non-overlapping windows): IC -0.097 -> -0.143 (overlapping -0.101 -> -0.123), hit 0.426 -> 0.447, Brier 0.2500 -> 0.2510 (clim 0.2564). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_28d`.

**Failure modes for this horizon:**

- Sparse non-overlapping samples: 21 independent test windows; one regime shift can flip the sign (see IC by year).
- Thin history: model sample starts 2021-12-05 (binding source: addr_growth_28_365).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Unlock schedules are the current vintage; past revisions are not recoverable (look-ahead risk).
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 90d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 1679 labelled days through 2026-07-10, shrinkage kappa=100.0):

- P(up) = **48.4%**; ternary P(down / flat / up) = 32.7% / 37.9% / 29.4% (flat band = +-0.5 x 90d trailing vol scaled to 90d = +-14.94%)
- Expected log return +0.78% (simple +0.78%) = shrunk drift +0.01% (sample drift +0.01% x prior weight 0.53) + feature signal +0.77%; GBM comparator +40.15%
- 80% prediction interval: -33.0% to +45.1% -> $73.33 to $158.9 (horizon vol 27.8%)
- Note: P(up) and the expected return point in different directions because the empirical residual distribution is skewed (mean and median differ); both come from the same distribution.
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.152 (t=0.27) does not clear momentum (-0.084) with t>2; net Sharpe -0.53 vs momentum 0.67 / buy-and-hold -0.50 (t=-0.59)**
- Non-overlapping test windows: **5** - fewer than ~8, so this horizon is **indicative only**.

**Data actually used** (model sample 2021-12-05 -> 2026-10-08, 18 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `b204e48e-9812-43e6-a00b-5fcfd40e04a2` | 2020-10-03 -> 2026-10-08 | 0.0% | 0d | 1d |
| price | `asset-price` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-10 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |
| supply | `asset-supply` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-11 -> 2026-10-08 | 0.0% | 0d | 1d |
| unlocks | `token-unlocks` / `b3d5d66c-26a2-404c-9325-91dc714a722b` | 2020-04-11 -> 2027-12-31 | 0.0% | 0d | schedule (forward-known) |

**Walk-forward out-of-sample results** (3 purged folds, retrain every 182d, embargo 90d, test 2025-06-02 -> 2026-07-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 404 | 5 | 5 | 5 | 5 |
| IC, signal only (Spearman, vol-scaled) | +0.266 | +0.152 (t=0.27) | -0.084 | n/a (constant) | -0.236 |
| IC, full forecast incl. per-fold drift | -0.012 | -0.195 | | | |
| Hit rate | 0.329 | 0.418 | 0.595 | drift 0.418 / long 0.418 | 0.341 |
| Drift-only R2 vs zero | | -0.0882 | | | |
| R2 OOS vs zero / vs drift | -0.0464 / +0.0873 | -0.0263 / +0.0567 | | | -0.4003 |
| Brier P(up) (clim / 0.5) | 0.2450 (0.2574 / 0.2500) | 0.2482 (0.2573) | | | 0.4522 |
| 80% interval coverage | 0.869 | 0.782 | | | |
| Ternary log-loss (clim) | 1.1209 (1.1553) | 1.1742 (1.1899) | | | |
| Mean net return / trade after 50bps | | -696bps (t=-0.59) | +89bps (t=0.74) | buy-and-hold -646bps | -1322bps |
| Annualised Sharpe (net) | | -0.53 | 0.67 | buy-and-hold -0.50 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 95 | 1.1 | +0.845 | 0.389 | 0.389 | 0.389 |
| bear | 198 | 2.2 | -0.439 | 0.232 | 0.768 | 0.232 |
| chop | 111 | 1.2 | +0.325 | 0.450 | 0.414 | 0.450 |

IC by calendar year: 2025: +0.66, 2026: -0.46

Calibration of P(up) (OOS quintiles): p=0.47 -> 0.00 (n=82); p=0.49 -> 0.72 (n=83); p=0.49 -> 0.36 (n=124); p=0.50 -> 0.37 (n=51); p=0.50 -> 0.14 (n=64)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `unlock_next_90d` | -0.0054 | -0.0070 | 100% | yes | +0.01 | Scheduled unlocks over the next 90d as % of circulating supply (current schedule vintage) |
| `btcdom_chg_90d` | +0.0052 | +0.0054 | 100% | yes | +0.18 | BTC dominance 90d change, trailing z |
| `rel_btc_90d` | +0.0042 | +0.0048 | 100% | yes | +0.99 | SOL/BTC 90d relative return, trailing z |
| `fee_growth_28_365` | +0.0036 | +0.0040 | 100% | yes | +1.17 | Network fee revenue 28d vs 365d average, log, trailing z |
| `drawdown_365` | +0.0034 | +0.0049 | 100% | yes | +0.08 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.0032 | +0.0044 | 100% | yes | +0.54 | Distance to 200d MA, 4y trailing z |
| `beta_btc_180d` | -0.0027 | -0.0030 | 100% | yes | -0.76 | Rolling 180d beta to BTC, 4y trailing z |
| `price_to_activity` | -0.0026 | -0.0018 | 100% | yes | -1.08 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.584 | 28d log return, vol-scaled |
| `ret_90d` | +1.051 | 90d log return, vol-scaled |
| `drawdown_365` | +0.077 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.540 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | -0.112 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `btc_ret_90d` | +1.254 | BTC 90d return, vol-scaled |
| `rel_btc_90d` | +0.986 | SOL/BTC 90d relative return, trailing z |
| `btcdom_chg_90d` | +0.181 | BTC dominance 90d change, trailing z |
| `addr_growth_28_365` | +0.269 | Active addresses 28d vs 365d average, log, trailing z |
| `fee_growth_28_365` | +1.168 | Network fee revenue 28d vs 365d average, log, trailing z |
| `price_to_activity` | -1.077 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `beta_btc_180d` | -0.763 | Rolling 180d beta to BTC, 4y trailing z |
| `unlock_next_90d` | +0.007 | Scheduled unlocks over the next 90d as % of circulating supply (current schedule vintage) |
| `supply_infl_90d` | -1.516 | Trailing 90d circulating supply inflation, trailing z |

</details>

**Short-history ablation**: window 2024-06-23 -> 2026-10-08 leaves fewer than 4 non-overlapping 90d test windows; no meaningful comparison possible
Too short to evaluate (< 700 days): `spy_ret_90d`.

**Failure modes for this horizon:**

- Sparse non-overlapping samples: 5 independent test windows; one regime shift can flip the sign (see IC by year).
- Thin history: model sample starts 2021-12-05 (binding source: addr_growth_28_365).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Unlock schedules are the current vintage; past revisions are not recoverable (look-ahead risk).
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
