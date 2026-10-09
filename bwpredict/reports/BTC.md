# BTC-USD probabilistic forward-return forecast

Decision time: **2026-10-09 00:00 UTC** (last completed daily bar 2026-10-08, VWAP close $81,689). Snapshot of Blockworks MCP data; nothing after the last completed bar is used.

Primary model: ridge regression on vol-scaled forward returns (shrinkage picked by an inner, purged time split), with the predictive distribution taken from empirical residuals. Comparators: zero return, prevailing mean (drift), trailing same-horizon momentum, and a depth-2 gradient-boosted model. Walk-forward is purged with an embargo of one full horizon. Costs (round trip per non-overlapping trade): 1d 10bps, 7d 15bps, 14d 20bps, 28d 30bps, 90d 50bps.

## Summary

| horizon | P(up) | P(down/flat/up) | expected return | 80% interval | price range | non-overlap IC (N) | status |
|---|---|---|---|---|---|---|---|
| 1d | 50.0% | 25/49/26% | +0.05% | -2.2% / +2.4% | $79,856 - $83,683 | +0.019 (1647) | no edge claimed |
| 7d | 51.6% | 25/46/29% | +0.37% | -6.2% / +7.8% | $76,630 - $88,024 | +0.047 (231) | no edge claimed |
| 14d | 51.3% | 25/45/30% | +0.74% | -9.0% / +13.0% | $74,306 - $92,268 | +0.045 (114) | no edge claimed |
| 28d | 52.7% | 26/41/33% | +1.26% | -15.2% / +21.3% | $69,249 - $99,077 | +0.191 (45) | regime tilt |
| 90d | 53.2% | 32/33/35% | +4.44% | -22.3% / +42.5% | $63,489 - $116,425 | +0.160 (12) | regime tilt |

## Horizon 1d

**Current forecast** (as of 2026-10-08, model trained on 2379 labelled days through 2026-10-07, shrinkage kappa=100.0):

- P(up) = **50.0%**; ternary P(down / flat / up) = 24.6% / 49.4% / 25.9% (flat band = +-0.5 x 30d trailing vol scaled to 1d = +-1.02%)
- Expected log return +0.05% (simple +0.05%) = shrunk drift +0.05% (sample drift +0.08% x prior weight 0.62) + feature signal -0.00%; GBM comparator -0.04%
- 80% prediction interval: -2.2% to +2.4% -> $79,856 to $83,683 (horizon vol 2.0%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.019 (t=0.77) does not clear momentum (-0.028) with t>2; net Sharpe -0.47 vs momentum -0.65 / buy-and-hold 0.26 (t=-1.00)**

**Data actually used** (model sample 2020-04-03 -> 2026-10-08, 19 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| eth | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| perp | `perpetual-markets` / `binance_BTCUSDT` | 2019-09-08 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6057-140-68` | 2017-08-17 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (19 purged folds, retrain every 91d, embargo 1d, test 2022-04-05 -> 2026-10-07):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1647 | 1647 | 1647 | 1647 | 1647 |
| IC, signal only (Spearman, vol-scaled) | +0.019 | +0.019 (t=0.77) | -0.028 | n/a (constant) | +0.011 |
| IC, full forecast incl. per-fold drift | -0.028 | -0.028 | | | |
| Hit rate | 0.495 | 0.495 | 0.478 | drift 0.495 / long 0.495 | 0.491 |
| Drift-only R2 vs zero | | -0.0051 | | | |
| R2 OOS vs zero / vs drift | -0.0008 / +0.0042 | -0.0008 / +0.0042 | | | -0.0213 |
| Brier P(up) (clim / 0.5) | 0.2503 (0.2511 / 0.2500) | 0.2503 (0.2511) | | | 0.2569 |
| 80% interval coverage | 0.808 | 0.808 | | | |
| Ternary log-loss (clim) | 1.0421 (1.0430) | 1.0421 (1.0430) | | | |
| Mean net return / trade after 10bps | | -6bps (t=-1.00) | -9bps (t=-1.39) | buy-and-hold +4bps | -8bps |
| Annualised Sharpe (net) | | -0.47 | -0.65 | buy-and-hold 0.26 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 546 | 546.0 | -0.014 | 0.495 | 0.465 | 0.495 |
| bear | 425 | 425.0 | -0.031 | 0.506 | 0.492 | 0.506 |
| chop | 676 | 676.0 | +0.071 | 0.488 | 0.479 | 0.488 |

IC by calendar year: 2022: -0.10, 2023: -0.03, 2024: +0.01, 2025: +0.02, 2026: +0.03

Calibration of P(up) (OOS quintiles): p=0.50 -> 0.53 (n=353); p=0.50 -> 0.49 (n=321); p=0.50 -> 0.50 (n=314); p=0.50 -> 0.50 (n=375); p=0.51 -> 0.45 (n=284)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_3d` | +0.0006 | +0.0005 | 100% | yes | -1.39 | 3d log return, vol-scaled |
| `tech_trend` | +0.0006 | +0.0007 | 100% | yes | +0.10 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `ret_7d` | +0.0006 | +0.0006 | 100% | yes | -0.71 | 7d log return, vol-scaled |
| `perp_spot_vol` | +0.0006 | +0.0006 | 100% | yes | +0.99 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `int_cexflow_x_negmom` | -0.0005 | -0.0003 | 100% | yes | +0.28 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `stab_growth_7d` | +0.0004 | +0.0003 | 95% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `ethbtc_1d` | -0.0004 | -0.0006 | 100% | yes | -0.96 | ETH/BTC 1d change (alt risk appetite), trailing z |
| `ret_1d` | +0.0004 | +0.0003 | 100% | yes | -1.01 | 1d log return, vol-scaled |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_1d` | -1.012 | 1d log return, vol-scaled |
| `ret_3d` | -1.387 | 3d log return, vol-scaled |
| `ret_7d` | -0.708 | 7d log return, vol-scaled |
| `vol_7v90` | -0.242 | 7d vs 90d realised vol, log, trailing z (vol spike) |
| `range_1d` | +0.514 | 1d high/low range, log, trailing z (liquidation-cluster proxy) |
| `tech_trend` | +0.098 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.155 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_3d` | -0.611 | Perp premium (funding proxy) 3d mean, trailing z |
| `prem_chg_3d` | +0.336 | Change in perp premium over 3d, trailing z |
| `perp_spot_vol` | +0.989 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_1d` | -0.123 | Cumulative 1d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_1d` | -0.622 | USD-stablecoin supply growth over 1d (mint minus burn), trailing z |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `ethbtc_1d` | -0.959 | ETH/BTC 1d change (alt risk appetite), trailing z |
| `btcdom_chg_1d` | +0.497 | BTC dominance 1d change, trailing z |
| `gold_ret_1d` | +0.402 | PAXG (gold) 1d return, trailing z (only long-history macro proxy) |
| `int_cexflow_x_negmom` | +0.283 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `int_volspike_x_range` | +0.000 | Vol spike x large daily range (liquidation-cluster proxy) |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_1d`, `int_prem_x_oi`, `etf_flow_1d`; same rows, window 2024-05-01 -> 2026-10-07, test 2025-05-03 -> 2026-10-07, 523 non-overlapping windows): IC +0.038 -> +0.057 (overlapping +0.038 -> +0.057), hit 0.491 -> 0.501, Brier 0.2500 -> 0.2499 (clim 0.2501). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `hl_liq_3d`, `spy_ret_1d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.10 to +0.03; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-04-03 (binding source: gold_ret_1d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 7d

**Current forecast** (as of 2026-10-08, model trained on 2367 labelled days through 2026-10-01, shrinkage kappa=100.0):

- P(up) = **51.6%**; ternary P(down / flat / up) = 25.3% / 46.0% / 28.7% (flat band = +-0.5 x 30d trailing vol scaled to 7d = +-2.72%)
- Expected log return +0.37% (simple +0.37%) = shrunk drift +0.37% (sample drift +0.60% x prior weight 0.62) + feature signal -0.00%; GBM comparator +0.96%
- 80% prediction interval: -6.2% to +7.8% -> $76,630 to $88,024 (horizon vol 5.4%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.047 (t=0.72) does not clear momentum (-0.017) with t>2; net Sharpe 0.09 vs momentum -0.24 / buy-and-hold 0.34 (t=0.20)**

**Data actually used** (model sample 2020-04-09 -> 2026-10-08, 18 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| eth | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `54e8d8d0-979a-4297-887f-2ea8a989753c` | 2014-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| perp | `perpetual-markets` / `binance_BTCUSDT` | 2019-09-08 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6057-140-68` | 2017-08-17 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (18 purged folds, retrain every 91d, embargo 7d, test 2022-04-23 -> 2026-10-01):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1623 | 231 | 231 | 231 | 231 |
| IC, signal only (Spearman, vol-scaled) | +0.045 | +0.047 (t=0.72) | -0.017 | n/a (constant) | +0.012 |
| IC, full forecast incl. per-fold drift | -0.064 | -0.062 | | | |
| Hit rate | 0.521 | 0.521 | 0.477 | drift 0.523 / long 0.523 | 0.513 |
| Drift-only R2 vs zero | | -0.0230 | | | |
| R2 OOS vs zero / vs drift | -0.0022 / +0.0203 | -0.0022 / +0.0203 | | | -0.0692 |
| Brier P(up) (clim / 0.5) | 0.2502 (0.2508 / 0.2500) | 0.2502 (0.2508) | | | 0.2755 |
| 80% interval coverage | 0.808 | 0.808 | | | |
| Ternary log-loss (clim) | 1.0633 (1.0671) | 1.0633 (1.0671) | | | |
| Mean net return / trade after 15bps | | +14bps (t=0.20) | -19bps (t=-0.50) | buy-and-hold +33bps | +44bps |
| Annualised Sharpe (net) | | 0.09 | -0.24 | buy-and-hold 0.34 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 540 | 77.1 | -0.037 | 0.559 | 0.439 | 0.559 |
| bear | 425 | 60.7 | +0.040 | 0.529 | 0.459 | 0.532 |
| chop | 658 | 94.0 | +0.085 | 0.483 | 0.520 | 0.488 |

IC by calendar year: 2022: -0.08, 2023: -0.09, 2024: -0.04, 2025: -0.03, 2026: +0.09

Calibration of P(up) (OOS quintiles): p=0.50 -> 0.54 (n=335); p=0.50 -> 0.60 (n=334); p=0.51 -> 0.51 (n=309); p=0.51 -> 0.46 (n=321); p=0.52 -> 0.51 (n=324)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_28d` | +0.0014 | +0.0024 | 100% | yes | +0.60 | 28d log return, vol-scaled |
| `tech_trend` | +0.0013 | +0.0022 | 100% | yes | +0.10 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `perp_spot_vol` | +0.0013 | +0.0016 | 100% | yes | +0.99 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `ret_14d` | +0.0011 | +0.0018 | 100% | yes | -0.43 | 14d log return, vol-scaled |
| `prem_chg_7d` | +0.0010 | +0.0016 | 100% | yes | -0.43 | Change in perp premium over 7d, trailing z |
| `stab_growth_7d` | +0.0009 | +0.0011 | 100% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `prem_lvl_7d` | +0.0008 | +0.0007 | 94% | yes | -0.77 | Perp premium (funding proxy) 7d mean, trailing z |
| `btcdom_chg_7d` | +0.0008 | +0.0016 | 100% | yes | +0.18 | BTC dominance 7d change, trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -0.708 | 7d log return, vol-scaled |
| `ret_14d` | -0.426 | 14d log return, vol-scaled |
| `ret_28d` | +0.605 | 28d log return, vol-scaled |
| `vol_14v90` | -1.264 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | -0.051 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | +0.098 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.155 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | -0.765 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | -0.427 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | +0.000 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.989 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `ethbtc_7d` | -0.891 | ETH/BTC 7d change (alt risk appetite), trailing z |
| `btcdom_chg_7d` | +0.178 | BTC dominance 7d change, trailing z |
| `gold_ret_7d` | -0.462 | PAXG (gold) 7d return, trailing z (only long-history macro proxy) |
| `addr_growth_7_90` | +0.634 | Active addresses 7d vs 90d average, log, trailing z |
| `fee_growth_7_90` | +1.069 | Network fee revenue 7d vs 90d average, log, trailing z |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_7d`, `etf_flow_7d`; same rows, window 2024-05-04 -> 2026-10-08, test 2025-05-18 -> 2026-10-01, 71 non-overlapping windows): IC -0.013 -> -0.041 (overlapping -0.016 -> -0.043), hit 0.418 -> 0.414, Brier 0.2499 -> 0.2498 (clim 0.2549). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `hl_liq_3d`, `spy_ret_7d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.09 to +0.09; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-04-09 (binding source: gold_ret_7d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 14d

**Current forecast** (as of 2026-10-08, model trained on 2354 labelled days through 2026-09-24, shrinkage kappa=100.0):

- P(up) = **51.3%**; ternary P(down / flat / up) = 25.2% / 45.2% / 29.6% (flat band = +-0.5 x 30d trailing vol scaled to 14d = +-3.87%)
- Expected log return +0.73% (simple +0.74%) = shrunk drift +0.73% (sample drift +1.18% x prior weight 0.62) + feature signal +0.00%; GBM comparator +1.06%
- 80% prediction interval: -9.0% to +13.0% -> $74,306 to $92,268 (horizon vol 7.6%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.045 (t=0.48) does not clear momentum (+0.019) with t>2; net Sharpe 0.39 vs momentum 0.14 / buy-and-hold 0.50 (t=0.82); Brier 0.2515 not below climatology 0.2490**

**Data actually used** (model sample 2020-04-15 -> 2026-10-08, 18 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| eth | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `54e8d8d0-979a-4297-887f-2ea8a989753c` | 2014-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| perp | `perpetual-markets` / `binance_BTCUSDT` | 2019-09-08 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6057-140-68` | 2017-08-17 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (18 purged folds, retrain every 91d, embargo 14d, test 2022-05-13 -> 2026-09-24):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1596 | 114 | 114 | 114 | 114 |
| IC, signal only (Spearman, vol-scaled) | +0.041 | +0.045 (t=0.48) | +0.019 | n/a (constant) | -0.056 |
| IC, full forecast incl. per-fold drift | -0.089 | -0.085 | | | |
| Hit rate | 0.547 | 0.547 | 0.484 | drift 0.547 / long 0.547 | 0.481 |
| Drift-only R2 vs zero | | -0.0317 | | | |
| R2 OOS vs zero / vs drift | +0.0019 / +0.0325 | +0.0019 / +0.0326 | | | -0.1838 |
| Brier P(up) (clim / 0.5) | 0.2515 (0.2490 / 0.2500) | 0.2515 (0.2490) | | | 0.3014 |
| 80% interval coverage | 0.825 | 0.825 | | | |
| Ternary log-loss (clim) | 1.0604 (1.0610) | 1.0604 (1.0610) | | | |
| Mean net return / trade after 20bps | | +71bps (t=0.82) | -31bps (t=0.30) | buy-and-hold +91bps | -1bps |
| Annualised Sharpe (net) | | 0.39 | 0.14 | buy-and-hold 0.50 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 533 | 38.1 | -0.137 | 0.621 | 0.508 | 0.621 |
| bear | 419 | 29.9 | +0.038 | 0.580 | 0.434 | 0.580 |
| chop | 644 | 46.0 | +0.090 | 0.464 | 0.495 | 0.464 |

IC by calendar year: 2022: -0.09, 2023: -0.09, 2024: -0.07, 2025: +0.04, 2026: +0.08

Calibration of P(up) (OOS quintiles): p=0.47 -> 0.56 (n=320); p=0.49 -> 0.63 (n=323); p=0.49 -> 0.55 (n=317); p=0.50 -> 0.51 (n=324); p=0.51 -> 0.48 (n=312)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `tech_trend` | +0.0019 | +0.0021 | 100% | yes | +0.10 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `ret_28d` | +0.0018 | +0.0020 | 100% | yes | +0.60 | 28d log return, vol-scaled |
| `perp_spot_vol` | +0.0017 | +0.0015 | 100% | yes | +0.99 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `ret_14d` | +0.0015 | +0.0016 | 100% | yes | -0.43 | 14d log return, vol-scaled |
| `btcdom_chg_14d` | +0.0014 | +0.0020 | 100% | yes | +0.19 | BTC dominance 14d change, trailing z |
| `ret_7d` | +0.0011 | +0.0014 | 100% | yes | -0.71 | 7d log return, vol-scaled |
| `ethbtc_14d` | -0.0010 | -0.0015 | 100% | yes | -0.57 | ETH/BTC 14d change (alt risk appetite), trailing z |
| `stab_growth_14d` | +0.0009 | +0.0008 | 94% | yes | -0.46 | USD-stablecoin supply growth over 14d (mint minus burn), trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -0.708 | 7d log return, vol-scaled |
| `ret_14d` | -0.426 | 14d log return, vol-scaled |
| `ret_28d` | +0.605 | 28d log return, vol-scaled |
| `vol_14v90` | -1.264 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | -0.051 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | +0.098 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.155 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | -0.765 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | -0.427 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | +0.000 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.989 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_14d` | -0.508 | Cumulative 14d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_14d` | -0.461 | USD-stablecoin supply growth over 14d (mint minus burn), trailing z |
| `ethbtc_14d` | -0.574 | ETH/BTC 14d change (alt risk appetite), trailing z |
| `btcdom_chg_14d` | +0.185 | BTC dominance 14d change, trailing z |
| `gold_ret_14d` | -0.826 | PAXG (gold) 14d return, trailing z (only long-history macro proxy) |
| `addr_growth_7_90` | +0.634 | Active addresses 7d vs 90d average, log, trailing z |
| `fee_growth_7_90` | +1.069 | Network fee revenue 7d vs 90d average, log, trailing z |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_14d`, `etf_flow_14d`; same rows, window 2024-05-08 -> 2026-10-08, test 2025-06-05 -> 2026-09-24, 34 non-overlapping windows): IC +0.011 -> +0.105 (overlapping +0.022 -> +0.103), hit 0.472 -> 0.492, Brier 0.2483 -> 0.2481 (clim 0.2548). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_14d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.09 to +0.08; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-04-15 (binding source: gold_ret_14d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 28d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 2434 labelled days through 2026-09-10, shrinkage kappa=100.0):

- P(up) = **52.7%**; ternary P(down / flat / up) = 26.4% / 41.0% / 32.5% (flat band = +-0.5 x 30d trailing vol scaled to 28d = +-5.52%)
- Expected log return +1.25% (simple +1.26%) = shrunk drift +1.17% (sample drift +1.87% x prior weight 0.63) + feature signal +0.08%; GBM comparator +7.57%
- 80% prediction interval: -15.2% to +21.3% -> $69,249 to $99,077 (horizon vol 10.8%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.191 (t=1.28) does not clear momentum (+0.046) with t>2; net Sharpe 0.78 vs momentum 0.12 / buy-and-hold 0.79 (t=1.47); Brier 0.2500 not below climatology 0.2469**
- Non-overlapping test windows: **45** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2020-01-12 -> 2026-10-08, 14 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| eth | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `54e8d8d0-979a-4297-887f-2ea8a989753c` | 2014-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| price | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (8 purged folds, retrain every 182d, embargo 28d, test 2023-03-08 -> 2026-09-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1283 | 45 | 45 | 45 | 45 |
| IC, signal only (Spearman, vol-scaled) | +0.181 | +0.191 (t=1.28) | +0.046 | n/a (constant) | +0.032 |
| IC, full forecast incl. per-fold drift | -0.092 | -0.092 | | | |
| Hit rate | 0.574 | 0.573 | 0.482 | drift 0.568 / long 0.568 | 0.555 |
| Drift-only R2 vs zero | | +0.0239 | | | |
| R2 OOS vs zero / vs drift | +0.0265 / +0.0026 | +0.0265 / +0.0027 | | | -0.1420 |
| Brier P(up) (clim / 0.5) | 0.2500 (0.2468 / 0.2500) | 0.2500 (0.2469) | | | 0.2823 |
| 80% interval coverage | 0.825 | 0.825 | | | |
| Ternary log-loss (clim) | 1.0794 (1.0745) | 1.0794 (1.0745) | | | |
| Mean net return / trade after 30bps | | +228bps (t=1.47) | +30bps (t=0.22) | buy-and-hold +251bps | +104bps |
| Annualised Sharpe (net) | | 0.78 | 0.12 | buy-and-hold 0.79 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 486 | 17.4 | +0.083 | 0.607 | 0.455 | 0.591 |
| bear | 250 | 8.9 | -0.053 | 0.676 | 0.476 | 0.676 |
| chop | 547 | 19.5 | +0.187 | 0.497 | 0.510 | 0.499 |

IC by calendar year: 2023: +0.11, 2024: +0.24, 2025: +0.20, 2026: -0.39

Calibration of P(up) (OOS quintiles): p=0.49 -> 0.55 (n=275); p=0.50 -> 0.61 (n=271); p=0.51 -> 0.64 (n=247); p=0.51 -> 0.47 (n=252); p=0.54 -> 0.57 (n=238)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `btcdom_chg_28d` | +0.0023 | +0.0123 | 100% | yes | +0.12 | BTC dominance 28d change, trailing z |
| `price_to_activity` | -0.0023 | -0.0090 | 100% | yes | -0.11 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `ethbtc_28d` | -0.0017 | -0.0147 | 100% | yes | -0.34 | ETH/BTC 28d change (alt risk appetite), trailing z |
| `ret_28d` | +0.0017 | +0.0087 | 100% | yes | +0.60 | 28d log return, vol-scaled |
| `fee_growth_28_365` | +0.0015 | +0.0076 | 100% | yes | +2.08 | Network fee revenue 28d vs 365d average, log, trailing z |
| `addr_growth_28_365` | +0.0010 | +0.0063 | 100% | yes | +0.16 | Active addresses 28d vs 365d average, log, trailing z |
| `trend_200` | +0.0010 | +0.0036 | 100% | yes | +0.30 | Distance to 200d MA, 4y trailing z |
| `stab_growth_90d` | +0.0010 | +0.0040 | 100% | yes | -0.37 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.605 | 28d log return, vol-scaled |
| `ret_90d` | +1.254 | 90d log return, vol-scaled |
| `drawdown_365` | -0.314 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.300 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | -0.051 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `ethbtc_28d` | -0.341 | ETH/BTC 28d change (alt risk appetite), trailing z |
| `btcdom_chg_28d` | +0.123 | BTC dominance 28d change, trailing z |
| `addr_growth_28_365` | +0.160 | Active addresses 28d vs 365d average, log, trailing z |
| `fee_growth_28_365` | +2.084 | Network fee revenue 28d vs 365d average, log, trailing z |
| `price_to_activity` | -0.115 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |

</details>

**Short-history ablation** (core vs core + `gold_ret_28d`, `hl_oi_chg_28d`, `etf_flow_28d`, `mstr_buys_28d`; same rows, window 2024-05-16 -> 2026-10-08, test 2025-07-11 -> 2026-09-10, 15 non-overlapping windows): IC -0.072 -> -0.164 (overlapping -0.066 -> -0.151), hit 0.403 -> 0.403, Brier 0.2438 -> 0.2438 (clim 0.2607). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_28d`.

**Failure modes for this horizon:**

- Sparse non-overlapping samples: 45 independent test windows; one regime shift can flip the sign (see IC by year).
- Thin history: model sample starts 2020-01-12 (binding source: stab_cexflow_90d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 90d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 2372 labelled days through 2026-07-10, shrinkage kappa=100.0):

- P(up) = **53.2%**; ternary P(down / flat / up) = 31.6% / 33.1% / 35.4% (flat band = +-0.5 x 90d trailing vol scaled to 90d = +-10.07%)
- Expected log return +4.34% (simple +4.44%) = shrunk drift +4.12% (sample drift +6.65% x prior weight 0.62) + feature signal +0.22%; GBM comparator +22.16%
- 80% prediction interval: -22.3% to +42.5% -> $63,489 to $116,425 (horizon vol 19.2%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.160 (t=0.51) does not clear momentum (+0.078) with t>2; net Sharpe 0.56 vs momentum -0.32 / buy-and-hold 0.60 (t=1.01)**
- Non-overlapping test windows: **12** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2020-01-12 -> 2026-10-08, 14 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| eth | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `54e8d8d0-979a-4297-887f-2ea8a989753c` | 2014-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| price | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (7 purged folds, retrain every 182d, embargo 90d, test 2023-07-10 -> 2026-07-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1097 | 12 | 12 | 12 | 12 |
| IC, signal only (Spearman, vol-scaled) | +0.107 | +0.160 (t=0.51) | +0.078 | n/a (constant) | +0.454 |
| IC, full forecast incl. per-fold drift | -0.435 | -0.435 | | | |
| Hit rate | 0.582 | 0.583 | 0.503 | drift 0.583 / long 0.583 | 0.695 |
| Drift-only R2 vs zero | | +0.0396 | | | |
| R2 OOS vs zero / vs drift | +0.0373 / -0.0070 | +0.0398 / -0.0007 | | | +0.1576 |
| Brier P(up) (clim / 0.5) | 0.2542 (0.2544 / 0.2500) | 0.2542 (0.2544) | | | 0.2322 |
| 80% interval coverage | 0.816 | 0.816 | | | |
| Ternary log-loss (clim) | 1.1180 (1.1089) | 1.1181 (1.1090) | | | |
| Mean net return / trade after 50bps | | +740bps (t=1.01) | -45bps (t=-0.57) | buy-and-hold +790bps | +1183bps |
| Annualised Sharpe (net) | | 0.56 | -0.32 | buy-and-hold 0.60 | |

Complexity check: the depth-2 GBM beats ridge on non-overlapping IC, Brier and net return together, so it is worth a closer look, but with this few windows it is not adopted.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 393 | 4.4 | +0.211 | 0.588 | 0.588 | 0.588 |
| bear | 215 | 2.4 | -0.608 | 0.395 | 0.605 | 0.395 |
| chop | 489 | 5.4 | +0.102 | 0.661 | 0.389 | 0.661 |

IC by calendar year: 2023: +0.20, 2024: -0.43, 2025: +0.07, 2026: -0.62

Calibration of P(up) (OOS quintiles): p=0.50 -> 0.86 (n=225); p=0.52 -> 0.73 (n=227); p=0.53 -> 0.58 (n=206); p=0.55 -> 0.36 (n=259); p=0.57 -> 0.37 (n=180)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `price_to_activity` | -0.0040 | -0.0123 | 100% | yes | -0.11 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `btcdom_chg_90d` | +0.0036 | +0.0100 | 100% | yes | +0.18 | BTC dominance 90d change, trailing z |
| `fee_growth_28_365` | +0.0029 | +0.0132 | 100% | yes | +2.08 | Network fee revenue 28d vs 365d average, log, trailing z |
| `addr_growth_28_365` | +0.0024 | +0.0118 | 100% | yes | +0.16 | Active addresses 28d vs 365d average, log, trailing z |
| `ret_28d` | +0.0020 | +0.0085 | 100% | yes | +0.60 | 28d log return, vol-scaled |
| `ethbtc_90d` | -0.0020 | +0.0029 | 43% | no | +0.48 | ETH/BTC 90d change (alt risk appetite), trailing z |
| `drawdown_365` | +0.0019 | +0.0083 | 100% | yes | -0.31 | Drawdown from trailing 1y high, 4y trailing z |
| `ret_90d` | +0.0017 | +0.0057 | 100% | yes | +1.25 | 90d log return, vol-scaled |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.605 | 28d log return, vol-scaled |
| `ret_90d` | +1.254 | 90d log return, vol-scaled |
| `drawdown_365` | -0.314 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.300 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | -0.051 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `ethbtc_90d` | +0.481 | ETH/BTC 90d change (alt risk appetite), trailing z |
| `btcdom_chg_90d` | +0.181 | BTC dominance 90d change, trailing z |
| `addr_growth_28_365` | +0.160 | Active addresses 28d vs 365d average, log, trailing z |
| `fee_growth_28_365` | +2.084 | Network fee revenue 28d vs 365d average, log, trailing z |
| `price_to_activity` | -0.115 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |

</details>

**Short-history ablation**: window 2024-06-23 -> 2026-10-08 leaves fewer than 4 non-overlapping 90d test windows; no meaningful comparison possible
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
