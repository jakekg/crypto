# ETH-USD probabilistic forward-return forecast

Decision time: **2026-10-09 00:00 UTC** (last completed daily bar 2026-10-08, VWAP close $2,472). Snapshot of Blockworks MCP data; nothing after the last completed bar is used.

Primary model: ridge regression on vol-scaled forward returns (shrinkage picked by an inner, purged time split), with the predictive distribution taken from empirical residuals. Comparators: zero return, prevailing mean (drift), trailing same-horizon momentum, and a depth-2 gradient-boosted model. Walk-forward is purged with an embargo of one full horizon. Costs (round trip per non-overlapping trade): 1d 10bps, 7d 15bps, 14d 20bps, 28d 30bps, 90d 50bps.

## Summary

| horizon | P(up) | P(down/flat/up) | expected return | 80% interval | price range | non-overlap IC (N) | status |
|---|---|---|---|---|---|---|---|
| 1d | 50.9% | 25/48/27% | +0.04% | -2.8% / +2.9% | $2,403 - $2,545 | +0.001 (1592) | no edge claimed |
| 7d | 51.3% | 26/46/29% | +0.35% | -7.2% / +9.0% | $2,295 - $2,695 | +0.076 (224) | no edge claimed |
| 14d | 50.1% | 27/42/31% | +0.60% | -10.4% / +14.8% | $2,215 - $2,837 | +0.121 (110) | no edge claimed |
| 28d | 52.7% | 27/40/33% | +1.21% | -15.7% / +23.4% | $2,085 - $3,051 | +0.006 (45) | regime tilt |
| 90d | 55.9% | 30/33/37% | +4.00% | -27.4% / +49.2% | $1,796 - $3,690 | -0.242 (12) | regime tilt |

## Horizon 1d

**Current forecast** (as of 2026-10-08, model trained on 2324 labelled days through 2026-10-07, shrinkage kappa=100.0):

- P(up) = **50.9%**; ternary P(down / flat / up) = 25.3% / 47.7% / 27.0% (flat band = +-0.5 x 30d trailing vol scaled to 1d = +-1.18%)
- Expected log return +0.04% (simple +0.04%) = shrunk drift +0.04% (sample drift +0.07% x prior weight 0.61) + feature signal -0.01%; GBM comparator -0.44%
- 80% prediction interval: -2.8% to +2.9% -> $2,403 to $2,545 (horizon vol 2.4%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.001 (t=0.05) does not clear momentum (-0.032) with t>2; net Sharpe -0.35 vs momentum -0.80 / buy-and-hold 0.07 (t=-0.72); Brier 0.2512 not below climatology 0.2508**

**Data actually used** (model sample 2020-05-28 -> 2026-10-08, 20 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| perp | `perpetual-markets` / `binance_ETHUSDT` | 2019-11-27 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6059-140-68` | 2017-08-17 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (18 purged folds, retrain every 91d, embargo 1d, test 2022-05-30 -> 2026-10-07):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1592 | 1592 | 1592 | 1592 | 1592 |
| IC, signal only (Spearman, vol-scaled) | +0.001 | +0.001 (t=0.05) | -0.032 | n/a (constant) | -0.020 |
| IC, full forecast incl. per-fold drift | -0.030 | -0.030 | | | |
| Hit rate | 0.503 | 0.503 | 0.469 | drift 0.504 / long 0.504 | 0.498 |
| Drift-only R2 vs zero | | -0.0037 | | | |
| R2 OOS vs zero / vs drift | -0.0056 / -0.0019 | -0.0056 / -0.0019 | | | -0.0209 |
| Brier P(up) (clim / 0.5) | 0.2512 (0.2508 / 0.2500) | 0.2512 (0.2508) | | | 0.2589 |
| 80% interval coverage | 0.814 | 0.814 | | | |
| Ternary log-loss (clim) | 1.0417 (1.0419) | 1.0417 (1.0419) | | | |
| Mean net return / trade after 10bps | | -6bps (t=-0.72) | -15bps (t=-1.67) | buy-and-hold +1bps | -1bps |
| Annualised Sharpe (net) | | -0.35 | -0.80 | buy-and-hold 0.07 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 577 | 577.0 | -0.000 | 0.511 | 0.444 | 0.510 |
| bear | 559 | 559.0 | -0.039 | 0.504 | 0.479 | 0.508 |
| chop | 456 | 456.0 | +0.043 | 0.491 | 0.489 | 0.493 |

IC by calendar year: 2022: -0.03, 2023: -0.04, 2024: +0.04, 2025: +0.01, 2026: +0.02

Calibration of P(up) (OOS quintiles): p=0.50 -> 0.52 (n=341); p=0.51 -> 0.52 (n=348); p=0.51 -> 0.51 (n=266); p=0.51 -> 0.49 (n=325); p=0.53 -> 0.48 (n=312)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `ret_3d` | +0.0006 | +0.0037 | 100% | yes | -2.25 | 3d log return, vol-scaled |
| `perp_spot_vol` | +0.0006 | +0.0052 | 100% | yes | +0.87 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `int_cexflow_x_negmom` | -0.0006 | -0.0018 | 83% | yes | +0.58 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `stab_growth_7d` | +0.0005 | +0.0016 | 100% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `tech_trend` | +0.0005 | +0.0003 | 94% | yes | -0.14 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `ret_7d` | +0.0005 | +0.0004 | 94% | yes | -1.45 | 7d log return, vol-scaled |
| `stab_growth_1d` | +0.0004 | -0.0009 | 94% | yes | -0.62 | USD-stablecoin supply growth over 1d (mint minus burn), trailing z |
| `prem_lvl_3d` | +0.0004 | +0.0043 | 100% | yes | +0.21 | Perp premium (funding proxy) 3d mean, trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_1d` | -1.746 | 1d log return, vol-scaled |
| `ret_3d` | -2.254 | 3d log return, vol-scaled |
| `ret_7d` | -1.450 | 7d log return, vol-scaled |
| `vol_7v90` | +0.031 | 7d vs 90d realised vol, log, trailing z (vol spike) |
| `range_1d` | +0.931 | 1d high/low range, log, trailing z (liquidation-cluster proxy) |
| `tech_trend` | -0.139 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.418 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_3d` | +0.209 | Perp premium (funding proxy) 3d mean, trailing z |
| `prem_chg_3d` | +0.607 | Change in perp premium over 3d, trailing z |
| `perp_spot_vol` | +0.870 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_1d` | -0.123 | Cumulative 1d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_1d` | -0.622 | USD-stablecoin supply growth over 1d (mint minus burn), trailing z |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `btc_ret_1d` | -1.012 | BTC 1d return, vol-scaled |
| `rel_btc_1d` | -0.959 | ETH/BTC 1d relative return, trailing z |
| `btcdom_chg_1d` | +0.497 | BTC dominance 1d change, trailing z |
| `gold_ret_1d` | +0.402 | PAXG (gold) 1d return, trailing z (only long-history macro proxy) |
| `int_cexflow_x_negmom` | +0.579 | Stablecoin CEX inflow (7d) x negative 7d momentum |
| `int_volspike_x_range` | +0.029 | Vol spike x large daily range (liquidation-cluster proxy) |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_1d`, `int_prem_x_oi`, `etf_flow_1d`; same rows, window 2024-05-01 -> 2026-10-07, test 2025-05-03 -> 2026-10-07, 523 non-overlapping windows): IC +0.047 -> +0.053 (overlapping +0.047 -> +0.053), hit 0.486 -> 0.493, Brier 0.2493 -> 0.2495 (clim 0.2501). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `hl_liq_3d`, `spy_ret_1d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.04 to +0.04; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-05-28 (binding source: perp_spot_vol).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 7d

**Current forecast** (as of 2026-10-08, model trained on 2315 labelled days through 2026-10-01, shrinkage kappa=10.0):

- P(up) = **51.3%**; ternary P(down / flat / up) = 25.6% / 45.5% / 28.9% (flat band = +-0.5 x 30d trailing vol scaled to 7d = +-3.16%)
- Expected log return +0.35% (simple +0.35%) = shrunk drift +0.32% (sample drift +0.52% x prior weight 0.61) + feature signal +0.03%; GBM comparator +2.13%
- 80% prediction interval: -7.2% to +9.0% -> $2,295 to $2,695 (horizon vol 6.2%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.076 (t=1.14) does not clear momentum (-0.010) with t>2; net Sharpe 0.06 vs momentum -0.83 / buy-and-hold 0.27 (t=0.11)**

**Data actually used** (model sample 2020-05-31 -> 2026-10-08, 19 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `09f15639-06cf-4460-83b8-a86de2f2db28` | 2016-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| perp | `perpetual-markets` / `binance_ETHUSDT` | 2019-11-27 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6059-140-68` | 2017-08-17 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (18 purged folds, retrain every 91d, embargo 7d, test 2022-06-14 -> 2026-10-01):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1571 | 224 | 224 | 224 | 224 |
| IC, signal only (Spearman, vol-scaled) | +0.077 | +0.076 (t=1.14) | -0.010 | n/a (constant) | +0.051 |
| IC, full forecast incl. per-fold drift | -0.056 | -0.057 | | | |
| Hit rate | 0.511 | 0.511 | 0.477 | drift 0.514 / long 0.514 | 0.531 |
| Drift-only R2 vs zero | | -0.0137 | | | |
| R2 OOS vs zero / vs drift | -0.0064 / +0.0072 | -0.0064 / +0.0072 | | | -0.0922 |
| Brier P(up) (clim / 0.5) | 0.2503 (0.2506 / 0.2500) | 0.2503 (0.2506) | | | 0.2710 |
| 80% interval coverage | 0.791 | 0.791 | | | |
| Ternary log-loss (clim) | 1.0642 (1.0658) | 1.0642 (1.0658) | | | |
| Mean net return / trade after 15bps | | +18bps (t=0.11) | -18bps (t=-1.72) | buy-and-hold +38bps | +70bps |
| Annualised Sharpe (net) | | 0.06 | -0.83 | buy-and-hold 0.27 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 571 | 81.6 | +0.086 | 0.536 | 0.459 | 0.534 |
| bear | 544 | 77.7 | +0.072 | 0.546 | 0.500 | 0.555 |
| chop | 456 | 65.1 | +0.132 | 0.439 | 0.474 | 0.441 |

IC by calendar year: 2022: +0.14, 2023: +0.00, 2024: +0.04, 2025: +0.24, 2026: -0.01

Calibration of P(up) (OOS quintiles): p=0.48 -> 0.56 (n=323); p=0.50 -> 0.45 (n=312); p=0.50 -> 0.49 (n=313); p=0.50 -> 0.57 (n=309); p=0.52 -> 0.50 (n=314)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `prem_persist_14d` | +0.0135 | +0.0118 | 100% | yes | +0.07 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.0123 | +0.0138 | 100% | yes | +0.87 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `ret_28d` | +0.0116 | +0.0052 | 100% | yes | +0.12 | 28d log return, vol-scaled |
| `prem_lvl_7d` | +0.0099 | +0.0066 | 100% | yes | -0.17 | Perp premium (funding proxy) 7d mean, trailing z |
| `stab_growth_7d` | +0.0088 | +0.0064 | 100% | yes | -0.11 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `tech_trend` | +0.0080 | +0.0042 | 100% | yes | -0.14 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `vol_14v90` | -0.0073 | -0.0055 | 100% | yes | -1.06 | 14d vs 90d realised vol, log, trailing z |
| `addr_growth_7_90` | +0.0072 | +0.0092 | 100% | yes | -0.17 | Active addresses 7d vs 90d average, log, trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -1.450 | 7d log return, vol-scaled |
| `ret_14d` | -0.945 | 14d log return, vol-scaled |
| `ret_28d` | +0.116 | 28d log return, vol-scaled |
| `vol_14v90` | -1.059 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | -0.396 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | -0.139 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.418 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | -0.168 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | -0.038 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | +0.071 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.870 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_7d` | -0.399 | Cumulative 7d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_7d` | -0.106 | USD-stablecoin supply growth over 7d (mint minus burn), trailing z |
| `btc_ret_7d` | -0.708 | BTC 7d return, vol-scaled |
| `rel_btc_7d` | -0.891 | ETH/BTC 7d relative return, trailing z |
| `btcdom_chg_7d` | +0.178 | BTC dominance 7d change, trailing z |
| `gold_ret_7d` | -0.462 | PAXG (gold) 7d return, trailing z (only long-history macro proxy) |
| `addr_growth_7_90` | -0.173 | Active addresses 7d vs 90d average, log, trailing z |
| `fee_growth_7_90` | +1.342 | Network fee revenue 7d vs 90d average, log, trailing z |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_7d`, `etf_flow_7d`; same rows, window 2024-05-04 -> 2026-10-08, test 2025-05-18 -> 2026-10-01, 71 non-overlapping windows): IC -0.032 -> -0.035 (overlapping -0.035 -> -0.033), hit 0.412 -> 0.414, Brier 0.2594 -> 0.2602 (clim 0.2527). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `hl_liq_3d`, `spy_ret_7d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.01 to +0.24; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-05-31 (binding source: prem_chg_7d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 14d

**Current forecast** (as of 2026-10-08, model trained on 2308 labelled days through 2026-09-24, shrinkage kappa=100.0):

- P(up) = **50.1%**; ternary P(down / flat / up) = 27.0% / 42.1% / 30.8% (flat band = +-0.5 x 30d trailing vol scaled to 14d = +-4.50%)
- Expected log return +0.60% (simple +0.60%) = shrunk drift +0.61% (sample drift +0.99% x prior weight 0.61) + feature signal -0.00%; GBM comparator +1.55%
- 80% prediction interval: -10.4% to +14.8% -> $2,215 to $2,837 (horizon vol 8.8%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.121 (t=1.26) does not clear momentum (+0.024) with t>2; net Sharpe 0.44 vs momentum 0.17 / buy-and-hold 0.31 (t=0.90)**

**Data actually used** (model sample 2020-05-31 -> 2026-10-08, 19 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| gold | `asset-price` / `be5685e4-c09d-4671-8e50-abbe1dd50b43` | 2019-10-04 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `09f15639-06cf-4460-83b8-a86de2f2db28` | 2016-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| perp | `perpetual-markets` / `binance_ETHUSDT` | 2019-11-27 -> 2026-10-08 | 0.0% | 0d | 0d |
| price | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| spot | `bqen` / `6059-140-68` | 2017-08-17 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (18 purged folds, retrain every 91d, embargo 14d, test 2022-06-28 -> 2026-09-24):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1550 | 110 | 110 | 110 | 110 |
| IC, signal only (Spearman, vol-scaled) | +0.122 | +0.121 (t=1.26) | +0.024 | n/a (constant) | +0.115 |
| IC, full forecast incl. per-fold drift | -0.033 | -0.034 | | | |
| Hit rate | 0.518 | 0.518 | 0.504 | drift 0.507 / long 0.507 | 0.550 |
| Drift-only R2 vs zero | | -0.0245 | | | |
| R2 OOS vs zero / vs drift | -0.0151 / +0.0091 | -0.0150 / +0.0092 | | | -0.1001 |
| Brier P(up) (clim / 0.5) | 0.2505 (0.2531 / 0.2500) | 0.2505 (0.2531) | | | 0.2734 |
| 80% interval coverage | 0.792 | 0.792 | | | |
| Ternary log-loss (clim) | 1.0735 (1.0812) | 1.0735 (1.0812) | | | |
| Mean net return / trade after 20bps | | +99bps (t=0.90) | +6bps (t=0.36) | buy-and-hold +77bps | +155bps |
| Annualised Sharpe (net) | | 0.44 | 0.17 | buy-and-hold 0.31 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 564 | 40.3 | +0.112 | 0.535 | 0.521 | 0.544 |
| bear | 530 | 37.9 | +0.019 | 0.568 | 0.447 | 0.564 |
| chop | 456 | 32.6 | +0.360 | 0.439 | 0.550 | 0.395 |

IC by calendar year: 2022: +0.20, 2023: +0.23, 2024: -0.02, 2025: +0.33, 2026: -0.20

Calibration of P(up) (OOS quintiles): p=0.43 -> 0.47 (n=313); p=0.50 -> 0.63 (n=311); p=0.50 -> 0.42 (n=319); p=0.51 -> 0.54 (n=298); p=0.57 -> 0.47 (n=309)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `prem_persist_14d` | +0.0021 | +0.0347 | 100% | yes | +0.07 | Share of last 14d with premium above its 1y median (funding persistence) |
| `ret_28d` | +0.0018 | +0.0043 | 94% | yes | +0.12 | 28d log return, vol-scaled |
| `perp_spot_vol` | +0.0018 | +0.0299 | 100% | yes | +0.87 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `btc_ret_14d` | +0.0017 | +0.0060 | 94% | yes | -0.43 | BTC 14d return, vol-scaled |
| `tech_trend` | +0.0017 | +0.0056 | 100% | yes | -0.14 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `prem_lvl_7d` | +0.0016 | +0.0128 | 100% | yes | -0.17 | Perp premium (funding proxy) 7d mean, trailing z |
| `ret_14d` | +0.0012 | -0.0042 | 89% | yes | -0.95 | 14d log return, vol-scaled |
| `addr_growth_7_90` | +0.0010 | +0.0192 | 100% | yes | -0.17 | Active addresses 7d vs 90d average, log, trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_7d` | -1.450 | 7d log return, vol-scaled |
| `ret_14d` | -0.945 | 14d log return, vol-scaled |
| `ret_28d` | +0.116 | 28d log return, vol-scaled |
| `vol_14v90` | -1.059 | 14d vs 90d realised vol, log, trailing z |
| `vol_regime` | -0.396 | 30d realised vol percentile within trailing 2y (centred) |
| `tech_trend` | -0.139 | Composite of RSI14, distance to MA50 and MA200 (mean of trailing z-scores) |
| `tech_bbwidth` | -1.418 | Bollinger(20,2) band width, log, trailing z (compression vs expansion) |
| `prem_lvl_7d` | -0.168 | Perp premium (funding proxy) 7d mean, trailing z |
| `prem_chg_7d` | -0.038 | Change in perp premium over 7d, trailing z |
| `prem_persist_14d` | +0.071 | Share of last 14d with premium above its 1y median (funding persistence) |
| `perp_spot_vol` | +0.870 | Binance perp 7d volume / aggregate spot VWAP volume, log, trailing z (leverage share) |
| `stab_cexflow_14d` | -0.508 | Cumulative 14d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_14d` | -0.461 | USD-stablecoin supply growth over 14d (mint minus burn), trailing z |
| `btc_ret_14d` | -0.426 | BTC 14d return, vol-scaled |
| `rel_btc_14d` | -0.574 | ETH/BTC 14d relative return, trailing z |
| `btcdom_chg_14d` | +0.185 | BTC dominance 14d change, trailing z |
| `gold_ret_14d` | -0.826 | PAXG (gold) 14d return, trailing z (only long-history macro proxy) |
| `addr_growth_7_90` | -0.173 | Active addresses 7d vs 90d average, log, trailing z |
| `fee_growth_7_90` | +1.342 | Network fee revenue 7d vs 90d average, log, trailing z |

</details>

**Short-history ablation** (core vs core + `hl_oi_chg_14d`, `etf_flow_14d`; same rows, window 2024-05-08 -> 2026-10-08, test 2025-06-05 -> 2026-09-24, 34 non-overlapping windows): IC +0.006 -> -0.100 (overlapping +0.013 -> -0.112), hit 0.392 -> 0.392, Brier 0.2934 -> 0.2522 (clim 0.2563). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_14d`.

**Failure modes for this horizon:**

- Regime shift: IC by year ranges -0.20 to +0.33; drift learned from the training window dominates the sign of the forecast.
- Thin history: model sample starts 2020-05-31 (binding source: prem_chg_7d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Exchange-specific volume/premium: perp premium and perp/spot volume come from Binance only; venue-specific wash or incentive volume and index-construction quirks feed straight into them.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 28d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 2434 labelled days through 2026-09-10, shrinkage kappa=100.0):

- P(up) = **52.7%**; ternary P(down / flat / up) = 27.2% / 40.1% / 32.7% (flat band = +-0.5 x 30d trailing vol scaled to 28d = +-6.43%)
- Expected log return +1.20% (simple +1.21%) = shrunk drift +1.15% (sample drift +1.84% x prior weight 0.63) + feature signal +0.06%; GBM comparator +10.63%
- 80% prediction interval: -15.7% to +23.4% -> $2,085 to $3,051 (horizon vol 12.5%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC +0.006 (t=0.04) does not clear momentum (+0.186) with t>2; net Sharpe 0.14 vs momentum -0.08 / buy-and-hold 0.20 (t=0.26); Brier 0.2708 not below climatology 0.2567**
- Non-overlapping test windows: **45** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2020-01-12 -> 2026-10-08, 16 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `09f15639-06cf-4460-83b8-a86de2f2db28` | 2016-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| price | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (8 purged folds, retrain every 182d, embargo 28d, test 2023-03-08 -> 2026-09-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1283 | 45 | 45 | 45 | 45 |
| IC, signal only (Spearman, vol-scaled) | +0.002 | +0.006 (t=0.04) | +0.186 | n/a (constant) | +0.165 |
| IC, full forecast incl. per-fold drift | -0.122 | -0.123 | | | |
| Hit rate | 0.503 | 0.503 | 0.563 | drift 0.503 / long 0.503 | 0.556 |
| Drift-only R2 vs zero | | -0.0335 | | | |
| R2 OOS vs zero / vs drift | -0.1241 / -0.0880 | -0.1249 / -0.0884 | | | -0.3804 |
| Brier P(up) (clim / 0.5) | 0.2708 (0.2567 / 0.2500) | 0.2708 (0.2567) | | | 0.2896 |
| 80% interval coverage | 0.751 | 0.751 | | | |
| Ternary log-loss (clim) | 1.1289 (1.0947) | 1.1288 (1.0947) | | | |
| Mean net return / trade after 30bps | | +62bps (t=0.26) | +175bps (t=-0.16) | buy-and-hold +93bps | +150bps |
| Annualised Sharpe (net) | | 0.14 | -0.08 | buy-and-hold 0.20 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 491 | 17.5 | +0.211 | 0.540 | 0.570 | 0.540 |
| bear | 432 | 15.4 | -0.261 | 0.616 | 0.507 | 0.616 |
| chop | 360 | 12.9 | -0.069 | 0.317 | 0.619 | 0.319 |

IC by calendar year: 2023: +0.48, 2024: -0.26, 2025: +0.46, 2026: -0.55

Calibration of P(up) (OOS quintiles): p=0.51 -> 0.56 (n=266); p=0.52 -> 0.54 (n=258); p=0.53 -> 0.43 (n=277); p=0.53 -> 0.55 (n=226); p=0.68 -> 0.45 (n=256)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `btc_ret_28d` | +0.0024 | +0.0197 | 100% | yes | +0.60 | BTC 28d return, vol-scaled |
| `price_to_activity` | -0.0014 | -0.0119 | 100% | yes | -0.75 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `ret_28d` | +0.0013 | -0.0008 | 88% | yes | +0.12 | 28d log return, vol-scaled |
| `fee_growth_28_365` | +0.0013 | +0.0052 | 100% | yes | +1.94 | Network fee revenue 28d vs 365d average, log, trailing z |
| `stab_cexflow_90d` | +0.0011 | +0.0265 | 100% | yes | -0.46 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `btcdom_chg_28d` | +0.0010 | +0.0066 | 100% | yes | +0.12 | BTC dominance 28d change, trailing z |
| `stab_growth_90d` | +0.0009 | +0.0083 | 100% | yes | -0.37 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `ret_90d` | +0.0009 | -0.0024 | 88% | yes | +1.43 | 90d log return, vol-scaled |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.116 | 28d log return, vol-scaled |
| `ret_90d` | +1.430 | 90d log return, vol-scaled |
| `drawdown_365` | -0.297 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.569 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | -0.396 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `btc_ret_28d` | +0.605 | BTC 28d return, vol-scaled |
| `rel_btc_28d` | -0.341 | ETH/BTC 28d relative return, trailing z |
| `btcdom_chg_28d` | +0.123 | BTC dominance 28d change, trailing z |
| `addr_growth_28_365` | -0.808 | Active addresses 28d vs 365d average, log, trailing z |
| `fee_growth_28_365` | +1.943 | Network fee revenue 28d vs 365d average, log, trailing z |
| `price_to_activity` | -0.754 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `beta_btc_180d` | +0.299 | Rolling 180d beta to BTC, 4y trailing z |

</details>

**Short-history ablation** (core vs core + `gold_ret_28d`, `hl_oi_chg_28d`, `etf_flow_28d`; same rows, window 2024-05-16 -> 2026-10-08, test 2025-07-11 -> 2026-09-10, 15 non-overlapping windows): IC -0.055 -> -0.071 (overlapping -0.031 -> -0.053), hit 0.499 -> 0.436, Brier 0.2806 -> 0.2503 (clim 0.2692). Not used in the primary forecast: too few independent windows to trust.
Too short to evaluate (< 700 days): `spy_ret_28d`.

**Failure modes for this horizon:**

- Sparse non-overlapping samples: 45 independent test windows; one regime shift can flip the sign (see IC by year).
- Thin history: model sample starts 2020-01-12 (binding source: stab_cexflow_90d).
- Stale / revised onchain data: chain and stablecoin series are lagged 1 day; CEX wallet labels for stablecoin balances are applied retroactively (today's label set), a look-ahead risk that likely flatters `stab_cexflow_*`.
- Missing slow fundamentals and macro: no DXY, real yields or long-history equities in the MCP; gold (PAXG) is the only macro proxy.

## Horizon 90d - regime tilt, not a price path

**Current forecast** (as of 2026-10-08, model trained on 2372 labelled days through 2026-07-10, shrinkage kappa=100.0):

- P(up) = **55.9%**; ternary P(down / flat / up) = 29.6% / 33.3% / 37.1% (flat band = +-0.5 x 90d trailing vol scaled to 90d = +-14.11%)
- Expected log return +3.92% (simple +4.00%) = shrunk drift +3.73% (sample drift +6.02% x prior weight 0.62) + feature signal +0.20%; GBM comparator +19.44%
- 80% prediction interval: -27.4% to +49.2% -> $1,796 to $3,690 (horizon vol 26.4%)
- Verdict: **NO TRADABLE EDGE CLAIMED - non-overlapping IC -0.242 (t=-0.79) does not clear momentum (-0.150) with t>2; net Sharpe 0.13 vs momentum -0.30 / buy-and-hold 0.16 (t=0.23)**
- Non-overlapping test windows: **12** - treat as a regime tilt; windows are few and regimes do not repeat.

**Data actually used** (model sample 2020-01-12 -> 2026-10-08, 16 core features):

| source | MCP model / series | raw history | missing in model window | staleness at latest bar | publication lag |
|---|---|---|---|---|---|
| btc | `asset-price` / `1e31218a-e44e-4285-820c-8282ee222035` | 2010-07-21 -> 2026-10-08 | 0.0% | 0d | 0d |
| dominance | `asset-marketcap` / `1e31218a-e44e-4285-820c-8282ee222035` | 2013-03-01 -> 2026-10-08 | 0.0% | 0d | 0d |
| onchain | `blockchains` / `09f15639-06cf-4460-83b8-a86de2f2db28` | 2016-01-01 -> 2026-10-08 | 0.0% | 0d | 1d |
| price | `asset-price` / `21c795f5-1bfd-40c3-858e-e9d7e820c6d0` | 2015-08-07 -> 2026-10-08 | 0.0% | 0d | 0d |
| stables | `stablecoins-by-currency` / `usd` | 2018-01-01 -> 2026-10-07 | 0.0% | 0d | 1d |

**Walk-forward out-of-sample results** (7 purged folds, retrain every 182d, embargo 90d, test 2023-07-10 -> 2026-07-10):

| metric | model (ridge) overlapping | model non-overlapping | momentum non-overlap | zero / drift / long-only non-overlap | GBM non-overlap |
|---|---|---|---|---|---|
| N | 1097 | 12 | 12 | 12 | 12 |
| IC, signal only (Spearman, vol-scaled) | -0.205 | -0.242 (t=-0.79) | -0.150 | n/a (constant) | +0.204 |
| IC, full forecast incl. per-fold drift | -0.228 | -0.282 | | | |
| Hit rate | 0.526 | 0.526 | 0.431 | drift 0.526 / long 0.526 | 0.562 |
| Drift-only R2 vs zero | | -0.0753 | | | |
| R2 OOS vs zero / vs drift | -0.0231 / +0.0416 | -0.0258 / +0.0455 | | | -0.1965 |
| Brier P(up) (clim / 0.5) | 0.2529 (0.2597 / 0.2500) | 0.2528 (0.2596) | | | 0.3161 |
| 80% interval coverage | 0.681 | 0.681 | | | |
| Ternary log-loss (clim) | 1.1678 (1.1552) | 1.1674 (1.1547) | | | |
| Mean net return / trade after 50bps | | +157bps (t=0.23) | -712bps (t=-0.53) | buy-and-hold +207bps | +383bps |
| Annualised Sharpe (net) | | 0.13 | -0.30 | buy-and-hold 0.16 | |

Complexity check: the depth-2 GBM does not beat ridge on non-overlapping IC, Brier and net return together, so the simpler linear model is kept.

Regime stability (regime = trailing 90d return >+15% bull, <-15% bear, else chop; known at T; overlapping rows):

| regime | rows | ~independent | IC | hit | momentum hit | long-only hit |
|---|---|---|---|---|---|---|
| bull | 406 | 4.5 | +0.158 | 0.493 | 0.493 | 0.493 |
| bear | 398 | 4.4 | -0.500 | 0.603 | 0.397 | 0.603 |
| chop | 293 | 3.3 | -0.283 | 0.468 | 0.392 | 0.468 |

IC by calendar year: 2023: +0.61, 2024: -0.27, 2025: +0.13, 2026: -0.48

Calibration of P(up) (OOS quintiles): p=0.57 -> 0.40 (n=226); p=0.58 -> 0.53 (n=228); p=0.59 -> 0.70 (n=207); p=0.59 -> 0.64 (n=219); p=0.60 -> 0.37 (n=217)

**Top features** (standardised ridge coefficient in the final fit; sign agreement = share of walk-forward folds with the same sign; 'stable' if >= 80%):

| feature | coef (final) | mean coef (folds) | sign agreement | stable? | latest value (z) | description |
|---|---|---|---|---|---|---|
| `btcdom_chg_90d` | +0.0032 | +0.0024 | 100% | yes | +0.18 | BTC dominance 90d change, trailing z |
| `price_to_activity` | -0.0028 | -0.0044 | 100% | yes | -0.75 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `fee_growth_28_365` | +0.0025 | +0.0034 | 100% | yes | +1.94 | Network fee revenue 28d vs 365d average, log, trailing z |
| `btc_ret_90d` | +0.0019 | +0.0025 | 100% | yes | +1.25 | BTC 90d return, vol-scaled |
| `stab_cexflow_28d` | +0.0017 | +0.0025 | 100% | yes | -0.47 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_cexflow_90d` | +0.0016 | +0.0037 | 100% | yes | -0.46 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `rel_btc_90d` | -0.0011 | +0.0003 | 43% | no | +0.48 | ETH/BTC 90d relative return, trailing z |
| `beta_btc_180d` | -0.0008 | +0.0015 | 14% | no | +0.30 | Rolling 180d beta to BTC, 4y trailing z |

<details><summary>Latest feature snapshot (all core features)</summary>

| feature | value | description |
|---|---|---|
| `ret_28d` | +0.116 | 28d log return, vol-scaled |
| `ret_90d` | +1.430 | 90d log return, vol-scaled |
| `drawdown_365` | -0.297 | Drawdown from trailing 1y high, 4y trailing z |
| `trend_200` | +0.569 | Distance to 200d MA, 4y trailing z |
| `vol_regime` | -0.396 | 30d realised vol percentile within trailing 2y (centred) |
| `stab_cexflow_28d` | -0.466 | Cumulative 28d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_28d` | -0.503 | USD-stablecoin supply growth over 28d (mint minus burn), trailing z |
| `stab_cexflow_90d` | -0.458 | Cumulative 90d USD-stablecoin net flow into CEX wallets / supply, trailing z |
| `stab_growth_90d` | -0.370 | USD-stablecoin supply growth over 90d (mint minus burn), trailing z |
| `btc_ret_90d` | +1.254 | BTC 90d return, vol-scaled |
| `rel_btc_90d` | +0.481 | ETH/BTC 90d relative return, trailing z |
| `btcdom_chg_90d` | +0.181 | BTC dominance 90d change, trailing z |
| `addr_growth_28_365` | -0.808 | Active addresses 28d vs 365d average, log, trailing z |
| `fee_growth_28_365` | +1.943 | Network fee revenue 28d vs 365d average, log, trailing z |
| `price_to_activity` | -0.754 | log(price / 28d avg transactions), 4y trailing z (slow valuation vs activity) |
| `beta_btc_180d` | +0.299 | Rolling 180d beta to BTC, 4y trailing z |

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
