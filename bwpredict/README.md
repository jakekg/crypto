# bwpredict: probabilistic crypto forward-return forecasts from Blockworks MCP data

This replaces the May 2024 LSTM models in the repository root (`ethpriceprediction*.py`).
Those models used one input (past closes). They also leaked the future: the scaler was fit on the
full history, and `random_split` shuffled overlapping 60-day windows into train and validation.
This pipeline forecasts the **distribution of forward log returns** at 1d, 7d, 14d, 28d and 90d.
For each horizon it outputs P(up), ternary P(down/flat/up), the expected return, an 80% prediction
interval and an explicit tradable-edge verdict. Inputs are market, derivatives, onchain, stablecoin,
ETF and cross-asset series discovered in the Blockworks MCP.

**Bottom line (snapshot 2026-10-09):** none of the 20 asset × horizon combinations (BTC, ETH, SOL, XRP × 5 horizons) beats
the zero, drift, momentum and buy-and-hold baselines after costs with statistical support. The
forecasts are therefore well-calibrated *distributions* (80% intervals cover 75–84% out of sample,
except ETH/XRP 90d at 68–72%) that sit close to a shrunk-drift climatology. They are not trading
signals. Signal-only ICs average +0.04 across the 20 cases (14 positive), which is weak and not
significant per case. That honest null is the main result.

```
pip install -r requirements.txt
python run.py BTC ETH SOL XRP      # -> reports/<SYM>.md, reports/<SYM>.json, reports/<SYM>_forecast.png
python -m pytest -q tests          # point-in-time / leakage tests
python -m bwpredict.manifest XRP   # MCP calls needed to (re)fetch an asset's data
```

Reports: [BTC](reports/BTC.md) · [ETH](reports/ETH.md) · [SOL](reports/SOL.md) · [XRP](reports/XRP.md)

---

## Step 1: Data discovery (Blockworks MCP)

MCP tools available: `timeseries_catalog`, `timeseries_data` (≤ 5,000 points per call, ≤ 10–20 series),
`tabular_catalog`, `tabular_data` (≤ 200 rows), `search_documents` (text only, not used).
The catalogs list 61 time-series models and 46 tabular models. Not used on purpose: `asset-volatility` / `asset-sharpe` (recomputed locally so the windows are point-in-time controlled), lending, DEX, vault, launchpad, fundraising and TCG models (protocol-level, not asset-price drivers at these horizons), and `search_documents` / news (text, not cleanly timestamped as features). IDs were resolved through the tabular
`assets`, `markets`, `perpetual-markets`, `blockchains`, `etf-assets` and `dat-company-assets`
models. No identifier is assumed; every perp id in `config.py` was checked against `perpetual-markets`.

| Bucket | Requested | What the MCP has (model → fields) | Freq / history | Used? | Gaps & look-ahead flags |
|---|---|---|---|---|---|
| Spot OHLCV | yes | `asset-price` → VWAP OHLC + volume (cross-venue) | 5m–1w; BTC 2010-07+, ETH 2015-08+, SOL 2020-04+, XRP 2015-01+ | **core** | VWAP candle [T, T+1d): complete at T+1 00:00 UTC. Partial bars dropped |
| Per-venue spot | yes | `bqen` → Binance USDT pair close/volume | 1d; 2017-08+ (BTC/ETH) | **core** (basis leg) | single venue |
| Perp OHLCV / OI | yes | `perpetual-markets` → close, volume, OI (end-of-day + avg) | 1d; price 2019-09+, **CEX OI only ~300 days (2025-12+)** | price/volume **core**; CEX OI too short | dated futures excluded |
| Funding rates | yes | **not available** | — | proxy | proxy = Binance perp close / Binance spot close − 1 at the same timestamp |
| Basis | yes | **not available** (no dated futures) | — | same proxy | |
| Liquidations | yes | `perp-dex-protocols` (Hyperliquid) → total liquidation volume per pair | 1d; **2025-05+**, no long/short split | ablation only (too short) | CEX liquidations missing |
| Taker buy/sell, spreads | yes | **not available** for CEX (DEX buy/sell exists only for stablecoin aggregates) | — | no | |
| Options IV/skew | if present | **not available** (only chain-level options volume) | — | no | |
| OI (long history) | yes | Hyperliquid pair OI | 1d; 2023-05+ | ablation | venue-specific |
| Exchange in/outflows (native asset) | yes | **not available** | — | no | |
| Stablecoin flows | yes | `stablecoins-by-currency/usd` → supply, mints, burns, **CEX balance** | 1d; 2018-01+ (CEX balance 2019-05+) | **core** | **look-ahead risk**: CEX wallet labels applied retroactively; lag 1d |
| Active addresses, fees, txs | yes | `blockchains` → active-addresses, total-transactions, fee-revenue | 1d; BTC 2014+, ETH 2016+, SOL 2020+ | **core** | last day often partial → lag 1d; possible revisions |
| Supply / unlocks | yes | `asset-supply`, `token-unlocks` (forward schedule) | 1d | core where schedule covers today | **look-ahead risk**: current schedule vintage. XRP schedule ends 2022-06 → dropped |
| Whale / labeled-wallet flows | yes | only `dat-companies-by-asset` (treasury purchases, e.g. MSTR BTC) | 1d; 2020-08+ | ablation (28/90d) | dated on 8-K announcement day; lag 1d |
| ETF flows | (institutional) | `etf-assets` → spot flow, AUM | US trading days; 2024-01+ | ablation | latest row provisional (AUM present, flow null); lag 1d |
| BTC dominance, ETH/BTC | yes | `asset-marketcap` → dominance; ETH/BTC from `asset-price` | 1d; 2013+ | **core** | dominance universe may be backfilled |
| Equities | yes | `tokenized-equities-by-equity` → underlying close (SPY, QQQ…) | **SPY 2025-01+**, QQQ 2025-06+ | ablation only | too short to train |
| DXY | yes | **not available** | — | no | |
| Yields / real yields | yes | tokenized TIP/IEF/TLT closes only from 2025 | — | no | too short |
| Gold | yes | `asset-price` PAXG | 1d; 2019-10+ | **core** (1–14d), ablation (28/90d) | the only long-history macro proxy |
| Sentiment / attention | if present | `asset-sentiment` (2025-01+), `asset-mindshare` (2024-07+) | 1d | no | too short for 28d/90d; not used |

Snapshots of every call used are stored verbatim under `data/raw/` (MCP JSON). The loader indexes
them by (model, series key), so refreshing just means dropping in new files.

## Step 2: Point-in-time dataset (`bwpredict/dataset.py`)

* Row T is the UTC daily bar [T, T+1d). Decision time is T+1d 00:00 UTC, and **only completed bars
  are used** (bars ending after the snapshot time are dropped).
* Publication lags (`config.LAG`): prices, perps, dominance, gold, Hyperliquid 0d. Onchain,
  stablecoins, ETF, treasury buys, supply, equities 1d.
* Resampling rules: all inputs are native 1d bars. Volumes and flows are **summed**; OI, balances,
  AUM and supply take the **last** value; the reference price is the **VWAP** close. ETF flows on
  non-trading days are genuine zeros (sum rule).
* Forward-fill only applies to stock-like series (balances, OI, AUM, prices), up to `config.FFILL`
  days. Flows are never forward-filled. A `stale__<source>` column records days since the last real
  observation.
* Target `y_h = log(P[T+h] / P[T])`. Ternary label: up/down if `|y_h| > 0.5 × σ_d × √h`, else flat.
  σ_d is the trailing daily log-return vol over `max(30, h)` days ending at T, so "flat" means
  within normal noise.
* Tests (`tests/test_point_in_time.py`) rebuild the panel and features with the clock set to three
  past dates. They require every feature and σ on or before the cut to be identical to the full
  build, which catches centred windows, back-fills and negative shifts. They also check label
  arithmetic and the absence of partial bars, and confirm that a deliberately leaked target is
  detected (IC > 0.8).

## Step 3: Features (`bwpredict/features.py`)

Features are engineered per horizon, not dumped as raw columns. Levels become trailing (past-only)
z-scores, clipped at ±4, and the model re-standardises on each training window. Price technicals
are compressed into two composites (`tech_trend` = RSI14 + MA50 and MA200 distance; `tech_bbwidth`).

* **1d:** 1/3/7d vol-scaled returns, 7d-vs-90d vol spike, daily-range shock (liquidation-cluster
  proxy), perp premium level and 3d change (funding proxy), perp/spot volume share, 1d and 7d
  stablecoin CEX netflow and supply growth z-scores, ETH/BTC (or BTC and relative-to-BTC for alts),
  dominance, gold. Interactions: stablecoin CEX inflow × negative momentum, and vol spike × range
  shock. *Ablation:* Hyperliquid OI change, premium × rising OI, Hyperliquid liquidations, ETF flow,
  SPY.
* **7d / 14d:** 7/14/28d returns, vol ratio, vol-percentile regime, premium level/change and
  **funding persistence** (share of days above the 1y median), cumulative h-day stablecoin flows,
  cross-asset momentum, active-address and fee growth. *Ablation:* OI trend, ETF flows, SPY.
* **28d / 90d:** 28/90d trend, drawdown from 1y high, distance to MA200, vol regime, cumulative
  28d and 90d stablecoin CEX flows and supply growth, dominance and ETH/BTC trends, 28d-vs-365d
  activity and fee growth, a slow valuation ratio (price / transactions), rolling beta to BTC (alts),
  **forward unlocks over the horizon** and trailing supply inflation (where available). No order-book
  or single-day liquidation features. *Ablation:* gold, ETF flows, treasury-company buys, OI trend.

## Step 4: Model and validation (`bwpredict/model.py`)

* **Target in vol units:** `z = y_h / (σ_d √h)`. Forecasts map back via `μ = ẑ · σ_d √h`.
* **Baselines:** zero return, prevailing mean (drift), trailing same-horizon momentum, and long-only.
* **Primary model:** ridge on train-standardised features. Shrinkage κ (penalty = κ·n) comes from an
  inner purged time split with the **one-standard-error rule**, using h-day block SEs. The drift
  (intercept) is the training mean **shrunk toward zero by a Bayesian prior** (annualised-Sharpe
  prior sd 0.5).
* **Complexity comparator:** a depth-2 histogram GBM. It is reported but kept out of the forecast,
  because it never beats ridge on IC, Brier and net return together.
* **Predictive distribution:** `ẑ + e`, with `e` the demeaned empirical training residuals. P(up),
  the ternary probabilities and the 80% PI all come from this one distribution, so skew is kept.
  This is why XRP can show P(up) < 50% alongside a positive mean.
* **Walk-forward:** expanding window. Training rows must satisfy `T + h + embargo ≤ test start`
  (purge plus an embargo of one full horizon). Tests run in 91-day blocks for 1–14d and 182-day
  blocks with fewer folds for 28/90d. Minimum training is 2y (1–14d) or 3y (28/90d).
* **Metrics:** reported both overlapping and non-overlapping (step = h, averaged over all h start
  offsets).
  * IC (Spearman) on vol-standardised quantities. Headline IC is signal-only; full-forecast IC is
    shown too.
  * Hit rate vs momentum, drift and long-only.
  * R²_OOS vs zero and vs drift.
  * Brier vs climatology and 0.5, and ternary log-loss vs climatology.
  * PI coverage.
  * Net return after costs (1d 10, 7d 15, 14d 20, 28d 30, 90d 50 bps round trip per window) and net
    Sharpe vs momentum and buy-and-hold.
  * IC by year, regime splits (trailing-90d bull/bear/chop, known at T), and fold-level coefficient
    sign stability.
* **Edge rule:** an edge is claimed only if non-overlapping IC beats momentum with t>2, net Sharpe
  beats momentum, buy-and-hold and zero with t>2, *and* Brier beats climatology. If a horizon has
  fewer than ~8 non-overlapping windows it is labelled **indicative only** (SOL 90d: 5).

### Changes made after seeing out-of-sample results (disclosed)

1. **IC is computed on vol-standardised forecasts.** Raw-μ IC mostly ranked trailing volatility
   whenever drift dominated, which produced a spurious −0.32 SOL 14d IC.
2. **Headline IC is signal-only.** The per-fold drift level made pooled IC a test of "did last
   window's drift persist" (full-forecast IC is still reported).
3. **κ selection moved from "within 0.2% of best" to the 1-SE rule.** The former chose noisy κ, and
   Brier was worse than climatology.
4. **Drift prior added.** Prevailing-mean drift beat zero in only 2 of 15 cases.
5. **Unlock schedules that end before today are dropped** (XRP).
6. **Bug fix:** GBM IC was scored on the ridge signal.

Every change moves the model toward the null or fixes measurement. None was tuned to raise an IC.
A zero-drift variant was also tested: average Brier was 0.2524 vs 0.2523 and R²_OOS vs zero
−0.0035 vs −0.0106, so it is not clearly better or worse. The pre-specified shrunk drift was kept,
and every forecast prints its drift and signal parts separately.

## Step 5: Results (non-overlapping, out-of-sample)

| asset | h | sample start | N non-overlap | IC signal (t) | IC momentum | hit model / long-only | net Sharpe model / mom / B&H | Brier model / clim | 80% PI coverage | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 1d | 2020-04-03 | 1647 | +0.019 (0.8) | -0.028 | 0.495 / 0.495 | -0.47 / -0.65 / 0.26 | 0.2503 / 0.2511 | 0.81 | no edge |
| BTC | 7d | 2020-04-09 | 231 | +0.047 (0.7) | -0.017 | 0.521 / 0.523 | 0.09 / -0.24 / 0.34 | 0.2502 / 0.2508 | 0.81 | no edge |
| BTC | 14d | 2020-04-15 | 114 | +0.045 (0.5) | +0.019 | 0.547 / 0.547 | 0.39 / 0.14 / 0.50 | 0.2515 / 0.2490 | 0.83 | no edge |
| BTC | 28d | 2020-01-12 | 45 | +0.191 (1.3) | +0.046 | 0.573 / 0.568 | 0.78 / 0.12 / 0.79 | 0.2500 / 0.2469 | 0.82 | no edge |
| BTC | 90d | 2020-01-12 | 12 | +0.160 (0.5) | +0.078 | 0.583 / 0.583 | 0.56 / -0.32 / 0.60 | 0.2542 / 0.2544 | 0.82 | no edge |
| ETH | 1d | 2020-05-28 | 1592 | +0.001 (0.1) | -0.032 | 0.503 / 0.504 | -0.35 / -0.80 / 0.07 | 0.2512 / 0.2508 | 0.81 | no edge |
| ETH | 7d | 2020-05-31 | 224 | +0.076 (1.1) | -0.010 | 0.511 / 0.514 | 0.06 / -0.83 / 0.27 | 0.2503 / 0.2506 | 0.79 | no edge |
| ETH | 14d | 2020-05-31 | 110 | +0.121 (1.3) | +0.024 | 0.518 / 0.507 | 0.44 / 0.17 / 0.31 | 0.2505 / 0.2531 | 0.79 | no edge |
| ETH | 28d | 2020-01-12 | 45 | +0.006 (0.0) | +0.186 | 0.503 / 0.503 | 0.14 / -0.08 / 0.20 | 0.2708 / 0.2567 | 0.75 | no edge |
| ETH | 90d | 2020-01-12 | 12 | -0.242 (-0.8) | -0.150 | 0.526 / 0.526 | 0.13 / -0.30 / 0.16 | 0.2528 / 0.2596 | 0.68 | no edge |
| SOL | 1d | 2021-03-16 | 1300 | +0.026 (0.9) | +0.002 | 0.507 / 0.497 | -0.02 / -0.38 / 0.57 | 0.2502 / 0.2501 | 0.79 | no edge |
| SOL | 7d | 2021-07-21 | 165 | -0.015 (-0.2) | +0.010 | 0.492 / 0.511 | 0.00 / 0.64 / 0.69 | 0.2515 / 0.2518 | 0.81 | no edge |
| SOL | 14d | 2021-07-21 | 81 | -0.026 (-0.2) | +0.127 | 0.463 / 0.537 | -0.38 / 1.17 / 0.69 | 0.2545 / 0.2553 | 0.82 | no edge |
| SOL | 28d | 2021-12-05 | 21 | +0.156 (0.7) | -0.072 | 0.432 / 0.493 | -1.45 / -0.88 / -0.57 | 0.2478 / 0.2509 | 0.82 | no edge |
| SOL | 90d | 2021-12-05 | 5 | +0.152 (0.3) | -0.084 | 0.418 / 0.418 | -0.53 / 0.67 / -0.50 | 0.2482 / 0.2573 | 0.78 | no edge (indicative only) |
| XRP | 1d | 2020-07-07 | 1552 | -0.003 (-0.1) | -0.061 | 0.484 / 0.484 | -0.05 / -0.60 / 0.41 | 0.2498 / 0.2504 | 0.80 | no edge |
| XRP | 7d | 2020-07-10 | 218 | -0.075 (-1.1) | -0.076 | 0.464 / 0.464 | 0.34 / -0.13 / 0.44 | 0.2497 / 0.2491 | 0.81 | no edge |
| XRP | 14d | 2020-07-10 | 107 | -0.085 (-0.9) | -0.082 | 0.446 / 0.446 | 0.34 / -0.08 / 0.40 | 0.2499 / 0.2478 | 0.84 | no edge |
| XRP | 28d | 2020-01-12 | 45 | +0.080 (0.5) | -0.196 | 0.450 / 0.450 | 0.34 / -0.71 / 0.39 | 0.2468 / 0.2487 | 0.81 | no edge |
| XRP | 90d | 2020-01-12 | 12 | +0.248 (0.8) | -0.016 | 0.464 / 0.464 | 0.39 / 0.29 / 0.41 | 0.2607 / 0.2636 | 0.72 | no edge |

How to read the table:
* **1–14d:** IC is near zero, and hit rates equal the long-only base rate. The features (funding
  proxy, stablecoin flows, onchain activity, cross-asset momentum) carry no exploitable short-horizon
  information beyond drift in this sample.
* **28d / 90d:** a few positive signal ICs (BTC 28d +0.19 over 45 windows) are not significant and
  flip sign by year (BTC 28d: 2026 −0.39). Treat these horizons as **regime tilts**: the useful
  output is the width and skew of the distribution, not the point.
* **Short-history ablations:** adding Hyperliquid OI, ETF flows and treasury buys gives no consistent
  improvement on the windows where they exist (see each report).

Each asset report has, per horizon:
* the data actually used (source, MCP ids, history, missingness, staleness, lag);
* top features with sign stability across folds;
* the full baseline table, regime and yearly stability, calibration and the ablation;
* the latest feature snapshot, the current forecast and the horizon's failure modes.

## Failure modes and limitations

* **Thin history:**
  * core samples start 2020 (perp premium from 2019-09, stablecoin CEX balance from 2019-05);
  * the 90d horizon has 12 non-overlapping test windows (5 for SOL);
  * the 2020–2026 sample holds roughly one and a half crypto cycles.
* **Stale or revised data:**
  * onchain and stablecoin series are lagged a day;
  * CEX wallet labels and unlock schedules are today's vintage, a look-ahead risk that likely
    flatters `stab_cexflow_*` and `unlock_next_*`;
  * ETF flows have a provisional last row.
* **Exchange-specific / wash volume:** the funding proxy and perp/spot volume share use Binance
  only, and Hyperliquid OI and liquidations are a single venue.
* **Regime shift:**
  * the drift learned in training dominates the forecast sign;
  * the shrinkage prior halves it but cannot know the next regime;
  * IC by year swings ±0.3 at 28d.
* **Missing slow fundamentals:** no DXY, real yields, long-history equities, options, CEX funding,
  CEX liquidations or native exchange flows in the MCP.
* **Cross-horizon consistency is not enforced.** Each horizon is a separate model with its own vol
  window, so a 28d interval can be wider than the 90d one after a vol spike (XRP today).

## Refreshing data / adding assets

The Blockworks MCP is reached through an MCP client (e.g. Claude with the Blockworks connector).
`python -m bwpredict.manifest <SYM>` prints the exact `timeseries_data` calls. Save each verbatim
JSON response into `data/raw/` and update `data/raw/_snapshot.json`. Assets listed in
`bwpredict/config.py` are BTC, ETH, SOL, XRP, BNB, DOGE, ADA, AVAX, LINK, TRX, SUI and HYPE (ids
verified). Snapshots are included for BTC, ETH, SOL and XRP. For another asset, resolve its
`assetID`, Binance spot `marketID` and perp id via `tabular_data` and add an entry. Missing sources
are skipped automatically and reported.
