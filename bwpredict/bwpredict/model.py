"""Models, purged walk-forward validation and evaluation metrics.

Target: z_T = y_h(T) / sigma_h(T), the forward h-day log return in units of trailing
vol scaled to the horizon. Modelling in vol units keeps the target roughly stationary
across a 2019-2026 sample in which BTC vol more than halved; predictions are mapped back
with mu = z_hat * sigma_h.

Predictive distribution: mu + sigma_h * e, with e drawn from the empirical standardised
training residuals. P(up), the ternary probabilities (flat band = +-0.5 in z units, which
is exactly the label definition) and the prediction interval all come from this one
distribution, so they are mutually consistent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C

# Up to kappa=100 the model can collapse to (almost) the prevailing mean when features carry no signal.
KAPPAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


# ----------------------------------------------------------------------------- models
PRIOR_SHARPE_SD = 0.5   # prior sd of the annualised drift Sharpe, centred on zero


def drift_weight(n: int, h: int) -> float:
    """Posterior weight on the sample drift of the vol-scaled target (var ~ 1 per window).

    Prior: drift (in z units per h-day window) ~ N(0, tau^2), tau = PRIOR_SHARPE_SD * sqrt(h/365).
    Data: n daily rows with h-day overlapping labels ~ n/h independent windows.
    """
    tau2 = (PRIOR_SHARPE_SD ** 2) * h / 365
    n_eff = max(n / h, 1.0)
    return tau2 / (tau2 + 1.0 / n_eff)


class Ridge:
    """Closed-form ridge on train-standardised features; penalty alpha = kappa * n.

    The intercept (drift) is the training mean of the target shrunk toward zero by a
    Bayesian prior (drift_weight); coefficients are fit on the demeaned target.
    """

    def __init__(self, kappa: float, h: int = 1):
        self.kappa, self.h = kappa, h

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1
        Z = (X - self.mu_) / self.sd_
        n, k = Z.shape
        ybar = y.mean()
        self.drift_w_ = drift_weight(n, self.h)
        self.y0_ = ybar * self.drift_w_
        self.coef_ = np.linalg.solve(Z.T @ Z + self.kappa * n * np.eye(k), Z.T @ (y - ybar))
        return self

    def signal(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu_) / self.sd_) @ self.coef_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.y0_ + self.signal(X)


def _gap_split(t_index: np.ndarray, h: int, frac: float = 0.25):
    """Inner time split of a training window with a 2h purge gap."""
    n = len(t_index)
    cut = t_index[int(n * (1 - frac))]
    tr = t_index <= cut - 2 * h
    va = t_index > cut
    return tr, va


def choose_kappa(X, y, t_days, h) -> float:
    """Inner purged time split + one-standard-error rule.

    The standard error of the validation-MSE difference is computed from h-day blocks so
    that overlapping labels are not counted as independent. We take the most regularised
    kappa whose validation MSE is within 1 SE of the best one; with weak signals this lands
    near the prevailing-drift model, which is the intended conservative default.
    """
    tr, va = _gap_split(t_days, h)
    if tr.sum() < 200 or va.sum() < 60:
        return KAPPAS[-1]
    errs = []
    for k in KAPPAS:
        m = Ridge(k, h).fit(X[tr], y[tr])
        errs.append((y[va] - m.predict(X[va])) ** 2)
    errs = np.array(errs)                      # kappa x n_val
    mses = errs.mean(1)
    best = int(mses.argmin())
    blk = (t_days[va] - t_days[va].min()) // max(h, 5)
    se = []
    for i in range(len(KAPPAS)):
        d = pd.Series(errs[i] - errs[best]).groupby(blk).mean()
        se.append(d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 2 else np.inf)
    ok = [i for i in range(len(KAPPAS)) if mses[i] <= mses[best] + se[i]]
    return KAPPAS[max(ok)]


def fit_ridge(X, y, t_days, h):
    k = choose_kappa(X, y, t_days, h)
    return Ridge(k, h).fit(X, y)


def fit_gbm(X, y):
    n = len(y)
    m = HistGradientBoostingRegressor(max_depth=2, max_iter=150, learning_rate=0.03,
                                      min_samples_leaf=max(60, n // 15), l2_regularization=1.0,
                                      random_state=0)
    return m.fit(X, y)


# ----------------------------------------------------------------------------- distribution
def dist_stats(zhat: np.ndarray, resid: np.ndarray, band: float = C.FLAT_BAND_K):
    """Probabilities/quantiles from zhat + e, e ~ empirical residuals (demeaned, so the
    predictive mean is exactly zhat; residual skew and fat tails are kept)."""
    e = np.sort(resid - resid.mean())
    n = len(e)
    def cdf(x):  # P(zhat + e <= x) for each zhat
        return np.searchsorted(e, x - zhat, side="right") / n
    p_up = 1 - cdf(0.0)
    p3_up = 1 - cdf(band)
    p3_dn = cdf(-band)
    lo_q, hi_q = (1 - C.PI_LEVEL) / 2, 1 - (1 - C.PI_LEVEL) / 2
    q_lo, q_hi = np.quantile(e, lo_q), np.quantile(e, hi_q)
    return dict(p_up=p_up, p3_up=p3_up, p3_dn=p3_dn, p3_fl=1 - p3_up - p3_dn,
                z_lo=zhat + q_lo, z_hi=zhat + q_hi)


# ----------------------------------------------------------------------------- walk-forward
@dataclass
class WFResult:
    oos: pd.DataFrame
    coefs: pd.DataFrame          # fold x feature (standardised ridge coefficients)
    kappas: list
    folds: list = field(default_factory=list)


def make_design(panel: pd.DataFrame, X: pd.DataFrame, h: int):
    D = X.copy()
    D["y"] = panel[f"y_{h}"]
    D["sigma"] = panel[f"sigma_{h}"]
    D["yz"] = D["y"] / D["sigma"]
    D["label"] = panel[f"label_{h}"]
    lc = np.log(panel["price__close"])
    D["mom"] = lc - lc.shift(h)
    D["trail90"] = lc - lc.shift(90)
    feat_ok = X.notna().all(axis=1) & D["sigma"].notna() & D["mom"].notna()
    D = D[feat_ok]
    return D


def walk_forward(D: pd.DataFrame, feats: list[str], h: int, min_train: int | None = None,
                 retrain: int | None = None, with_gbm: bool = True) -> WFResult:
    w = C.WALK[h]
    min_train = min_train or w["min_train"]
    retrain = retrain or w["retrain"]
    emb = h  # embargo = one full horizon, on top of the h-day label purge
    t0 = D.index.min()
    days = ((D.index - t0) / pd.Timedelta(days=1)).astype(int).to_numpy()
    lab = D["yz"].notna().to_numpy()
    last_lab_day = days[lab].max()
    first_test = days.min() + min_train + h + emb
    rows, coefs, kappas, folds = [], [], [], []
    X_all = D[feats].to_numpy(float)
    yz_all = D["yz"].to_numpy(float)
    for b0 in range(first_test, last_lab_day + 1, retrain):
        b1 = b0 + retrain
        tr = lab & (days + h + emb <= b0)
        te = lab & (days >= b0) & (days < b1)
        if tr.sum() < 250 or te.sum() == 0:
            continue
        Xtr, ytr = X_all[tr], yz_all[tr]
        rid = fit_ridge(Xtr, ytr, days[tr], h)
        resid = ytr - rid.predict(Xtr)
        zr = rid.predict(X_all[te])
        st = dist_stats(zr, resid)
        out = D.loc[te, ["y", "yz", "sigma", "label", "mom", "trail90"]].copy()
        out["zhat"] = zr
        out["zsig"] = rid.signal(X_all[te])          # signal part, excluding the drift term
        out["drift"] = rid.y0_
        out["mu"] = zr * out["sigma"]
        for k, v in st.items():
            out[k] = v
        out["p_clim"] = (ytr > 0).mean()
        out["mu_pm"] = ytr.mean() * out["sigma"]   # prevailing-mean (unshrunk drift) forecast
        lab_tr = D.loc[tr, "label"].to_numpy()
        out["c_dn"], out["c_fl"], out["c_up"] = [(lab_tr == c).mean() for c in (-1, 0, 1)]
        if with_gbm:
            g = fit_gbm(Xtr, ytr)
            gz = g.predict(X_all[te])
            out["zhat_gbm"] = gz
            out["mu_gbm"] = gz * out["sigma"]
            out["p_up_gbm"] = dist_stats(gz, ytr - g.predict(Xtr))["p_up"]
        out["fold"] = len(folds)
        rows.append(out)
        coefs.append(pd.Series(rid.coef_, index=feats, name=len(folds)))
        kappas.append(rid.kappa)
        folds.append(dict(test_start=str(D.index[te].min().date()), test_end=str(D.index[te].max().date()),
                          n_train=int(tr.sum()), n_test=int(te.sum()), kappa=rid.kappa,
                          train_end=str(D.index[tr].max().date())))
    oos = pd.concat(rows) if rows else pd.DataFrame()
    return WFResult(oos=oos, coefs=pd.DataFrame(coefs), kappas=kappas, folds=folds)


# ----------------------------------------------------------------------------- metrics
def _ic(a, b):
    if len(a) < 5 or np.std(a) == 0:
        return np.nan
    return spearmanr(a, b).correlation


def _hit(pred, y):
    m = (y != 0) & (pred != 0)
    return float((np.sign(pred[m]) == np.sign(y[m])).mean()) if m.sum() else np.nan


def _offsets(oos: pd.DataFrame, h: int):
    """Yield non-overlapping subsamples (one per start offset) of the OOS predictions."""
    t0 = oos.index.min()
    d = ((oos.index - t0) / pd.Timedelta(days=1)).astype(int)
    for k in range(h):
        yield oos[(d % h) == k]


def _strategy(pos, y, cost):
    net = pos * y - np.where(pos != 0, cost, 0.0)
    return net


def metric_block(o: pd.DataFrame, h: int, mu_col="mu", p_col="p_up") -> dict:
    y = o["y"].to_numpy()
    mu = o[mu_col].to_numpy()
    mom = o["mom"].to_numpy()
    # IC is computed on vol-standardised quantities (forecast z vs realised z, momentum / sigma).
    # Ranking raw mu = zhat * sigma would mostly rank trailing volatility whenever the drift term
    # dominates zhat, which measures "did high-vol periods go up", not forecasting skill.
    sig = o["sigma"].to_numpy()
    zf, yz = mu / sig, y / sig
    # Headline IC = signal-only (excludes the per-fold drift level, which otherwise turns pooled
    # IC into a test of whether last window's drift persisted).
    zs = o["zsig"].to_numpy() if "zsig" in o and mu_col == "mu" else zf
    out = dict(n=len(o), ic=_ic(zs, yz), ic_full=_ic(zf, yz), ic_mom=_ic(mom / sig, yz), ic_raw=_ic(mu, y),
               hit=_hit(mu, y), hit_mom=_hit(mom, y),
               hit_long=float((y > 0).mean()) if len(y) else np.nan)
    out["r2_oos_vs_zero"] = 1 - np.sum((y - mu) ** 2) / np.sum(y ** 2) if len(y) else np.nan
    if "mu_pm" in o:
        pm = o["mu_pm"].to_numpy()
        out["r2_oos_vs_mean"] = 1 - np.sum((y - mu) ** 2) / np.sum((y - pm) ** 2)
        out["hit_pm"] = _hit(pm, y)
        out["r2_pm_vs_zero"] = 1 - np.sum((y - pm) ** 2) / np.sum(y ** 2)
    p = o[p_col].to_numpy()
    up = (y > 0).astype(float)
    out["brier"] = float(np.mean((p - up) ** 2))
    out["brier_clim"] = float(np.mean((o["p_clim"].to_numpy() - up) ** 2))
    out["brier_half"] = float(np.mean((0.5 - up) ** 2))
    if "p3_up" in o:
        L = o["label"].to_numpy()
        P = np.c_[o["p3_dn"], o["p3_fl"], o["p3_up"]].clip(1e-4, 1)
        Cc = np.c_[o["c_dn"], o["c_fl"], o["c_up"]].clip(1e-4, 1)
        j = (L + 1).astype(int)
        out["logloss3"] = float(-np.mean(np.log(P[np.arange(len(L)), j])))
        out["logloss3_clim"] = float(-np.mean(np.log(Cc[np.arange(len(L)), j])))
    if "z_lo" in o and mu_col == "mu":
        out["pi_coverage"] = float(((yz >= o["z_lo"].to_numpy()) & (yz <= o["z_hi"].to_numpy())).mean())
    cost = C.COST_BPS[h] / 1e4
    s_mod = _strategy(np.sign(mu), y, cost)
    s_mom = _strategy(np.sign(mom), y, cost)
    s_bh = y.copy()  # passive buy-and-hold: no per-window trading cost
    for name, s in (("model", s_mod), ("mom", s_mom), ("bh", s_bh)):
        out[f"net_{name}_bps"] = float(np.mean(s) * 1e4) if len(s) else np.nan
        out[f"net_{name}_hit"] = float((s > 0).mean()) if len(s) else np.nan
    return out


def evaluate(oos: pd.DataFrame, h: int) -> dict:
    res = {"overlapping": metric_block(oos, h)}
    # Non-overlapping: step = h; average metrics across the h possible start offsets.
    blocks = [metric_block(sub, h) for sub in _offsets(oos, h) if len(sub) >= 5]
    if blocks:
        nb = pd.DataFrame(blocks)
        res["nonoverlap"] = nb.mean(numeric_only=True).to_dict()
        res["nonoverlap"]["n"] = int(nb["n"].min())
        # t-stats on the offset-0 path (independent windows).
        sub = next(_offsets(oos, h))
        cost = C.COST_BPS[h] / 1e4
        net = _strategy(np.sign(sub["mu"].to_numpy()), sub["y"].to_numpy(), cost)
        netm = _strategy(np.sign(sub["mom"].to_numpy()), sub["y"].to_numpy(), cost)
        res["nonoverlap"]["net_model_t"] = float(net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(len(net)))
        res["nonoverlap"]["net_mom_t"] = float(netm.mean() / (netm.std(ddof=1) + 1e-12) * np.sqrt(len(netm)))
        ic0 = res["nonoverlap"]["ic"]
        n0 = res["nonoverlap"]["n"]
        res["nonoverlap"]["ic_t"] = float(ic0 * np.sqrt(max(n0 - 2, 1)) / np.sqrt(max(1 - ic0 ** 2, 1e-6))) if n0 > 2 else np.nan
        res["nonoverlap"]["sharpe_model"] = float(net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(365 / h))
        res["nonoverlap"]["sharpe_mom"] = float(netm.mean() / (netm.std(ddof=1) + 1e-12) * np.sqrt(365 / h))
        bh = sub["y"].to_numpy()
        res["nonoverlap"]["sharpe_bh"] = float(bh.mean() / (bh.std(ddof=1) + 1e-12) * np.sqrt(365 / h))
    if "mu_gbm" in oos:
        # GBM has no separate drift term: its IC is computed on its own full forecast (mu_gbm / sigma).
        g = oos.drop(columns=["zsig"]).rename(columns={"mu": "mu_r", "mu_gbm": "mu", "p_up": "p_up_r",
                                                      "p_up_gbm": "p_up"})
        gb = [metric_block(sub, h) for sub in _offsets(g, h) if len(sub) >= 5]
        res["gbm_overlapping"] = metric_block(g, h)
        res["gbm_nonoverlap"] = pd.DataFrame(gb).mean(numeric_only=True).to_dict() if gb else {}
    # Regimes from information known at T: trailing 90d log return.
    reg = np.where(oos["trail90"] > 0.15, "bull", np.where(oos["trail90"] < -0.15, "bear", "chop"))
    res["regimes"] = {}
    for r in ("bull", "bear", "chop"):
        sub = oos[reg == r]
        if len(sub) >= 20:
            mb = metric_block(sub, h)
            res["regimes"][r] = dict(n=len(sub), n_indep=round(len(sub) / h, 1), ic=mb["ic"], hit=mb["hit"],
                                     hit_mom=mb["hit_mom"], hit_long=mb["hit_long"])
    # Calibration table (quintiles of P(up)).
    q = pd.qcut(oos["p_up"], 5, duplicates="drop")
    cal = oos.groupby(q, observed=True).apply(lambda d: pd.Series(dict(p_mean=d["p_up"].mean(),
                                                                       realised=(d["y"] > 0).mean(),
                                                                       n=len(d))))
    res["calibration"] = cal.reset_index(drop=True).to_dict(orient="records")
    # IC by calendar year (stability).
    res["ic_by_year"] = {int(y): _ic(g["zsig"].to_numpy(), g["yz"].to_numpy())
                         for y, g in oos.groupby(oos.index.year) if len(g) >= 30}
    return res


def coef_stability(coefs: pd.DataFrame, final: pd.Series) -> pd.DataFrame:
    if coefs.empty:
        return pd.DataFrame()
    sign_final = np.sign(final)
    agree = (np.sign(coefs) == sign_final).mean()
    return pd.DataFrame(dict(coef_final=final, coef_mean_folds=coefs.mean(), sign_agreement=agree,
                             pos_share=(coefs > 0).mean())).sort_values("coef_final", key=np.abs, ascending=False)


# ----------------------------------------------------------------------------- final fit
def final_fit(D: pd.DataFrame, feats: list[str], h: int) -> dict:
    """Fit on every labelled row and score the latest feature row (decision time = now)."""
    lab = D["yz"].notna().to_numpy()
    t0 = D.index.min()
    days = ((D.index - t0) / pd.Timedelta(days=1)).astype(int).to_numpy()
    X = D[feats].to_numpy(float)
    y = D["yz"].to_numpy(float)
    rid = fit_ridge(X[lab], y[lab], days[lab], h)
    resid = y[lab] - rid.predict(X[lab])
    last = D.index.max()
    xl = D.loc[[last], feats].to_numpy(float)
    z = rid.predict(xl)
    st = {k: float(v[0]) for k, v in dist_stats(z, resid).items()}
    sigma = float(D.loc[last, "sigma"])
    g = fit_gbm(X[lab], y[lab])
    zg = float(g.predict(xl)[0])
    return dict(asof=str(last.date()), kappa=rid.kappa, zhat=float(z[0]), sigma=sigma, mu=float(z[0]) * sigma,
                drift_mu=float(rid.y0_) * sigma, signal_mu=float(rid.signal(xl)[0]) * sigma, drift_w=rid.drift_w_,
                mu_gbm=zg * sigma, drift_only_mu=float(y[lab].mean()) * sigma,
                coef=pd.Series(rid.coef_, index=feats), n_train=int(lab.sum()),
                train_end=str(D.index[lab].max().date()), **st)
