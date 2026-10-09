"""Point-in-time guarantees.

1. Truncation invariance: rebuilding the panel and features with the snapshot clock set to
   an earlier date must reproduce exactly the same feature values for every row on or
   before that date. Any centred window, backfill or negative shift would break this.
2. Labels use only future prices from completed bars and match the documented definition.
3. A deliberately leaked target is detected by the walk-forward (sanity of the harness).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bwpredict import dataset, io  # noqa: E402
from bwpredict import model as M  # noqa: E402
from bwpredict.features import build_features  # noqa: E402

CUTS = ["2021-06-30", "2023-03-15", "2025-11-20"]


def _build(symbol, h, cut=None, monkeypatch=None):
    if cut is not None:
        monkeypatch.setattr(dataset, "snapshot_time", lambda: pd.Timestamp(cut) + pd.Timedelta(days=1))
    p, _ = dataset.build_panel(symbol)
    fs = build_features(p, symbol, h)
    return p, fs


@pytest.mark.parametrize("symbol", ["BTC", "SOL"])
@pytest.mark.parametrize("h", [1, 7, 28, 90])
def test_truncation_invariance(symbol, h, monkeypatch):
    p_full, fs_full = _build(symbol, h)
    for cut in CUTS:
        p_cut, fs_cut = _build(symbol, h, cut, monkeypatch)
        assert p_cut.index.max() <= pd.Timestamp(cut)
        cols = [c for c in fs_cut.X.columns if not c.startswith("unlock_next")]  # schedule is forward-known by design
        a = fs_cut.X[cols]
        b = fs_full.X.loc[a.index, cols]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)
        # sigma used for the target scaling / flat band is also point-in-time
        pd.testing.assert_series_equal(p_cut[f"sigma_{h}"], p_full.loc[p_cut.index, f"sigma_{h}"])


def test_labels_definition():
    p, _ = dataset.build_panel("BTC")
    h = 7
    lc = np.log(p["price__close"])
    y = lc.shift(-h) - lc
    pd.testing.assert_series_equal(p[f"y_{h}"], y, check_names=False)
    assert p[f"y_{h}"].iloc[-h:].isna().all()          # no label without a completed bar at T+h
    band = 0.5 * p[f"sigma_{h}"]
    lab = p[f"label_{h}"].dropna()
    assert set(lab.unique()) <= {-1.0, 0.0, 1.0}
    yy, bb = y.loc[lab.index], band.loc[lab.index]
    assert ((lab == 1) == (yy > bb)).all() and ((lab == -1) == (yy < -bb)).all()


def test_no_partial_bar(monkeypatch):
    monkeypatch.setattr(dataset, "snapshot_time", lambda: pd.Timestamp("2024-05-10 13:00"))
    p, _ = dataset.build_panel("BTC")
    assert p.index.max() == pd.Timestamp("2024-05-09")


def test_leak_is_detected():
    p, _ = dataset.build_panel("BTC")
    h = 7
    fs = build_features(p, "BTC", h)
    core = [f for f, t in fs.tiers.items() if t == "core"]
    X = fs.X[core].copy()
    X["leak"] = p[f"y_{h}"].fillna(0)
    D = M.make_design(p, X, h)
    wf = M.walk_forward(D, ["leak"], h, with_gbm=False)
    assert M.evaluate(wf.oos, h)["overlapping"]["ic"] > 0.8
