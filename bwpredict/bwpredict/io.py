"""Load Blockworks MCP `timeseries_data` snapshots into pandas.

Every raw file under data/raw is the verbatim JSON returned by the Blockworks MCP
`timeseries_data` tool ({model, granularity, point_schema, series:[{key, points}]}).
File names are irrelevant: the loader indexes every file by (model, series key) and
concatenates/deduplicates windows that were fetched in several calls (the MCP caps one
call at 5,000 points).
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


@lru_cache(maxsize=1)
def _index(raw_dir: str = RAW_DIR) -> dict:
    idx: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        if os.path.basename(path).startswith("_"):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        fields = [c["field"] for c in doc["point_schema"]]
        for s in doc["series"]:
            if not s["points"]:
                continue
            df = pd.DataFrame(s["points"], columns=fields)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
            df = df.set_index("time").sort_index()
            idx.setdefault((doc["model"], s["key"]), []).append(df)
    out = {}
    for k, frames in idx.items():
        df = pd.concat(frames, axis=0)
        # When windows overlap prefer the later fetch (last), then align columns.
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[k] = df
    return out


def available() -> list[tuple[str, str]]:
    return sorted(_index().keys())


def load(model: str, key: str, fields: list[str] | None = None) -> pd.DataFrame | None:
    """Return a daily DataFrame for (model, series key), or None if no snapshot exists."""
    df = _index().get((model, key))
    if df is None:
        return None
    df = df.copy()
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    if fields is not None:
        missing = [f for f in fields if f not in df.columns]
        for f in missing:
            df[f] = float("nan")
        df = df[fields]
    return df.astype(float)


def snapshot_time() -> pd.Timestamp:
    """Wall-clock time at which the raw snapshot was taken (UTC, naive)."""
    meta = os.path.join(RAW_DIR, "_snapshot.json")
    if os.path.exists(meta):
        with open(meta) as fh:
            return pd.Timestamp(json.load(fh)["fetched_at_utc"]).tz_localize(None)
    return pd.Timestamp.utcnow().tz_localize(None)
