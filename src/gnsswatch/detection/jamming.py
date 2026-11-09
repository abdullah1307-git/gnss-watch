"""Jamming indicator: share of aircraft with degraded GNSS integrity, per cell and window.

Jamming does not move aircraft; it makes their GNSS receivers report low integrity (NIC)
and accuracy (NACp), or lose the fix entirely. One aircraft with a bad receiver proves
nothing, but when many independent aircraft in the same area degrade at the same time,
interference is the likely cause. So the indicator:

1. counts aircraft, not reports (an aircraft is degraded in a cell-window if most of its
   reports there are),
2. removes aircraft with faulty receivers using a peer check (degraded where the other
   aircraft around them are healthy), and
3. needs a minimum number of aircraft and of degraded aircraft before raising a level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import JammingConfig
from ..grid import Grid
from ..parsing import gnss_mask

CELL_COLUMNS = [
    "window_start", "cell", "lat", "lon", "n_aircraft", "n_degraded", "degraded_ratio", "level",
]
OVERALL_COLUMNS = [
    "cell", "lat", "lon", "n_aircraft", "n_degraded", "degraded_ratio", "level",
    "peak_ratio", "windows_high",
]
LEVELS = ("insufficient", "low", "medium", "high")


@dataclass
class JammingResult:
    cells: pd.DataFrame
    faulty_aircraft: set[str]


def degraded_reports(df: pd.DataFrame, cfg: JammingConfig) -> pd.Series:
    """True where a report shows degraded GNSS, False where healthy, NA where unknown."""
    nic_bad = df["nic"] < cfg.nic_bad_below
    nacp_bad = df["nac_p"] < cfg.nacp_bad_below
    known = df["nic"].notna() | df["nac_p"].notna() | df["gps_lost"]
    degraded = nic_bad | nacp_bad | df["gps_lost"]
    return degraded.where(known)


def classify(n_aircraft, n_degraded, ratio, cfg: JammingConfig) -> np.ndarray:
    n_aircraft, n_degraded, ratio = (np.asarray(x) for x in (n_aircraft, n_degraded, ratio))
    enough_evidence = n_degraded >= cfg.min_degraded_aircraft
    return np.select(
        [
            n_aircraft < cfg.min_aircraft,
            enough_evidence & (ratio >= cfg.high_ratio),
            enough_evidence & (ratio >= cfg.medium_ratio),
        ],
        ["insufficient", "high", "medium"],
        default="low",
    )


def per_aircraft_status(df: pd.DataFrame, grid: Grid, cfg: JammingConfig) -> pd.DataFrame:
    """One row per (window_start, cell, icao) with a boolean ``degraded``."""
    d = df[gnss_mask(df) & ~df["on_ground"].to_numpy()].copy()
    d["degraded"] = degraded_reports(d, cfg)
    d = d[d["degraded"].notna()]
    if d.empty:
        return pd.DataFrame(columns=["window_start", "cell", "icao", "degraded"])
    d["degraded"] = d["degraded"].astype(bool)
    d["cell"] = grid.cells(d["lat"].to_numpy(), d["lon"].to_numpy())
    d["window_start"] = (d["ts"] // cfg.window_s * cfg.window_s).astype("int64")
    share = d.groupby(["window_start", "cell", "icao"], sort=False)["degraded"].mean()
    return (share >= cfg.degraded_report_share).rename("degraded").reset_index()


def faulty_aircraft(status: pd.DataFrame, cfg: JammingConfig) -> set[str]:
    """Aircraft whose degradation is repeatedly contradicted by healthy peers."""
    if status.empty:
        return set()
    key = ["window_start", "cell"]
    totals = status.groupby(key)["degraded"].agg(n="size", n_bad="sum").reset_index()
    s = status.merge(totals, on=key, how="left")
    peers = s["n"] - 1
    peers_bad = s["n_bad"] - s["degraded"].astype(int)
    comparable = s["degraded"] & (peers >= cfg.peer_min_aircraft)
    contradicted = comparable & (peers_bad / peers.where(peers > 0) < 0.5)
    per = pd.DataFrame({"icao": s["icao"], "comparable": comparable, "contradicted": contradicted})
    per = per.groupby("icao")[["comparable", "contradicted"]].sum()
    bad = per[
        (per["contradicted"] >= cfg.faulty_min_contradicted)
        & (per["contradicted"] >= cfg.faulty_contradicted_share * per["comparable"])
    ]
    return set(bad.index)


def _add_centers(cells: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    centers = {c: grid.center(c) for c in cells["cell"].unique()}
    cells["lat"] = cells["cell"].map(lambda c: centers[c][0]).astype(float)
    cells["lon"] = cells["cell"].map(lambda c: centers[c][1]).astype(float)
    return cells


def analyze_jamming(df: pd.DataFrame, grid: Grid, cfg: JammingConfig) -> JammingResult:
    status = per_aircraft_status(df, grid, cfg)
    faulty = faulty_aircraft(status, cfg)
    status = status[~status["icao"].isin(faulty)]
    if status.empty:
        return JammingResult(pd.DataFrame(columns=CELL_COLUMNS), faulty)

    cells = (
        status.groupby(["window_start", "cell"], sort=True)["degraded"]
        .agg(n_aircraft="size", n_degraded="sum")
        .reset_index()
    )
    cells["n_degraded"] = cells["n_degraded"].astype("int64")
    cells["degraded_ratio"] = cells["n_degraded"] / cells["n_aircraft"]
    cells["level"] = classify(cells["n_aircraft"], cells["n_degraded"], cells["degraded_ratio"], cfg)
    return JammingResult(_add_centers(cells, grid)[CELL_COLUMNS], faulty)


def jamming_cells(df: pd.DataFrame, grid: Grid, cfg: JammingConfig) -> pd.DataFrame:
    """Per (time window, cell): aircraft seen, aircraft degraded, ratio and level."""
    return analyze_jamming(df, grid, cfg).cells


def aggregate_cells(cells: pd.DataFrame, grid: Grid, cfg: JammingConfig) -> pd.DataFrame:
    """Collapse all windows into one row per cell (for a whole-period map).

    Counts are aircraft-windows, so a cell that was jammed for 3 of 24 hours shows a ratio
    near 1/8. ``peak_ratio`` and ``windows_high`` keep short, intense events visible, and a
    cell that was "high" in any window is shown at least as "medium" overall.
    """
    if cells.empty:
        return pd.DataFrame(columns=OVERALL_COLUMNS)
    agg = cells.groupby("cell", sort=True).agg(
        n_aircraft=("n_aircraft", "sum"), n_degraded=("n_degraded", "sum")
    )
    usable = cells[cells["level"] != "insufficient"]
    peak = usable.groupby("cell")["degraded_ratio"].max().rename("peak_ratio")
    high = (usable["level"] == "high").groupby(usable["cell"]).sum().rename("windows_high")
    out = agg.join(peak).join(high).reset_index()
    out["peak_ratio"] = out["peak_ratio"].fillna(0.0)
    out["windows_high"] = out["windows_high"].fillna(0).astype("int64")
    out["degraded_ratio"] = out["n_degraded"] / out["n_aircraft"]
    out["level"] = classify(out["n_aircraft"], out["n_degraded"], out["degraded_ratio"], cfg)
    bump = (out["windows_high"] > 0) & (out["level"] == "low")
    out.loc[bump, "level"] = "medium"
    return _add_centers(out, grid)[OVERALL_COLUMNS]
