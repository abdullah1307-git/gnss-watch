"""Spoofing indicators from per-aircraft kinematic plausibility.

Spoofing feeds the receiver a fake but healthy-looking signal, so NIC/NACp often stay
good. What gives it away is physics:

1. Position jumps: consecutive positions imply a speed the aircraft cannot fly.
2. GNSS vs barometric altitude: the difference between them changes slowly in real
   flight (it follows the weather), so a sudden step means the GNSS altitude is wrong.
3. Hotspots: several independent aircraft "jumping" into the same place at around the
   same time is the classic signature of a spoofer broadcasting one fake location.

A single jump can also be a decoding glitch, which is why hotspots (many aircraft) are
reported separately from individual events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SpoofingConfig
from ..grid import Grid
from ..parsing import gnss_mask

EARTH_RADIUS_NM = 3440.065

EVENT_COLUMNS = [
    "kind", "icao", "ts", "lat", "lon", "from_lat", "from_lon", "magnitude", "implied_kt",
    "gs_kt", "dt_s",
]
HOTSPOT_COLUMNS = [
    "window_start", "cell", "lat", "lon", "n_aircraft", "n_events", "first_ts", "last_ts",
    "aircraft",
]


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _with_previous(df: pd.DataFrame) -> pd.DataFrame:
    d = df[gnss_mask(df)].sort_values(["icao", "ts"], kind="stable").reset_index(drop=True)
    prev = d.groupby("icao", sort=False).shift(1)
    for col in ["ts", "lat", "lon", "gs", "alt_baro", "alt_geom", "on_ground"]:
        d[f"prev_{col}"] = prev[col]
    d["dt_s"] = d["ts"] - d["prev_ts"]
    return d


def position_jumps(df: pd.DataFrame, cfg: SpoofingConfig) -> pd.DataFrame:
    """Consecutive positions of one aircraft that imply an impossible ground speed."""
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    d = _with_previous(df)
    valid = (
        d["prev_ts"].notna()
        & (d["dt_s"] > 0)
        & (d["dt_s"] <= cfg.max_gap_s)
        & ~d["new_leg"]
    )
    d = d[valid]
    if d.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    dist = haversine_nm(d["prev_lat"], d["prev_lon"], d["lat"], d["lon"])
    implied = dist / d["dt_s"].to_numpy() * 3600.0
    gs_ref = np.fmax(d["gs"].to_numpy(), d["prev_gs"].to_numpy())
    limit = np.fmax(cfg.max_plausible_kt, np.nan_to_num(gs_ref, nan=0.0) * cfg.speed_factor
                    + cfg.speed_margin_kt)
    jump = (dist >= cfg.min_jump_nm) & (implied > limit)

    hits = d[jump]
    return pd.DataFrame(
        {
            "kind": "position_jump",
            "icao": hits["icao"].to_numpy(),
            "ts": hits["ts"].to_numpy(),
            "lat": hits["lat"].to_numpy(),
            "lon": hits["lon"].to_numpy(),
            "from_lat": hits["prev_lat"].to_numpy(),
            "from_lon": hits["prev_lon"].to_numpy(),
            "magnitude": dist[jump],
            "implied_kt": implied[jump],
            "gs_kt": gs_ref[jump],
            "dt_s": hits["dt_s"].to_numpy(),
        },
        columns=EVENT_COLUMNS,
    )


def altitude_divergence(df: pd.DataFrame, cfg: SpoofingConfig) -> pd.DataFrame:
    """Sudden steps in (GNSS altitude - barometric altitude) within a short time."""
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    d = _with_previous(df)
    delta = d["alt_geom"] - d["alt_baro"]
    prev_delta = d["prev_alt_geom"] - d["prev_alt_baro"]
    step = (delta - prev_delta).abs()
    airborne = ~d["on_ground"] & d["prev_on_ground"].eq(False)
    hit = (
        airborne
        & (d["dt_s"] > 0)
        & (d["dt_s"] <= cfg.alt_step_window_s)
        & (step > cfg.alt_step_ft)
    ).fillna(False).to_numpy(dtype=bool)
    hits = d[hit]
    return pd.DataFrame(
        {
            "kind": "geo_baro_step",
            "icao": hits["icao"].to_numpy(),
            "ts": hits["ts"].to_numpy(),
            "lat": hits["lat"].to_numpy(),
            "lon": hits["lon"].to_numpy(),
            "from_lat": hits["prev_lat"].to_numpy(),
            "from_lon": hits["prev_lon"].to_numpy(),
            "magnitude": step[hit].to_numpy(),
            "implied_kt": np.nan,
            "gs_kt": hits["gs"].to_numpy(),
            "dt_s": hits["dt_s"].to_numpy(),
        },
        columns=EVENT_COLUMNS,
    )


def cluster_jumps(jumps: pd.DataFrame, cfg: SpoofingConfig) -> np.ndarray:
    """Greedy spatio-temporal clustering of jump destinations.

    Jumps are visited in time order. A jump joins the nearest open cluster whose centroid
    is within ``hotspot_radius_nm`` and whose latest jump is at most ``hotspot_window_s``
    earlier; otherwise it starts a new cluster. Unlike fixed grid cells, this does not
    split a hotspot that happens to sit on a cell border.
    """
    order = np.argsort(jumps["ts"].to_numpy(), kind="stable")
    lat = jumps["lat"].to_numpy()[order]
    lon = jumps["lon"].to_numpy()[order]
    ts = jumps["ts"].to_numpy()[order]
    labels_sorted = np.empty(len(order), dtype=np.int64)
    c_lat: list[float] = []
    c_lon: list[float] = []
    c_n: list[int] = []
    c_last: list[float] = []
    for i in range(len(order)):
        best = -1
        if c_lat:
            is_open = ts[i] - np.asarray(c_last) <= cfg.hotspot_window_s
            d = haversine_nm(lat[i], lon[i], np.asarray(c_lat), np.asarray(c_lon))
            d = np.where(is_open, d, np.inf)
            k = int(np.argmin(d))
            if d[k] <= cfg.hotspot_radius_nm:
                best = k
        if best < 0:
            c_lat.append(lat[i])
            c_lon.append(lon[i])
            c_n.append(1)
            c_last.append(ts[i])
            labels_sorted[i] = len(c_lat) - 1
        else:
            n = c_n[best]
            c_lat[best] = (c_lat[best] * n + lat[i]) / (n + 1)
            c_lon[best] = (c_lon[best] * n + lon[i]) / (n + 1)
            c_n[best] = n + 1
            c_last[best] = ts[i]
            labels_sorted[i] = best
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels


def spoofing_hotspots(jumps: pd.DataFrame, grid: Grid, cfg: SpoofingConfig) -> pd.DataFrame:
    """Places where several distinct aircraft jumped to around the same time."""
    if jumps.empty:
        return pd.DataFrame(columns=HOTSPOT_COLUMNS)
    j = jumps.assign(cluster=cluster_jumps(jumps, cfg))
    spots = j.groupby("cluster", sort=False).agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        n_aircraft=("icao", "nunique"),
        n_events=("icao", "size"),
        first_ts=("ts", "min"),
        last_ts=("ts", "max"),
        aircraft=("icao", lambda s: ",".join(sorted(set(s)))),
    )
    spots = spots[spots["n_aircraft"] >= cfg.hotspot_min_aircraft].reset_index(drop=True)
    if spots.empty:
        return pd.DataFrame(columns=HOTSPOT_COLUMNS)
    spots["cell"] = grid.cells(spots["lat"].to_numpy(), spots["lon"].to_numpy())
    spots["window_start"] = (
        spots["first_ts"] // cfg.hotspot_window_s * cfg.hotspot_window_s
    ).astype("int64")
    return spots.sort_values("first_ts")[HOTSPOT_COLUMNS].reset_index(drop=True)
