"""Live snapshots from the free adsb.lol API (https://api.adsb.lol/docs).

The API answers circle queries of up to 250 NM, so a region is covered by tiling it with
overlapping circles and de-duplicating aircraft. Requests are spaced out to be polite.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator

import pandas as pd
import requests

from ..config import BBox
from ..parsing import parse_live_aircraft

log = logging.getLogger(__name__)

API_BASE = "https://api.adsb.lol"
USER_AGENT = "gnss-watch/0.1 (+https://github.com/abdullah1307-git/gnss-watch)"
MAX_RADIUS_NM = 250


def fetch_point(
    lat: float, lon: float, radius_nm: int = MAX_RADIUS_NM, session: requests.Session | None = None
) -> dict:
    radius_nm = max(1, min(int(radius_nm), MAX_RADIUS_NM))
    s = session or requests.Session()
    url = f"{API_BASE}/v2/point/{lat:.4f}/{lon:.4f}/{radius_nm}"
    resp = s.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def tile_centers(bbox: BBox, radius_nm: int = MAX_RADIUS_NM) -> Iterator[tuple[float, float]]:
    """Circle centres whose circles fully cover the bbox (square grid, spacing r*sqrt(2))."""
    step_lat = radius_nm * math.sqrt(2) / 60.0  # 1 degree of latitude is about 60 NM
    lat = bbox.min_lat + step_lat / 2
    while lat - step_lat / 2 < bbox.max_lat:
        step_lon = step_lat / max(math.cos(math.radians(min(abs(lat) + step_lat / 2, 89))), 1e-6)
        lon = bbox.min_lon + step_lon / 2
        while lon - step_lon / 2 < bbox.max_lon:
            yield (min(lat, 90.0), max(-180.0, min(lon, 180.0)))
            lon += step_lon
        lat += step_lat


def fetch_region(
    bbox: BBox,
    radius_nm: int = MAX_RADIUS_NM,
    pause_s: float = 1.0,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """One snapshot of all aircraft in ``bbox``."""
    s = session or requests.Session()
    frames = []
    for i, (lat, lon) in enumerate(tile_centers(bbox, radius_nm)):
        if i:
            time.sleep(pause_s)
        try:
            frames.append(parse_live_aircraft(fetch_point(lat, lon, radius_nm, s)))
        except requests.RequestException as exc:
            log.warning("tile %.2f,%.2f failed: %s", lat, lon, exc)
    if not frames:
        return parse_live_aircraft({"ac": []})
    df = pd.concat(frames, ignore_index=True)
    inside = (
        df["lat"].between(bbox.min_lat, bbox.max_lat) & df["lon"].between(bbox.min_lon, bbox.max_lon)
    )
    df = df[inside].sort_values("ts").drop_duplicates("icao", keep="last")
    return df.reset_index(drop=True)
