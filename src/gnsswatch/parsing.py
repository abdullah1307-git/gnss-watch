"""Turn raw readsb / adsb.lol data into one tidy table of position reports.

Every data source ends up as a pandas DataFrame with the columns in ``COLUMNS``:

    icao       str    ICAO 24-bit address (lowercase hex)
    ts         float  Unix time of the position (seconds)
    lat, lon   float  WGS84 position
    alt_baro   float  Barometric altitude in feet (NaN if unknown or on ground)
    alt_geom   float  GNSS (geometric) altitude in feet (NaN if unknown)
    on_ground  bool
    gs         float  Ground speed in knots
    track      float  Track in degrees
    nic        float  Navigation Integrity Category (NaN if unknown)
    nac_p      float  Navigation Accuracy Category for position (NaN if unknown)
    source     str    Position source, e.g. "adsb_icao", "mlat" ("" if unknown)
    stale      bool   readsb flag: no position for 20 s before this one
    new_leg    bool   readsb flag: start of a new flight leg
    gps_lost   bool   readsb live flag: aircraft recently lost GNSS (gpsOkBefore present)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .config import BBox

COLUMNS = [
    "icao", "ts", "lat", "lon", "alt_baro", "alt_geom", "on_ground", "gs", "track",
    "nic", "nac_p", "source", "stale", "new_leg", "gps_lost",
]

# Positions derived from the aircraft's own GNSS receiver. MLAT and TIS-B positions are
# computed on the ground, so they say nothing about the aircraft's GNSS health.
GNSS_SOURCES = frozenset(
    {"adsb_icao", "adsb_icao_nt", "adsr_icao", "adsb_other", "adsr_other", ""}
)

_NAN = math.nan


def _num(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return _NAN
    if isinstance(value, (int, float)):
        return float(value)
    return _NAN


def parse_trace(doc: dict, bbox: BBox | None = None) -> list[tuple]:
    """Parse one readsb ``trace_full_<icao>.json`` document into row tuples.

    Trace entry layout (readsb README-json):
        0 seconds after doc["timestamp"], 1 lat, 2 lon, 3 altitude ft | "ground" | null,
        4 ground speed kt, 5 track deg, 6 flags bitfield, 7 vertical rate,
        8 aircraft details object or null, 9 position source, 10 geometric altitude, ...
    Flags: 1 = stale position, 2 = start of new leg, 8 = altitude is geometric.

    NIC / NACp only appear in the sparse details objects, so they are carried forward.
    """
    icao = str(doc.get("icao", "")).lower().lstrip("~")
    base = float(doc.get("timestamp", 0.0))
    nic = _NAN
    nac_p = _NAN
    rows: list[tuple] = []

    for entry in doc.get("trace") or ():
        if not isinstance(entry, list) or len(entry) < 7:
            continue

        details = entry[8] if len(entry) > 8 else None
        if isinstance(details, dict):
            if "nic" in details:
                nic = _num(details.get("nic"))
            if "nac_p" in details:
                nac_p = _num(details.get("nac_p"))

        lat, lon = entry[1], entry[2]
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)
        if bbox is not None and not bbox.contains(lat, lon):
            continue

        flags = int(entry[6] or 0)
        alt = entry[3]
        on_ground = alt == "ground"
        alt_value = _num(alt)
        alt_geom = _num(entry[10]) if len(entry) > 10 else _NAN
        if flags & 8:  # the altitude column is geometric, not barometric
            if math.isnan(alt_geom):
                alt_geom = alt_value
            alt_baro = _NAN
        else:
            alt_baro = alt_value

        source = entry[9] if len(entry) > 9 and isinstance(entry[9], str) else ""

        rows.append(
            (
                icao,
                base + float(entry[0]),
                lat,
                lon,
                alt_baro,
                alt_geom,
                on_ground,
                _num(entry[4]),
                _num(entry[5]),
                nic,
                nac_p,
                source,
                bool(flags & 1),
                bool(flags & 2),
                False,
            )
        )
    return rows


def frame_from_rows(rows: Iterable[tuple]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(list(rows), columns=COLUMNS)
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column order and dtypes so every source produces identical frames."""
    df = df.reindex(columns=COLUMNS)
    float_cols = ["ts", "lat", "lon", "alt_baro", "alt_geom", "gs", "track", "nic", "nac_p"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ["on_ground", "stale", "new_leg", "gps_lost"]:
        df[col] = df[col].fillna(False).astype(bool)
    df["icao"] = df["icao"].astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    return df.reset_index(drop=True)


def parse_live_aircraft(payload: dict) -> pd.DataFrame:
    """Parse an adsb.lol / readsb ``/v2/...`` response (``{"ac": [...], "now": ms}``)."""
    now_ms = payload.get("now") or 0
    now = float(now_ms) / 1000.0
    rows = []
    for ac in payload.get("ac") or ():
        lat, lon = ac.get("lat"), ac.get("lon")
        if lat is None or lon is None:
            continue
        alt = ac.get("alt_baro")
        seen_pos = _num(ac.get("seen_pos"))
        rows.append(
            (
                str(ac.get("hex", "")).lower().lstrip("~"),
                now - (0.0 if math.isnan(seen_pos) else seen_pos),
                float(lat),
                float(lon),
                _num(alt),
                _num(ac.get("alt_geom")),
                alt == "ground",
                _num(ac.get("gs")),
                _num(ac.get("track")),
                _num(ac.get("nic")),
                _num(ac.get("nac_p")),
                str(ac.get("type") or ""),
                False,
                False,
                "gpsOkBefore" in ac,
            )
        )
    return frame_from_rows(rows)


def gnss_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask of rows whose position came from the aircraft's own GNSS."""
    return df["source"].isin(GNSS_SOURCES).to_numpy()
