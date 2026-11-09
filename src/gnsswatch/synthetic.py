"""Synthetic readsb traces with known jamming and spoofing, for tests and the demo.

The generator writes data in the exact readsb ``trace_full`` format, so the demo exercises
the real parser and detectors end to end. Ground truth is returned alongside, which lets
tests check that the detectors find what was injected and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import REGIONS, BBox

NM_PER_DEG_LAT = 60.0


@dataclass
class Scenario:
    bbox: BBox = field(default_factory=lambda: REGIONS["baltic"])
    start_ts: float = 1_762_560_000.0  # 2025-11-08 00:00 UTC
    duration_s: int = 6 * 3600
    n_aircraft: int = 400
    report_every_s: int = 30
    # Jamming: degraded NIC/NACp inside a circle during a time interval.
    jam_center: tuple[float, float] = (57.0, 21.0)
    jam_radius_nm: float = 120.0
    jam_start_s: int = 2 * 3600
    jam_end_s: int = 4 * 3600
    # Spoofing: aircraft near the target are moved onto one fake location for a while.
    spoof_target: tuple[float, float] = (54.9, 18.0)
    spoof_capture_radius_nm: float = 200.0
    spoof_start_s: int = int(4.5 * 3600)
    spoof_duration_s: int = 600
    spoof_max_aircraft: int = 6
    spoof_alt_offset_ft: float = 2500.0
    # Background noise.
    share_bad_receivers: float = 0.02
    share_mlat: float = 0.05
    seed: int = 7


@dataclass
class GroundTruth:
    jammed_aircraft: set[str]
    spoofed_aircraft: set[str]
    faulty_aircraft: set[str]


def _dist_nm(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * NM_PER_DEG_LAT
    dlon = (lon2 - lon1) * NM_PER_DEG_LAT * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dlat, dlon)


def synthetic_traces(sc: Scenario | None = None) -> tuple[list[dict], GroundTruth]:
    sc = sc or Scenario()
    rng = np.random.default_rng(sc.seed)
    b = sc.bbox
    docs: list[dict] = []
    jammed: set[str] = set()
    spoofed: set[str] = set()
    faulty: set[str] = set()
    spoof_slots = sc.spoof_max_aircraft

    for k in range(sc.n_aircraft):
        icao = f"{0x3C0000 + k:06x}"
        dep = rng.uniform(-1.5 * 3600, sc.duration_s - 1800)
        flight_s = rng.uniform(1.5 * 3600, 3.5 * 3600)
        t0 = max(0.0, dep)
        t1 = min(sc.duration_s, dep + flight_s)
        if t1 - t0 < 600:
            continue
        speed = rng.uniform(420, 480)
        heading = rng.uniform(0, 360)
        cruise_alt = float(rng.choice(np.arange(28000, 40001, 1000)))
        isa_offset = rng.uniform(-400, 600)  # normal, slowly varying geo-baro difference
        bad_receiver = rng.random() < sc.share_bad_receivers
        source = "mlat" if rng.random() < sc.share_mlat else "adsb_icao"

        # Start point chosen so the track at time t0 lies inside the box.
        lat = rng.uniform(b.min_lat + 0.5, b.max_lat - 0.5)
        lon = rng.uniform(b.min_lon + 0.5, b.max_lon - 0.5)
        times = np.arange(t0, t1, sc.report_every_s)
        dist = speed * (times - t0) / 3600.0
        h = math.radians(heading)
        lats = lat + dist * math.cos(h) / NM_PER_DEG_LAT
        lons = lon + dist * math.sin(h) / (NM_PER_DEG_LAT * math.cos(math.radians(lat)))

        in_jam = (
            (_dist_nm(lats, lons, *sc.jam_center) <= sc.jam_radius_nm)
            & (times >= sc.jam_start_s) & (times < sc.jam_end_s)
        )

        spoof_mask = np.zeros(len(times), dtype=bool)
        if source == "adsb_icao" and spoof_slots > 0:
            near = _dist_nm(lats, lons, *sc.spoof_target) <= sc.spoof_capture_radius_nm
            during = (times >= sc.spoof_start_s) & (times < sc.spoof_start_s + sc.spoof_duration_s)
            if (near & during).any():
                spoof_mask = during
                spoof_slots -= 1
                spoofed.add(icao)
        if in_jam.any() and source == "adsb_icao":
            jammed.add(icao)
        if bad_receiver and source == "adsb_icao":
            faulty.add(icao)

        trace = []
        last_integrity = None
        for i, t in enumerate(times):
            plat, plon = float(lats[i]), float(lons[i])
            alt_geom = cruise_alt + isa_offset
            if spoof_mask[i]:
                plat = sc.spoof_target[0] + rng.normal(0, 0.01)
                plon = sc.spoof_target[1] + rng.normal(0, 0.01)
                alt_geom += sc.spoof_alt_offset_ft

            if source != "adsb_icao":
                integrity = None
            elif in_jam[i]:
                integrity = (0, 0)
            elif bad_receiver:
                integrity = (6, 7)
            else:
                integrity = (8, 9)

            details = None
            if integrity is not None and (integrity != last_integrity or i % 20 == 0):
                details = {"nic": integrity[0], "nac_p": integrity[1], "rc": 186}
                last_integrity = integrity

            flags = 2 if i == 0 else 0
            trace.append([
                round(float(t - t0), 1), round(plat, 6), round(plon, 6), int(cruise_alt),
                round(speed, 1), round(heading, 1), flags, 0, details, source,
                int(alt_geom) if source == "adsb_icao" else None,
            ])
        docs.append({"icao": icao, "timestamp": sc.start_ts + t0, "trace": trace})

    return docs, GroundTruth(jammed_aircraft=jammed, spoofed_aircraft=spoofed, faulty_aircraft=faulty)
