"""Write analysis results: tables (CSV, optional Parquet), GeoJSON, summary JSON, HTML map."""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import AnalysisResult

log = logging.getLogger(__name__)

MAX_JUMP_LINES = 3000
ATTRIBUTION = (
    "Flight data: adsb.lol contributors, Open Database License (ODbL) 1.0. "
    "Base map: Natural Earth (public domain)."
)


def _iso(ts: float | None) -> str | None:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _with_times(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("ts", "window_start", "first_ts", "last_ts"):
        if col in out.columns:
            out[f"{col}_utc"] = out[col].map(_iso)
    return out


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _ring(result: AnalysisResult, cell: str) -> list[list[float]]:
    """GeoJSON polygon ring ([lon, lat] pairs, closed)."""
    ring = [[round(lon, 5), round(lat, 5)] for lat, lon in result.grid.boundary(cell)]
    return ring + [ring[0]]


def to_geojson(result: AnalysisResult) -> dict:
    features: list[dict] = []
    for row in result.cells_overall.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_ring(result, row.cell)]},
            "properties": {
                "layer": "jamming",
                "cell": row.cell,
                "level": row.level,
                "n_aircraft": int(row.n_aircraft),
                "n_degraded": int(row.n_degraded),
                "degraded_ratio": round(float(row.degraded_ratio), 4),
                "peak_ratio": round(float(row.peak_ratio), 4),
                "windows_high": int(row.windows_high),
            },
        })
    for row in result.jumps.head(MAX_JUMP_LINES).itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[row.from_lon, row.from_lat], [row.lon, row.lat]],
            },
            "properties": {
                "layer": "position_jump",
                "icao": row.icao,
                "time_utc": _iso(row.ts),
                "jump_nm": round(float(row.magnitude), 1),
                "implied_kt": round(float(row.implied_kt), 0),
            },
        })
    for row in result.hotspots.itertuples(index=False):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
            "properties": {
                "layer": "spoofing_hotspot",
                "n_aircraft": int(row.n_aircraft),
                "n_events": int(row.n_events),
                "first_utc": _iso(row.first_ts),
                "last_utc": _iso(row.last_ts),
                "aircraft": row.aircraft,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _report_payload(result: AnalysisResult, title: str) -> dict:
    """Compact data for the HTML map: geometry once per cell, stats per window."""
    usable = result.cells[result.cells["level"] != "insufficient"]
    cells_needed = set(result.cells_overall["cell"])
    geometry = {c: _ring(result, c) for c in sorted(cells_needed)}
    windows: dict[str, list] = {}
    for ws, group in usable.groupby("window_start", sort=True):
        windows[_iso(ws)] = [
            [r.cell, int(r.n_aircraft), int(r.n_degraded), r.level]
            for r in group.itertuples(index=False)
        ]
    overall = [
        [r.cell, int(r.n_aircraft), int(r.n_degraded), r.level, round(float(r.peak_ratio), 3),
         int(r.windows_high)]
        for r in result.cells_overall.itertuples(index=False)
        if r.level != "insufficient"
    ]
    jumps = [
        [round(r.from_lat, 4), round(r.from_lon, 4), round(r.lat, 4), round(r.lon, 4), r.icao,
         _iso(r.ts), round(float(r.magnitude), 1)]
        for r in result.jumps.head(MAX_JUMP_LINES).itertuples(index=False)
    ]
    hotspots = [
        [round(r.lat, 4), round(r.lon, 4), int(r.n_aircraft), int(r.n_events), _iso(r.first_ts),
         _iso(r.last_ts), r.aircraft]
        for r in result.hotspots.itertuples(index=False)
    ]
    summary = result.summary()
    summary.pop("settings", None)
    summary["period_start"] = _iso(summary["period_start"])
    summary["period_end"] = _iso(summary["period_end"])
    region = result.settings.region
    return {
        "title": title,
        "generated_utc": _iso(datetime.now(timezone.utc).timestamp()),
        "bounds": [[region.min_lat, region.min_lon], [region.max_lat, region.max_lon]],
        "summary": summary,
        "geometry": geometry,
        "overall": overall,
        "windows": windows,
        "jumps": jumps,
        "hotspots": hotspots,
        "attribution": ATTRIBUTION,
    }


def render_html(result: AnalysisResult, title: str) -> str:
    template = resources.files("gnsswatch").joinpath("templates/report.html").read_text("utf-8")
    payload = json.dumps(_report_payload(result, title), separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # never close the <script> tag early
    return template.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


def write_outputs(result: AnalysisResult, out_dir: Path, title: str = "GNSS Watch report") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "jamming_cells": result.cells,
        "jamming_cells_overall": result.cells_overall,
        "position_jumps": result.jumps,
        "altitude_events": result.altitude_events,
        "spoofing_hotspots": result.hotspots,
    }
    written: dict[str, Path] = {}
    use_parquet = _parquet_available()
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        _with_times(df).to_csv(path, index=False)
        written[name] = path
        if use_parquet:
            df.to_parquet(out_dir / f"{name}.parquet", index=False)

    summary = result.summary()
    summary["period_start_utc"] = _iso(summary["period_start"])
    summary["period_end_utc"] = _iso(summary["period_end"])
    summary["faulty_aircraft"] = result.faulty_aircraft
    summary["attribution"] = ATTRIBUTION
    written["summary"] = out_dir / "summary.json"
    written["summary"].write_text(json.dumps(summary, indent=2, default=str), "utf-8")

    written["geojson"] = out_dir / "map.geojson"
    written["geojson"].write_text(json.dumps(to_geojson(result)), "utf-8")

    written["report"] = out_dir / "report.html"
    written["report"].write_text(render_html(result, title), "utf-8")
    log.info("wrote %d files to %s", len(written), out_dir)
    return written
