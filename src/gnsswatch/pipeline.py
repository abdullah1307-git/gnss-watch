"""End-to-end analysis: position reports in, jamming cells and spoofing events out."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .config import Settings
from .detection import (
    aggregate_cells,
    altitude_divergence,
    position_jumps,
    spoofing_hotspots,
)
from .detection.jamming import analyze_jamming
from .grid import Grid, make_grid

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    settings: Settings
    grid: Grid
    n_reports: int
    n_aircraft: int
    period: tuple[float, float] | None
    cells: pd.DataFrame          # per window and cell
    cells_overall: pd.DataFrame  # whole period, per cell
    jumps: pd.DataFrame
    altitude_events: pd.DataFrame
    hotspots: pd.DataFrame
    faulty_aircraft: list[str]

    def summary(self) -> dict:
        levels = self.cells_overall["level"].value_counts().to_dict() if len(self.cells_overall) else {}
        return {
            "period_start": self.period[0] if self.period else None,
            "period_end": self.period[1] if self.period else None,
            "grid": self.grid.name,
            "position_reports": self.n_reports,
            "aircraft": self.n_aircraft,
            "cells": int(len(self.cells_overall)),
            "cells_by_level": {k: int(v) for k, v in levels.items()},
            "cell_windows_high": int((self.cells["level"] == "high").sum()) if len(self.cells) else 0,
            "aircraft_faulty_receivers": len(self.faulty_aircraft),
            "position_jumps": int(len(self.jumps)),
            "aircraft_with_jumps": int(self.jumps["icao"].nunique()) if len(self.jumps) else 0,
            "altitude_events": int(len(self.altitude_events)),
            "spoofing_hotspots": int(len(self.hotspots)),
            "settings": self.settings.to_dict(),
        }


def analyze(df: pd.DataFrame, settings: Settings | None = None) -> AnalysisResult:
    settings = settings or Settings()
    grid = make_grid(settings.grid_resolution, settings.grid_backend)
    log.info("analysing %d reports from %d aircraft on %s grid",
             len(df), df["icao"].nunique() if len(df) else 0, grid.name)

    jam = analyze_jamming(df, grid, settings.jamming)
    cells = jam.cells
    overall = aggregate_cells(cells, grid, settings.jamming)
    jumps = position_jumps(df, settings.spoofing)
    alt_events = altitude_divergence(df, settings.spoofing)
    hotspots = spoofing_hotspots(jumps, grid, settings.spoofing)

    period = (float(df["ts"].min()), float(df["ts"].max())) if len(df) else None
    return AnalysisResult(
        settings=settings,
        grid=grid,
        n_reports=int(len(df)),
        n_aircraft=int(df["icao"].nunique()) if len(df) else 0,
        period=period,
        cells=cells,
        cells_overall=overall,
        jumps=jumps,
        altitude_events=alt_events,
        hotspots=hotspots,
        faulty_aircraft=sorted(jam.faulty_aircraft),
    )
