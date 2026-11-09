"""Configuration: regions of interest and detection thresholds.

All thresholds are plain dataclass fields so they can be tuned from the CLI or tests
and are recorded in every report's summary for reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in decimal degrees (WGS84)."""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    def __post_init__(self) -> None:
        if not (-90 <= self.min_lat < self.max_lat <= 90):
            raise ValueError(f"invalid latitude range: {self.min_lat}..{self.max_lat}")
        if not (-180 <= self.min_lon < self.max_lon <= 180):
            raise ValueError(f"invalid longitude range: {self.min_lon}..{self.max_lon}")

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    @classmethod
    def parse(cls, text: str) -> BBox:
        """Parse 'min_lat,min_lon,max_lat,max_lon' or a named region (see REGIONS)."""
        key = text.strip().lower()
        if key in REGIONS:
            return REGIONS[key]
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "bbox must be a region name or 'min_lat,min_lon,max_lat,max_lon', "
                f"got {text!r}"
            )
        return cls(*parts)


REGIONS: dict[str, BBox] = {
    # Baltic Sea and surroundings: one of the most affected areas in Europe.
    "baltic": BBox(53.0, 9.0, 62.0, 32.0),
    "berlin": BBox(51.5, 12.0, 53.5, 15.0),
    "germany": BBox(47.0, 5.5, 55.2, 15.5),
    "eastern-med": BBox(30.0, 25.0, 38.0, 37.0),
    "black-sea": BBox(40.0, 27.0, 47.5, 42.0),
    "europe": BBox(34.0, -12.0, 72.0, 45.0),
}


@dataclass(frozen=True)
class JammingConfig:
    """Cell-level jamming indicator settings.

    A position report is "degraded" when its navigation integrity (NIC) or accuracy (NACp)
    falls below the thresholds. Healthy airliners usually report NIC >= 8 and NACp >= 9.
    """

    window_s: int = 3600
    nic_bad_below: int = 7
    nacp_bad_below: int = 8
    # An aircraft counts as degraded in a cell/window if at least this share of its reports are.
    degraded_report_share: float = 0.5
    # Cells with fewer aircraft than this are labelled "insufficient".
    min_aircraft: int = 3
    # One degraded aircraft is not evidence of interference: medium/high need at least this many.
    min_degraded_aircraft: int = 2
    medium_ratio: float = 0.02
    high_ratio: float = 0.10
    # Peer check: an aircraft that is degraded where its peers (other aircraft in the same
    # cell and window) are healthy has an equipment issue, not interference. If that happens
    # in at least `faulty_min_contradicted` cell-windows and in at least
    # `faulty_contradicted_share` of its comparable degraded cell-windows, the aircraft is
    # excluded from the jamming statistics and reported separately.
    peer_min_aircraft: int = 2
    faulty_min_contradicted: int = 2
    faulty_contradicted_share: float = 0.5


@dataclass(frozen=True)
class SpoofingConfig:
    """Per-aircraft kinematic plausibility checks used as spoofing indicators."""

    # Ignore pairs of reports further apart than this (coverage gaps).
    max_gap_s: float = 900.0
    # Minimum displacement before a jump is considered at all (filters decoder noise).
    min_jump_nm: float = 3.0
    # Implied speed must exceed max(max_plausible_kt, gs * speed_factor + speed_margin_kt).
    max_plausible_kt: float = 750.0
    speed_factor: float = 1.5
    speed_margin_kt: float = 100.0
    # Sudden change of (GNSS altitude - barometric altitude) between nearby reports.
    alt_step_window_s: float = 60.0
    alt_step_ft: float = 1000.0
    # Hotspot = at least N distinct aircraft whose jumps land within radius of each other,
    # each within window seconds of the previous jump in the cluster.
    hotspot_min_aircraft: int = 3
    hotspot_radius_nm: float = 15.0
    hotspot_window_s: int = 3600


@dataclass(frozen=True)
class Settings:
    region: BBox = field(default_factory=lambda: REGIONS["baltic"])
    # H3 resolution 4 cells are about 1,770 km2 (edge about 22 km).
    grid_resolution: int = 4
    grid_backend: str = "auto"  # "auto" | "h3" | "square"
    jamming: JammingConfig = field(default_factory=JammingConfig)
    spoofing: SpoofingConfig = field(default_factory=SpoofingConfig)

    def to_dict(self) -> dict:
        return asdict(self)
