import pandas as pd
import pytest

from gnsswatch.config import REGIONS, Settings
from gnsswatch.parsing import frame_from_rows, parse_trace
from gnsswatch.pipeline import analyze
from gnsswatch.synthetic import Scenario, synthetic_traces


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return Scenario()


@pytest.fixture(scope="session")
def synthetic(scenario):
    docs, truth = synthetic_traces(scenario)
    df = frame_from_rows(r for d in docs for r in parse_trace(d, REGIONS["baltic"]))
    return docs, df, truth


@pytest.fixture(scope="session")
def demo_result(synthetic):
    _, df, truth = synthetic
    return analyze(df, Settings(grid_backend="square")), truth


def make_track(
    icao: str = "abc123",
    start_ts: float = 1_700_000_000.0,
    lat: float = 55.0,
    lon: float = 15.0,
    n: int = 20,
    every_s: float = 30.0,
    gs: float = 450.0,
    heading_east: bool = True,
    **overrides,
) -> pd.DataFrame:
    """A straight, physically plausible track as a normalised frame."""
    import math

    rows = []
    for i in range(n):
        dist_nm = gs * i * every_s / 3600.0
        dlon = dist_nm / (60.0 * math.cos(math.radians(lat))) if heading_east else 0.0
        dlat = 0.0 if heading_east else dist_nm / 60.0
        row = dict(
            icao=icao, ts=start_ts + i * every_s, lat=lat + dlat, lon=lon + dlon,
            alt_baro=35000.0, alt_geom=35400.0, on_ground=False, gs=gs, track=90.0,
            nic=8.0, nac_p=9.0, source="adsb_icao", stale=False, new_leg=False, gps_lost=False,
        )
        row.update(overrides)
        rows.append(row)
    return frame_from_rows([tuple(r.values()) for r in rows])
