import pandas as pd
import pytest

from gnsswatch.config import SpoofingConfig
from gnsswatch.detection.spoofing import (
    altitude_divergence,
    haversine_nm,
    position_jumps,
    spoofing_hotspots,
)
from gnsswatch.grid import SquareGrid
from gnsswatch.parsing import normalize

from .conftest import make_track

CFG = SpoofingConfig()


def test_haversine_known_distance():
    # Berlin (EDDB) to Warsaw (EPWA) is about 279 NM
    assert haversine_nm(52.36, 13.50, 52.17, 20.97) == pytest.approx(276, abs=6)


def test_normal_flight_has_no_events():
    df = make_track(n=60)
    assert position_jumps(df, CFG).empty
    assert altitude_divergence(df, CFG).empty


def test_teleport_is_detected():
    df = make_track(n=20)
    df.loc[10:, "lat"] += 1.0  # 60 NM in 30 seconds
    jumps = position_jumps(df, CFG)
    assert len(jumps) == 1
    j = jumps.iloc[0]
    assert j.magnitude == pytest.approx(60, abs=1)
    assert j.implied_kt > 7000


def test_coverage_gap_with_plausible_speed_is_not_a_jump():
    df = make_track(n=20)
    df = df.drop(index=range(5, 15)).reset_index(drop=True)  # 5.5 minutes without reports
    assert position_jumps(df, CFG).empty


def test_new_leg_and_mlat_are_ignored():
    leg = make_track(n=20)
    leg.loc[10:, "lat"] += 1.0
    leg.loc[10, "new_leg"] = True
    assert position_jumps(leg, CFG).empty

    mlat = make_track(n=20, source="mlat")
    mlat.loc[10:, "lat"] += 1.0
    assert position_jumps(mlat, CFG).empty


def test_altitude_step_is_detected():
    df = make_track(n=20)
    df.loc[12:, "alt_geom"] += 2500  # GNSS altitude suddenly wrong, baro unchanged
    events = altitude_divergence(df, CFG)
    assert len(events) == 1
    assert events.iloc[0].magnitude == pytest.approx(2500)


def _jumps_to(lat, lon, n_aircraft, t0=1_700_000_000.0):
    rows = []
    for i in range(n_aircraft):
        rows.append(dict(kind="position_jump", icao=f"ee{i:04x}", ts=t0 + 60 * i,
                         lat=lat + 0.02 * (-1) ** i, lon=lon + 0.02 * i, from_lat=lat - 2,
                         from_lon=lon, magnitude=120.0, implied_kt=9000.0, gs_kt=450.0, dt_s=30.0))
    return pd.DataFrame(rows)


def test_hotspot_on_cell_border_is_not_split():
    jumps = _jumps_to(55.0, 18.0, 4)  # 18.0 is a cell border on a 0.4 degree grid
    spots = spoofing_hotspots(jumps, SquareGrid(0.4), CFG)
    assert len(spots) == 1
    assert spots.iloc[0].n_aircraft == 4


def test_scattered_jumps_are_not_a_hotspot():
    far = pd.concat([_jumps_to(55.0, 18.0, 1), _jumps_to(57.0, 22.0, 1), _jumps_to(59.0, 12.0, 1)])
    far["icao"] = ["a", "b", "c"]
    assert spoofing_hotspots(far, SquareGrid(0.4), CFG).empty


def test_hotspot_needs_events_close_in_time():
    jumps = _jumps_to(55.0, 18.0, 3)
    jumps.loc[2, "ts"] += 5 * 3600  # third aircraft hours later
    assert spoofing_hotspots(jumps, SquareGrid(0.4), CFG).empty


def test_synthetic_spoofing_is_found_without_false_positives(demo_result, scenario):
    res, truth = demo_result
    assert set(res.jumps["icao"]) <= truth.spoofed_aircraft
    assert set(res.altitude_events["icao"]) <= truth.spoofed_aircraft
    assert len(res.hotspots) == 1
    spot = res.hotspots.iloc[0]
    assert haversine_nm(spot.lat, spot.lon, *scenario.spoof_target) < 5
    members = set(spot.aircraft.split(","))
    assert members <= truth.spoofed_aircraft
    assert len(members) >= len(truth.spoofed_aircraft) - 1


def test_empty_input():
    empty = normalize(pd.DataFrame())
    assert position_jumps(empty, CFG).empty
    assert altitude_divergence(empty, CFG).empty
