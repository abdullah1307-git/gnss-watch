import numpy as np
import pandas as pd

from gnsswatch.config import JammingConfig
from gnsswatch.detection.jamming import analyze_jamming, classify, faulty_aircraft
from gnsswatch.detection.spoofing import haversine_nm
from gnsswatch.grid import SquareGrid

from .conftest import make_track

CFG = JammingConfig()


def test_classify_levels():
    levels = classify(
        n_aircraft=[2, 10, 10, 10, 40],
        n_degraded=[2, 0, 1, 5, 2],
        ratio=[1.0, 0.0, 0.1, 0.5, 0.05],
        cfg=CFG,
    )
    # too few aircraft / healthy / one degraded aircraft is not evidence / high / medium
    assert levels.tolist() == ["insufficient", "low", "low", "high", "medium"]


def test_jammed_area_is_flagged_and_healthy_area_is_not():
    frames = []
    for i in range(6):  # six aircraft crossing the same area, all degraded
        frames.append(make_track(icao=f"aa{i:04x}", lat=56.0 + i * 0.01, nic=0.0, nac_p=0.0))
    for i in range(6):  # six healthy aircraft elsewhere
        frames.append(make_track(icao=f"bb{i:04x}", lat=59.0 + i * 0.01))
    df = pd.concat(frames, ignore_index=True)
    res = analyze_jamming(df, SquareGrid(1.0), CFG)
    by_lat = res.cells.groupby(res.cells["lat"].round())["level"].agg(set)
    assert by_lat.loc[56.0] == {"high"}
    assert by_lat.loc[60.0] == {"low"} or by_lat.loc[59.0] == {"low"}
    assert res.faulty_aircraft == set()


def test_mlat_and_ground_reports_are_ignored():
    jammed_mlat = [make_track(icao=f"cc{i:04x}", nic=0.0, nac_p=0.0, source="mlat") for i in range(5)]
    on_ground = [make_track(icao=f"dd{i:04x}", nic=0.0, nac_p=0.0, on_ground=True) for i in range(5)]
    res = analyze_jamming(pd.concat(jammed_mlat + on_ground), SquareGrid(1.0), CFG)
    assert res.cells.empty


def test_faulty_receiver_detected_by_peers():
    status = pd.DataFrame({
        "window_start": [0, 0, 0, 3600, 3600, 3600, 7200, 7200, 7200],
        "cell": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "icao": ["bad", "p1", "p2", "bad", "p3", "p4", "x1", "x2", "x3"],
        "degraded": [True, False, False, True, False, False, True, True, True],
    })
    # 'bad' is degraded twice where its peers are healthy; x1..x3 are degraded together
    assert faulty_aircraft(status, CFG) == {"bad"}


def test_synthetic_scenario_no_false_jamming(demo_result, scenario):
    res, _ = demo_result
    flagged = res.cells[res.cells["level"].isin(["medium", "high"])]
    assert len(flagged) > 10
    dist = haversine_nm(flagged["lat"], flagged["lon"], *scenario.jam_center)
    margin_nm = 40  # half a cell diagonal of slack around the circle edge
    assert (dist <= scenario.jam_radius_nm + margin_nm).all()
    hours = (flagged["window_start"] - scenario.start_ts) / 3600
    assert np.isin(hours, [2, 3]).all()  # jamming was injected between 02:00 and 04:00


def test_synthetic_scenario_faulty_receivers(demo_result):
    res, truth = demo_result
    found = set(res.faulty_aircraft)
    assert found <= truth.faulty_aircraft  # precision: no healthy aircraft blamed
    assert len(found) >= 0.75 * len(truth.faulty_aircraft)
