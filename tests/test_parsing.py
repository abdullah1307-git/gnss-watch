import gzip
import io
import json
import math
import tarfile

import pytest

from gnsswatch.config import BBox
from gnsswatch.parsing import COLUMNS, frame_from_rows, gnss_mask, parse_live_aircraft, parse_trace
from gnsswatch.sources.adsblol_history import iter_trace_docs, load_history

DOC = {
    "icao": "3c6444",
    "timestamp": 1_700_000_000.0,
    "trace": [
        [0.0, 52.36, 13.50, "ground", 12.0, 250.0, 2, 0, None, "adsb_icao", None],
        [30.0, 52.40, 13.55, 3000, 180.0, 250.0, 0, 1500,
         {"nic": 8, "nac_p": 9, "rc": 186}, "adsb_icao", 3150],
        [60.0, 52.45, 13.60, 6000, 220.0, 250.0, 0, 1500, None, "adsb_icao", 6200],
        [90.0, 52.50, 13.65, 9000, 250.0, 250.0, 8, 1500, None, "adsb_icao", None],
        [120.0, None, None, 9500, 250.0, 250.0, 0, 0, None, "adsb_icao", None],
        [150.0, 52.55, 13.70, 10000, 260.0, 250.0, 1, 0, {"nic": 0, "nac_p": 0}, "mlat", None],
    ],
}


def test_parse_trace_fields():
    df = frame_from_rows(parse_trace(DOC))
    assert list(df.columns) == COLUMNS
    assert len(df) == 5  # the entry without a position is skipped

    first = df.iloc[0]
    assert first.icao == "3c6444"
    assert first.on_ground and math.isnan(first.alt_baro)
    assert first.new_leg
    assert math.isnan(first.nic)  # no details object seen yet

    second = df.iloc[1]
    assert second.ts == pytest.approx(1_700_000_030.0)
    assert second.alt_baro == 3000 and second.alt_geom == 3150
    assert second.nic == 8 and second.nac_p == 9


def test_integrity_is_carried_forward():
    df = frame_from_rows(parse_trace(DOC))
    assert df.iloc[2].nic == 8  # no details on this entry, value carried from the previous one


def test_geometric_altitude_flag():
    df = frame_from_rows(parse_trace(DOC))
    row = df.iloc[3]  # flags & 8: the altitude column is geometric
    assert math.isnan(row.alt_baro)
    assert row.alt_geom == 9000


def test_stale_flag_and_source():
    df = frame_from_rows(parse_trace(DOC))
    last = df.iloc[-1]
    assert last.stale
    assert last.source == "mlat"
    assert not gnss_mask(df)[-1]
    assert gnss_mask(df)[:-1].all()


def test_bbox_filter_keeps_integrity_state():
    bbox = BBox(52.44, 13.0, 53.0, 14.0)  # excludes the first two entries
    df = frame_from_rows(parse_trace(DOC, bbox))
    assert df["lat"].min() >= 52.44
    assert df.iloc[0].nic == 8  # details came from an entry outside the bbox


def test_parse_live_aircraft():
    payload = {
        "now": 1_700_000_100_000,
        "ac": [
            {"hex": "ABC123", "lat": 55.0, "lon": 20.0, "alt_baro": 36000, "alt_geom": 36500,
             "gs": 450.1, "track": 90, "nic": 0, "nac_p": 0, "type": "adsb_icao", "seen_pos": 2.0,
             "gpsOkBefore": 1_700_000_000.0},
            {"hex": "def456", "lat": 54.0, "lon": 19.0, "alt_baro": "ground", "type": "adsb_icao",
             "seen_pos": 0.5, "nic": 8, "nac_p": 10},
            {"hex": "fff000", "type": "mode_s"},  # no position
        ],
    }
    df = parse_live_aircraft(payload)
    assert len(df) == 2
    a = df.set_index("icao").loc["abc123"]
    assert a.ts == pytest.approx(1_700_000_098.0)
    assert a.gps_lost
    b = df.set_index("icao").loc["def456"]
    assert b.on_ground and not b.gps_lost


def _write_split_tar(directory, docs, n_parts=2):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for doc in docs:
            data = gzip.compress(json.dumps(doc).encode())
            info = tarfile.TarInfo(f"./traces/{doc['icao'][-2:]}/trace_full_{doc['icao']}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        noise = b"not a trace"
        info = tarfile.TarInfo("./heatmap/00.bin.ttf")
        info.size = len(noise)
        tar.addfile(info, io.BytesIO(noise))
    raw = buf.getvalue()
    size = math.ceil(len(raw) / n_parts)
    letters = "abcdefgh"
    for i in range(n_parts):
        (directory / f"v2026.01.01-planes-readsb-prod-0.tar.a{letters[i]}").write_bytes(
            raw[i * size:(i + 1) * size]
        )


def test_split_tar_is_streamed_as_one_archive(tmp_path):
    docs = [dict(DOC, icao=f"3c64{i:02x}") for i in range(3)]
    _write_split_tar(tmp_path, docs, n_parts=3)
    found = list(iter_trace_docs(tmp_path))
    assert sorted(d["icao"] for d in found) == sorted(d["icao"] for d in docs)


def test_extracted_directory_is_supported(tmp_path):
    sub = tmp_path / "traces" / "44"
    sub.mkdir(parents=True)
    (sub / "trace_full_3c6444.json").write_bytes(gzip.compress(json.dumps(DOC).encode()))
    df = load_history(tmp_path)
    assert len(df) == 5
    assert df["icao"].unique().tolist() == ["3c6444"]


def test_load_history_max_traces(tmp_path):
    docs = [dict(DOC, icao=f"3c64{i:02x}") for i in range(4)]
    _write_split_tar(tmp_path, docs)
    df = load_history(tmp_path, max_traces=2)
    assert df["icao"].nunique() == 2
