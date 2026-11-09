import json

from gnsswatch.cli import main
from gnsswatch.outputs import write_outputs


def test_write_outputs(tmp_path, demo_result):
    res, _ = demo_result
    written = write_outputs(res, tmp_path, title="Test <report>")
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0

    geo = json.loads(written["geojson"].read_text())
    layers = {f["properties"]["layer"] for f in geo["features"]}
    assert layers == {"jamming", "position_jump", "spoofing_hotspot"}
    polygon = next(f for f in geo["features"] if f["geometry"]["type"] == "Polygon")
    ring = polygon["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # closed ring, GeoJSON order [lon, lat]

    summary = json.loads(written["summary"].read_text())
    assert summary["spoofing_hotspots"] == 1
    assert "ODbL" in summary["attribution"]

    html = written["report"].read_text()
    assert "__DATA__" not in html and "__TITLE__" not in html
    assert html.count("</script>") == 3  # Leaflet, topojson-client, inline script; nothing injected
    assert "<title>Test &lt;report&gt;</title>" in html


def test_cli_demo(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path), "--grid", "square"]) == 0
    out = capsys.readouterr().out
    assert "spoofing_hotspots" in out
    assert (tmp_path / "report.html").exists()


def test_cli_analyze_missing_input(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope")]) == 2
