import numpy as np
import pytest

from gnsswatch.config import BBox
from gnsswatch.grid import SquareGrid, h3_available, make_grid


def test_square_grid_cells_and_geometry():
    g = SquareGrid(0.5)
    cells = g.cells(np.array([55.1, 55.4, 55.6]), np.array([20.1, 20.4, 20.1]))
    assert cells[0] == cells[1] != cells[2]
    lat, lon = g.center(cells[0])
    assert (lat, lon) == pytest.approx((55.25, 20.25))
    ring = g.boundary(cells[0])
    assert len(ring) == 4
    lats = [p[0] for p in ring]
    assert min(lats) <= 55.1 <= max(lats)


def test_square_grid_negative_coordinates():
    g = SquareGrid(1.0)
    cell = g.cells(np.array([-0.5]), np.array([-0.5]))[0]
    assert g.center(cell) == pytest.approx((-0.5, -0.5))


def test_make_grid_square_backend():
    g = make_grid(4, "square")
    assert isinstance(g, SquareGrid)
    with pytest.raises(ValueError):
        make_grid(4, "triangles")


@pytest.mark.skipif(not h3_available(), reason="h3 not installed")
def test_h3_grid_roundtrip():
    g = make_grid(4, "h3")
    cell = g.cells(np.array([52.52]), np.array([13.40]))[0]
    lat, lon = g.center(cell)
    assert abs(lat - 52.52) < 0.5 and abs(lon - 13.40) < 0.5
    assert len(g.boundary(cell)) in (5, 6)


def test_bbox_parse():
    assert BBox.parse("baltic").min_lat == 53.0
    b = BBox.parse("50,10,55,20")
    assert b.contains(52, 15) and not b.contains(56, 15)
    with pytest.raises(ValueError):
        BBox.parse("1,2,3")
    with pytest.raises(ValueError):
        BBox(10, 0, 5, 1)
