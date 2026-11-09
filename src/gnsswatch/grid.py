"""Spatial grids used to aggregate observations into cells.

H3 (Uber's hexagonal grid) is the default because equal-area hexagons make per-cell ratios
comparable. A dependency-free square grid is provided as a fallback.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

# Approximate square-cell size (degrees) matching each H3 resolution's edge length.
_SQUARE_SIZE_FOR_RES = {2: 2.0, 3: 1.0, 4: 0.4, 5: 0.15, 6: 0.06, 7: 0.02}


class Grid(Protocol):
    name: str

    def cells(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray: ...

    def center(self, cell: str) -> tuple[float, float]: ...

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        """Polygon ring as (lat, lon) pairs, not closed."""
        ...


class H3Grid:
    def __init__(self, resolution: int) -> None:
        import h3  # imported lazily so the package works without it

        self._h3 = h3
        self.resolution = resolution
        self.name = f"h3:{resolution}"

    def cells(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        f = self._h3.latlng_to_cell
        r = self.resolution
        return np.array([f(a, b, r) for a, b in zip(lat, lon, strict=True)], dtype=object)

    def center(self, cell: str) -> tuple[float, float]:
        return self._h3.cell_to_latlng(cell)

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        return list(self._h3.cell_to_boundary(cell))


class SquareGrid:
    def __init__(self, size_deg: float) -> None:
        if size_deg <= 0:
            raise ValueError("size_deg must be positive")
        self.size = size_deg
        self.name = f"square:{size_deg}"

    def cells(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        i = np.floor(np.asarray(lat, dtype=float) / self.size).astype(int)
        j = np.floor(np.asarray(lon, dtype=float) / self.size).astype(int)
        return np.array([f"sq{self.size}_{a}_{b}" for a, b in zip(i, j, strict=True)], dtype=object)

    def _ij(self, cell: str) -> tuple[int, int]:
        _, i, j = cell.rsplit("_", 2)
        return int(i), int(j)

    def center(self, cell: str) -> tuple[float, float]:
        i, j = self._ij(cell)
        return ((i + 0.5) * self.size, (j + 0.5) * self.size)

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        i, j = self._ij(cell)
        s = self.size
        lat0, lon0 = i * s, j * s
        return [(lat0, lon0), (lat0, lon0 + s), (lat0 + s, lon0 + s), (lat0 + s, lon0)]


def h3_available() -> bool:
    try:
        import h3  # noqa: F401
    except ImportError:
        return False
    return True


def make_grid(resolution: int = 4, backend: str = "auto") -> Grid:
    """Create a grid. backend: 'auto' (H3 if installed), 'h3' or 'square'."""
    if backend not in {"auto", "h3", "square"}:
        raise ValueError(f"unknown grid backend {backend!r}")
    if backend == "h3" or (backend == "auto" and h3_available()):
        return H3Grid(resolution)
    size = _SQUARE_SIZE_FOR_RES.get(resolution)
    if size is None:
        # Each H3 resolution step shrinks edges by about sqrt(7).
        size = 0.4 / math.sqrt(7) ** (resolution - 4)
    return SquareGrid(size)
