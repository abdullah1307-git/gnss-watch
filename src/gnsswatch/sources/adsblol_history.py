"""adsb.lol daily history (ODbL-licensed) published as GitHub releases.

Each day is a tar archive of readsb's ``globe_history`` directory, split into parts
(``<tag>.tar.aa``, ``<tag>.tar.ab``, ...). Inside, every aircraft has a gzip-compressed
``traces/<xx>/trace_full_<icao>.json``.

A full day is several gigabytes, so this module:
  * downloads parts with resume support (safe to re-run after a dropped connection), and
  * reads the parts as one continuous stream, never extracting the archive to disk.

Data: https://github.com/adsblol (Open Database License 1.0, attribution required).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import tarfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..config import BBox
from ..parsing import frame_from_rows, parse_trace

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
USER_AGENT = "gnss-watch/0.1 (+https://github.com/abdullah1307-git/gnss-watch)"
CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int


def repo_for(day: date) -> str:
    return f"adsblol/globe_history_{day.year}"


def release_tag(day: date, pod: str = "prod-0") -> str:
    return f"v{day:%Y.%m.%d}-planes-readsb-{pod}"


def list_assets(day: date, pod: str = "prod-0", session: requests.Session | None = None) -> list[Asset]:
    """Return the tar part assets of one day's release, in order."""
    s = session or requests.Session()
    url = f"{GITHUB_API}/repos/{repo_for(day)}/releases/tags/{release_tag(day, pod)}"
    resp = s.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
                 timeout=30)
    if resp.status_code == 404:
        raise FileNotFoundError(
            f"No release {release_tag(day, pod)} in {repo_for(day)}. Data is uploaded the day "
            "after it is recorded; try an earlier date or another pod (prod-0, prod-1)."
        )
    resp.raise_for_status()
    assets = [
        Asset(a["name"], a["browser_download_url"], int(a.get("size", 0)))
        for a in resp.json().get("assets", [])
        if ".tar" in a["name"]
    ]
    if not assets:
        raise FileNotFoundError(f"Release {release_tag(day, pod)} has no tar assets")
    return sorted(assets, key=lambda a: a.name)


def download_day(
    day: date,
    dest_dir: Path,
    pod: str = "prod-0",
    session: requests.Session | None = None,
) -> list[Path]:
    """Download all parts of one day into ``dest_dir`` (resuming partial files)."""
    s = session or requests.Session()
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for asset in list_assets(day, pod, s):
        path = dest_dir / asset.name
        _download_resumable(s, asset, path)
        paths.append(path)
    return paths


RETRYABLE = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _download_resumable(
    s: requests.Session,
    asset: Asset,
    path: Path,
    max_attempts: int = 10,
    backoff_s: float = 5.0,
) -> None:
    """Download one asset, resuming after dropped connections (up to ``max_attempts``)."""
    for attempt in range(1, max_attempts + 1):
        try:
            _download_once(s, asset, path)
            return
        except RETRYABLE as exc:
            if attempt == max_attempts:
                raise
            wait = min(backoff_s * attempt, 60.0)
            have = path.stat().st_size if path.exists() else 0
            log.warning(
                "%s: connection dropped at %.0f MiB (%s). Resuming in %.0f s (attempt %d of %d)",
                asset.name, have / CHUNK, type(exc).__name__, wait, attempt + 1, max_attempts,
            )
            time.sleep(wait)


def _download_once(s: requests.Session, asset: Asset, path: Path) -> None:
    have = path.stat().st_size if path.exists() else 0
    if asset.size and have == asset.size:
        log.info("already complete: %s", path.name)
        return
    if asset.size and have > asset.size:
        path.unlink()
        have = 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    with s.get(asset.url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        mode = "ab" if have and resp.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        done = have
        next_report = done + 100 * CHUNK
        with path.open(mode) as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    pct = f"{100 * done / asset.size:.0f}%" if asset.size else "?"
                    log.info("%s: %.0f MiB (%s)", asset.name, done / CHUNK, pct)
                    next_report = done + 100 * CHUNK
    log.info("downloaded %s (%.0f MiB)", path.name, path.stat().st_size / CHUNK)


class ConcatReader(io.RawIOBase):
    """Read several files back to back as if they were one (for split tar archives)."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = list(paths)
        self._idx = 0
        self._fh = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        while self._idx < len(self._paths):
            if self._fh is None:
                self._fh = open(self._paths[self._idx], "rb")
            n = self._fh.readinto(buffer)
            if n:
                return n
            self._fh.close()
            self._fh = None
            self._idx += 1
        return 0

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def _decode_trace(raw: bytes) -> dict:
    if raw[:2] == b"\x1f\x8b":  # files are gzip even though they end in .json
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _is_trace(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith("trace_full_") and base.endswith(".json")


def iter_trace_docs(source: Path | Sequence[Path]) -> Iterator[dict]:
    """Yield trace documents from tar part files, a directory of parts, or an extracted dir."""
    if isinstance(source, Path) and source.is_dir():
        parts = sorted(p for p in source.iterdir() if ".tar" in p.name)
        if parts:
            yield from _iter_tar(parts)
        else:
            yield from _iter_extracted(source)
        return
    paths = [source] if isinstance(source, Path) else list(source)
    yield from _iter_tar(sorted(paths))


def _iter_tar(parts: Sequence[Path]) -> Iterator[dict]:
    stream = io.BufferedReader(ConcatReader(parts), buffer_size=CHUNK)
    with tarfile.open(fileobj=stream, mode="r|*") as tar:
        for member in tar:
            if not member.isfile() or not _is_trace(member.name):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            try:
                yield _decode_trace(fh.read())
            except (OSError, ValueError) as exc:
                log.warning("skipping unreadable %s: %s", member.name, exc)


def _iter_extracted(root: Path) -> Iterator[dict]:
    for path in sorted(root.rglob("trace_full_*.json")):
        try:
            yield _decode_trace(path.read_bytes())
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable %s: %s", path, exc)


def load_history(
    source: Path | Sequence[Path],
    bbox: BBox | None = None,
    max_traces: int | None = None,
) -> pd.DataFrame:
    """Load position reports from one day of history, keeping only those inside ``bbox``."""
    rows: list[tuple] = []
    n_docs = 0
    for doc in iter_trace_docs(source):
        n_docs += 1
        rows.extend(parse_trace(doc, bbox))
        if n_docs % 20000 == 0:
            log.info("read %d traces, %d positions in region", n_docs, len(rows))
        if max_traces is not None and n_docs >= max_traces:
            break
    log.info("read %d traces, %d positions in region", n_docs, len(rows))
    return frame_from_rows(rows)
