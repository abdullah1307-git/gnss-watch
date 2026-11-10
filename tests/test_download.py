"""Downloader behaviour with a fake HTTP session (no network needed)."""

import pytest
import requests

from gnsswatch.sources import adsblol_history as hist
from gnsswatch.sources.adsblol_history import Asset

PAYLOAD = bytes(range(256)) * 40  # 10 KiB of known bytes


class FakeResponse:
    def __init__(self, body: bytes, status: int, fail_after: int | None):
        self.body, self.status_code, self.fail_after = body, status, fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        sent = 0
        for i in range(0, len(self.body), 1024):
            if self.fail_after is not None and sent >= self.fail_after:
                raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
            piece = self.body[i:i + 1024]
            sent += len(piece)
            yield piece


class FlakySession:
    """Serves PAYLOAD, honouring Range headers, and drops the connection on chosen calls."""

    def __init__(self, drops: list[int | None]):
        self.drops = list(drops)
        self.ranges: list[str | None] = []

    def get(self, url, headers=None, stream=False, timeout=None):
        rng = (headers or {}).get("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        fail_after = self.drops.pop(0) if self.drops else None
        return FakeResponse(PAYLOAD[start:], 206 if rng else 200, fail_after)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(hist.time, "sleep", lambda s: None)


ASSET = Asset("day.tar.aa", "https://example.invalid/day.tar.aa", len(PAYLOAD))


def test_resumes_after_dropped_connection(tmp_path):
    path = tmp_path / ASSET.name
    session = FlakySession(drops=[3072, 2048, None])  # drop twice, then succeed
    hist._download_resumable(session, ASSET, path)
    assert path.read_bytes() == PAYLOAD
    assert session.ranges == [None, "bytes=3072-", "bytes=5120-"]


def test_skips_complete_file(tmp_path):
    path = tmp_path / ASSET.name
    path.write_bytes(PAYLOAD)
    session = FlakySession(drops=[])
    hist._download_resumable(session, ASSET, path)
    assert session.ranges == []


def test_gives_up_after_max_attempts(tmp_path):
    path = tmp_path / ASSET.name
    session = FlakySession(drops=[1024, 1024, 1024])
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        hist._download_resumable(session, ASSET, path, max_attempts=3)
    assert path.stat().st_size == 3072  # progress kept for the next run
