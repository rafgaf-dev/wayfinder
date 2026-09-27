"""HTTP, backoff and JSONL helpers shared by the fetch scripts.

The fetchers store raw responses as JSON Lines, one record per line, appended
and flushed as they arrive. That makes every fetch resumable: a crash loses at
most the line being written, and the next run skips what is already on disk.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, TextIO, TypeVar

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "wayfinder/0.1 (dataset build for a research project)"

T = TypeVar("T")


class TransientError(Exception):
    """A failure worth retrying after a pause: throttling, 5xx, network."""


def get_json(url: str, params: dict, retryable: set[int], timeout: float = 30.0):
    # Never log the full URL: for keyed APIs it contains the key.
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as e:  # subclass of URLError, so caught first
        if e.code in retryable:
            raise TransientError(f"HTTP {e.code}") from e
        raise
    except (urllib.error.URLError, TimeoutError) as e:
        raise TransientError(f"network error: {e}") from e

    payload = json.loads(body)
    # The Steam store sometimes returns a bare `null` while throttling.
    if payload is None:
        raise TransientError("null response body")
    return payload


def with_backoff(fn: Callable[[], T], start_s: float, max_s: float, retries: int = 5) -> T:
    delay = start_s
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except TransientError as e:
            log.warning("%s; backing off %.0fs (retry %d/%d)", e, delay, attempt, retries)
            time.sleep(delay)
            delay = min(delay * 2, max_s)
    # Persistent failure usually means we are blocked or out of quota.
    # Stopping is better than skipping: skipped items would silently thin the
    # sample, whereas a stopped run resumes from exactly this point.
    return fn()


def pace(started: float, interval: float) -> None:
    # Pace from the start of the request, so its own latency counts towards
    # the interval rather than being added on top.
    time.sleep(max(0.0, interval - (time.monotonic() - started)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield records one at a time, skipping a line truncated by a crash."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("ignoring a truncated line in %s", path)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def open_for_append(path: Path) -> TextIO:
    # If the last run died mid-write, start on a fresh line rather than
    # gluing the next record onto the truncated one.
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    out = path.open("a", encoding="utf-8")
    if needs_newline:
        out.write("\n")
    return out


def write_record(out: TextIO, record: dict) -> None:
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()
