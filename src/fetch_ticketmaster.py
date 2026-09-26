"""Fetch raw Ticketmaster Discovery API events for a fixed run of days.

Scope: events in one country (GB by default) starting within --days days of
--start-date, in the three segments that map to districts: Music, Sports and
Arts & Theatre. Film and Miscellaneous are left out because neither maps
cleanly to a district.

Why the query is sliced: the Discovery API will not page past the 1,000th
result of a query (page * size must stay below 1,000). A query for a whole
fortnight would be cut short without any error, and which events survived
would depend on the sort order. In date order, the late events of each day
would be the ones dropped, and start time is exactly what time_of_day is
labelled from. So each (segment, day) is fetched as its own slice, and any
slice with more than 1,000 events is halved in time until every piece fits.

Output, in data/raw/ticketmaster/:
  events.jsonl  each event exactly as received, with the slice it came from
  slices.jsonl  one line per completed (segment, day), so a rerun skips it

Ticketmaster only lists upcoming events, so a later run cannot reproduce an
earlier one. The raw files are the record of what was fetched.

Usage:
    TICKETMASTER_API_KEY=... python src/fetch_ticketmaster.py --days 14
"""

import argparse
import logging
import os
import time
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fetch_common import (
    REPO_ROOT,
    get_json,
    now_iso,
    open_for_append,
    pace,
    read_jsonl,
    with_backoff,
    write_record,
)

log = logging.getLogger("fetch_ticketmaster")

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "ticketmaster"
EVENTS_URL = "https://app.ticketmaster.com/discovery/v2/events.json"

# Segment ids are stable identifiers in Ticketmaster's classification tree.
# Every returned event is checked against the requested segment, so a wrong
# id would show up in the log rather than silently mislabel data.
SEGMENTS = {
    "music": ("KZFzniwnSyZfZ7v7nJ", "Music"),
    "sports": ("KZFzniwnSyZfZ7v7nE", "Sports"),
    "arts": ("KZFzniwnSyZfZ7v7na", "Arts & Theatre"),
}

PAGE_SIZE = 200  # the API maximum
MAX_REACHABLE = 1000  # deep-paging limit: page * size < 1000
MIN_WINDOW = timedelta(hours=1)  # stop halving here, and report truncation

# The free tier allows 5 requests per second and 5,000 per day. Spikes clear
# within seconds, so backoff starts short, unlike the Steam fetcher. If the
# daily quota is exhausted, the retries run out and the run stops; completed
# slices are already saved.
INTERVAL_S = 0.3
BACKOFF_START_S = 5.0
BACKOFF_MAX_S = 120.0
RETRYABLE = {429, 500, 502, 503, 504}
DEFAULT_MAX_CALLS = 4500  # stop between slices before touching the daily cap


class Discovery:
    """Paced, retrying access to the event search endpoint."""

    def __init__(self, key: str, country: str):
        self.key = key
        self.country = country
        self.calls = 0
        self._last_start = 0.0

    def search(self, segment_id: str, start: datetime, end: datetime, page: int) -> dict:
        params = {
            "apikey": self.key,
            "countryCode": self.country,
            "segmentId": segment_id,
            "startDateTime": fmt(start),
            "endDateTime": fmt(end),
            "size": PAGE_SIZE,
            "page": page,
            # A fully determined order keeps pagination stable between calls.
            # With relevance order, results can shift between pages and
            # events get skipped or repeated.
            "sort": "date,name,asc",
            "includeTest": "no",
        }

        def call() -> dict:
            pace(self._last_start, INTERVAL_S)
            self._last_start = time.monotonic()
            self.calls += 1
            return get_json(EVENTS_URL, params, RETRYABLE)

        return with_backoff(call, BACKOFF_START_S, BACKOFF_MAX_S)


def fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_window(api: Discovery, segment_id: str, start: datetime, end: datetime) -> tuple[list[dict], bool]:
    """Return every event in [start, end], halving the window as needed.

    The second value is True if some piece still exceeded the paging limit at
    MIN_WINDOW, so its events are incomplete.
    """
    first = api.search(segment_id, start, end, 0)
    total = first.get("page", {}).get("totalElements", 0)

    if total > MAX_REACHABLE and end - start > MIN_WINDOW:
        # Round to the second: the API takes whole seconds, and a fractional
        # midpoint would leave a sliver between the two halves.
        half = timedelta(seconds=int((end - start).total_seconds()) // 2)
        left, t1 = fetch_window(api, segment_id, start, start + half)
        right, t2 = fetch_window(api, segment_id, start + half, end)
        return left + right, t1 or t2

    truncated = total > MAX_REACHABLE
    if truncated:
        log.warning("%s to %s has %d events; only the first %d are reachable",
                    fmt(start), fmt(end), total, MAX_REACHABLE)

    events = first.get("_embedded", {}).get("events", [])
    pages = min(first.get("page", {}).get("totalPages", 0), MAX_REACHABLE // PAGE_SIZE)
    for page in range(1, pages):
        events += api.search(segment_id, start, end, page).get("_embedded", {}).get("events", [])
    return events, truncated


def primary_segment_id(event: dict) -> str | None:
    # About a quarter of events carry a single classification with no
    # `primary` flag. Treat the first classification as primary in that case.
    classifications = event.get("classifications", [])
    primary = next((c for c in classifications if c.get("primary")), None)
    chosen = primary or (classifications[0] if classifications else {})
    return chosen.get("segment", {}).get("id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None,
                        help="first day, YYYY-MM-DD (UTC); defaults to today")
    parser.add_argument("--country", default="GB")
    parser.add_argument("--segments", nargs="+", choices=sorted(SEGMENTS), default=sorted(SEGMENTS))
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                        help="stop between slices after this many API calls")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    key = os.environ.get("TICKETMASTER_API_KEY")
    if not key:
        raise SystemExit("TICKETMASTER_API_KEY is not set. Get a key at https://developer.ticketmaster.com.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "events.jsonl"
    slices_path = args.out_dir / "slices.jsonl"

    first_day = args.start_date or datetime.now(timezone.utc).date()
    days = [first_day + timedelta(days=i) for i in range(args.days)]
    done = {(r["segment"], r["day"]) for r in read_jsonl(slices_path)}
    seen = {r["id"] for r in read_jsonl(events_path)}
    log.info("resuming with %d slices done and %d events stored", len(done), len(seen))

    api = Discovery(key, args.country)
    new_events: list[dict] = []
    with open_for_append(events_path) as events_out, open_for_append(slices_path) as slices_out:
        # Days outer, segments inner, so a run stopped part-way still has
        # every segment covered over the same days.
        for day in days:
            for segment in args.segments:
                if (segment, day.isoformat()) in done:
                    continue
                if api.calls >= args.max_calls:
                    log.warning("reached --max-calls (%d); rerun tomorrow to continue", args.max_calls)
                    return summarise(new_events, api.calls)

                segment_id, segment_name = SEGMENTS[segment]
                start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
                calls_before = api.calls
                try:
                    events, truncated = fetch_window(api, segment_id, start, start + timedelta(days=1))
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        raise SystemExit("Ticketmaster rejected the API key (HTTP 401).") from e
                    raise

                # Windows share their boundary second, so an event can arrive
                # twice. Keep the first copy only.
                fresh = []
                for event in events:
                    if event["id"] not in seen:
                        seen.add(event["id"])
                        fresh.append(event)
                mismatched = sum(primary_segment_id(e) != segment_id for e in fresh)
                if mismatched:
                    log.warning("%s %s: %d events whose primary segment is not %s",
                                segment, day, mismatched, segment_name)

                fetched_at = now_iso()
                for event in fresh:
                    write_record(events_out, {
                        "id": event["id"],
                        "segment": segment,
                        "slice_day": day.isoformat(),
                        "fetched_at": fetched_at,
                        "event": event,
                    })
                # Written after the events, so a crash between the two
                # re-fetches the slice rather than marking it done unfinished.
                write_record(slices_out, {
                    "segment": segment,
                    "day": day.isoformat(),
                    "fetched_at": fetched_at,
                    "country": args.country,
                    "n_events": len(events),
                    "n_new": len(fresh),
                    "calls": api.calls - calls_before,
                    "truncated": truncated,
                })
                new_events += fresh
                log.info("%s %s: %d events (%d new), %d calls so far",
                         segment, day, len(events), len(fresh), api.calls)

    summarise(new_events, api.calls)


def summarise(events: list[dict], calls: int) -> None:
    """Log how much usable prose came back.

    Ticketmaster events often carry little or no descriptive text, and the
    model sees only prose. Seeing this at fetch time lets us decide early
    whether the domain is viable, rather than finding out in build_dataset.
    """
    n = len(events)
    if not n:
        log.info("no new events this run (%d calls)", calls)
        return

    def share(field: str) -> str:
        return f"{sum(bool((e.get(field) or '').strip()) for e in events) / n:.0%}"

    attractions = {a["id"] for e in events for a in e.get("_embedded", {}).get("attractions", [])}
    log.info("this run: %d new events, %d distinct attractions, %d calls", n, len(attractions), calls)
    log.info("non-empty text: description %s, info %s, pleaseNote %s",
             share("description"), share("info"), share("pleaseNote"))


if __name__ == "__main__":
    main()
