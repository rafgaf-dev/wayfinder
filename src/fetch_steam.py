"""Fetch raw Steam records for a reproducible random sample of games.

Three stages, each resumable on its own:

1. App list. Every game appid on the store, from IStoreService/GetAppList.
   This needs a free Steam Web API key in STEAM_API_KEY, because the old
   keyless ISteamApps/GetAppList/v2 endpoint has been withdrawn. The list is
   cached to applist.json, which freezes the sample frame: the live list
   changes daily. To use a list from elsewhere, write that file yourself as
   {"apps": [{"appid": 620}, ...]} and no key is needed.

2. App details. store.steampowered.com/api/appdetails, which is keyless. It
   takes one app per request, because it ignores multiple appids unless the
   response is filtered down to price data. Appended to appdetails.jsonl.

3. SteamSpy. steamspy.com/api.php?request=appdetails, keyless, for every game
   stored in stage 2. It is here for the user tags ("Relaxing", "Dark Humor",
   "Replay Value", ...), which carry the mood and commitment signal that the
   store's genres lack. Its playtime fields are zero for every game since
   Valve's 2018 privacy change, so they are useless. Appended to
   steamspy.jsonl.

Records are stored exactly as received. Cleaning, filtering and mapping to the
taxonomy happen in build_dataset.py, so changing a filter never requires a
re-fetch.

Usage:
    STEAM_API_KEY=... python src/fetch_steam.py --target-games 2000
"""

import argparse
import json
import logging
import os
import random
import time
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

log = logging.getLogger("fetch_steam")

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "steam"

APPLIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAMSPY_URL = "https://steamspy.com/api.php"

# The store answers throttling with 403 as well as 429. On the Web API a 403
# means a bad key, so retrying it there would only waste time.
STORE_RETRYABLE = {403, 429, 500, 502, 503, 504}
WEBAPI_RETRYABLE = {429, 500, 502, 503, 504}
STEAMSPY_RETRYABLE = {429, 500, 502, 503, 504}

# The store allows roughly 200 requests per rolling 5 minutes, i.e. one every
# 1.5 s. SteamSpy asks for at most one appdetails request per second. Both
# defaults leave a little headroom.
DEFAULT_STORE_INTERVAL_S = 1.6
STEAMSPY_INTERVAL_S = 1.1

# Once throttled, short retries only spend more of the same 5-minute budget,
# so the first backoff is a full minute and doubles from there.
BACKOFF_START_S = 60.0
BACKOFF_MAX_S = 600.0

PROGRESS_EVERY = 50


# --- Stage 1: app list -------------------------------------------------------

def fetch_applist(key: str) -> list[dict]:
    apps, last_appid = [], 0
    while True:
        params = {
            "key": key,
            "include_games": "true",
            "include_dlc": "false",
            "include_software": "false",
            "include_videos": "false",
            "include_hardware": "false",
            "max_results": 50000,
            "last_appid": last_appid,
        }
        page = with_backoff(
            lambda: get_json(APPLIST_URL, params, WEBAPI_RETRYABLE), BACKOFF_START_S, BACKOFF_MAX_S
        )["response"]
        apps.extend({"appid": a["appid"], "name": a.get("name", "")} for a in page.get("apps", []))
        log.info("app list: %d games so far", len(apps))
        if not page.get("have_more_results"):
            return apps
        last_appid = page["last_appid"]


def load_or_fetch_applist(path: Path) -> list[dict]:
    if path.exists():
        apps = json.loads(path.read_text())["apps"]
        log.info("using cached app list: %s (%d apps)", path, len(apps))
        return apps

    key = os.environ.get("STEAM_API_KEY")
    if not key:
        raise SystemExit(
            "STEAM_API_KEY is not set and there is no cached app list at "
            f"{path}. Get a key at https://steamcommunity.com/dev/apikey."
        )
    apps = fetch_applist(key)
    snapshot = {"fetched_at": now_iso(), "source": APPLIST_URL, "apps": apps}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot))
    tmp.replace(path)  # atomic, so a crash never leaves a half-written list
    return apps


# --- Stage 2: store app details ----------------------------------------------

def fetch_store_details(appid: int) -> dict:
    payload = get_json(
        APPDETAILS_URL,
        {"appids": appid, "l": "english", "cc": "gb"},
        STORE_RETRYABLE,
    )
    # The response is a one-entry dict keyed by an appid, but not always the
    # one requested: Steam sometimes keys it by a related app such as a DLC.
    # Take the single entry and rely on data.steam_appid instead.
    (entry,) = payload.values()

    record = {"requested_appid": appid, "fetched_at": now_iso()}
    data = entry.get("data") if entry.get("success") else None
    if data is None:
        record["status"] = "unavailable"  # delisted, region-locked, unreleased
    elif data.get("type") != "game":
        record.update(status="not_game", type=data.get("type"))
    else:
        record.update(status="ok", steam_appid=data["steam_appid"], data=data)
    return record


def run_store_stage(order: list[int], path: Path, target: int, interval: float) -> set[int]:
    """Fetch store details down `order` until `target` games are stored.

    Returns the steam_appids of every game stored, including earlier runs.
    """
    records = read_jsonl(path)
    requested = {r["requested_appid"] for r in records}
    stored = {r["steam_appid"] for r in records if r["status"] == "ok"}
    log.info("store: resuming with %d requested, %d games stored", len(requested), len(stored))

    counts: dict[str, int] = {}
    with open_for_append(path) as out:
        for appid in order:
            if len(stored) >= target:
                break
            if appid in requested:
                continue

            started = time.monotonic()
            record = with_backoff(lambda: fetch_store_details(appid), BACKOFF_START_S, BACKOFF_MAX_S)

            # Old appids can redirect to a current one, so two requests may
            # return the same game. Keep the first; later ones are markers.
            if record["status"] == "ok":
                if record["steam_appid"] in stored:
                    record = {**record, "status": "duplicate"}
                    del record["data"]
                else:
                    stored.add(record["steam_appid"])

            write_record(out, record)
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            n = sum(counts.values())
            if n % PROGRESS_EVERY == 0:
                log.info("store: %d requests this run %s; %d/%d games stored",
                         n, counts, len(stored), target)
            pace(started, interval)

    if len(stored) < target:
        log.warning("store: app list exhausted with %d games stored", len(stored))
    log.info("store: done %s; %d games stored in %s", counts, len(stored), path)
    return stored


# --- Stage 3: SteamSpy tags --------------------------------------------------

def fetch_steamspy(steam_appid: int) -> dict:
    data = get_json(
        STEAMSPY_URL,
        {"request": "appdetails", "appid": steam_appid},
        STEAMSPY_RETRYABLE,
    )
    # SteamSpy answers unknown apps with a normal-looking record whose name is
    # null. `tags` is a {tag: votes} dict, or [] when a game has no tags (an
    # empty PHP array); both are kept as received.
    status = "unknown" if data.get("name") is None else "ok"
    return {"steam_appid": steam_appid, "fetched_at": now_iso(), "status": status, "data": data}


def run_steamspy_stage(steam_appids: set[int], path: Path) -> None:
    done = {r["steam_appid"] for r in read_jsonl(path)}
    # Sorted so a run's progress is predictable; order has no effect on data.
    todo = sorted(steam_appids - done)
    log.info("steamspy: %d already fetched, %d to go", len(done), len(todo))

    counts: dict[str, int] = {}
    with open_for_append(path) as out:
        for steam_appid in todo:
            started = time.monotonic()
            record = with_backoff(lambda: fetch_steamspy(steam_appid), BACKOFF_START_S, BACKOFF_MAX_S)
            write_record(out, record)
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            n = sum(counts.values())
            if n % PROGRESS_EVERY == 0:
                log.info("steamspy: %d/%d %s", n, len(todo), counts)
            pace(started, STEAMSPY_INTERVAL_S)
    log.info("steamspy: done %s in %s", counts, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-games", type=int, default=2000,
                        help="stop once this many distinct games are stored in total")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--interval", type=float, default=DEFAULT_STORE_INTERVAL_S,
                        help="minimum seconds between store request starts")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    apps = load_or_fetch_applist(args.out_dir / "applist.json")

    # Sort before shuffling so the permutation depends only on the seed and
    # the set of appids, not on the order the API happened to return them in.
    # The sample is always a prefix of this one permutation, so raising
    # --target-games later extends the sample instead of redrawing it.
    order = sorted({a["appid"] for a in apps})
    random.Random(args.seed).shuffle(order)

    stored = run_store_stage(order, args.out_dir / "appdetails.jsonl", args.target_games, args.interval)
    run_steamspy_stage(stored, args.out_dir / "steamspy.jsonl")


if __name__ == "__main__":
    main()
