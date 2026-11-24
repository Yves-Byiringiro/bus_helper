"""Look up Centro bus schedule data from a GTFS folder.

GTFS is just a folder of CSV-style .txt files. We load the ones we need
with pandas and answer simple questions from them.

To switch to a newer schedule, replace the .txt files in the data/ folder
(or change DATA_DIR). Nothing else needs to change.
"""
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TZ = ZoneInfo("America/New_York")
LOOKAHEAD_DAYS = 14  # how far ahead we search for the next day with service
WALK_METERS = 400  # how far we assume someone walks to or from a stop (~5 minutes)
TRANSFER_METERS = 150  # how far someone walks between stops when changing buses
MIN_TRANSFER_SECS = 3 * 60  # time allowed to change buses
MAX_TRANSFER_WAIT_SECS = 45 * 60  # longest wait we suggest between buses
SEARCH_WINDOW_SECS = 3 * 3600  # trip planner first looks this far ahead
# Rough box around Syracuse (lon, lat, lon, lat), so an address can't match another city.
SYRACUSE_BOX = "-76.35,43.20,-75.90,42.90"

_cache = {}


def _load():
    """Read the files once and keep them in memory."""
    if not _cache:
        read = lambda name: pd.read_csv(
            os.path.join(DATA_DIR, name), dtype=str, keep_default_na=False
        )
        _cache["stops"] = read("stops.txt")
        _cache["routes"] = read("routes.txt")
        _cache["trips"] = read("trips.txt")
        _cache["stop_times"] = read("stop_times.txt")
        _cache["calendar"] = read("calendar.txt")
        _cache["calendar_dates"] = read("calendar_dates.txt")
        _cache["feed_info"] = read("feed_info.txt")

        stops = _cache["stops"]
        # The downtown Transit Hub's bays are named just "A1".."C22" in the feed; make them recognizable.
        bay = stops["stop_name"].str.fullmatch(r"[A-C]\d{1,2}")
        stops.loc[bay, "stop_name"] = "Centro Transit Hub, bay " + stops.loc[bay, "stop_name"]
        stops["lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

        # Stop times with trip and route info and times as numbers, for the trip planner.
        st = _cache["stop_times"][["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"]]
        st = st.merge(
            _cache["trips"][["trip_id", "route_id", "service_id", "trip_headsign"]], on="trip_id"
        ).merge(_cache["routes"][["route_id", "route_short_name"]], on="route_id")
        st["seq"] = st["stop_sequence"].astype(int)
        st["arr"] = st["arrival_time"].map(_to_seconds)
        st["dep"] = st["departure_time"].map(_to_seconds)
        _cache["planner"] = st.drop(columns=["stop_sequence", "arrival_time", "departure_time"])
    return _cache


def feed_coverage():
    """Which dates does this schedule file cover?"""
    info = _load()["feed_info"].iloc[0]
    return {
        "start": info["feed_start_date"],  # YYYYMMDD
        "end": info["feed_end_date"],
        "version": info["feed_version"],
    }


def _to_seconds(hms):
    """'6:08:09' -> seconds. GTFS allows 25:10:00 for after-midnight trips."""
    h, m, s = (int(p) for p in hms.split(":"))
    return h * 3600 + m * 60 + s


def _fmt12(hms):
    """'14:05:00' -> '2:05 PM'. Also handles after-midnight times like '25:10:00'."""
    h, m, _ = (int(p) for p in hms.strip().split(":"))
    h24 = h % 24
    suffix = "AM" if h24 < 12 else "PM"
    return f"{h24 % 12 or 12}:{m:02d} {suffix}"


def _now(date=None, time=None):
    """Use the real current time unless a test date/time is given."""
    now = datetime.now(TZ)
    if date:
        d = datetime.strptime(date, "%Y-%m-%d")
        now = now.replace(year=d.year, month=d.month, day=d.day)
    if time:
        t = datetime.strptime(time, "%H:%M")
        now = now.replace(hour=t.hour, minute=t.minute, second=0)
    return now


def _label(day):
    """'Monday 2026-11-23'"""
    return day.strftime("%A %Y-%m-%d")


def _in_feed(day):
    cov = feed_coverage()
    return cov["start"] <= day.strftime("%Y%m%d") <= cov["end"]


def _active_service_ids(day):
    """Which service_ids run on this calendar date?"""
    data = _load()
    ymd = day.strftime("%Y%m%d")
    weekday = day.strftime("%A").lower()
    cal = data["calendar"]
    running = set(
        cal[
            (cal["start_date"] <= ymd)
            & (cal["end_date"] >= ymd)
            & (cal[weekday] == "1")
        ]["service_id"]
    )
    exc = data["calendar_dates"]
    exc = exc[exc["date"] == ymd]
    running |= set(exc[exc["exception_type"] == "1"]["service_id"])  # added
    running -= set(exc[exc["exception_type"] == "2"]["service_id"])  # removed
    return running


def _stop_table(stop_id):
    """Every scheduled stop at this stop, with route, trip, and service info."""
    data = _load()
    st = data["stop_times"]
    st = st[st["stop_id"] == str(stop_id)]
    df = st.merge(data["trips"], on="trip_id").merge(data["routes"], on="route_id")
    df = df.copy()
    df["secs"] = df["departure_time"].map(_to_seconds)
    return df


def _departures_on(df, day, after_secs=0, limit=5, latest=False):
    """Departures on one date. latest=True returns the LAST ones of the day."""
    rows = df[df["service_id"].isin(_active_service_ids(day)) & (df["secs"] >= after_secs)]
    rows = rows.sort_values("secs")
    rows = rows.tail(limit) if latest else rows.head(limit)
    return [
        {
            "route": r.route_short_name,
            "trip_label": r.trip_headsign,  # Centro's own text, e.g. "40 Syracuse University to Hub"
            "time": _fmt12(r.departure_time)
            + (" (after midnight)" if r.secs >= 86400 else ""),
        }
        for r in rows.itertuples()
    ]


def find_stops(name, limit=15):
    """Search stops by part of their name, e.g. 'Marshall' or 'Destiny'."""
    stops = _load()["stops"]
    hits = stops[stops["stop_name"].str.contains(name, case=False, na=False, regex=False)]
    return [
        {"stop_id": r.stop_id, "stop_name": r.stop_name}
        for r in hits.head(limit).itertuples()
    ]


def routes_at_stop(stop_id, date=None):
    """Routes that serve this stop, and whether each one is actually running soon.

    A route can appear in the schedule file but have no service for weeks
    (for example a State Fair shuttle, or a route that skips Thanksgiving week).
    """
    now = _now(date)
    df = _stop_table(stop_id)
    days = [now + timedelta(days=i) for i in range(LOOKAHEAD_DAYS)]
    active = {d: _active_service_ids(d) for d in days if _in_feed(d)}
    out = []
    for (short, long_), g in df.groupby(["route_short_name", "route_long_name"]):
        sids = set(g["service_id"])
        running = [d for d, ids in active.items() if sids & ids]
        out.append(
            {
                "route": short,
                "name": long_,
                f"days_with_service_next_{LOOKAHEAD_DAYS}": len(running),
                "next_service_date": _label(running[0]) if running else None,
            }
        )
    return out


def next_departures(stop_id, limit=5, date=None, time=None, latest=False):
    """Scheduled departures at a stop (scheduled, NOT live).

    date  "YYYY-MM-DD" (default today)    time "HH:MM" 24-hour (default now)
    latest=True gives the last departures of that day instead of the next ones.
    If nothing is left on that date, also reports the next day with service.
    """
    now = _now(date, time)
    cov = feed_coverage()
    if not _in_feed(now):
        return {
            "error": "The schedule file does not cover this date.",
            "date_asked": now.strftime("%Y-%m-%d"),
            "schedule_runs_from": cov["start"],
            "schedule_runs_to": cov["end"],
        }

    df = _stop_table(stop_id)
    after = 0 if latest else now.hour * 3600 + now.minute * 60
    deps = _departures_on(df, now, after, limit, latest)
    result = {
        "stop_id": str(stop_id),
        "date": _label(now),
        "as_of": now.strftime("%I:%M %p").lstrip("0"),
        "source": "scheduled times from Centro schedule file (not live)",
        "departures": deps,
    }
    if latest:
        result.pop("as_of")  # the current clock time is irrelevant for "last bus" questions
    if not deps:
        result["note"] = "No scheduled departures left at this stop on this date."
        result["next_service"] = None
        for i in range(1, LOOKAHEAD_DAYS + 1):
            day = now + timedelta(days=i)
            if not _in_feed(day):
                break
            nxt = _departures_on(df, day, 0, limit)
            if nxt:
                result["next_service"] = {"date": _label(day), "departures": nxt}
                break
        if result["next_service"] is None:
            result["next_service_note"] = (
                f"No service at this stop in the next {LOOKAHEAD_DAYS} days."
            )
    return result


# ---------- Trip planner ----------


def _meters(lat1, lon1, lat2, lon2):
    """Straight-line distance in meters (works on numbers or arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def _near_point(lat, lon, radius):
    """Stops within `radius` meters of a point, closest first."""
    stops = _load()["stops"]
    out = stops.assign(meters=_meters(lat, lon, stops["lat"], stops["lon"]))
    return out[out["meters"] <= radius].sort_values("meters")


def _nearby(stop_id, radius):
    """Stops within `radius` meters of a stop (including itself), closest first."""
    stops = _load()["stops"]
    me = stops[stops["stop_id"] == str(stop_id)]
    if me.empty:
        return None
    return _near_point(me["lat"].iloc[0], me["lon"].iloc[0], radius)


PLACES_PATH = os.path.join(os.path.dirname(__file__), "places.csv")


def _match_places(name):
    """Rows of places.csv whose name, address, or category contain every word of `name`."""
    _load()  # schedule first, so the cache is not mistaken for already loaded
    if "places" not in _cache:
        _cache["places"] = pd.read_csv(PLACES_PATH, dtype={"name": str, "address": str}, keep_default_na=False)
    places = _cache["places"]
    words = name.lower().replace("'", "").split()
    text = (places["name"] + " " + places["address"] + " " + places["category"]).str.lower().str.replace("'", "")
    return places[text.map(lambda t: all(w in t for w in words))]


def find_place(name, limit=5):
    """Search places.csv (stores and other destinations) by name or category.

    The schedule file only knows stop names, so this is how we connect a store
    to the bus stops closest to it.
    """
    hits = _match_places(name)
    out = []
    for r in hits.head(limit).itertuples():
        near = _near_point(r.lat, r.lon, WALK_METERS).head(4)
        out.append(
            {
                "name": r.name,
                "address": r.address,
                "category": r.category,
                "source": r.source,
                "nearest_stops": [
                    {"stop_id": s.stop_id, "stop_name": s.stop_name, "meters": int(s.meters)}
                    for s in near.itertuples()
                ],
            }
        )
    return out


def stops_near(stop_id, radius_m=WALK_METERS, limit=10):
    """Other stops within walking distance of a stop (straight-line distance)."""
    near = _nearby(stop_id, radius_m)
    if near is None:
        return {"error": f"Unknown stop id {stop_id}"}
    return [
        {"stop_id": r.stop_id, "stop_name": r.stop_name, "meters": int(r.meters)}
        for r in near.head(limit).itertuples()
    ]


def _running_on(day):
    """Stop times of every trip running on this date, in seconds after midnight of that date.

    Trips after midnight belong to the previous day's schedule (times like 24:30:00),
    so they are added with run = -1 and their times shifted back by a day.
    """
    st = _load()["planner"]
    today = st[st["service_id"].isin(_active_service_ids(day))].assign(run=0)
    late = st[st["service_id"].isin(_active_service_ids(day - timedelta(days=1))) & (st["arr"] >= 86400)]
    late = late.assign(run=-1, arr=late["arr"] - 86400, dep=late["dep"] - 86400)
    return pd.concat([today, late])


def _close_pairs(a_ids, b_ids, radius):
    """Pairs of stops (a, b) within `radius` meters of each other, for changing buses."""
    stops = _load()["stops"].set_index("stop_id")
    a, b = stops.loc[list(a_ids)], stops.loc[list(b_ids)]
    d = _meters(a["lat"].values[:, None], a["lon"].values[:, None], b["lat"].values[None, :], b["lon"].values[None, :])
    i, j = np.nonzero(d <= radius)
    return pd.DataFrame({"stop_a": a.index[i], "stop_b": b.index[j], "transfer_m": d[i, j]})


_TRIP_COLS = ["trip_id", "run", "route_id", "route_short_name", "trip_headsign"]


def _boardings(st, stop_ids, after, until, tag):
    """Buses leaving these stops between after and until. Columns stop_id_<tag>, seq_<tag>, dep_<tag>."""
    rows = st[st["stop_id"].isin(stop_ids) & st["dep"].between(after, until)]
    return rows[_TRIP_COLS + ["stop_id", "seq", "dep"]].rename(
        columns={"stop_id": f"stop_id_{tag}", "seq": f"seq_{tag}", "dep": f"dep_{tag}"}
    )


def _arrivals(st, stop_ids, tag, cols=_TRIP_COLS):
    """Buses reaching these stops. Columns stop_id_<tag>, seq_<tag>, arr_<tag>."""
    rows = st if stop_ids is None else st[st["stop_id"].isin(stop_ids)]
    return rows[cols + ["stop_id", "seq", "arr"]].rename(
        columns={"stop_id": f"stop_id_{tag}", "seq": f"seq_{tag}", "arr": f"arr_{tag}"}
    )


def _direct_options(st, origins, dests, after, until):
    o = _boardings(st, origins.index, after, until, "o")
    d = _arrivals(st, dests.index, "d", cols=["trip_id", "run"])
    m = o.merge(d, on=["trip_id", "run"])
    m = m[m["seq_d"] > m["seq_o"]]
    # For each bus: the earliest stop where you can get off, and the latest stop where you can get on.
    m = m.sort_values(["arr_d", "dep_o"], ascending=[True, False]).drop_duplicates(["trip_id", "run"])
    return [
        {
            "arrive_secs": r.arr_d,
            "legs": [_leg(r.route_short_name, r.trip_headsign, r.stop_id_o, r.dep_o, r.stop_id_d, r.arr_d)],
            "walk_to_first_stop_m": int(origins[r.stop_id_o]),
            "walk_from_last_stop_m": int(dests[r.stop_id_d]),
        }
        for r in m.itertuples()
    ]


def _transfer_options(st, origins, dests, after, until):
    """Trips with exactly one change of bus."""
    # First bus: from the start to any later stop on that trip (a = where you get off).
    leg1 = _boardings(st, origins.index, after, until, "o").merge(
        _arrivals(st, None, "a", cols=["trip_id", "run"]), on=["trip_id", "run"]
    )
    leg1 = leg1[leg1["seq_a"] > leg1["seq_o"]]

    # Second bus: from any earlier stop on that trip (b = where you get on) to the destination.
    leg2 = _arrivals(st, dests.index, "d").merge(
        _boardings(st, st["stop_id"].unique(), after, 10**6, "b")[["trip_id", "run", "stop_id_b", "seq_b", "dep_b"]],
        on=["trip_id", "run"],
    )
    leg2 = leg2[leg2["seq_b"] < leg2["seq_d"]]
    if leg1.empty or leg2.empty:
        return []

    pairs = _close_pairs(set(leg1["stop_id_a"]), set(leg2["stop_id_b"]), TRANSFER_METERS)
    m = leg1.merge(pairs, left_on="stop_id_a", right_on="stop_a").merge(
        leg2, left_on="stop_b", right_on="stop_id_b", suffixes=("_1", "_2")
    )
    wait = m["dep_b"] - m["arr_a"]
    other_bus = (m["trip_id_1"] != m["trip_id_2"]) | (m["run_1"] != m["run_2"])
    m = m[wait.between(MIN_TRANSFER_SECS, MAX_TRANSFER_WAIT_SECS) & other_bus]
    m = m.sort_values(["arr_d", "dep_o", "transfer_m"], ascending=[True, False, True])
    m = m.drop_duplicates(["trip_id_2", "run_2"]).drop_duplicates(["trip_id_1", "run_1"])
    return [
        {
            "arrive_secs": r.arr_d,
            "legs": [
                _leg(r.route_short_name_1, r.trip_headsign_1, r.stop_id_o, r.dep_o, r.stop_id_a, r.arr_a),
                _leg(r.route_short_name_2, r.trip_headsign_2, r.stop_id_b, r.dep_b, r.stop_id_d, r.arr_d),
            ],
            "walk_between_buses_m": int(r.transfer_m),
            "walk_to_first_stop_m": int(origins[r.stop_id_o]),
            "walk_from_last_stop_m": int(dests[r.stop_id_d]),
        }
        for r in m.itertuples()
    ]


def _stop_name(stop_id):
    stops = _load()["stops"]
    return stops.loc[stops["stop_id"] == stop_id, "stop_name"].iloc[0]


def _fmt_secs(secs):
    secs = int(secs)
    return _fmt12(f"{secs // 3600}:{secs % 3600 // 60}:00")


def _leg(route, label, board, dep, alight, arr):
    return {
        "route": route,
        "trip_label": label,
        "board_at": f"{_stop_name(board)} (stop {board})",
        "departs": _fmt_secs(dep),
        "get_off_at": f"{_stop_name(alight)} (stop {alight})",
        "arrives": _fmt_secs(arr),
    }


@lru_cache(maxsize=256)
def _geocode(address):
    """Street address -> (lat, lon, matched address) from OpenStreetMap, or None if not found."""
    q = urllib.parse.urlencode(
        {"q": address, "format": "json", "limit": 1, "viewbox": SYRACUSE_BOX, "bounded": 1, "countrycodes": "us"}
    )
    req = urllib.request.Request(
        "https://nominatim.openstreetmap.org/search?" + q,
        headers={"User-Agent": "su-bus-helper-class-project/1.0"},  # required by OpenStreetMap
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        hits = json.load(resp)
    if not hits:
        return None
    return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0]["display_name"]


def find_address(address):
    """Nearest bus stops to a street address in Syracuse, e.g. '130 Redfield Pl'."""
    hit = _geocode(address)
    if hit is None:
        return {"error": f"Could not find the address '{address}' in Syracuse. Ask for a nearby intersection."}
    lat, lon, matched = hit
    near = _near_point(lat, lon, WALK_METERS).head(4)
    return {
        "matched_address": matched,
        "source": "location from OpenStreetMap",
        "nearest_stops": [
            {"stop_id": s.stop_id, "stop_name": s.stop_name, "meters": int(s.meters)} for s in near.itertuples()
        ],
    }


def _endpoint(stop_id, place, address=None):
    """(label, stops within walking distance with meters) for a stop id, place name, or street address."""
    if address:
        hit = _geocode(address)
        if hit is None:
            raise LookupError(f"Could not find the address '{address}' in Syracuse.")
        lat, lon, matched = hit
        label, near = matched, _near_point(lat, lon, WALK_METERS)
        if near.empty:
            raise LookupError(f"No bus stop within {WALK_METERS} m of {matched}.")
    elif place:
        hits = _match_places(place)
        if hits.empty:
            raise LookupError(f"No place matching '{place}'. Use find_place, or give a stop id.")
        r = hits.iloc[0]
        label, near = f"{r['name']} ({r['address']})", _near_point(r["lat"], r["lon"], WALK_METERS)
        if near.empty:
            raise LookupError(f"No bus stop within {WALK_METERS} m of {r['name']}.")
    else:
        near = _nearby(stop_id, WALK_METERS)
        if near is None:
            raise LookupError(f"Unknown stop id {stop_id}. Use find_stops to get a valid stop id.")
        label = f"{_stop_name(str(stop_id))} (stop {stop_id})"
    return label, near.set_index("stop_id")["meters"]


def _steps(option, to_label):
    """Plain-language directions for one trip option, so the answer always says where to get off."""
    legs = option["legs"]
    steps = []
    if option["walk_to_first_stop_m"]:
        steps.append(f"Walk about {option['walk_to_first_stop_m']} m to {legs[0]['board_at']}.")
    for i, leg in enumerate(legs):
        if i:
            walk = option.get("walk_between_buses_m", 0)
            if walk:
                steps.append(f"Walk about {walk} m to {leg['board_at']}.")
            else:
                steps.append(f"Change buses at the same stop, {leg['board_at']}.")
        steps.append(f"Take {leg['route']} (\"{leg['trip_label']}\") at {leg['departs']} from {leg['board_at']}.")
        steps.append(f"Get off at {leg['get_off_at']} at {leg['arrives']}.")
    if option["walk_from_last_stop_m"]:
        steps.append(f"Walk about {option['walk_from_last_stop_m']} m to {to_label}.")
    return steps


def plan_trip(
    from_stop_id=None, to_stop_id=None, date=None, time=None, limit=3,
    from_place=None, to_place=None, from_address=None, to_address=None,
):
    """Scheduled bus trips from one stop (or place, or address) to another: direct buses first, else one change.

    Stops within WALK_METERS of the start and the destination are also considered;
    for a place or address, walking distances are measured from that spot itself.
    If nothing runs in the rest of that day, the next day with service is searched.
    """
    try:
        from_label, origins = _endpoint(from_stop_id, from_place, from_address)
        to_label, dests = _endpoint(to_stop_id, to_place, to_address)
    except LookupError as e:
        return {"error": str(e)}

    now = _now(date, time)
    cov = feed_coverage()
    if not _in_feed(now):
        return {
            "error": "The schedule file does not cover this date.",
            "date_asked": now.strftime("%Y-%m-%d"),
            "schedule_runs_from": cov["start"],
            "schedule_runs_to": cov["end"],
        }
    if set(origins.index) & set(dests.index):
        return {"note": "These stops are within walking distance of each other (about 5 minutes). No bus needed."}

    after = now.hour * 3600 + now.minute * 60
    searches = [(0, after, after + SEARCH_WINDOW_SECS), (0, after, 10**6)]
    searches += [(i, 0, 10**6) for i in range(1, LOOKAHEAD_DAYS + 1)]
    for offset, start, end in searches:
        day = now + timedelta(days=offset)
        if not _in_feed(day):
            break
        st = _running_on(day)
        options = _direct_options(st, origins, dests, start, end)
        if len(options) < limit:
            options += _transfer_options(st, origins, dests, start, end)
        if options:
            options = sorted(options, key=lambda o: (o["arrive_secs"], len(o["legs"])))[:limit]
            for o in options:
                del o["arrive_secs"]
                o["steps"] = _steps(o, to_label)
            result = {
                "from": from_label,
                "to": to_label,
                "date": _label(day),
                "source": "scheduled times from Centro schedule file (not live)",
                "options": options,
            }
            if offset == 0:
                result["searched_after"] = now.strftime("%I:%M %p").lstrip("0")
            else:
                result["note"] = f"No trips found on {_label(now)} after that time. These are the first trips on the next day with service."
            return result
    return {
        "note": f"No scheduled trip found between these stops (direct or with one change) in the next {LOOKAHEAD_DAYS} days.",
        "suggestion": "Try a stop close to the start or destination, or check Centro's trip planner.",
    }


if __name__ == "__main__":
    # Quick self-test: py gtfs_tools.py
    print("Coverage:", feed_coverage())
    stop = find_stops("Marshall")[0]
    print("Stop:", stop)
    print("Routes:", routes_at_stop(stop["stop_id"]))
    print(next_departures(stop["stop_id"]))  # today
    dest = find_stops("Destiny")[0]
    print(plan_trip(stop["stop_id"], dest["stop_id"]))
