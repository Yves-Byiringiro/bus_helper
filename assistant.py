"""The assistant: an OpenAI model that can call our bus lookup functions.

ask()        -> grounded assistant (uses tools + notes.md)
ask_plain()  -> plain chatbot with no tools (the baseline to compare against)
"""
import json
import os

import gtfs_tools

# Change this to a model your OpenAI account can use (check OpenAI's model list).
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

NOTES_PATH = os.path.join(os.path.dirname(__file__), "notes.md")


def _notes():
    with open(NOTES_PATH, encoding="utf-8") as f:
        return f.read()


def system_prompt(test_date=None, test_time=None):
    cov = gtfs_tools.feed_coverage()
    now = gtfs_tools._now(test_date, test_time)
    today = now.strftime("%A %Y-%m-%d, %I:%M %p")
    return f"""You are a bus helper for new Syracuse University students who do not have a car and want to ride Centro buses.

Rules:
1. For routes, stops, and departure times, ALWAYS use the tools. Never answer from memory and never invent routes, stops, or times.
2. For fares, student ID rules, and disruptions, use ONLY the notes below. If the notes do not cover it, say you don't know and suggest checking centro.org or calling Centro.
3. The schedule data is scheduled times only, NOT live arrivals. Say so when giving times. The schedule file runs {cov['start']} to {cov['end']}.
4. If a tool returns an error or no results, say that plainly. Do not guess.
5. Say where each answer came from (schedule file or notes).
6. Each departure has a "trip_label" copied from Centro's schedule (for example "40 Syracuse University to Hub"). Quote it exactly. Never rephrase it, and never guess the bus's direction or final destination from it. If the label is unclear, say so.
7. When a stop search returns several stops, check all of them that are relevant. Say which stop each answer is about.
8. Today is {today}. For questions about another day, pass that date to next_departures (YYYY-MM-DD). For the first bus of a day use time "00:00". For the last bus of a day set latest to true.
9. routes_at_stop shows whether each route actually runs soon. If a route has 0 days of service in the next 14 days, say it is in the schedule but has no service in the next two weeks, and do not recommend it.
10. If next_departures returns no departures but has next_service, tell the user when service resumes. A time marked "after midnight" belongs to the previous day's schedule; say so.
11. Getting from A to B ("how do I get to ...", "bus to ..."):
   a. You need BOTH a starting point and a destination. If the user did not say where they are starting from, ask exactly "What's your current location?" (a street address, an intersection, or a place name all work) and do not look anything up yet. Assume they are leaving now unless they say another day or time. Do not ask when they want to leave, and do not mention that you assumed now.
   b. The schedule file only has stop names (mostly street intersections), not stores, buildings, or addresses. For a street address with a house number (e.g. "130 Redfield Pl"), use find_address, or pass it to plan_trip as from_address/to_address. Do not search for addresses with find_stops. For a store or building, call find_place first and use its nearest_stops (tell the user the address and the walk from the stop). If find_place has nothing, check the "Places" list in the notes. If none of these work, ask the user for the nearest cross street or intersection. Never guess which stop is near a business, and never use a stop just because part of its name matches (e.g. "Redfield Village" is not Redfield Pl).
   f. For "where can I buy ..." or "shops near ..." questions, find_place can search by type (e.g. "grocery", "book store") or street. Only suggest places it returns, and say they come from the local places list.
   c. If find_stops returns several possible stops and you cannot tell which one the user means, show a short list and ask them to pick.
   d. Then call plan_trip with the two stop ids, or with from_place/to_place for a place from find_place, or from_address/to_address for a street address (these measure walking from that spot itself). Always use plan_trip for these questions; never answer a trip question with only routes_at_stop or a list of stops near the destination. Present each option (at most 3) as numbered steps, copying the "steps" text from plan_trip: walk to the boarding stop, take the bus (route and trip_label) at the departure time, GET OFF at the named stop at the arrival time, any change of bus, and the final walk. Every option must say which stop to get off at. Mention walking distances are straight-line estimates.
   e. If plan_trip says the trip is on a later day, say clearly that there is no bus for this trip on the day asked and when the next one is. If it finds no trip, say so and suggest Centro's trip planner or calling Centro.
12. Keep answers short and clear.

NOTES (verified information from Centro and SU):
{_notes()}
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_stops",
            "description": "Search bus stops by part of the stop name, e.g. 'Marshall' or 'Destiny'. Returns stop ids.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "routes_at_stop",
            "description": "List the bus routes that serve a stop id.",
            "parameters": {
                "type": "object",
                "properties": {"stop_id": {"type": "string"}},
                "required": ["stop_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_departures",
            "description": "Scheduled departures at a stop id (scheduled times, not live). Defaults to the next departures today. Use date/time for other days, and latest for the last buses of a day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stop_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, default today"},
                    "time": {"type": "string", "description": "HH:MM in 24-hour time, default now"},
                    "latest": {"type": "boolean", "description": "true = the last departures of the day"},
                },
                "required": ["stop_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_place",
            "description": "Search known Syracuse stores and places by name or type (e.g. 'Price Rite', 'Books End', 'grocery', 'James St'). Returns the address and the nearest bus stops with walking distance.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_address",
            "description": "Nearest bus stops to a street address in Syracuse, e.g. '130 Redfield Pl'. Use this for addresses with a house number.",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stops_near",
            "description": "Other stops within walking distance (about 400 m) of a stop id, closest first.",
            "parameters": {
                "type": "object",
                "properties": {"stop_id": {"type": "string"}},
                "required": ["stop_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_trip",
            "description": "Plan a bus trip using scheduled times (not live). Give the start as from_stop_id or from_place, and the destination as to_stop_id or to_place (a place name from find_place). Finds direct buses, or trips with one change of bus. Also considers stops within a short walk of the start and destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_stop_id": {"type": "string"},
                    "to_stop_id": {"type": "string"},
                    "from_place": {"type": "string", "description": "exact place name from find_place"},
                    "to_place": {"type": "string", "description": "exact place name from find_place"},
                    "from_address": {"type": "string", "description": "street address, e.g. '130 Redfield Pl'"},
                    "to_address": {"type": "string", "description": "street address"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, default today"},
                    "time": {"type": "string", "description": "leave after HH:MM in 24-hour time, default now"},
                },
                "required": [],
            },
        },
    },
]


def run_tool(name, args, test_date=None, test_time=None):
    """Run one of our lookup functions. Returns something JSON-friendly.

    test_date/test_time stand in for "now"; a date or time the model asks for wins.
    """
    date = args.get("date") or test_date
    time = args.get("time") or test_time
    try:
        if name == "find_stops":
            return gtfs_tools.find_stops(args["name"])
        if name == "routes_at_stop":
            return gtfs_tools.routes_at_stop(args["stop_id"], date=test_date)
        if name == "next_departures":
            return gtfs_tools.next_departures(
                args["stop_id"],
                limit=args.get("limit", 5),
                date=date,
                time=time,
                latest=args.get("latest", False),
            )
        if name == "find_place":
            return gtfs_tools.find_place(args["name"])
        if name == "stops_near":
            return gtfs_tools.stops_near(args["stop_id"])
        if name == "plan_trip":
            return gtfs_tools.plan_trip(
                args.get("from_stop_id"),
                args.get("to_stop_id"),
                date=date,
                time=time,
                from_place=args.get("from_place"),
                to_place=args.get("to_place"),
                from_address=args.get("from_address"),
                to_address=args.get("to_address"),
            )
        if name == "find_address":
            return gtfs_tools.find_address(args["address"])
    except Exception as e:  # tell the model so it can fix its request instead of crashing
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": f"unknown tool {name}"}


def _client():
    from openai import OpenAI  # imported here so tests without the package still work

    return OpenAI()  # reads OPENAI_API_KEY from the environment


def ask(history, test_date=None, test_time=None, max_steps=8):
    """history = [{'role': 'user'|'assistant', 'content': '...'}, ...]

    Returns (answer_text, tool_log). tool_log lists every lookup the model made,
    which is useful to show sources and for your report.
    """
    client = _client()
    messages = [{"role": "system", "content": system_prompt(test_date, test_time)}] + history
    tool_log = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content, tool_log
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = run_tool(call.function.name, args, test_date, test_time)
            tool_log.append({"tool": call.function.name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
    # Out of lookups: answer with what was found so far (or ask a clarifying question) instead of failing.
    messages.append(
        {
            "role": "system",
            "content": "No more lookups are allowed. Using only the lookup results above, answer as well as you can. "
            "If you could not find what you needed, say what is missing and ask the user for it (for example a nearby intersection).",
        }
    )
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, tool_choice="none")
    return resp.choices[0].message.content, tool_log


def ask_plain(question):
    """Baseline: the same model with NO tools and NO notes."""
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content
