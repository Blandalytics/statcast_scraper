"""Fast pitch-level Statcast data from statsapi.mlb.com.

Four pull modes, all sharing one fetch/flatten core:

    mlb_season(2024)                                 every pitch in a season
    pitcher_season("Tarik Skubal", [2023, 2024])     one pitcher, one/many seasons
    pitcher_game("Skubal", game_date="2024-06-01")   one pitcher, one game
    mlb_day("2024-06-01")                            every pitch on a date

The two season pulls also accept an explicit date range instead of seasons:

    mlb_season(start="2024-06-01", end="2024-06-30")
    pitcher_season("Skubal", start="2024-06-01", end="2024-07-31")

Uses gzip, field-filtered JSON and a thread pool; roughly 6x faster and 6x
lighter than the Baseball Savant CSV export.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import orjson
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://statsapi.mlb.com/api/v1"


class NotFound(LookupError):
    """A requested player or game does not exist."""

# Whitelist of JSON keys. Covers every field we extract; omits only `link`
# (a URL rebuildable from `id`), the always-empty hot/cold zone arrays, and the
# runners/index arrays, which are one-to-many rather than per-pitch.
_FIELDS = ",".join((
    "allPlays",
    # play: about
    "about", "atBatIndex", "captivatingIndex", "endTime", "halfInning", "hasOut",
    "hasReview", "inning", "isComplete", "isScoringPlay", "isTopInning", "startTime",
    # play: matchup
    "matchup", "batSide", "code", "description", "batter", "id", "fullName",
    "pitchHand", "pitcher", "postOnFirst", "postOnSecond", "postOnThird",
    "splits", "menOnBase",
    # play: result + review
    "result", "awayScore", "homeScore", "event", "eventType", "isOut", "rbi", "type",
    "reviewDetails", "challengeTeamId", "inProgress", "isOverturned", "reviewType",
    # pitch event
    "playEvents", "index", "playId", "pitchNumber", "isPitch",
    "details", "call", "disengagementNum", "isBall", "isInPlay",
    "isStrike", "runnerGoing",
    "count", "balls", "strikes", "outs",
    "pitchData", "endSpeed", "extension", "plateTime", "startSpeed",
    "strikeZoneBottom", "strikeZoneTop", "typeConfidence", "zone",
    "breaks", "breakAngle", "breakHorizontal", "breakLength", "breakVertical",
    "breakVerticalInduced", "breakY", "spinDirection", "spinRate",
    "coordinates", "aX", "aY", "aZ", "pX", "pZ", "pfxX", "pfxZ",
    "vX0", "vY0", "vZ0", "x", "x0", "y", "y0", "z0",
    "hitData", "coordX", "coordY", "hardness", "launchAngle", "launchSpeed",
    "location", "totalDistance", "trajectory",
))
_SCHED_FIELDS = "dates,date,games,gamePk,status,codedGameState"
# The schedule endpoint has no "P" code; postseason is wild card / division /
# league championship / world series. The gameLog endpoint does accept "P".
_POSTSEASON = "F,D,L,W"

# Play-level fields, repeated for every pitch in the plate appearance.
_PLAY_COLS = (
    "game_pk", "game_date",
    "at_bat_index", "inning", "half", "is_top_inning",
    "play_start_time", "play_end_time", "captivating_index",
    "play_has_out", "play_is_complete", "is_scoring_play", "play_has_review",
    "pitcher", "pitcher_name", "p_throws", "p_throws_desc",
    "batter", "batter_name", "stand", "stand_desc",
    "on_1b", "on_2b", "on_3b",
    "split_batter", "split_pitcher", "men_on_base",
    "events", "event", "event_desc", "result_type", "result_is_out", "rbi",
    "away_score", "home_score",
    "review_type", "review_team_id", "review_overturned", "review_in_progress",
)

# Fields carried by the individual pitch.
_PITCH_COLS = (
    "pitch_number", "event_index", "play_id", "pitch_start_time", "pitch_end_time",
    "pitch_type", "pitch_name", "call_code", "call_name", "description", "det_code",
    "is_ball", "is_strike", "is_in_play", "is_out", "pitch_has_review",
    "runner_going", "disengagement_num",
    "balls", "strikes", "outs",
    "release_speed", "end_speed", "plate_time", "release_extension",
    "type_confidence", "zone", "sz_top", "sz_bot",
    "plate_x", "plate_z", "pfx_x", "pfx_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "pitch_coord_x", "pitch_coord_y",
    "break_angle", "break_length", "break_y", "break_vertical", "ivb", "hb",
    "release_spin_rate", "spin_axis",
    "launch_speed", "launch_angle", "hit_distance", "trajectory", "hardness",
    "hit_location", "hit_coord_x", "hit_coord_y",
)

_COLS = _PLAY_COLS + _PITCH_COLS
#: Every column a pull can return, in default order. Pass any subset as
#: ``columns=`` to the pull functions.
COLUMNS = _COLS
# Needed to order the result; materialised even when not requested.
_SORT = ("game_date", "game_pk", "at_bat_index", "pitch_number")

# float32 is ample: Statcast reports 1-2 decimals.
_F32 = ("release_speed", "end_speed", "plate_time", "release_extension",
        "type_confidence", "sz_top", "sz_bot",
        "plate_x", "plate_z", "pfx_x", "pfx_z",
        "vx0", "vy0", "vz0", "ax", "ay", "az",
        "release_pos_x", "release_pos_y", "release_pos_z",
        "pitch_coord_x", "pitch_coord_y",
        "break_angle", "break_length", "break_y", "break_vertical", "ivb", "hb",
        "release_spin_rate", "spin_axis",
        "launch_speed", "launch_angle", "hit_distance",
        "hit_coord_x", "hit_coord_y")
# UInt8 is nullable, so it covers the sparse small ints too.
_U8 = ("balls", "strikes", "outs", "inning", "pitch_number", "zone",
       "event_index", "captivating_index", "rbi", "away_score", "home_score",
       "disengagement_num")
_I32 = ("game_pk", "pitcher", "batter", "at_bat_index")
_I32N = ("on_1b", "on_2b", "on_3b", "review_team_id")
_BOOL = ("is_top_inning", "play_has_out", "play_is_complete", "is_scoring_play",
         "play_has_review", "result_is_out", "is_ball", "is_strike", "is_in_play",
         "is_out", "pitch_has_review", "runner_going", "review_overturned",
         "review_in_progress")
_CAT = ("pitch_type", "pitch_name", "call_code", "call_name", "description",
        "det_code", "events", "event", "event_desc", "result_type",
        "stand", "stand_desc", "p_throws", "p_throws_desc", "half",
        "pitcher_name", "batter_name", "split_batter", "split_pitcher",
        "men_on_base", "trajectory", "hardness", "hit_location", "review_type")
_TIME = ("play_start_time", "play_end_time", "pitch_start_time", "pitch_end_time")

_NUMERIC = ("float32", "UInt8", "int32", "Int32")
_CASTS = ((_F32, "float32"), (_U8, "UInt8"), (_I32, "int32"), (_I32N, "Int32"),
          (_BOOL, "boolean"), (_CAT, "category"))


def _session(pool: int = 16) -> requests.Session:
    s = requests.Session()
    s.headers["Accept-Encoding"] = "gzip"
    retry = Retry(total=3, backoff_factor=0.3,
                  status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(pool_maxsize=pool, pool_connections=pool,
                                    max_retries=retry))
    return s


def _seasons(value: int | str | Iterable[int]) -> tuple[int, ...]:
    """Normalise a season, or a list of seasons, to a tuple of ints."""
    if isinstance(value, int | str):
        return (int(value),)
    return tuple(int(v) for v in value)


def _years(start: str, end: str) -> tuple[int, ...]:
    """Every calendar year touched by a date range."""
    return tuple(range(int(str(start)[:4]), int(str(end)[:4]) + 1))


def _year_chunks(start: str, end: str) -> tuple[tuple[str, str], ...]:
    """Split a date range at calendar-year boundaries.

    The schedule endpoint silently truncates a multi-year range to its first
    season, so each year has to be asked for separately.
    """
    return tuple((max(str(start), f"{y}-01-01"), min(str(end), f"{y}-12-31"))
                 for y in _years(start, end))


def _in_range(games: dict[int, str], start: str | None,
              end: str | None) -> dict[int, str]:
    """Trim a gamePk -> date map to an inclusive date range."""
    if start is None:
        return games
    return {pk: d for pk, d in games.items() if str(start) <= d <= str(end)}


def _check_span(seasons, start, end) -> None:
    """Require exactly one of seasons=... or start=/end=."""
    if (seasons is None) == (start is None):
        raise ValueError("pass either seasons=... or both start= and end=")
    if (start is None) != (end is None):
        raise ValueError("start= and end= must be given together")


def _span_label(seasons, start, end) -> str:
    if seasons is None:
        return f"{start}..{end}"
    return ", ".join(str(y) for y in _seasons(seasons))


# ---------------------------------------------------------------------------
# game discovery
# ---------------------------------------------------------------------------

def _sched_game_type(game_type: str) -> str:
    """Expand "P" to the four postseason round codes the schedule endpoint uses."""
    return ",".join(_POSTSEASON if part.strip().upper() == "P" else part.strip()
                    for part in str(game_type).split(","))


def _schedule(s: requests.Session, params: dict) -> dict[int, str]:
    """Map gamePk -> date for a schedule query, keeping only games actually played.

    codedGameState "F" is the real filter: abstractGameState reports "Final"
    for postponed and cancelled games too.
    """
    q = {"sportId": 1, "fields": _SCHED_FIELDS, **params}
    if "gameType" in q:
        q["gameType"] = _sched_game_type(q["gameType"])
    dates = s.get(f"{API}/schedule", params=q, timeout=60).json().get("dates", [])
    return {g["gamePk"]: d["date"]
            for d in dates for g in d["games"]
            if g.get("status", {}).get("codedGameState") == "F"}


def _mlb_queries(seasons, start, end, game_type: str) -> list[dict]:
    """Schedule queries covering the requested span, one per season/year."""
    if seasons is not None:
        return [{"season": y, "gameType": game_type} for y in _seasons(seasons)]
    return [{"startDate": a, "endDate": b, "gameType": game_type}
            for a, b in _year_chunks(start, end)]


def _mlb_games(s: requests.Session, seasons, start, end,
               game_type: str) -> dict[int, str]:
    """Every played game in the requested span."""
    games: dict[int, str] = {}
    for q in _mlb_queries(seasons, start, end, game_type):
        games |= _schedule(s, q)
    return _in_range(games, start, end)


def _pitcher_span_games(s: requests.Session, pid: int, seasons, start, end,
                        game_type: str) -> dict[int, str]:
    """Every game the pitcher appeared in across the requested span."""
    years = _seasons(seasons) if seasons is not None else _years(start, end)
    games: dict[int, str] = {}
    for year in years:
        games |= _pitcher_games(s, pid, year, game_type)
    return _in_range(games, start, end)


def _pitcher_games(s: requests.Session, pitcher_id: int, season: int,
                   game_type: str) -> dict[int, str]:
    """Map gamePk -> date for every game a pitcher appeared in."""
    log = s.get(f"{API}/people/{pitcher_id}/stats",
                params={"stats": "gameLog", "group": "pitching",
                        "season": season, "gameType": game_type},
                timeout=30).json()
    splits = log["stats"][0]["splits"] if log.get("stats") else []
    return {sp["game"]["gamePk"]: sp["date"] for sp in splits}


# ---------------------------------------------------------------------------
# fetch + flatten
# ---------------------------------------------------------------------------

def _play_head(play: dict, pk: int, date: str, pid: int | None) -> tuple:
    """Play-level values, identical for every pitch in the plate appearance."""
    mu = play.get("matchup", {})
    ab = play.get("about", {})
    res = play.get("result", {})
    rev = play.get("reviewDetails", {})
    spl = mu.get("splits", {})
    return (
        pk, date,
        ab.get("atBatIndex"), ab.get("inning"), ab.get("halfInning"),
        ab.get("isTopInning"), ab.get("startTime"), ab.get("endTime"),
        ab.get("captivatingIndex"), ab.get("hasOut"), ab.get("isComplete"),
        ab.get("isScoringPlay"), ab.get("hasReview"),
        pid, (mu.get("pitcher") or {}).get("fullName"),
        (mu.get("pitchHand") or {}).get("code"),
        (mu.get("pitchHand") or {}).get("description"),
        (mu.get("batter") or {}).get("id"), (mu.get("batter") or {}).get("fullName"),
        (mu.get("batSide") or {}).get("code"),
        (mu.get("batSide") or {}).get("description"),
        (mu.get("postOnFirst") or {}).get("id"),
        (mu.get("postOnSecond") or {}).get("id"),
        (mu.get("postOnThird") or {}).get("id"),
        spl.get("batter"), spl.get("pitcher"), spl.get("menOnBase"),
        res.get("eventType"), res.get("event"), res.get("description"),
        res.get("type"), res.get("isOut"), res.get("rbi"),
        res.get("awayScore"), res.get("homeScore"),
        rev.get("reviewType"), rev.get("challengeTeamId"),
        rev.get("isOverturned"), rev.get("inProgress"),
    )


def _play_rows(play: dict, pk: int, date: str, pid: int | None) -> Iterator[tuple]:
    """Yield one row per tracked pitch in a single plate appearance."""
    head = _play_head(play, pk, date, pid)
    for e in play.get("playEvents", ()):
        pit = e.get("pitchData")
        if not pit:                      # skips IBB / timer-violation phantoms
            continue
        co = pit.get("coordinates", {})
        br = pit.get("breaks", {})
        hd = e.get("hitData", {})
        hc = hd.get("coordinates", {})
        det = e.get("details", {})
        cnt = e.get("count", {})
        typ = det.get("type", {})
        call = det.get("call", {})
        yield head + (
            e.get("pitchNumber"), e.get("index"), e.get("playId"),
            e.get("startTime"), e.get("endTime"),
            typ.get("code"), typ.get("description"),
            call.get("code"), call.get("description"),
            det.get("description"), det.get("code"),
            det.get("isBall"), det.get("isStrike"), det.get("isInPlay"),
            det.get("isOut"), det.get("hasReview"), det.get("runnerGoing"),
            det.get("disengagementNum"),
            cnt.get("balls"), cnt.get("strikes"), cnt.get("outs"),
            pit.get("startSpeed"), pit.get("endSpeed"), pit.get("plateTime"),
            pit.get("extension"), pit.get("typeConfidence"), pit.get("zone"),
            pit.get("strikeZoneTop"), pit.get("strikeZoneBottom"),
            co.get("pX"), co.get("pZ"), co.get("pfxX"), co.get("pfxZ"),
            co.get("vX0"), co.get("vY0"), co.get("vZ0"),
            co.get("aX"), co.get("aY"), co.get("aZ"),
            co.get("x0"), co.get("y0"), co.get("z0"),
            co.get("x"), co.get("y"),
            br.get("breakAngle"), br.get("breakLength"), br.get("breakY"),
            br.get("breakVertical"), br.get("breakVerticalInduced"),
            br.get("breakHorizontal"), br.get("spinRate"), br.get("spinDirection"),
            hd.get("launchSpeed"), hd.get("launchAngle"), hd.get("totalDistance"),
            hd.get("trajectory"), hd.get("hardness"), hd.get("location"),
            hc.get("coordX"), hc.get("coordY"),
        )


def _game_rows(blob: bytes, pk: int, date: str,
               pitcher_id: int | None) -> Iterator[tuple]:
    """Yield rows for one game, optionally limited to a single pitcher."""
    for play in orjson.loads(blob).get("allPlays", ()):
        pid = play.get("matchup", {}).get("pitcher", {}).get("id")
        if pitcher_id is None or pid == pitcher_id:
            yield from _play_rows(play, pk, date, pid)


def _check_columns(columns) -> None:
    """Reject unknown column names up front."""
    if columns is None:
        return
    unknown = [c for c in columns if c not in _COLS]
    if unknown:
        raise ValueError(f"unknown column(s): {', '.join(unknown)}. "
                         f"See statfast.COLUMNS for the {len(_COLS)} valid names.")


def _work_columns(columns) -> list[str]:
    """Columns to materialise: those requested, plus the sort keys."""
    if columns is None:
        return list(_COLS)
    wanted = set(columns) | set(_SORT)
    return [c for c in _COLS if c in wanted]


def _select(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Narrow to the requested columns, in the caller's order."""
    return df if columns is None else df[list(columns)]


def _cast_group(df: pd.DataFrame, cols: tuple[str, ...], dtype: str) -> None:
    numeric = dtype in _NUMERIC
    for c in cols:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce") if numeric else df[c]
        df[c] = col.astype(dtype)


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink to compact dtypes in place, skipping columns not present."""
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d")
    for c in _TIME:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="ISO8601", utc=True)
    for cols, dtype in _CASTS:
        _cast_group(df, cols, dtype)
    return df


def _collect(s: requests.Session, games: dict[int, str], pitcher_id: int | None,
             workers: int, columns=None) -> pd.DataFrame:
    """Fetch each game's playByPlay in parallel and flatten to a DataFrame."""
    _check_columns(columns)
    if not games:
        return _select(pd.DataFrame(columns=_COLS), columns)

    def fetch(pk: int) -> bytes:
        r = s.get(f"{API}/game/{pk}/playByPlay", params={"fields": _FIELDS}, timeout=30)
        r.raise_for_status()
        return r.content

    with ThreadPoolExecutor(max_workers=workers) as ex:
        blobs = list(zip(games, ex.map(fetch, games), strict=True))

    rows = [row
            for pk, blob in blobs
            for row in _game_rows(blob, pk, games[pk], pitcher_id)]
    df = pd.DataFrame(rows, columns=_COLS)
    if df.empty:
        return _select(df, columns)
    df = _cast(df[_work_columns(columns)]).sort_values(list(_SORT), ignore_index=True)
    return _select(df, columns)


# ---------------------------------------------------------------------------
# player lookup
# ---------------------------------------------------------------------------

def _person(s: requests.Session, pid: int) -> tuple[int, str]:
    """Look up a player by MLBAM id."""
    r = s.get(f"{API}/people/{pid}", params={"fields": "people,id,fullName"}, timeout=30)
    people = r.json().get("people", []) if r.ok else []
    if not people:
        raise NotFound(f"no MLB player with id {pid}")
    return pid, people[0]["fullName"]


def resolve_pitcher(name: str | int, season: int | None = None,
                    session: requests.Session | None = None) -> tuple[int, str]:
    """Resolve a pitcher name (or a raw MLBAM id) to ``(id, full_name)``."""
    s = session or _session()
    if isinstance(name, int) or str(name).isdigit():
        return _person(s, int(name))

    params = {"names": name, "sportIds": "1"}
    if season:
        params["season"] = season
    hits = s.get(f"{API}/people/search", params=params,
                 timeout=30).json().get("people", [])
    if not hits:
        raise NotFound(f"no MLB player matching {name!r}")

    pool = [p for p in hits
            if p.get("primaryPosition", {}).get("abbreviation") in ("P", "TWP")] or hits
    if len(pool) > 1:
        opts = ", ".join(f"{p['fullName']} ({p['id']})" for p in pool[:10])
        raise NotFound(f"{name!r} is ambiguous - pass an id. Candidates: {opts}")
    return pool[0]["id"], pool[0]["fullName"]


# ---------------------------------------------------------------------------
# pull modes
# ---------------------------------------------------------------------------

def mlb_season(seasons: int | Iterable[int] | None = None, *,
               start: str | None = None, end: str | None = None,
               game_type: str = "R", workers: int = 12,
               columns: Iterable[str] | None = None,
               session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch in whole seasons, or between two dates."""
    _check_span(seasons, start, end)
    s = session or _session(workers)
    games = _mlb_games(s, seasons, start, end, game_type)
    df = _collect(s, games, None, workers, columns)
    df.attrs.update(scope="mlb_season", span=_span_label(seasons, start, end),
                    start=start, end=end, game_type=game_type)
    return df


def pitcher_season(pitcher: str | int,
                   seasons: int | Iterable[int] | None = None, *,
                   start: str | None = None, end: str | None = None,
                   game_type: str = "R", workers: int = 12,
                   columns: Iterable[str] | None = None,
                   session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch by one pitcher, in whole seasons or between dates."""
    _check_span(seasons, start, end)
    s = session or _session(workers)
    pid, full = resolve_pitcher(pitcher, session=s)
    games = _pitcher_span_games(s, pid, seasons, start, end, game_type)
    df = _collect(s, games, pid, workers, columns)
    df.attrs.update(scope="pitcher_season", pitcher_id=pid, pitcher_name=full,
                    span=_span_label(seasons, start, end),
                    start=start, end=end, game_type=game_type)
    return df


def mlb_day(date: str, *, game_type: str = "R", workers: int = 12,
            columns: Iterable[str] | None = None,
            session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch thrown on a single date (YYYY-MM-DD)."""
    s = session or _session(workers)
    games = _schedule(s, {"date": str(date), "gameType": game_type})
    df = _collect(s, games, None, workers, columns)
    df.attrs.update(scope="mlb_day", date=str(date), game_type=game_type)
    return df


def _one_game(s: requests.Session, pid: int, game_pk: int | None,
              game_date: str | None, game_type: str) -> dict[int, str]:
    """Resolve exactly one game for a pitcher, by gamePk or by date."""
    if game_pk is not None:
        found = _schedule(s, {"gamePk": int(game_pk)})
        if not found:
            raise NotFound(f"no completed game with gamePk {game_pk}")
        return found
    season = int(str(game_date)[:4])
    games = {pk: d for pk, d in _pitcher_games(s, pid, season, game_type).items()
             if d == str(game_date)}
    if not games:
        raise NotFound(f"pitcher {pid} did not appear on {game_date}")
    return games


def pitcher_game(pitcher: str | int, *, game_pk: int | None = None,
                 game_date: str | None = None, game_type: str = "R",
                 workers: int = 12, columns: Iterable[str] | None = None,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch by one pitcher in a single game (by gamePk or date)."""
    if (game_pk is None) == (game_date is None):
        raise ValueError("pass exactly one of game_pk= or game_date=")
    s = session or _session(workers)
    pid, full = resolve_pitcher(pitcher, session=s)
    games = _one_game(s, pid, game_pk, game_date, game_type)
    df = _collect(s, games, pid, workers, columns)
    first = next(iter(games))
    df.attrs.update(scope="pitcher_game", pitcher_id=pid, pitcher_name=full,
                    game_pk=first, game_date=games[first], game_type=game_type)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _title(df: pd.DataFrame) -> str:
    at = df.attrs
    when = at.get("date") or at.get("game_date") or at.get("span", "")
    return (f"{at.get('pitcher_name') or 'MLB'} | {at.get('scope', '?')} "
            f"{when} [{at.get('game_type', 'R')}]")


def _report_span(df: pd.DataFrame) -> None:
    if "game_pk" in df:
        print(f"  {df.game_pk.nunique():,} games")
    if "game_date" in df:
        print(f"  {df.game_date.min().date()} -> {df.game_date.max().date()}")
    if "pitcher" in df and df.pitcher.nunique() > 1:
        print(f"  {df.pitcher.nunique():,} pitchers")


def _report_mix(df: pd.DataFrame) -> None:
    if "pitch_type" not in df or "release_speed" not in df:
        return
    mix = df.pitch_type.value_counts(normalize=True).mul(100).head(6)
    velo = df.groupby("pitch_type", observed=True).release_speed.mean()
    print("  mix:", ", ".join(f"{p} {mix[p]:.1f}% @{velo[p]:.1f}" for p in mix.index))


def _report(df: pd.DataFrame, elapsed: float) -> None:
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(_title(df))
    print(f"  {len(df):,} pitches / {df.shape[1]} cols / {elapsed:.2f}s / {mb:,.1f} MB")
    _report_span(df)
    _report_mix(df)


def _split_columns(value: str | None) -> list[str] | None:
    return [c.strip() for c in value.split(",")] if value else None


def _build_parser():
    import argparse

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-t", "--game-type", default="R",
                        help='"R" (default), "P", "S", or e.g. "R,P"')
    common.add_argument("-w", "--workers", type=int, default=12)
    common.add_argument("-o", "--out", help="write .parquet / .csv")
    common.add_argument("-c", "--columns",
                        help="comma-separated subset of columns (default: all)")

    ap = argparse.ArgumentParser(description="Pull pitch-level Statcast data.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("mlb-season", parents=[common],
                       help="all pitches in whole seasons, or between two dates")
    p.add_argument("seasons", nargs="*", type=int)
    p.add_argument("--start", help="YYYY-MM-DD (use instead of seasons)")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.set_defaults(run=lambda a: mlb_season(a.seasons or None, start=a.start,
                                            end=a.end, game_type=a.game_type,
                                            workers=a.workers,
                                            columns=_split_columns(a.columns)))

    p = sub.add_parser("pitcher-season", parents=[common],
                       help="one pitcher, whole seasons or between two dates")
    p.add_argument("pitcher", help='name ("Tarik Skubal") or MLBAM id (669373)')
    p.add_argument("seasons", nargs="*", type=int)
    p.add_argument("--start", help="YYYY-MM-DD (use instead of seasons)")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.set_defaults(run=lambda a: pitcher_season(a.pitcher, a.seasons or None,
                                                start=a.start, end=a.end,
                                                game_type=a.game_type,
                                                workers=a.workers,
                                                columns=_split_columns(a.columns)))

    p = sub.add_parser("pitcher-game", parents=[common], help="one pitcher, one game")
    p.add_argument("pitcher", help='name ("Tarik Skubal") or MLBAM id (669373)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="YYYY-MM-DD")
    g.add_argument("--pk", type=int, help="gamePk")
    p.set_defaults(run=lambda a: pitcher_game(a.pitcher, game_pk=a.pk,
                                              game_date=a.date,
                                              game_type=a.game_type,
                                              workers=a.workers,
                                              columns=_split_columns(a.columns)))

    p = sub.add_parser("mlb-day", parents=[common], help="all pitches on one date")
    p.add_argument("date", help="YYYY-MM-DD")
    p.set_defaults(run=lambda a: mlb_day(a.date, game_type=a.game_type,
                                         workers=a.workers,
                                         columns=_split_columns(a.columns)))
    return ap


def _main(argv: list[str] | None = None) -> int:
    import time

    a = _build_parser().parse_args(argv)
    t0 = time.perf_counter()
    try:
        df = a.run(a)
    except (NotFound, ValueError) as exc:
        print(exc)
        return 2
    elapsed = time.perf_counter() - t0

    if df.empty:
        print("no pitches matched")
        return 1
    _report(df, elapsed)
    if a.out:
        (df.to_parquet if a.out.endswith(".parquet") else df.to_csv)(a.out, index=False)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
