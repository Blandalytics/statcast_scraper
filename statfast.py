"""Fast pitch-level Statcast data from statsapi.mlb.com.

Four pull modes, all sharing one fetch/flatten core:

    mlb_season(2024)                                 every pitch in a season
    pitcher_season("Tarik Skubal", [2023, 2024])     one pitcher, one/many seasons
    pitcher_game("Skubal", game_date="2024-06-01")   one pitcher, one game
    mlb_day("2024-06-01")                            every pitch on a date

The season pulls also take start=/end= dates instead of seasons. Dates accept
any common form. Uses gzip, field-filtered JSON and a thread pool; roughly 6x
faster and 6x lighter than the Baseball Savant CSV export.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from typing import NamedTuple

import numpy as np
import orjson
import pandas as pd
import requests
from pandas.api.types import union_categoricals
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://statsapi.mlb.com/api/v1"


class NotFound(LookupError):
    """A requested player or game does not exist."""


class _Game(NamedTuple):
    """What discovery knows about a game before its play-by-play is fetched."""

    date: str
    home: int  # team id
    away: int  # team id


Sess = requests.Session
Games = dict[int, _Game]
Teams = dict[int, tuple[str, str]]  # gamePk -> (home abbr, away abbr)
Rows = Iterator[tuple]
DateLike = str | _dt.date  # anything _date() accepts; pandas Timestamp is a datetime


def _names(text: str, sep: str | None = None) -> tuple[str, ...]:
    return tuple(text.split(sep))


# JSON keys to request. Covers every field extracted; omits only `link` (rebuildable
# from `id`), the always-empty hot/cold zone arrays, and the one-to-many runner arrays.
_FIELDS = ",".join(_names("""
    allPlays about atBatIndex captivatingIndex endTime halfInning hasOut hasReview inning
    isComplete isScoringPlay isTopInning startTime matchup batSide code description batter id
    fullName pitchHand pitcher postOnFirst postOnSecond postOnThird splits menOnBase result
    awayScore homeScore event eventType isOut rbi type reviewDetails challengeTeamId inProgress
    isOverturned reviewType playEvents index playId pitchNumber isPitch details call
    disengagementNum isBall isInPlay isStrike runnerGoing count balls strikes outs pitchData
    endSpeed extension plateTime startSpeed strikeZoneBottom strikeZoneTop typeConfidence zone
    breaks breakAngle breakHorizontal breakLength breakVertical breakVerticalInduced breakY
    spinDirection spinRate coordinates aX aY aZ pX pZ pfxX pfxZ vX0 vY0 vZ0 x x0 y y0 z0
    hitData coordX coordY hardness launchAngle launchSpeed location totalDistance trajectory
"""))  # fmt: skip
_SCHED_FIELDS = "dates,date,games,gamePk,status,codedGameState,teams,home,away,team,id"
# The schedule endpoint has no "P"; postseason is WC/DS/LCS/WS. gameLog does accept "P".
_POSTSEASON = "F,D,L,W"

# Play-level fields, repeated for every pitch in the plate appearance.
_PLAY_COLS = _names("""
    game_pk game_date home_team away_team bat_team field_team
    at_bat_index inning half is_top_inning play_start_time play_end_time captivating_index
    play_has_out play_is_complete is_scoring_play play_has_review
    pitcher pitcher_name p_throws p_throws_desc batter batter_name stand stand_desc
    on_1b on_2b on_3b split_batter split_pitcher men_on_base
    events event event_desc result_type result_is_out rbi
    away_score home_score bat_score field_score
    post_away_score post_home_score post_bat_score post_field_score
    review_type review_team_id review_overturned review_in_progress
""")
# Fields carried by the individual pitch.
_PITCH_COLS = _names("""
    pitch_number event_index play_id pitch_start_time pitch_end_time
    pitch_type pitch_name call_code call_name description det_code
    is_ball is_strike is_in_play is_out pitch_has_review runner_going disengagement_num
    balls strikes outs
    release_speed end_speed plate_time release_extension type_confidence zone sz_top sz_bot
    plate_x plate_z pfx_x pfx_z vx0 vy0 vz0 ax ay az release_pos_x release_pos_y release_pos_z
    pitch_coord_x pitch_coord_y
    break_angle break_length break_y break_vertical ivb hb release_spin_rate spin_axis
    launch_speed launch_angle hit_distance trajectory hardness hit_location hit_coord_x hit_coord_y
""")
_ROW_COLS = _PLAY_COLS + _PITCH_COLS  # what _play_rows yields
_COLS = _ROW_COLS + ("spray_angle",)  # plus columns derived after the frame is built
#: Every column a pull can return, in default order; pass any subset as ``columns=``.
COLUMNS = _COLS
# Needed to order the result; materialised even when not requested.
_SORT = _names("game_date game_pk at_bat_index pitch_number")

# float32 is ample: Statcast reports 1-2 decimals. UInt8 is nullable, so it covers the sparse ints.
_F32 = _names("""
    release_speed end_speed plate_time release_extension type_confidence sz_top sz_bot
    plate_x plate_z pfx_x pfx_z vx0 vy0 vz0 ax ay az release_pos_x release_pos_y release_pos_z
    pitch_coord_x pitch_coord_y break_angle break_length break_y break_vertical ivb hb
    release_spin_rate spin_axis launch_speed launch_angle hit_distance hit_coord_x hit_coord_y
    spray_angle
""")
_U8 = _names("""
    balls strikes outs inning pitch_number zone event_index captivating_index rbi disengagement_num
    away_score home_score bat_score field_score
    post_away_score post_home_score post_bat_score post_field_score
""")
_I32 = _names("game_pk pitcher batter at_bat_index")
_I32N = _names("on_1b on_2b on_3b review_team_id")
_BOOL = _names("""
    is_top_inning play_has_out play_is_complete is_scoring_play play_has_review result_is_out
    is_ball is_strike is_in_play is_out pitch_has_review runner_going review_overturned
    review_in_progress
""")
_CAT = _names("""
    home_team away_team bat_team field_team pitch_type pitch_name call_code call_name description
    det_code events event event_desc result_type stand stand_desc p_throws p_throws_desc half
    pitcher_name batter_name split_batter split_pitcher men_on_base trajectory hardness
    hit_location review_type
""")
_TIME = _names("play_start_time play_end_time pitch_start_time pitch_end_time")

# Per-chunk casts. Category is handled by _categorize, which needs a uniform intermediate.
_DTYPES = ("float32", "UInt8", "int32", "Int32", "boolean")
_VALUE_CASTS = tuple(zip((_F32, _U8, _I32, _I32N, _BOOL), _DTYPES, strict=True))
# Games flattened per chunk; bounds the transient row-tuple cost (~100 games is ~30k pitches).
_CHUNK_GAMES = 100


def _session(pool: int = 16) -> Sess:
    s = requests.Session()
    s.headers["Accept-Encoding"] = "gzip"
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(pool_maxsize=pool, pool_connections=pool, max_retries=retry))
    return s


def _seasons(value: int | str | Iterable[int]) -> tuple[int, ...]:
    """Normalise a season, or a list of seasons, to a tuple of ints."""
    if isinstance(value, int | str):
        return (int(value),)
    return tuple(int(v) for v in value)


# ---- dates --------------------------------------------------------------------------
# String forms tried after ISO 8601. Each needs a four-digit year and a full Y/M/D so nothing
# is silently defaulted; slash forms are US (month first); %d and %m accept unpadded digits.
_DATE_FORMATS = _names("%Y-%m-%d|%Y/%m/%d|%Y.%m.%d|%Y%m%d|%m/%d/%Y|%m-%d-%Y|%m.%d.%Y", "|")
_DATE_FORMATS += _names("%B %d, %Y|%B %d %Y|%b %d, %Y|%b %d %Y|%d %B %Y|%d %b %Y|%d-%b-%Y", "|")
_DATE_PARSERS = (
    _dt.datetime.fromisoformat,
    *(lambda t, f=f: _dt.datetime.strptime(t, f) for f in _DATE_FORMATS),
)


def _date(value) -> str:
    """Normalise a date/datetime/Timestamp or any common date string to 'YYYY-MM-DD';
    partial or two-digit-year dates are rejected rather than guessed."""
    if isinstance(value, _dt.date):  # datetime is a date subclass
        return value.isoformat()[:10]
    text = str(value).strip()
    for parse in _DATE_PARSERS:
        try:
            return parse(text).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {value!r}; use YYYY-MM-DD")


def _norm_span(start, end) -> tuple[str | None, str | None]:
    """Canonicalise a start/end pair, rejecting a reversed range."""
    if start is None:
        return None, None
    start, end = _date(start), _date(end)
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    return start, end


def _check_span(seasons, start, end) -> tuple[str | None, str | None]:
    """Require exactly one of seasons=... or start=/end=; return them canonical."""
    if (seasons is None) == (start is None):
        raise ValueError("pass either seasons=... or both start= and end=")
    if (start is None) != (end is None):
        raise ValueError("start= and end= must be given together")
    return _norm_span(start, end)


def _years(start: str, end: str) -> tuple[int, ...]:
    """Every calendar year touched by a date range."""
    return tuple(range(int(start[:4]), int(end[:4]) + 1))


def _year_chunks(start: str, end: str) -> tuple[tuple[str, str], ...]:
    """Split a date range at calendar-year boundaries: the schedule endpoint silently
    truncates a multi-year range to its first season."""
    return tuple((max(start, f"{y}-01-01"), min(end, f"{y}-12-31")) for y in _years(start, end))


def _in_range(games: Games, start: str | None, end: str | None) -> Games:
    """Trim a game map to an inclusive date range."""
    if start is None:
        return games
    return {pk: g for pk, g in games.items() if start <= g.date <= end}


def _span_label(seasons, start, end) -> str:
    if seasons is None:
        return f"{start}..{end}"
    return ", ".join(str(y) for y in _seasons(seasons))


# ---- game discovery -----------------------------------------------------------------
def _sched_game_type(game_type: str) -> str:
    """Expand "P" to the four postseason round codes the schedule endpoint uses."""
    parts = (p.strip() for p in str(game_type).split(","))
    return ",".join(_POSTSEASON if p.upper() == "P" else p for p in parts)


def _schedule(s: Sess, params: dict) -> Games:
    """gamePk -> _Game for a schedule query, keeping only games actually played
    (codedGameState "F"; abstractGameState says "Final" for postponed games too)."""
    q = {"sportId": 1, "fields": _SCHED_FIELDS, **params}
    if "gameType" in q:
        q["gameType"] = _sched_game_type(q["gameType"])
    dates = s.get(f"{API}/schedule", params=q, timeout=60).json().get("dates", [])
    return {
        g["gamePk"]: _Game(
            d["date"], g["teams"]["home"]["team"]["id"], g["teams"]["away"]["team"]["id"]
        )
        for d in dates
        for g in d["games"]
        if g.get("status", {}).get("codedGameState") == "F"
    }


def _mlb_queries(seasons, start, end, game_type: str) -> list[dict]:
    """Schedule queries covering the requested span, one per season/year."""
    if seasons is not None:
        return [{"season": y, "gameType": game_type} for y in _seasons(seasons)]
    return [
        {"startDate": a, "endDate": b, "gameType": game_type} for a, b in _year_chunks(start, end)
    ]


def _mlb_games(s: Sess, seasons, start, end, game_type: str) -> Games:
    """Every played game in the requested span."""
    games: Games = {}
    for q in _mlb_queries(seasons, start, end, game_type):
        games |= _schedule(s, q)
    return _in_range(games, start, end)


def _pitcher_span_games(s: Sess, pid: int, seasons, start, end, game_type: str) -> Games:
    """Every game the pitcher appeared in across the requested span."""
    years = _seasons(seasons) if seasons is not None else _years(start, end)
    games: Games = {}
    for year in years:
        games |= _pitcher_games(s, pid, year, game_type)
    return _in_range(games, start, end)


def _split_game(sp: dict) -> _Game:
    """Build a _Game from a gameLog split; `team` is the pitcher's own club."""
    own, opp = sp["team"]["id"], sp["opponent"]["id"]
    home, away = (own, opp) if sp.get("isHome") else (opp, own)
    return _Game(sp["date"], home, away)


def _pitcher_games(s: Sess, pitcher_id: int, season: int, game_type: str) -> Games:
    """gamePk -> _Game for every game a pitcher appeared in."""
    params = {"stats": "gameLog", "group": "pitching", "season": season, "gameType": game_type}
    log = s.get(f"{API}/people/{pitcher_id}/stats", params=params, timeout=30).json()
    splits = log["stats"][0]["splits"] if log.get("stats") else []
    return {sp["game"]["gamePk"]: _split_game(sp) for sp in splits}


def _team_abbrs(s: Sess, years: Iterable[str]) -> dict[tuple[str, int], str]:
    """(season, team id) -> abbreviation. Season-keyed because clubs rebrand under the
    same id (Oakland was OAK in 2024 and ATH in 2025)."""
    out: dict[tuple[str, int], str] = {}
    for y in years:
        params = {"sportId": 1, "season": y, "fields": "teams,id,abbreviation"}
        r = s.get(f"{API}/teams", params=params, timeout=30).json()
        out.update({(str(y), t["id"]): t["abbreviation"] for t in r.get("teams", ())})
    return out


def _team_names(s: Sess, games: Games) -> Teams:
    """gamePk -> (home abbreviation, away abbreviation)."""
    abbr = _team_abbrs(s, {g.date[:4] for g in games.values()})
    return {
        pk: (
            abbr.get((g.date[:4], g.home), str(g.home)),
            abbr.get((g.date[:4], g.away), str(g.away)),
        )
        for pk, g in games.items()
    }


def _sides(top: bool, away, home) -> tuple:
    """Order an (away, home) pair as (batting, fielding)."""
    return (away, home) if top else (home, away)


# Home plate in the Gameday hit-chart pixel system (identical to Savant hc_x/hc_y). With this
# origin the foul lines fall at exactly +/-45 degrees, so the raw angle needs no rescaling.
_HOME = (125.42, 198.27)


def _spray_angle(df: pd.DataFrame) -> pd.Series:
    """Horizontal angle of a batted ball, in degrees: 0 is straight-away centre field and the
    pull side is negative, so the sign is flipped for left-handed hitters."""
    x = pd.to_numeric(df["hit_coord_x"], errors="coerce") - _HOME[0]
    y = _HOME[1] - pd.to_numeric(df["hit_coord_y"], errors="coerce")
    raw = np.degrees(np.arctan2(x, y))
    return raw.where(df["stand"] != "L", -raw)


# ---- fetch + flatten ----------------------------------------------------------------
def _play_head(play: dict, pk: int, game: _Game, teams: tuple, pid: int | None, pre, post) -> tuple:
    """Play-level values, identical for every pitch in the PA; pre/post are (away, home) scores."""
    mu = play.get("matchup", {})
    ab = play.get("about", {})
    res = play.get("result", {})
    rev = play.get("reviewDetails", {})
    spl = mu.get("splits", {})
    top = ab.get("isTopInning", ab.get("halfInning") == "top")
    home, away = teams
    bat_team, field_team = _sides(top, away, home)
    bat_score, field_score = _sides(top, *pre)
    post_bat, post_field = _sides(top, *post)
    # fmt: off
    return (
        pk, game.date, home, away, bat_team, field_team,
        ab.get("atBatIndex"), ab.get("inning"), ab.get("halfInning"), top,
        ab.get("startTime"), ab.get("endTime"), ab.get("captivatingIndex"),
        ab.get("hasOut"), ab.get("isComplete"), ab.get("isScoringPlay"), ab.get("hasReview"),
        pid, (mu.get("pitcher") or {}).get("fullName"),
        (mu.get("pitchHand") or {}).get("code"), (mu.get("pitchHand") or {}).get("description"),
        (mu.get("batter") or {}).get("id"), (mu.get("batter") or {}).get("fullName"),
        (mu.get("batSide") or {}).get("code"), (mu.get("batSide") or {}).get("description"),
        (mu.get("postOnFirst") or {}).get("id"), (mu.get("postOnSecond") or {}).get("id"),
        (mu.get("postOnThird") or {}).get("id"),
        spl.get("batter"), spl.get("pitcher"), spl.get("menOnBase"),
        res.get("eventType"), res.get("event"), res.get("description"),
        res.get("type"), res.get("isOut"), res.get("rbi"),
        pre[0], pre[1], bat_score, field_score, post[0], post[1], post_bat, post_field,
        rev.get("reviewType"), rev.get("challengeTeamId"), rev.get("isOverturned"),
        rev.get("inProgress"),
    )
    # fmt: on


def _play_rows(play: dict, pk: int, game: _Game, teams: tuple, pid: int | None, pre, post) -> Rows:
    """Yield one row per tracked pitch in a single plate appearance."""
    head = _play_head(play, pk, game, teams, pid, pre, post)
    for e in play.get("playEvents", ()):
        pit = e.get("pitchData")
        if not pit:  # skips IBB / timer-violation phantoms
            continue
        co, br = pit.get("coordinates", {}), pit.get("breaks", {})
        hd = e.get("hitData", {})
        hc, det, cnt = hd.get("coordinates", {}), e.get("details", {}), e.get("count", {})
        typ, call = det.get("type", {}), det.get("call", {})
        # fmt: off
        yield head + (
            e.get("pitchNumber"), e.get("index"), e.get("playId"), e.get("startTime"),
            e.get("endTime"), typ.get("code"), typ.get("description"), call.get("code"),
            call.get("description"), det.get("description"), det.get("code"),
            det.get("isBall"), det.get("isStrike"), det.get("isInPlay"), det.get("isOut"),
            det.get("hasReview"), det.get("runnerGoing"), det.get("disengagementNum"),
            cnt.get("balls"), cnt.get("strikes"), cnt.get("outs"),
            pit.get("startSpeed"), pit.get("endSpeed"), pit.get("plateTime"), pit.get("extension"),
            pit.get("typeConfidence"), pit.get("zone"), pit.get("strikeZoneTop"),
            pit.get("strikeZoneBottom"),
            co.get("pX"), co.get("pZ"), co.get("pfxX"), co.get("pfxZ"),
            co.get("vX0"), co.get("vY0"), co.get("vZ0"), co.get("aX"), co.get("aY"), co.get("aZ"),
            co.get("x0"), co.get("y0"), co.get("z0"), co.get("x"), co.get("y"),
            br.get("breakAngle"), br.get("breakLength"), br.get("breakY"), br.get("breakVertical"),
            br.get("breakVerticalInduced"), br.get("breakHorizontal"), br.get("spinRate"),
            br.get("spinDirection"), hd.get("launchSpeed"), hd.get("launchAngle"),
            hd.get("totalDistance"), hd.get("trajectory"), hd.get("hardness"), hd.get("location"),
            hc.get("coordX"), hc.get("coordY"),
        )
        # fmt: on


def _game_rows(blob: bytes, pk: int, game: _Game, teams: tuple, pitcher_id: int | None) -> Rows:
    """Yield rows for one game, optionally limited to a single pitcher. The API reports the
    score *after* each play, so the pre-play score is the previous play's result; every play
    is walked, filtered or not, to keep that running total right."""
    pre = (0, 0)
    for play in orjson.loads(blob).get("allPlays", ()):
        res = play.get("result", {})
        post = (res.get("awayScore", pre[0]), res.get("homeScore", pre[1]))
        pid = play.get("matchup", {}).get("pitcher", {}).get("id")
        if pitcher_id is None or pid == pitcher_id:
            yield from _play_rows(play, pk, game, teams, pid, pre, post)
        pre = post


def _check_columns(columns) -> None:
    """Reject unknown column names up front."""
    if columns is None:
        return
    unknown = [c for c in columns if c not in _COLS]
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown column(s): {names}. See statfast.COLUMNS for the valid names.")


def _work_columns(columns) -> list[str]:
    """Columns to materialise: those requested, plus the sort keys."""
    if columns is None:
        return list(_COLS)
    wanted = set(columns) | set(_SORT)
    return [c for c in _COLS if c in wanted]


def _narrow(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Frame limited to the working columns, owning its data: the casts assign into it, and a
    bare ``df[cols]`` slice is a child of ``df`` that raises SettingWithCopyWarning on pandas < 3
    when mutated. The default path was just built from row tuples and needs no copy."""
    if columns is None:
        return df
    return df[_work_columns(columns)].copy()


def _select(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Narrow to the requested columns, in the caller's order."""
    return df if columns is None else df[list(columns)]


def _cast_group(df: pd.DataFrame, cols: tuple[str, ...], dtype: str) -> None:
    numeric = dtype != "boolean"
    for c in cols:
        if c not in df.columns:
            continue
        col = pd.to_numeric(df[c], errors="coerce") if numeric else df[c]
        df[c] = col.astype(dtype)


def _cast_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compact every non-string column in place, skipping columns not present."""
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d")
    for c in _TIME:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="ISO8601", utc=True)
    for cols, dtype in _VALUE_CASTS:
        _cast_group(df, cols, dtype)
    return df


def _categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Dictionary-encode the string columns in place. The NA-preserving "string" step gives every
    chunk the same category dtype; an all-null column would otherwise get object categories,
    which union_categoricals refuses to merge."""
    for c in _CAT:
        if c in df.columns:
            df[c] = df[c].astype("string").astype("category")
    return df


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate chunk frames; pd.concat would degrade differing categoricals to object."""
    cats = [c for c in _CAT if c in frames[0].columns]
    df = pd.concat([f.drop(columns=cats) for f in frames], ignore_index=True)
    for c in cats:
        df[c] = union_categoricals([f[c] for f in frames])
    return df[list(frames[0].columns)]  # restore column order


def _fetch_all(s: Sess, games: Games, workers: int) -> Iterator[tuple[int, bytes]]:
    """Yield (gamePk, playByPlay bytes) as downloads complete, in game order. Consumed lazily:
    Executor.map drops each future once yielded and flattening outpaces the network, so only
    a handful of blobs are resident at once (a materialised list held ~1 GB per season)."""

    def fetch(pk: int) -> bytes:
        r = s.get(f"{API}/game/{pk}/playByPlay", params={"fields": _FIELDS}, timeout=30)
        r.raise_for_status()
        return r.content

    with ThreadPoolExecutor(max_workers=workers) as ex:
        yield from zip(games, ex.map(fetch, games), strict=True)


def _chunk_frames(
    s: Sess, games: Games, teams: Teams, pid: int | None, workers: int, columns
) -> Iterator[pd.DataFrame]:
    """Flatten games in chunks, yielding a fully compacted frame per chunk, so one chunk's
    row tuples are the only Python-object-heavy state alive at a time."""
    for batch in batched(_fetch_all(s, games, workers), _CHUNK_GAMES):
        rows = [
            row for pk, blob in batch for row in _game_rows(blob, pk, games[pk], teams[pk], pid)
        ]
        if rows:
            df = pd.DataFrame(rows, columns=_ROW_COLS)
            df["spray_angle"] = _spray_angle(df)
            yield _categorize(_cast_values(_narrow(df, columns)))


def _collect(s: Sess, games: Games, pid: int | None, workers: int, columns=None) -> pd.DataFrame:
    """Fetch each game's playByPlay in parallel and flatten to a DataFrame."""
    _check_columns(columns)
    if not games:
        return _select(pd.DataFrame(columns=_COLS), columns)
    frames = list(_chunk_frames(s, games, _team_names(s, games), pid, workers, columns))
    if not frames:
        return _select(pd.DataFrame(columns=_COLS), columns)
    df = _concat(frames)
    del frames  # drop the chunk copies early
    return _select(df.sort_values(list(_SORT), ignore_index=True), columns)


# ---- player lookup ------------------------------------------------------------------
def _person(s: Sess, pid: int) -> tuple[int, str]:
    """Look up a player by MLBAM id."""
    r = s.get(f"{API}/people/{pid}", params={"fields": "people,id,fullName"}, timeout=30)
    people = r.json().get("people", []) if r.ok else []
    if not people:
        raise NotFound(f"no MLB player with id {pid}")
    return pid, people[0]["fullName"]


def resolve_pitcher(
    name: str | int, season: int | None = None, session: Sess | None = None
) -> tuple[int, str]:
    """Resolve a pitcher name (or a raw MLBAM id) to ``(id, full_name)``."""
    s = session or _session()
    if isinstance(name, int) or str(name).isdigit():
        return _person(s, int(name))
    params = {"names": name, "sportIds": "1", **({"season": season} if season else {})}
    hits = s.get(f"{API}/people/search", params=params, timeout=30).json().get("people", [])
    if not hits:
        raise NotFound(f"no MLB player matching {name!r}")
    pool = [p for p in hits if p.get("primaryPosition", {}).get("abbreviation") in ("P", "TWP")]
    pool = pool or hits
    if len(pool) > 1:
        opts = ", ".join(f"{p['fullName']} ({p['id']})" for p in pool[:10])
        raise NotFound(f"{name!r} is ambiguous - pass an id. Candidates: {opts}")
    return pool[0]["id"], pool[0]["fullName"]


# ---- pull modes ---------------------------------------------------------------------
def mlb_season(
    seasons: int | Iterable[int] | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    game_type: str = "R",
    workers: int = 12,
    columns: Iterable[str] | None = None,
    session: Sess | None = None,
) -> pd.DataFrame:
    """Every tracked pitch in whole seasons, or between two dates."""
    start, end = _check_span(seasons, start, end)
    s = session or _session(workers)
    df = _collect(s, _mlb_games(s, seasons, start, end, game_type), None, workers, columns)
    span = _span_label(seasons, start, end)
    df.attrs.update(scope="mlb_season", span=span, start=start, end=end, game_type=game_type)
    return df


def pitcher_season(
    pitcher: str | int,
    seasons: int | Iterable[int] | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    game_type: str = "R",
    workers: int = 12,
    columns: Iterable[str] | None = None,
    session: Sess | None = None,
) -> pd.DataFrame:
    """Every tracked pitch by one pitcher, in whole seasons or between dates."""
    start, end = _check_span(seasons, start, end)
    s = session or _session(workers)
    pid, full = resolve_pitcher(pitcher, session=s)
    games = _pitcher_span_games(s, pid, seasons, start, end, game_type)
    df = _collect(s, games, pid, workers, columns)
    span = _span_label(seasons, start, end)
    df.attrs.update(scope="pitcher_season", pitcher_id=pid, pitcher_name=full, span=span)
    df.attrs.update(start=start, end=end, game_type=game_type)
    return df


def mlb_day(
    date: DateLike,
    *,
    game_type: str = "R",
    workers: int = 12,
    columns: Iterable[str] | None = None,
    session: Sess | None = None,
) -> pd.DataFrame:
    """Every tracked pitch thrown on a single date (any common date form)."""
    date = _date(date)
    s = session or _session(workers)
    df = _collect(s, _schedule(s, {"date": date, "gameType": game_type}), None, workers, columns)
    df.attrs.update(scope="mlb_day", date=date, game_type=game_type)
    return df


def _one_game(
    s: Sess, pid: int, game_pk: int | None, game_date: DateLike | None, game_type: str
) -> Games:
    """Resolve exactly one game for a pitcher, by gamePk or by date."""
    if game_pk is not None:
        found = _schedule(s, {"gamePk": int(game_pk)})
        if not found:
            raise NotFound(f"no completed game with gamePk {game_pk}")
        return found
    game_date = _date(game_date)
    games = _pitcher_games(s, pid, int(game_date[:4]), game_type)
    games = {pk: g for pk, g in games.items() if g.date == game_date}
    if not games:
        raise NotFound(f"pitcher {pid} did not appear on {game_date}")
    return games


def pitcher_game(
    pitcher: str | int,
    *,
    game_pk: int | None = None,
    game_date: DateLike | None = None,
    game_type: str = "R",
    workers: int = 12,
    columns: Iterable[str] | None = None,
    session: Sess | None = None,
) -> pd.DataFrame:
    """Every tracked pitch by one pitcher in a single game (by gamePk or date)."""
    if (game_pk is None) == (game_date is None):
        raise ValueError("pass exactly one of game_pk= or game_date=")
    s = session or _session(workers)
    pid, full = resolve_pitcher(pitcher, session=s)
    games = _one_game(s, pid, game_pk, game_date, game_type)
    df = _collect(s, games, pid, workers, columns)
    first = next(iter(games))
    df.attrs.update(scope="pitcher_game", pitcher_id=pid, pitcher_name=full, game_pk=first)
    df.attrs.update(game_date=games[first].date, game_type=game_type)
    return df


# ---- CLI ----------------------------------------------------------------------------
def _title(df: pd.DataFrame) -> str:
    at = df.attrs
    when = at.get("date") or at.get("game_date") or at.get("span", "")
    who = at.get("pitcher_name") or "MLB"
    return f"{who} | {at.get('scope', '?')} {when} [{at.get('game_type', 'R')}]"


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
    if "game_pk" in df:
        print(f"  {df.game_pk.nunique():,} games")
    if "game_date" in df:
        print(f"  {df.game_date.min().date()} -> {df.game_date.max().date()}")
    if "pitcher" in df and df.pitcher.nunique() > 1:
        print(f"  {df.pitcher.nunique():,} pitchers")
    _report_mix(df)


def _kw(a) -> dict:
    """Keyword arguments for a pull, from the parsed CLI namespace."""
    kw = {"game_type": a.game_type, "workers": a.workers}
    kw["columns"] = [c.strip() for c in a.columns.split(",")] if a.columns else None
    if hasattr(a, "start"):
        kw.update(start=a.start, end=a.end)
    return kw


def _build_parser():
    import argparse

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-t", "--game-type", default="R", help='"R" (default), "P", "S" or "R,P"')
    common.add_argument("-w", "--workers", type=int, default=12)
    common.add_argument("-o", "--out", help="write .parquet / .csv")
    common.add_argument("-c", "--columns", help="comma-separated subset of columns (default: all)")
    ap = argparse.ArgumentParser(description="Pull pitch-level Statcast data.")
    sub = ap.add_subparsers(dest="mode", required=True)

    def mode(name: str, help: str, pitcher: bool = False, span: bool = False):
        p = sub.add_parser(name, parents=[common], help=help)
        if pitcher:
            p.add_argument("pitcher", help='name ("Tarik Skubal") or MLBAM id (669373)')
        if span:
            p.add_argument("seasons", nargs="*", type=int)
            p.add_argument("--start", help="date (any common form), instead of seasons")
            p.add_argument("--end", help="date (any common form)")
        return p

    p = mode("mlb-season", "all pitches, by season or date range", span=True)
    p.set_defaults(run=lambda a: mlb_season(a.seasons or None, **_kw(a)))
    p = mode("pitcher-season", "one pitcher, by season or date range", pitcher=True, span=True)
    p.set_defaults(run=lambda a: pitcher_season(a.pitcher, a.seasons or None, **_kw(a)))
    p = mode("pitcher-game", "one pitcher, one game", pitcher=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="date (any common form)")
    g.add_argument("--pk", type=int, help="gamePk")
    p.set_defaults(run=lambda a: pitcher_game(a.pitcher, game_pk=a.pk, game_date=a.date, **_kw(a)))
    p = mode("mlb-day", "all pitches on one date")
    p.add_argument("date", help="date (any common form)")
    p.set_defaults(run=lambda a: mlb_day(a.date, **_kw(a)))
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
    if df.empty:
        print("no pitches matched")
        return 1
    _report(df, time.perf_counter() - t0)
    if a.out:
        (df.to_parquet if a.out.endswith(".parquet") else df.to_csv)(a.out, index=False)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
