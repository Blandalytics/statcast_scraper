"""Fast pitch-level Statcast data from statsapi.mlb.com.

Four pull modes, all sharing one fetch/flatten core:

    mlb_season(2024)                                 every pitch in a season
    pitcher_season("Tarik Skubal", [2023, 2024])     one pitcher, one/many seasons
    pitcher_game("Skubal", game_date="2024-06-01")   one pitcher, one game
    mlb_day("2024-06-01")                            every pitch on a date

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

# Whitelist of JSON keys; roughly halves the playByPlay payload.
_FIELDS = ",".join((
    "allPlays", "about", "inning", "halfInning", "atBatIndex",
    "matchup", "batter", "pitcher", "id", "batSide", "pitchHand", "code",
    "result", "eventType",
    "playEvents", "isPitch", "pitchNumber", "playId", "details", "call",
    "description", "type", "eventType",
    "count", "balls", "strikes", "outs",
    "pitchData", "startSpeed", "endSpeed", "zone", "extension",
    "strikeZoneTop", "strikeZoneBottom", "coordinates",
    "pX", "pZ", "pfxX", "pfxZ", "vX0", "vY0", "vZ0", "aX", "aY", "aZ",
    "x0", "y0", "z0",
    "breaks", "spinRate", "spinDirection", "breakVerticalInduced", "breakHorizontal",
    "hitData", "launchSpeed", "launchAngle", "totalDistance",
))
_SCHED_FIELDS = "dates,date,games,gamePk,status,codedGameState"
# The schedule endpoint has no "P" code; postseason is wild card / division /
# league championship / world series. The gameLog endpoint does accept "P".
_POSTSEASON = "F,D,L,W"

_COLS = ("game_pk", "game_date", "inning", "half", "at_bat_index", "pitch_number",
         "pitch_type", "call_code", "description", "events", "balls",
         "strikes", "outs", "pitcher", "batter", "stand", "p_throws",
         "release_speed", "end_speed", "zone", "release_extension",
         "plate_x", "plate_z", "pfx_x", "pfx_z", "vx0", "vy0", "vz0",
         "ax", "ay", "az", "release_pos_x", "release_pos_y", "release_pos_z",
         "release_spin_rate", "spin_axis", "ivb", "hb", "sz_top", "sz_bot",
         "launch_speed", "launch_angle", "hit_distance", "play_id")

# float32 is ample: Statcast reports 1-2 decimals.
_F32 = ("release_speed", "end_speed", "release_extension", "plate_x", "plate_z",
        "pfx_x", "pfx_z", "vx0", "vy0", "vz0",
        "ax", "ay", "az", "release_pos_x", "release_pos_y", "release_pos_z",
        "ivb", "hb", "sz_top", "sz_bot", "launch_speed", "launch_angle",
        "hit_distance", "release_spin_rate", "spin_axis")
_U8 = ("balls", "strikes", "outs", "inning", "pitch_number", "zone")
_I32 = ("game_pk", "pitcher", "batter", "at_bat_index")
_CAT = ("pitch_type", "description", "call_code", "events", "stand", "p_throws", "half")
_CASTS = ((_F32, "float32"), (_U8, "UInt8"), (_I32, "int32"), (_CAT, "category"))


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

def _play_rows(play: dict, pk: int, date: str, pid: int | None) -> Iterator[tuple]:
    """Yield one row per tracked pitch in a single plate appearance."""
    mu = play.get("matchup", {})
    ab = play.get("about", {})
    inn, half, abi = ab.get("inning"), ab.get("halfInning"), ab.get("atBatIndex")
    bat = mu.get("batter", {}).get("id")
    stand = mu.get("batSide", {}).get("code")
    throws = mu.get("pitchHand", {}).get("code")
    ev = play.get("result", {}).get("eventType")
    for e in play.get("playEvents", ()):
        pit = e.get("pitchData")
        if not pit:                      # skips IBB / timer-violation phantoms
            continue
        co = pit.get("coordinates", {})
        br = pit.get("breaks", {})
        hd = e.get("hitData", {})
        det = e.get("details", {})
        cnt = e.get("count", {})
        yield (pk, date, inn, half, abi, e.get("pitchNumber"),
               det.get("type", {}).get("code"), det.get("call", {}).get("code"),
               det.get("description"), ev,
               cnt.get("balls"), cnt.get("strikes"), cnt.get("outs"),
               pid, bat, stand, throws,
               pit.get("startSpeed"), pit.get("endSpeed"), pit.get("zone"),
               pit.get("extension"),
               co.get("pX"), co.get("pZ"), co.get("pfxX"), co.get("pfxZ"),
               co.get("vX0"), co.get("vY0"), co.get("vZ0"),
               co.get("aX"), co.get("aY"), co.get("aZ"),
               co.get("x0"), co.get("y0"), co.get("z0"),
               br.get("spinRate"), br.get("spinDirection"),
               br.get("breakVerticalInduced"), br.get("breakHorizontal"),
               pit.get("strikeZoneTop"), pit.get("strikeZoneBottom"),
               hd.get("launchSpeed"), hd.get("launchAngle"), hd.get("totalDistance"),
               e.get("playId"))


def _game_rows(blob: bytes, pk: int, date: str,
               pitcher_id: int | None) -> Iterator[tuple]:
    """Yield rows for one game, optionally limited to a single pitcher."""
    for play in orjson.loads(blob).get("allPlays", ()):
        pid = play.get("matchup", {}).get("pitcher", {}).get("id")
        if pitcher_id is None or pid == pitcher_id:
            yield from _play_rows(play, pk, date, pid)


def _cast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink to compact dtypes in place."""
    df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d")
    for cols, dtype in _CASTS:
        numeric = dtype != "category"
        for c in cols:
            col = pd.to_numeric(df[c], errors="coerce") if numeric else df[c]
            df[c] = col.astype(dtype)
    return df


def _collect(s: requests.Session, games: dict[int, str], pitcher_id: int | None,
             workers: int) -> pd.DataFrame:
    """Fetch each game's playByPlay in parallel and flatten to a DataFrame."""
    if not games:
        return pd.DataFrame(columns=_COLS)

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
        return df
    return _cast(df).sort_values(["game_date", "game_pk", "at_bat_index", "pitch_number"],
                                 ignore_index=True)


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

def mlb_season(seasons: int | Iterable[int], *, game_type: str = "R",
               workers: int = 12,
               session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch thrown in one or more full seasons."""
    s = session or _session(workers)
    years = _seasons(seasons)
    games: dict[int, str] = {}
    for year in years:
        games |= _schedule(s, {"season": year, "gameType": game_type})
    df = _collect(s, games, None, workers)
    df.attrs.update(scope="mlb_season", seasons=years, game_type=game_type)
    return df


def pitcher_season(pitcher: str | int, seasons: int | Iterable[int], *,
                   game_type: str = "R", workers: int = 12,
                   session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch by one pitcher across one or more full seasons."""
    s = session or _session(workers)
    pid, full = resolve_pitcher(pitcher, session=s)
    years = _seasons(seasons)
    games: dict[int, str] = {}
    for year in years:
        games |= _pitcher_games(s, pid, year, game_type)
    df = _collect(s, games, pid, workers)
    df.attrs.update(scope="pitcher_season", pitcher_id=pid, pitcher_name=full,
                    seasons=years, game_type=game_type)
    return df


def mlb_day(date: str, *, game_type: str = "R", workers: int = 12,
            session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch thrown on a single date (YYYY-MM-DD)."""
    s = session or _session(workers)
    games = _schedule(s, {"date": str(date), "gameType": game_type})
    df = _collect(s, games, None, workers)
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
                 workers: int = 12,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Every tracked pitch by one pitcher in a single game (by gamePk or date)."""
    if (game_pk is None) == (game_date is None):
        raise ValueError("pass exactly one of game_pk= or game_date=")
    s = session or _session(workers)
    pid, full = resolve_pitcher(pitcher, session=s)
    games = _one_game(s, pid, game_pk, game_date, game_type)
    df = _collect(s, games, pid, workers)
    first = next(iter(games))
    df.attrs.update(scope="pitcher_game", pitcher_id=pid, pitcher_name=full,
                    game_pk=first, game_date=games[first], game_type=game_type)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _title(df: pd.DataFrame) -> str:
    at = df.attrs
    when = at.get("date") or at.get("game_date") or ", ".join(
        str(y) for y in at.get("seasons", ()))
    return (f"{at.get('pitcher_name') or 'MLB'} | {at.get('scope', '?')} "
            f"{when} [{at.get('game_type', 'R')}]")


def _report(df: pd.DataFrame, elapsed: float) -> None:
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(_title(df))
    print(f"  {len(df):,} pitches / {df.game_pk.nunique():,} games / {df.shape[1]} cols")
    print(f"  {df.game_date.min().date()} -> {df.game_date.max().date()}")
    print(f"  {elapsed:.2f}s, {mb:,.1f} MB")
    if df.pitcher.nunique() > 1:
        print(f"  {df.pitcher.nunique():,} pitchers, {df.batter.nunique():,} batters")
    mix = df.pitch_type.value_counts(normalize=True).mul(100).head(6)
    velo = df.groupby("pitch_type", observed=True).release_speed.mean()
    print("  mix:", ", ".join(f"{p} {mix[p]:.1f}% @{velo[p]:.1f}" for p in mix.index))


def _build_parser():
    import argparse

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-t", "--game-type", default="R",
                        help='"R" (default), "P", "S", or e.g. "R,P"')
    common.add_argument("-w", "--workers", type=int, default=12)
    common.add_argument("-o", "--out", help="write .parquet / .csv")

    ap = argparse.ArgumentParser(description="Pull pitch-level Statcast data.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("mlb-season", parents=[common], help="all pitches in one or more seasons")
    p.add_argument("seasons", nargs="+", type=int)
    p.set_defaults(run=lambda a: mlb_season(a.seasons, game_type=a.game_type,
                                            workers=a.workers))

    p = sub.add_parser("pitcher-season", parents=[common], help="one pitcher, one or more seasons")
    p.add_argument("pitcher", help='name ("Tarik Skubal") or MLBAM id (669373)')
    p.add_argument("seasons", nargs="+", type=int)
    p.set_defaults(run=lambda a: pitcher_season(a.pitcher, a.seasons,
                                                game_type=a.game_type,
                                                workers=a.workers))

    p = sub.add_parser("pitcher-game", parents=[common], help="one pitcher, one game")
    p.add_argument("pitcher", help='name ("Tarik Skubal") or MLBAM id (669373)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="YYYY-MM-DD")
    g.add_argument("--pk", type=int, help="gamePk")
    p.set_defaults(run=lambda a: pitcher_game(a.pitcher, game_pk=a.pk,
                                              game_date=a.date,
                                              game_type=a.game_type,
                                              workers=a.workers))

    p = sub.add_parser("mlb-day", parents=[common], help="all pitches on one date")
    p.add_argument("date", help="YYYY-MM-DD")
    p.set_defaults(run=lambda a: mlb_day(a.date, game_type=a.game_type,
                                         workers=a.workers))
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
