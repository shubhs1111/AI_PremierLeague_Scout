"""Load the raw CSVs into tidy pandas frames."""
from functools import lru_cache

import pandas as pd

from .config import RAW


@lru_cache(maxsize=None)
def teams(season: str) -> pd.DataFrame:
    return pd.read_csv(RAW / season / "teams.csv")


@lru_cache(maxsize=None)
def players(season: str) -> pd.DataFrame:
    return pd.read_csv(RAW / season / "players.csv")


@lru_cache(maxsize=None)
def playerstats_latest(season: str) -> pd.DataFrame:
    """Latest cumulative FPL snapshot per player (price, status, news)."""
    ps = pd.read_csv(RAW / season / "playerstats.csv", low_memory=False)
    ps = ps.sort_values("gw").groupby("id", as_index=False).tail(1)
    return ps.rename(columns={"id": "player_id"})


@lru_cache(maxsize=None)
def matches(season: str) -> pd.DataFrame:
    files = sorted((RAW / season / "pl").glob("matches_gw*.csv"))
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m = m[m["finished"] == True].drop_duplicates("match_id")  # noqa: E712
    for col in ("home_team", "away_team"):
        m[col] = m[col].astype(int)
    return m


@lru_cache(maxsize=None)
def player_matches(season: str) -> pd.DataFrame:
    files = sorted((RAW / season / "pl").glob("playermatchstats_gw*.csv"))
    pm = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pm = pm.drop_duplicates(["player_id", "match_id"])
    pm = pm[pm["minutes_played"].fillna(0) > 0].copy()
    pl = players(season)[["player_id", "player_code", "team_code", "position"]]
    pm = pm.merge(pl, on="player_id", how="left")
    pm["season"] = season
    return pm
