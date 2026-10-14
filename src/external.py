"""Link the uploaded 2025/26 player stats (SofaScore-style export) to FPL players.

Used two ways: as an independent check on our ratings (its match rating is
built by a different provider), and for error and possession-loss counts that
FPL Core does not carry.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from . import load
from .config import PREVIOUS_SEASON, ROOT

EXTERNAL = ROOT / "data" / "external"

TEAM_MAP = {
    "Liverpool FC": "Liverpool", "Manchester City": "Man City", "Manchester United": "Man Utd",
    "Brighton &amp; Hove Albion": "Brighton", "Newcastle United": "Newcastle", "Leeds United": "Leeds",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Spurs", "West Ham United": "West Ham",
    "Wolverhampton": "Wolves",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def sofascore_2025_26() -> pd.DataFrame:
    d = pd.read_csv(EXTERNAL / "pl_player_stats_2025-26_gd35.csv")
    d["team"] = d.team_name.replace(TEAM_MAP)
    return d


def link_to_fpl(d: pd.DataFrame) -> pd.DataFrame:
    """Attach player_code by full name, then by last name within the same club."""
    fpl = load.players(PREVIOUS_SEASON).copy()
    teams = load.teams(PREVIOUS_SEASON).set_index("code").name
    fpl["team"] = fpl.team_code.map(teams)
    fpl["full"] = (fpl.first_name + " " + fpl.second_name).map(_norm)
    fpl["last"] = fpl.second_name.map(lambda s: _norm(s).split(" ")[-1])
    fpl["web"] = fpl.web_name.map(_norm)

    d = d.copy()
    d["full"] = d.player_name.map(_norm)
    d["last"] = d.full.str.split(" ").str[-1]
    by_full = fpl.drop_duplicates("full").set_index("full").player_code
    d["player_code"] = d.full.map(by_full)
    # Single-name players ("Rodri", "João Pedro") match on FPL's display name.
    by_web = fpl.drop_duplicates("web").set_index("web").player_code
    d["player_code"] = d.player_code.fillna(d.full.map(by_web))

    for col in ("last", "web"):
        keyed = fpl.drop_duplicates(["team", col]).set_index(["team", col]).player_code
        miss = d.player_code.isna()
        key = list(zip(d.loc[miss, "team"], d.loc[miss, "last"]))
        d.loc[miss, "player_code"] = pd.Series([keyed.get(k) for k in key], index=d.index[miss], dtype="float64")
    return d


def validation_table(players: pd.DataFrame) -> pd.DataFrame:
    """Our rating next to the external 2025/26 match rating for regulars."""
    ext = link_to_fpl(sofascore_2025_26())
    ext = ext[(ext.minutesPlayed >= 900) & ext.player_code.notna() & (ext.rating > 0)]
    # A last-name match can hit two rows; an ambiguous link is dropped rather than guessed.
    ext = ext[~ext.player_code.duplicated(keep=False)]
    cols = ["player_code", "rating", "minutesPlayed", "errorLeadToGoal", "errorLeadToShot", "possessionLost"]
    m = players.merge(ext[cols].rename(columns={"rating": "ext_rating", "minutesPlayed": "ext_minutes"}),
                      on="player_code", how="inner")
    return m


HISTORY_TEAM_MAP = {"Nott'm Forest": "Nott'm Forest", "Tottenham": "Spurs", "Man United": "Man Utd",
                    "Newcastle": "Newcastle", "Leeds": "Leeds", "Wolves": "Wolves",
                    "Brighton": "Brighton", "West Ham": "West Ham", "Hull": "Hull City",
                    "Ipswich": "Ipswich Town", "Coventry": "Coventry City"}


def points_history(first_season: str = "2020/21") -> pd.DataFrame:
    """Points per game per club per season from the uploaded 2000–2025 results file."""
    e = pd.read_csv(EXTERNAL / "epl_matches_2000_2025.csv")
    e = e[e.Season >= first_season]
    rows = []
    for side, gf, ga in (("HomeTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"),
                         ("AwayTeam", "FullTimeAwayGoals", "FullTimeHomeGoals")):
        pts = (e[gf] > e[ga]) * 3 + (e[gf] == e[ga]) * 1
        rows.append(pd.DataFrame({"season": e.Season, "team": e[side].replace(HISTORY_TEAM_MAP),
                                  "points": pts, "gf": e[gf], "ga": e[ga]}))
    h = pd.concat(rows).groupby(["team", "season"]).agg(
        played=("points", "size"), points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
    h["points_pg"] = h.points / h.played
    return h


TM_TEAM_MAP = {
    "Arsenal FC": "Arsenal", "Aston Villa": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brighton & Hove Albion": "Brighton", "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea", "Crystal Palace": "Crystal Palace", "Everton FC": "Everton",
    "Fulham FC": "Fulham", "Leeds United": "Leeds", "Liverpool FC": "Liverpool",
    "Manchester City": "Man City", "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Sunderland AFC": "Sunderland", "Tottenham Hotspur": "Spurs",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves", "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester", "Southampton FC": "Southampton", "Hull City": "Hull",
    "Coventry City": "Coventry",
}


def transfermarkt() -> pd.DataFrame:
    """Transfermarkt player profiles (Kaggle davidcariboo/player-scores players.csv)."""
    d = pd.read_csv(EXTERNAL / "transfermarkt_players.csv",
                    parse_dates=["date_of_birth", "contract_expiration_date"])
    d = d[d.last_season >= 2024].copy()  # still active in the dataset's latest seasons
    d["team"] = d.current_club_name.map(TM_TEAM_MAP)
    d["full"] = d.name.map(_norm)
    d["last"] = d.full.str.split(" ").str[-1]
    return d


def link_transfermarkt(players: pd.DataFrame) -> pd.DataFrame:
    """Age, market value and contract for each FPL player, keyed by player_code.

    Matching order: full name at the same club, full name anywhere if unique,
    then last name or FPL display name at the same club. Anything ambiguous stays unmatched.
    """
    tm = transfermarkt()
    p = players[["player_code", "first_name", "second_name", "web_name", "team", "team_code_previous"]].copy()
    prev_team = load.teams(PREVIOUS_SEASON).set_index("code").name
    p["team_any"] = p.team.fillna(p.team_code_previous.map(prev_team))
    p["full"] = (p.first_name.fillna("") + " " + p.second_name.fillna("")).map(_norm)
    p["last"] = p.second_name.fillna("").map(lambda s: _norm(s).split(" ")[-1])
    p["web"] = p.web_name.fillna("").map(_norm)
    # Iberian and Brazilian names: FPL lists both surnames, Transfermarkt usually one.
    first = p.first_name.fillna("").map(lambda s: (_norm(s).split(" ") or [""])[0])
    sur = p.second_name.fillna("").map(lambda s: _norm(s).split(" "))
    p["first_sur1"] = first + " " + sur.str[0]
    p["first_surlast"] = first + " " + sur.str[-1]

    def unique_map(keys, pool=None):
        g = (tm if pool is None else pool).groupby(keys).player_id
        return g.first()[g.size() == 1]

    p["tm_id"] = pd.Series([unique_map(["team", "full"]).get(k) for k in zip(p.team_any, p.full)],
                           index=p.index, dtype="float64")
    steps = [(["full"], lambda r: r.full),
             (["team", "last"], lambda r: (r.team_any, r["last"])),
             (["team", "last"], lambda r: (r.team_any, r.web.split(" ")[-1])),
             (["full"], lambda r: r.web),
             (["full"], lambda r: r.first_sur1),
             (["full"], lambda r: r.first_surlast),
             (["full"], lambda r: _norm(r.first_name or ""))]
    english = tm[tm.team.notna()]
    steps += [(["full"], lambda r: r.first_sur1, english), (["full"], lambda r: r.first_surlast, english)]
    for keys, key_of, *pool in steps:
        m = unique_map(keys, *pool)
        miss = p.tm_id.isna()
        p.loc[miss, "tm_id"] = pd.Series([m.get(key_of(r)) for _, r in p[miss].iterrows()],
                                         index=p.index[miss], dtype="float64")

    # One Transfermarkt profile can only belong to one FPL player.
    p.loc[p.tm_id.duplicated(keep=False), "tm_id"] = np.nan
    tm = tm.set_index("player_id")
    out = p[["player_code", "tm_id"]].dropna().copy()
    row = tm.loc[out.tm_id.astype(int)]
    out["date_of_birth"] = row.date_of_birth.values
    out["market_value_m"] = row.market_value_in_eur.values / 1e6
    out["contract_until"] = row.contract_expiration_date.values
    out["foot"] = row.foot.values
    out["height_cm"] = row.height_in_cm.values
    out["tm_url"] = row.url.values
    out["sub_position"] = row.sub_position.values
    return out.drop(columns="tm_id")
