"""Per-90 player profiles, style-based roles and role-specific ratings.

Ratings blend this season with last season (last season's minutes count for
less), then score each player against everyone else in the same role.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import load
from .config import AS_OF_DATE, CURRENT_SEASON, PREVIOUS_SEASON, PREVIOUS_SEASON_WEIGHT, SHRINKAGE_MINUTES

# Counting stats summed over matches, then converted to per-90.
COUNT_STATS = [
    "goals", "assists", "xg", "xa", "total_shots", "shots_on_target", "xgot", "chances_created",
    "successful_dribbles", "touches_opposition_box", "touches", "accurate_passes", "final_third_passes",
    "accurate_crosses", "accurate_long_balls", "tackles_won", "tackles", "interceptions", "recoveries",
    "blocks", "clearances", "dribbled_past", "duels_won", "duels_lost", "ground_duels_won",
    "aerial_duels_won", "was_fouled", "fouls_committed", "dispossessed", "big_chances_missed",
    "saves", "goals_conceded", "xgot_faced", "goals_prevented", "sweeper_actions", "high_claim",
    "gk_accurate_long_balls", "defensive_contributions",
]


def _attempts(made: pd.Series, pct: pd.Series) -> pd.Series:
    """Rebuild attempts from a made count and a success percentage."""
    with np.errstate(divide="ignore", invalid="ignore"):
        att = made / (pct / 100)
    return att.where(pct > 0, made).fillna(0)


PADJ_STATS = ["tackles_won", "tackles", "interceptions", "recoveries", "blocks", "clearances"]


def _season_totals(season: str, adjust: bool = True) -> pd.DataFrame:
    pm = load.player_matches(season).copy()
    pm["pass_attempts"] = _attempts(pm.accurate_passes.fillna(0), pm.accurate_passes_percent)
    pm["aerial_attempts"] = _attempts(pm.aerial_duels_won.fillna(0), pm.aerial_duels_won_percent)
    pm["cross_attempts"] = _attempts(pm.accurate_crosses.fillna(0), pm.accurate_crosses_percent)
    pm["dribble_attempts"] = _attempts(pm.successful_dribbles.fillna(0), pm.successful_dribbles_percent)
    pm["shots_faced_on_target"] = pm.saves.fillna(0) + pm.goals_conceded.fillna(0)
    pm["is_start"] = (pm.start_min.fillna(0) == 0).astype(int)
    # Possession-adjust defensive actions: a defender on a side that keeps the ball has
    # fewer chances to tackle. Scale to what he would make against 50% opposition possession.
    m = load.matches(season)[["match_id", "home_team", "home_possession", "away_possession"]]
    pm = pm.merge(m, on="match_id", how="left")
    opp = np.where(pm.team_code == pm.home_team, pm.away_possession, pm.home_possession)
    factor = np.clip(50 / pd.Series(opp, index=pm.index).replace(0, np.nan), 0.6, 1.6).fillna(1.0)
    if adjust:
        for c in PADJ_STATS:
            pm[c] = pm[c] * factor
    extra = ["pass_attempts", "aerial_attempts", "cross_attempts", "dribble_attempts",
             "shots_faced_on_target", "is_start"]
    cols = COUNT_STATS + extra
    pm[cols] = pm[cols].fillna(0)
    agg = pm.groupby("player_code").agg(
        minutes=("minutes_played", "sum"), apps=("match_id", "nunique"), **{c: (c, "sum") for c in cols})
    agg = agg.rename(columns={"is_start": "starts"})
    return agg


def blended_totals() -> pd.DataFrame:
    """Season totals with last season's numbers down-weighted, plus both raw minute counts."""
    cur = _season_totals(CURRENT_SEASON)
    prev = _season_totals(PREVIOUS_SEASON)
    w = PREVIOUS_SEASON_WEIGHT
    both = cur.add(prev * w, fill_value=0)
    both["minutes_current"] = cur.minutes.reindex(both.index).fillna(0)
    both["minutes_previous"] = prev.minutes.reindex(both.index).fillna(0)
    both["starts_current"] = cur.starts.reindex(both.index).fillna(0)
    both["apps_current"] = cur.apps.reindex(both.index).fillna(0)
    return both


def per90_profile(tot: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame(index=tot.index)
    m = tot.minutes.replace(0, np.nan)
    for c in COUNT_STATS:
        p[f"{c}_p90"] = tot[c] / m * 90
    with np.errstate(divide="ignore", invalid="ignore"):
        p["pass_accuracy"] = tot.accurate_passes / tot.pass_attempts * 100
        p["aerial_win_pct"] = tot.aerial_duels_won / tot.aerial_attempts * 100
        p["duel_win_pct"] = tot.duels_won / (tot.duels_won + tot.duels_lost) * 100
        p["cross_accuracy"] = tot.accurate_crosses / tot.cross_attempts * 100
        p["dribble_success"] = tot.successful_dribbles / tot.dribble_attempts * 100
        p["save_pct"] = tot.saves / tot.shots_faced_on_target * 100
        p["finishing_p90"] = (tot.goals - tot.xg) / m * 90
        p["passes_attempted_p90"] = tot.pass_attempts / m * 90
        p["tackles_int_p90"] = (tot.tackles_won + tot.interceptions) / m * 90
        p["xg_xa_p90"] = (tot.xg + tot.xa) / m * 90
    return p


# --- Roles -------------------------------------------------------------------
# FPL only gives four positions. Within each, k-means on playing-style
# features separates centre-backs from full-backs, and holding, box-to-box and
# attacking midfielders from each other.
ROLE_FEATURES = {
    "Defender": ["accurate_crosses_p90", "final_third_passes_p90", "touches_opposition_box_p90",
                 "successful_dribbles_p90", "clearances_p90", "aerial_duels_won_p90", "chances_created_p90"],
    "Midfielder": ["xg_p90", "xa_p90", "chances_created_p90", "successful_dribbles_p90",
                   "touches_opposition_box_p90", "tackles_won_p90", "interceptions_p90",
                   "recoveries_p90", "accurate_long_balls_p90", "accurate_crosses_p90"],
}


# Transfermarkt's listed position, where a player links to it, decides the role;
# k-means on FPL position is the fallback for the rest.
TM_ROLE = {"Goalkeeper": "GK", "Centre-Back": "CB", "Left-Back": "FB", "Right-Back": "FB",
           "Defensive Midfield": "DM", "Central Midfield": "CM", "Attacking Midfield": "AM/W",
           "Left Midfield": "AM/W", "Right Midfield": "AM/W", "Left Winger": "AM/W", "Right Winger": "AM/W",
           "Centre-Forward": "ST", "Second Striker": "ST"}


def assign_roles(prof: pd.DataFrame, positions: pd.Series, minutes: pd.Series,
                 listed: pd.Series | None = None) -> pd.Series:
    role = pd.Series(index=prof.index, dtype=object)
    role[positions == "Goalkeeper"] = "GK"
    role[positions == "Forward"] = "ST"

    for pos, k in (("Defender", 2), ("Midfielder", 3)):
        feats = ROLE_FEATURES[pos]
        idx = positions.index[positions == pos]
        X = prof.loc[idx, feats].fillna(0)
        fit_idx = idx[minutes.loc[idx] >= 450]
        scaler = StandardScaler().fit(X.loc[fit_idx])
        km = KMeans(n_clusters=k, n_init=20, random_state=7).fit(scaler.transform(X.loc[fit_idx]))
        labels = km.predict(scaler.transform(X))
        cent = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=feats)
        if pos == "Defender":
            fb = cent.accurate_crosses_p90.idxmax()
            names = {c: ("FB" if c == fb else "CB") for c in range(k)}
        else:
            attack = cent.xg_p90 + cent.xa_p90 + cent.touches_opposition_box_p90 / 5
            am = attack.idxmax()
            rest = [c for c in range(k) if c != am]
            defence = cent.tackles_won_p90 + cent.interceptions_p90 + cent.accurate_long_balls_p90 / 3
            dm = max(rest, key=lambda c: defence[c] - attack[c])
            names = {c: ("AM/W" if c == am else "DM" if c == dm else "CM") for c in range(k)}
        role.loc[idx] = [names[l] for l in labels]
    if listed is not None:
        tm = listed.map(TM_ROLE)
        # Never move a goalkeeper out of goal or an outfielder into it.
        ok = tm.notna() & ((tm == "GK") == (positions == "Goalkeeper"))
        role[ok] = tm[ok]
    return role


# --- Ratings -----------------------------------------------------------------
# Weights per role. Negative weight = fewer is better (e.g. being dribbled past).
ROLE_WEIGHTS = {
    "GK": {"goals_prevented_p90": 0.40, "save_pct": 0.20, "sweeper_actions_p90": 0.10,
           "high_claim_p90": 0.10, "gk_accurate_long_balls_p90": 0.10, "pass_accuracy": 0.10},
    "CB": {"aerial_duels_won_p90": 0.12, "aerial_win_pct": 0.10, "tackles_int_p90": 0.14,
           "clearances_p90": 0.08, "blocks_p90": 0.06, "duel_win_pct": 0.14, "pass_accuracy": 0.12,
           "accurate_long_balls_p90": 0.10, "dribbled_past_p90": -0.10, "xg_p90": 0.04},
    "FB": {"accurate_crosses_p90": 0.12, "chances_created_p90": 0.14, "xa_p90": 0.12,
           "final_third_passes_p90": 0.14, "tackles_int_p90": 0.14, "duel_win_pct": 0.10,
           "dribbled_past_p90": -0.10, "successful_dribbles_p90": 0.06, "recoveries_p90": 0.08},
    "DM": {"tackles_won_p90": 0.14, "interceptions_p90": 0.14, "recoveries_p90": 0.14,
           "duel_win_pct": 0.14, "pass_accuracy": 0.14, "final_third_passes_p90": 0.10,
           "accurate_long_balls_p90": 0.10, "dispossessed_p90": -0.05, "dribbled_past_p90": -0.05},
    "CM": {"final_third_passes_p90": 0.18, "chances_created_p90": 0.14, "xa_p90": 0.10,
           "xg_p90": 0.10, "pass_accuracy": 0.10, "recoveries_p90": 0.10, "tackles_int_p90": 0.10,
           "successful_dribbles_p90": 0.10, "dispossessed_p90": -0.08},
    "AM/W": {"xg_p90": 0.18, "xa_p90": 0.18, "chances_created_p90": 0.14, "successful_dribbles_p90": 0.14,
             "touches_opposition_box_p90": 0.12, "shots_on_target_p90": 0.10, "dispossessed_p90": -0.08,
             "finishing_p90": 0.06},
    "ST": {"xg_p90": 0.28, "finishing_p90": 0.10, "shots_on_target_p90": 0.10,
           "touches_opposition_box_p90": 0.14, "xa_p90": 0.10, "chances_created_p90": 0.08,
           "aerial_duels_won_p90": 0.08, "duel_win_pct": 0.06, "big_chances_missed_p90": -0.06},
}

# The short list shown on the dashboard for each role.
ROLE_KEY_STATS = {
    "GK": ["goals_prevented_p90", "save_pct", "sweeper_actions_p90", "pass_accuracy"],
    "CB": ["aerial_duels_won_p90", "tackles_int_p90", "duel_win_pct", "pass_accuracy", "accurate_long_balls_p90"],
    "FB": ["chances_created_p90", "accurate_crosses_p90", "tackles_int_p90", "final_third_passes_p90", "xa_p90"],
    "DM": ["tackles_won_p90", "interceptions_p90", "recoveries_p90", "duel_win_pct", "pass_accuracy"],
    "CM": ["final_third_passes_p90", "chances_created_p90", "recoveries_p90", "xg_xa_p90", "pass_accuracy"],
    "AM/W": ["xg_p90", "xa_p90", "chances_created_p90", "successful_dribbles_p90", "touches_opposition_box_p90"],
    "ST": ["xg_p90", "goals_p90", "shots_on_target_p90", "touches_opposition_box_p90", "finishing_p90"],
}


POSITION_ROLES = {"Goalkeeper": ["GK"], "Defender": ["CB", "FB"],
                  "Midfielder": ["DM", "CM", "AM/W"], "Forward": ["ST"]}


def rate_players(df: pd.DataFrame, min_pool_minutes: int = 450) -> pd.DataFrame:
    """Percentile each metric against players whose main role it is, weight, then shrink by minutes.

    Every outfield player is also scored for the other outfield roles
    (rating_CB, rating_FB, ...), so a centre-midfielder covering the holding
    role can be judged as a holding midfielder.
    """
    df = df.copy()
    k = df.minutes / (df.minutes + SHRINKAGE_MINUTES)
    for role, weights in ROLE_WEIGHTS.items():
        scope = (df.position == "Goalkeeper") if role == "GK" else (df.position != "Goalkeeper")
        pool = df[(df.role == role) & (df.minutes >= min_pool_minutes)]
        score = pd.Series(0.0, index=df.index[scope])
        total_w = sum(abs(w) for w in weights.values())
        for metric, w in weights.items():
            ref = pool[metric].dropna().sort_values().values
            vals = df.loc[scope, metric].fillna(np.nanmedian(ref) if len(ref) else 0)
            pct = np.searchsorted(ref, vals.values, side="right") / max(len(ref), 1) * 100
            if w < 0:
                pct = 100 - pct
            score += abs(w) * pct
        raw = score / total_w
        role_mean = raw[(df.loc[scope, "role"] == role) & (df.loc[scope, "minutes"] >= min_pool_minutes)].mean()
        df.loc[scope, f"rating_{role}"] = k[scope] * raw + (1 - k[scope]) * role_mean
    df["rating"] = [df.at[i, f"rating_{r}"] for i, r in df.role.items()]
    # Percentile against the role's regulars, so every page of the dashboard uses the same yardstick.
    df["role_percentile"] = np.nan
    for role, g in df.groupby("role"):
        ref = np.sort(g[g.minutes >= min_pool_minutes].rating.dropna().values)
        df.loc[g.index, "role_percentile"] = np.searchsorted(ref, g.rating.values, side="right") / max(len(ref), 1) * 100
    return df


def build_player_table() -> pd.DataFrame:
    tot = blended_totals()
    prof = per90_profile(tot)

    cur_players = load.players(CURRENT_SEASON).set_index("player_code")
    prev_players = load.players(PREVIOUS_SEASON).set_index("player_code")
    meta = cur_players[["player_id", "web_name", "first_name", "second_name", "team_code", "position"]]
    # Players who left the league are rated (they shape the role pools) but carry no current team.
    meta = meta.combine_first(prev_players[["web_name", "first_name", "second_name", "position"]])

    df = tot[["minutes", "minutes_current", "minutes_previous", "starts_current", "apps_current"]].join(prof)
    df = df.join(meta, how="left")
    df = df[df.position.notna()]

    # Age, market value, contract and listed position from Transfermarkt.
    from .external import link_transfermarkt
    team_names = load.teams(CURRENT_SEASON).set_index("code").name
    prev_team = load.players(PREVIOUS_SEASON).set_index("player_code").team_code
    keys = df.reset_index()[["player_code", "first_name", "second_name", "web_name", "team_code"]]
    keys["team"] = keys.team_code.map(team_names)
    keys["team_code_previous"] = keys.player_code.map(prev_team)
    tm = link_transfermarkt(keys).set_index("player_code")
    df = df.join(tm)
    df["role"] = assign_roles(df, df.position, df.minutes, df.sub_position)
    df["role_source"] = np.where(df.sub_position.map(TM_ROLE).notna(), "listed", "clustered")
    df = rate_players(df)
    # The same rating built from one season at a time, for the season switch on the dashboard.
    for season, tag, pool_min in ((CURRENT_SEASON, "cur", 180), (PREVIOUS_SEASON, "prev", 450)):
        tot1 = _season_totals(season)
        one = tot1[["minutes"]].join(per90_profile(tot1)).join(df[["position", "role"]], how="inner")
        one = rate_players(one, min_pool_minutes=pool_min)
        df[f"rating_{tag}"] = one.rating
        df[f"role_percentile_{tag}"] = one.role_percentile

    snap = load.playerstats_latest(CURRENT_SEASON)[["player_id", "now_cost", "status", "news", "selected_by_percent"]]
    df = df.reset_index().merge(snap, on="player_id", how="left")
    df["price_m"] = df.now_cost  # FPL Core already stores price in £m
    # FPL keeps departed players listed as unavailable with a "Has joined ..." note.
    df["left_league"] = (df.status == "u") & df.news.fillna("").str.contains("joined|left the club", case=False)
    df.loc[df.left_league, "team_code"] = np.nan
    df["minutes_raw_total"] = df.minutes_current + df.minutes_previous
    # Headline counts (FPL season totals), for this season and the two seasons together.
    for season, tag in ((CURRENT_SEASON, "cur"), (PREVIOUS_SEASON, "prev")):
        ps = load.playerstats_latest(season)[["player_id", "goals_scored", "assists", "clean_sheets", "starts"]]
        ps = ps.merge(load.players(season)[["player_id", "player_code"]], on="player_id").drop(columns="player_id")
        ps = ps.groupby("player_code").sum()
        for c in ("goals_scored", "assists", "clean_sheets", "starts"):
            df[f"{c}_{tag}"] = df.player_code.map(ps[c]).fillna(0)
    for c in ("goals_scored", "assists", "clean_sheets", "starts"):
        df[f"{c}_2s"] = df[f"{c}_cur"] + df[f"{c}_prev"]
    df["team"] = df.team_code.map(team_names)
    df["team_code_previous"] = df.player_code.map(prev_team)
    today = pd.Timestamp(AS_OF_DATE)
    df["age"] = (today - df.date_of_birth).dt.days / 365.25
    df["contract_years_left"] = (df.contract_until - today).dt.days / 365.25
    # Stats shown to people (scouting reports, comparison bars): plain per-90 numbers over both
    # seasons, with no season weighting and no possession adjustment, so they can be checked
    # against any stats site. The rating itself still uses the weighted, adjusted numbers.
    raw = _season_totals(CURRENT_SEASON, adjust=False).add(_season_totals(PREVIOUS_SEASON, adjust=False), fill_value=0)
    praw = per90_profile(raw)
    for s in sorted({s for v in ROLE_KEY_STATS.values() for s in v}):
        df[f"show_{s}"] = df.player_code.map(praw[s])
    return df
