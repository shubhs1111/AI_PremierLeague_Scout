"""Where each squad is weakest, judged against the other 19 clubs."""
import numpy as np
import pandas as pd

from .config import ROLE_SLOTS

# Which team-level ranks put pressure on each role (rank 20 = worst in league).
ROLE_TEAM_SIGNALS = {
    "GK": ["rank_shot_stopping_delta"],
    "CB": ["rank_xg_against_pg", "rank_big_chances_against_pg", "rank_set_piece_xg_against"],
    "FB": ["rank_box_touches_against_pg", "rank_big_chances_for_pg"],
    "DM": ["rank_shots_against_pg", "rank_xg_against_pg", "rank_tackles_won_pg"],
    "CM": ["rank_pass_accuracy", "rank_big_chances_for_pg", "rank_xg_diff_pg"],
    "AM/W": ["rank_big_chances_for_pg", "rank_box_touches_pg", "rank_xg_for_pg"],
    "ST": ["rank_xg_for_pg", "rank_finishing_delta"],
}

# Which roles can plausibly stand in for each role.
COVER_ROLES = {"GK": [], "CB": ["FB", "DM"], "FB": ["CB", "AM/W"], "DM": ["CM", "CB"],
               "CM": ["DM", "AM/W"], "AM/W": ["CM", "ST", "FB"], "ST": ["AM/W"]}

# Minutes this season before a player counts as a first choice in his role.
MIN_REGULAR_MINUTES = 135

NEED_WEIGHTS = {"role_gap": 0.50, "team_pressure": 0.20, "depth_gap": 0.15, "age_risk": 0.15}


def club_slots(squad: pd.DataFrame, games: int) -> dict:
    """How many starters each role gets, read from how the club has actually lined up this season.

    Minutes played by each listed role, divided by 90 x games, give the average number
    of players the club fields there. Back three or back four, two or three central
    midfielders, one or two strikers; attacking midfield and wing take what is left of 10 outfielders.
    """
    full = 90 * max(games, 1)
    share = squad.groupby("role").minutes_current.sum() / full
    if share.sum() == 0:  # pre-season
        return dict(ROLE_SLOTS)
    cb = 3 if share.get("CB", 0) >= 2.5 else 2
    st = 2 if share.get("ST", 0) >= 1.6 else 1
    central = 3 if share.get("DM", 0) + share.get("CM", 0) >= 2.5 else 2
    amw = max(1, 10 - cb - 2 - central - st)
    return {"GK": 1, "CB": cb, "FB": 2, "DM": 1, "CM": central - 1, "AM/W": amw, "ST": st}


def _pick_pivot(central: pd.DataFrame, pools: dict) -> pd.Index:
    """The holding midfielder: a regular listed as DM if there is one, otherwise the
    regular central midfielder whose game leans most toward the holding role."""
    listed = central[central.role == "DM"]
    if len(listed):
        return listed.sort_values("minutes_current", ascending=False).index[:1]
    # How much better a player scores on the holding-role weights than the central-role weights.
    # Raw scores, not percentiles, so a stronger all-round player is not pushed into the pivot.
    lean = central.rating_DM - central.rating_CM
    return lean.sort_values(ascending=False).index[:1]


def first_choice_xi(squad: pd.DataFrame, slots: dict, pools: dict) -> dict:
    """The players who hold each role this season, by minutes. Each player fills one role only.

    Specialists with real minutes go first. Holding and central midfield are filled
    together from the club's regular central midfielders. Roles still short are then
    filled by unused players from neighbouring roles, judged by that role's rating.
    """
    picks, used = {}, set()
    preseason = not (squad.minutes_current > 0).any()
    regular = squad if preseason else squad[squad.minutes_current >= MIN_REGULAR_MINUTES]
    for role, n in slots.items():
        if role in ("DM", "CM"):
            continue
        main = regular[regular.role == role].sort_values(["minutes_current", "rating"], ascending=False).head(n)
        picks[role] = main.assign(role_rating=main[f"rating_{role}"], covering=False, pivot=False)
        used.update(main.index)
    central = regular[regular.role.isin(["DM", "CM"])].sort_values("minutes_current", ascending=False)
    central = central.head(slots["DM"] + slots["CM"])
    dm_idx = _pick_pivot(central, pools) if len(central) else central.index
    dm = central.loc[dm_idx]
    cm = central.drop(index=dm_idx).head(slots["CM"])
    # A listed central midfielder playing the holding role is the club's set-up, not a stop-gap.
    picks["DM"] = dm.assign(role_rating=dm.rating_DM, covering=False, pivot=dm.role != "DM")
    picks["CM"] = cm.assign(role_rating=cm.rating_CM, covering=False, pivot=cm.role != "CM")
    used.update(dm.index); used.update(cm.index)
    for role, n in slots.items():
        short = n - len(picks[role])
        if short <= 0:
            continue
        col = f"rating_{role}"
        cover = squad[~squad.index.isin(used) & squad[col].notna() & squad.role.isin(COVER_ROLES[role])
                      & (squad.minutes_current >= MIN_REGULAR_MINUTES)]
        cover = cover.sort_values(["minutes_current", col], ascending=False).head(short)
        picks[role] = pd.concat([picks[role], cover.assign(role_rating=cover[col], covering=True, pivot=False)])
        used.update(cover.index)
    return picks


def role_strength(players: pd.DataFrame, games: dict) -> pd.DataFrame:
    rows = []
    active = players[players.team_code.notna()]
    # An empty slot counts as a fringe player: the 10th percentile of the role.
    replacement = active.groupby("role").rating.quantile(0.10)
    # The league's regulars in each role, to place any starter (including cover) on a percentile.
    pool_ratings = {r: players[(players.role == r) & (players.minutes >= 450)].rating for r in ROLE_SLOTS}
    for team, squad in active.groupby("team_code"):
        club = club_slots(squad, games.get(int(team), 1))
        xi = first_choice_xi(squad, club, pool_ratings)
        for role, slots in club.items():
            fc = xi[role]
            ratings = list(fc.role_rating) + [replacement[role]] * (slots - len(fc))
            role_players = squad[(squad.role == role) & (squad.minutes_raw_total >= 450)]
            rows.append({
                "team_code": int(team), "role": role, "slots": int(slots),
                # Half the average, half the weakest link: one poor starter is a real gap.
                "strength": float(0.5 * np.mean(ratings) + 0.5 * min(ratings)),
                "weakest_starter_rating": float(min(ratings)),
                "weakest_starter": fc.sort_values("role_rating").web_name.iloc[0] if len(fc) else None,
                "covering": [n for n, c in zip(fc.web_name, fc.covering) if c],
                "empty_slots": int(slots - len(fc)),
                "starters": list(fc.web_name),
                "starters_age": float(fc.age.mean()) if fc.age.notna().any() else np.nan,
                # 0 for starters 29 or younger (32 for goalkeepers), rising to 100 four years later.
                "age_risk": float(((fc.age - (32 if role == "GK" else 29)) / 4).clip(0, 1).mean() * 100)
                            if fc.age.notna().any() else 0.0,
                "starter_detail": [{"name": r.web_name, "age": None if pd.isna(r.age) else round(float(r.age), 1),
                                    "pct": round(float(min((pool_ratings[role] <= r.role_rating).mean() * 100, 100))),
                                    "covering": bool(r.covering), "pivot": bool(r.pivot), "own_role": r.role}
                                   for r in fc.itertuples()],
                # Injured or suspended players are not cover.
                "backups_above_median": int((squad[(squad.role == role) & ~squad.index.isin(fc.index)
                                                   & ~squad.status.isin(["i", "s", "u", "n"])].rating
                                             >= pool_ratings[role].median()).sum()),
                "squad_count": int(len(role_players)),
                "squad_count_rated_above_median": int((role_players.rating >= active[active.role == role].rating.median()).sum()),
            })
    return pd.DataFrame(rows)


PRIORITY_LABELS = {3: "Critical", 2: "High", 1: "Moderate", 0: "Low"}
WEAK_LINK_PCT = 35      # a starter below this role percentile is a weak link
AGEING_STARTER = 31     # a starter this old or older
AGEING_GK = 34          # goalkeepers peak later, so their age flag starts later
SIGNING_FLAGS = ("empty", "covering", "weak_link", "league")  # problems a new starter fixes


def ageing_age(role: str) -> int:
    return AGEING_GK if role == "GK" else AGEING_STARTER
LEAGUE_BOTTOM_RANK = 15  # role group ranked this low or lower among 20 clubs


def _ord(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def role_flags(r) -> list:
    """Concrete, checkable problems with a role. Each flag is one plain sentence."""
    from .config import ROLE_NAMES
    name = ROLE_NAMES[r.role].lower()
    flags = []
    if r.empty_slots:
        flags.append({"key": "empty", "text": f"No regular {name} this season: no specialist has played 135+ minutes there."})
    if r.covering:
        flags.append({"key": "covering", "text": f"{', '.join(r.covering)} {'is' if len(r.covering) == 1 else 'are'} "
                                                  f"filling in out of position."})
    for d in r.starter_detail:
        if d["pct"] < WEAK_LINK_PCT:
            flags.append({"key": "weak_link", "text": f"{d['name']} rates in the bottom third of Premier League "
                                                       f"{name}s ({_ord(d['pct'])} percentile)."})
    old = [d for d in r.starter_detail if d["age"] and d["age"] >= ageing_age(r.role)]
    if old:
        flags.append({"key": "ageing", "text": "Ageing: " + ", ".join(f"{d['name']} is {int(d['age'])}" for d in old)
                      + (f"; the starters average {r.starters_age:.0f}." if r.starters_age == r.starters_age else ".")})
    if r.role != "GK" and r.backups_above_median < min(r.slots, 2):
        flags.append({"key": "thin", "text": f"Thin cover: {r.backups_above_median or 'no'} fit above-average "
                                              f"backup{'' if r.backups_above_median == 1 else 's'} behind the starters."})
    if r.league_rank >= LEAGUE_BOTTOM_RANK:
        flags.append({"key": "league", "text": f"Only {_ord(r.league_rank)} of 20 clubs for this position."})
    return flags


def needs_signing(priority: int, flags: list) -> bool:
    """A position gets transfer targets only when a signing fixes something: it is High or
    Critical, or a Moderate whose one problem is the level of a starter (not age or depth alone)."""
    return priority >= 2 or any(f["key"] in SIGNING_FLAGS for f in flags)


def priority_tier(flags: list) -> int:
    keys = [f["key"] for f in flags]
    structural = any(k in ("empty", "covering") for k in keys)
    if structural and len(keys) >= 2:
        return 3
    if structural or len(keys) >= 2:
        return 2
    return 1 if keys else 0


def squad_needs(players: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    rs = role_strength(players, table.set_index("team_code").played.astype(int).to_dict())
    # Percentile of each club's role strength among all 20 clubs; gap = 100 - percentile.
    rs["league_percentile"] = rs.groupby("role").strength.rank(pct=True) * 100
    rs["role_gap"] = 100 - rs.league_percentile

    t = table.set_index("team_code")
    pressure = []
    for _, r in rs.iterrows():
        ranks = [t.at[r.team_code, s] for s in ROLE_TEAM_SIGNALS[r.role]]
        pressure.append((np.mean(ranks) - 1) / 19 * 100)
    rs["team_pressure"] = pressure

    slots = rs.slots
    rs["depth_gap"] = (1 - (rs.squad_count_rated_above_median / (slots * 1.5)).clip(upper=1)) * 100
    # Playing someone out of position is itself a depth problem.
    rs["depth_gap"] = np.maximum(rs.depth_gap, (rs.covering.map(len) + rs.empty_slots) / slots * 100)

    rs["need_score"] = (NEED_WEIGHTS["role_gap"] * rs.role_gap + NEED_WEIGHTS["team_pressure"] * rs.team_pressure
                        + NEED_WEIGHTS["depth_gap"] * rs.depth_gap + NEED_WEIGHTS["age_risk"] * rs.age_risk)
    rs["league_rank"] = rs.groupby("role").strength.rank(ascending=False, method="min").astype(int)
    rs["flags"] = [role_flags(r) for r in rs.itertuples()]
    rs["priority"] = rs["flags"].map(priority_tier)
    rs["needs_signing"] = [needs_signing(p, f) for p, f in zip(rs.priority, rs["flags"])]
    rs["priority_label"] = rs.priority.map(PRIORITY_LABELS)
    # Priority tier first (what is concretely wrong), need score breaks ties.
    rs = rs.sort_values(["team_code", "priority", "need_score"], ascending=[True, False, False])
    rs["need_rank"] = rs.groupby("team_code").cumcount() + 1
    return rs.sort_values(["team_code", "need_rank"])
