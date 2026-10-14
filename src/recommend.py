"""Rank transfer targets for a club's weakest roles."""
import numpy as np
import pandas as pd

from .config import TARGET_MIN_CURRENT_MINUTES, TARGET_MIN_MINUTES
from .player_ratings import ROLE_WEIGHTS

SCORE_WEIGHTS = {"upgrade": 0.30, "quality": 0.15, "style_fit": 0.15, "value": 0.10,
                 "age": 0.15, "reliability": 0.10, "realism": 0.05}

# Age score: full marks through the prime years, tapering for the youngest and oldest.
AGE_POINTS = ([17, 21, 28, 30, 32, 34], [0.7, 1.0, 1.0, 0.75, 0.35, 0.1])


def club_strength(table_cur: pd.DataFrame, table_prev: pd.DataFrame) -> pd.Series:
    """Points per game blended over both seasons; promoted clubs use this season only."""
    cur = table_cur.set_index("team_code").points_pg
    prev = table_prev.set_index("team_code").points_pg.reindex(cur.index)
    return (0.5 * cur + 0.5 * prev).fillna(cur)


def price_cap(squad: pd.DataFrame) -> float:
    """The club's own price structure: its three most expensive players, plus a little room."""
    top = squad.price_m.dropna().nlargest(3)
    return float(top.mean() + 0.5) if len(top) else 6.0


def value_cap(squad: pd.DataFrame) -> float | None:
    """Most a club pays for one player, judged by the market value it already holds:
    the average of its three most valuable players, plus 25%."""
    top = squad.market_value_m.dropna().nlargest(3)
    return float(top.mean() * 1.25) if len(top) == 3 else None


def age_score(age: pd.Series) -> pd.Series:
    return pd.Series(np.interp(age.fillna(27), *AGE_POINTS), index=age.index)


def _style_vectors(pool: pd.DataFrame, role: str) -> pd.DataFrame:
    metrics = list(ROLE_WEIGHTS[role])
    X = pool[metrics].astype(float)
    return (X - X.mean()) / X.std(ddof=0).replace(0, 1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def targets_for_role(players: pd.DataFrame, needs: pd.DataFrame, strength: pd.Series,
                     team_code: int, role: str, top_n: int = 5) -> pd.DataFrame:
    active = players[players.team_code.notna() & ~players.left_league.fillna(False)]
    squad = active[active.team_code == team_code]
    need = needs[(needs.team_code == team_code) & (needs.role == role)].iloc[0]
    bar = need.weakest_starter_rating

    role_pool = active[(active.role == role) & (active.minutes >= TARGET_MIN_MINUTES)]
    z = _style_vectors(role_pool, role)
    incumbents = squad[(squad.role == role) & squad.web_name.isin(need.starters)]
    inc_idx = [i for i in incumbents.index if i in z.index]
    club_profile = z.loc[inc_idx].mean().values if inc_idx else z.mean().values

    c = role_pool[(role_pool.team_code != team_code)
                  & (role_pool.minutes_current >= TARGET_MIN_CURRENT_MINUTES)
                  & (role_pool.status != "u")].copy()
    c["upgrade"] = c.rating - bar
    # A real upgrade, and a player good enough to start: at least the 60th percentile in the role.
    c = c[(c.upgrade >= 3) & (c.role_percentile >= 60)]
    cap = price_cap(squad)
    mv_cap = value_cap(squad)
    # Market value decides affordability where Transfermarkt has it; FPL price otherwise.
    affordable = c.market_value_m <= mv_cap if mv_cap else c.price_m <= cap
    c = c[affordable.where(c.market_value_m.notna(), c.price_m <= cap)]
    if c.empty:
        return c

    buyer = strength.get(team_code, strength.median())
    c["seller_strength"] = c.team_code.map(strength)
    # A club rarely prises a regular away from a clearly stronger club.
    c["realism"] = np.clip(1 - (c.seller_strength - buyer - 0.25).clip(lower=0) / 0.75, 0, 1)
    # A contract running down makes a sale more likely.
    c["realism"] = (c.realism + 0.15 * (c.contract_years_left <= 1.5)).clip(upper=1)
    c = c[c.realism >= 0.3]
    if c.empty:
        return c

    c["style_fit"] = [(_cosine(z.loc[i].values, club_profile) + 1) / 2 for i in c.index]
    # Rating gained per €m of market value (square root, so stars are not ruled out by price alone).
    c["value_ratio"] = c.upgrade / np.sqrt(c.market_value_m.clip(lower=1))
    c["age_score"] = age_score(c.age)
    parts = pd.DataFrame({
        "upgrade": (c.upgrade / 20).clip(upper=1),
        "quality": c.role_percentile / 100,
        "style_fit": c.style_fit,
        "value": c.value_ratio.rank(pct=True).fillna(0.5),
        "age": c.age_score,
        "reliability": (c.minutes / 2500).clip(upper=1),
        "realism": c.realism,
    })
    base = sum(parts[k] * w for k, w in SCORE_WEIGHTS.items()) * 100
    # Realism also scales the whole score: a great fit you cannot buy is not a target.
    c["target_score"] = base * (0.6 + 0.4 * c.realism)
    # Where the role's problem is an ageing group, an older signing does not fix it.
    if need.age_risk > 0:
        c["target_score"] *= 1 - (need.age_risk / 100) * (1 - c.age_score)
    c["replaces"] = need.weakest_starter if isinstance(need.weakest_starter, str) else None
    c["replaces_rating"] = bar
    c["price_cap"] = cap
    c["value_cap_m"] = mv_cap
    c["need_role"] = role
    return c.sort_values("target_score", ascending=False).head(top_n)


MAX_CLUBS_PER_PLAYER = 2


def recommend_all(players, needs, strength, roles_per_club: int = 7, top_n: int = 3) -> pd.DataFrame:
    """Targets for every position that needs a signing (the number varies by club).

    One player can join only one club, so the board does not offer the same few bargains
    to everyone. Candidates are handed out like a draft: in each round every position that
    needs a signing takes its best remaining candidate, the most urgent needs picking first,
    and no player is shown to more than MAX_CLUBS_PER_PLAYER clubs."""
    lists = []
    for team_code in sorted(needs.team_code.unique()):
        club_needs = needs[needs.team_code == team_code].sort_values("need_rank")
        if "needs_signing" in club_needs:
            club_needs = club_needs[club_needs.needs_signing]
        for r in club_needs.head(roles_per_club).itertuples():
            t = targets_for_role(players, needs, strength, team_code, r.role, top_n=25)
            if not t.empty:
                lists.append(((-r.priority, -r.need_score), t.assign(buyer_code=team_code)))
    if not lists:
        return pd.DataFrame()
    lists.sort(key=lambda x: x[0])
    taken, picked = {}, [[] for _ in lists]
    for _ in range(top_n):
        for k, (_, t) in enumerate(lists):
            for i, row in t.iterrows():
                if i in picked[k] or taken.get(row.player_code, 0) >= MAX_CLUBS_PER_PLAYER:
                    continue
                picked[k].append(i)
                taken[row.player_code] = taken.get(row.player_code, 0) + 1
                break
    out = [t.loc[p] for (_, t), p in zip(lists, picked) if p]
    res = pd.concat(out, ignore_index=True)
    # Positions left empty only because every candidate went to more urgent needs elsewhere.
    res.attrs["contested"] = {(int(t.buyer_code.iloc[0]), t.need_role.iloc[0]): list(t.web_name)
                              for (_, t), p in zip(lists, picked) if not p}
    return res
