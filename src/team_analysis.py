"""Team-level performance: results, expected goals and style, ranked across the league."""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import load

# Per-match team stats, named without the home_/away_ prefix.
TEAM_STATS = [
    "expected_goals_xg", "non_penalty_xg", "total_shots", "shots_on_target", "big_chances",
    "possession", "accurate_passes", "passes", "opposition_half", "touches_in_opposition_box",
    "accurate_crosses", "accurate_long_balls", "tackles_won", "interceptions", "duels_won",
    "aerial_duels_won", "xg_on_target_xgot", "keeper_saves", "corners", "xg_set_play",
]


def _expected_points(xg_for: float, xg_against: float, max_goals: int = 10) -> float:
    """Expected league points from a match's xG, treating goals as Poisson."""
    g = np.arange(max_goals + 1)
    p_for, p_against = poisson.pmf(g, xg_for), poisson.pmf(g, xg_against)
    grid = np.outer(p_for, p_against)
    return 3 * np.tril(grid, -1).sum() + np.trace(grid)


def team_match_rows(season: str) -> pd.DataFrame:
    """One row per team per match, with that team's stats and its opponent's."""
    m = load.matches(season)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        df = pd.DataFrame({
            "match_id": m["match_id"], "gameweek": m["gameweek"], "team_code": m[f"{side}_team"],
            "opponent_code": m[f"{opp}_team"], "is_home": side == "home",
            "goals_for": m[f"{side}_score"], "goals_against": m[f"{opp}_score"],
        })
        for s in TEAM_STATS:
            df[s] = m.get(f"{side}_{s}")
            df[f"opp_{s}"] = m.get(f"{opp}_{s}")
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["points"] = np.select(
        [out.goals_for > out.goals_against, out.goals_for == out.goals_against], [3, 1], 0)
    out["xpts"] = [_expected_points(a, b) for a, b in zip(out.expected_goals_xg, out.opp_expected_goals_xg)]
    return out


def team_table(season: str) -> pd.DataFrame:
    """Season summary per team, per-game averages and league ranks."""
    r = team_match_rows(season)
    g = r.groupby("team_code")
    t = pd.DataFrame({
        "played": g.size(),
        "won": g.apply(lambda d: (d.points == 3).sum()),
        "drawn": g.apply(lambda d: (d.points == 1).sum()),
        "lost": g.apply(lambda d: (d.points == 0).sum()),
        "points": g.points.sum(),
        "xpts": g.xpts.sum(),
        "goals_for": g.goals_for.sum(),
        "goals_against": g.goals_against.sum(),
        "xg_for": g.expected_goals_xg.sum(),
        "xg_against": g.opp_expected_goals_xg.sum(),
        "npxg_for": g.non_penalty_xg.sum(),
        "npxg_against": g.opp_non_penalty_xg.sum(),
        "shots_for_pg": g.total_shots.mean(),
        "shots_against_pg": g.opp_total_shots.mean(),
        "big_chances_for_pg": g.big_chances.mean(),
        "big_chances_against_pg": g.opp_big_chances.mean(),
        "box_touches_pg": g.touches_in_opposition_box.mean(),
        "box_touches_against_pg": g.opp_touches_in_opposition_box.mean(),
        "possession": g.possession.mean(),
        "pass_accuracy": g.accurate_passes.sum() / g.passes.sum() * 100,
        "passes_opp_half_share": g.opposition_half.sum() / g.passes.sum() * 100,
        "crosses_pg": g.accurate_crosses.mean(),
        "long_balls_pg": g.accurate_long_balls.mean(),
        "tackles_won_pg": g.tackles_won.mean(),
        "interceptions_pg": g.interceptions.mean(),
        "aerials_won_pg": g.aerial_duels_won.mean(),
        "xgot_faced": g.opp_xg_on_target_xgot.sum(),
        "set_piece_xg_for": g.xg_set_play.sum(),
        "set_piece_xg_against": g.opp_xg_set_play.sum(),
    })
    t["goal_diff"] = t.goals_for - t.goals_against
    t["xg_diff_pg"] = (t.xg_for - t.xg_against) / t.played
    t["xg_for_pg"] = t.xg_for / t.played
    t["xg_against_pg"] = t.xg_against / t.played
    t["points_pg"] = t.points / t.played
    # Positive finishing = scoring more than the chances were worth.
    t["finishing_delta"] = t.goals_for - t.xg_for
    # Positive shot stopping = conceding fewer than the shots on target were worth.
    t["shot_stopping_delta"] = t.xgot_faced - t.goals_against
    t["luck_delta_points"] = t.points - t.xpts

    names = load.teams(season).set_index("code")[["name", "short_name"]]
    t = t.join(names).reset_index()
    t = t.sort_values(["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
    t["position"] = np.arange(1, len(t) + 1)

    # Rank 1 = best in the league for every metric.
    higher_is_better = [
        "xg_for_pg", "npxg_for", "shots_for_pg", "big_chances_for_pg", "box_touches_pg", "possession",
        "pass_accuracy", "xg_diff_pg", "finishing_delta", "shot_stopping_delta", "xpts",
        "set_piece_xg_for", "tackles_won_pg", "interceptions_pg", "aerials_won_pg",
    ]
    lower_is_better = [
        "xg_against_pg", "npxg_against", "shots_against_pg", "big_chances_against_pg",
        "box_touches_against_pg", "set_piece_xg_against",
    ]
    for c in higher_is_better:
        t[f"rank_{c}"] = t[c].rank(ascending=False, method="min").astype(int)
    for c in lower_is_better:
        t[f"rank_{c}"] = t[c].rank(ascending=True, method="min").astype(int)
    return t


def gameweek_trend(season: str) -> pd.DataFrame:
    r = team_match_rows(season)
    return r[["team_code", "gameweek", "opponent_code", "is_home", "goals_for", "goals_against",
              "expected_goals_xg", "opp_expected_goals_xg", "points", "xpts"]].sort_values(["team_code", "gameweek"])
