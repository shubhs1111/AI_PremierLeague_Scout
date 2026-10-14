"""Run the whole scouting pipeline and write the dashboard data.

    python -m src.fetch_data      # refresh raw data (optional)
    python run_pipeline.py        # ratings, needs, targets, reports, dashboard
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from src import external, insights, load, player_ratings, recommend, reports, squad_needs, team_analysis
from src.config import CURRENT_SEASON, OUTPUTS, PREVIOUS_SEASON, PROCESSED, ROLE_NAMES
from src.player_ratings import ROLE_KEY_STATS
from src.reports import STAT_LABELS

SIGNAL_TEXT = {
    "rank_shot_stopping_delta": ("shot stopping (goals conceded vs xG on target faced)", "shot_stopping_delta", "{:+.1f} goals"),
    "rank_xg_against_pg": ("chances conceded, by xG per game", "xg_against_pg", "{:.2f}"),
    "rank_big_chances_against_pg": ("big chances conceded per game", "big_chances_against_pg", "{:.1f}"),
    "rank_box_touches_against_pg": ("opposition touches in their box per game", "box_touches_against_pg", "{:.1f}"),
    "rank_shots_against_pg": ("shots conceded per game", "shots_against_pg", "{:.1f}"),
    "rank_pass_accuracy": ("pass accuracy", "pass_accuracy", "{:.1f}%"),
    "rank_big_chances_for_pg": ("big chances created per game", "big_chances_for_pg", "{:.1f}"),
    "rank_xg_for_pg": ("xG created per game", "xg_for_pg", "{:.2f}"),
    "rank_finishing_delta": ("finishing (goals minus xG)", "finishing_delta", "{:+.1f}"),
}


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def clean(v):
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if v is pd.NaT:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def row_dict(r: pd.Series, cols) -> dict:
    return {c: clean(r.get(c)) for c in cols}


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    players = player_ratings.build_player_table()
    players["minutes_raw_total"] = players.minutes_current + players.minutes_previous
    table = team_analysis.team_table(CURRENT_SEASON)
    table_prev = team_analysis.team_table(PREVIOUS_SEASON)
    trend = team_analysis.gameweek_trend(CURRENT_SEASON)
    needs = squad_needs.squad_needs(players, table)
    strength = recommend.club_strength(table, table_prev)
    targets = recommend.recommend_all(players, needs, strength, roles_per_club=7, top_n=3)

    # Independent check: our ratings vs the uploaded 2025/26 match ratings.
    val = external.validation_table(players)
    validation = {
        "n": int(len(val)),
        "spearman": round(float(spearmanr(val.rating, val.ext_rating).statistic), 2),
        "by_role": {r: {"n": int(len(g)), "spearman": round(float(spearmanr(g.rating, g.ext_rating).statistic), 2)}
                    for r, g in val.groupby("role")},
        "points": [[clean(a), clean(b), r] for a, b, r in zip(val.rating, val.ext_rating, val.role)],
    }
    ext_rating = val.set_index("player_code").ext_rating
    players["ext_rating_2025_26"] = players.player_code.map(ext_rating)

    history = external.points_history()
    names_cur = load.teams(CURRENT_SEASON).set_index("code").name

    needs["age_rank"] = needs.groupby("role").starters_age.rank(ascending=False, method="min")
    players.to_csv(PROCESSED / "player_ratings.csv", index=False)
    table.to_csv(PROCESSED / "team_table.csv", index=False)
    needs.to_csv(PROCESSED / "squad_needs.csv", index=False)

    t_idx = table.set_index("team_code")
    prev_idx = table_prev.set_index("team_code")
    gw_max = int(trend.gameweek.max())
    stat_cols = sorted({s for v in ROLE_KEY_STATS.values() for s in v})

    def shown(row: dict) -> dict:
        """Swap in the plain per-90 numbers (both seasons, unweighted, unadjusted) for display."""
        return {**row, **{s: row.get(f"show_{s}") for s in stat_cols}}
    report_rows = []

    clubs = []
    for _, club in table.iterrows():
        code = int(club.team_code)
        club_needs = needs[needs.team_code == code].copy()

        def signal_text(role):
            sig = squad_needs.ROLE_TEAM_SIGNALS[role][0]
            label, col, fmt = SIGNAL_TEXT.get(sig, (sig, None, "{}"))
            rank = int(t_idx.at[code, sig])
            val = fmt.format(t_idx.at[code, col]) if col else ""
            place = "best" if rank <= 10 else "worst"
            n = rank if rank <= 10 else 21 - rank
            which = "the best" if n == 1 else f"{ordinal(n)} {place}"
            if n == 1 and place == "worst":
                which = "the worst"
            return f"Team context: {which} in the league for {label} ({val})."

        club_needs["team_signal_text"] = club_needs.role.map(signal_text)
        contested = targets.attrs.get("contested", {}) if not targets.empty else {}
        club_needs["contested"] = [contested.get((int(code), r), []) for r in club_needs.role]
        squad = players[(players.team_code == code)].copy()
        club_targets = targets[targets.buyer_code == code] if not targets.empty else pd.DataFrame()

        briefs = {}
        plays = {d["name"]: role for role, sd in zip(club_needs.role, club_needs.starter_detail) for d in sd}
        in_xi = set(plays)
        for n in club_needs.to_dict("records"):
            tg = club_targets[club_targets.need_role == n["role"]] if len(club_targets) else club_targets
            top = tg.sort_values("target_score", ascending=False).iloc[0].to_dict() if len(tg) else None
            ar = None if pd.isna(n["age_rank"]) else int(n["age_rank"])
            briefs[n["role"]] = insights.brief(club["name"], n, squad, t_idx.loc[code], top, ar, in_xi)
        club_needs["brief"] = club_needs.role.map(briefs)

        target_out = []
        for _, tg in club_targets.iterrows():
            need = club_needs[club_needs.role == tg.need_role].iloc[0].to_dict()
            inc = squad[squad.web_name == need["weakest_starter"]]
            inc_d = inc.iloc[0].to_dict() if len(inc) else None
            tg_d = shown(tg.to_dict())
            inc_d = shown(inc_d) if inc_d else None
            f = reports.facts(club.to_dict(), need, tg_d, inc_d)
            text, source = reports.write_report(f)
            report_rows.append({"club": club["name"], "target": tg.web_name, "role": tg.need_role,
                                "source": source, "report": text, "facts": f})
            d = row_dict(tg, ["player_code", "web_name", "first_name", "second_name", "team", "role", "need_role",
                              "rating", "role_percentile", "upgrade", "replaces", "replaces_rating", "price_m",
                              "price_cap", "value_cap_m", "market_value_m", "age", "age_score", "contract_until",
                              "contract_years_left", "foot", "height_cm", "tm_url",
                              "style_fit", "realism", "target_score", "minutes_current",
                              "minutes_raw_total", "status", "news", "sub_position", "goals_scored_2s", "assists_2s",
                              "clean_sheets_2s", "starts_2s"] + stat_cols)
            two = lambda v: None if v is None or v != v else float(f"{v:.2f}")
            d.update({s: two(tg_d[s]) for s in stat_cols})
            d["incumbent_stats"] = {s: two(inc_d[s]) for s in stat_cols} if inc_d else None
            d["report"], d["report_source"] = text, source
            target_out.append(d)

        squad_out = [row_dict(r, ["player_code", "web_name", "role", "position", "rating", "role_percentile",
                                  "minutes_current", "minutes_raw_total", "starts_current", "price_m", "market_value_m",
                                  "age", "contract_until", "status", "sub_position", "role_source",
                                  "goals_scored_2s", "assists_2s", "clean_sheets_2s", "starts_2s",
                                  "goals_scored_cur", "assists_cur", "clean_sheets_cur", "starts_cur",
                                  "goals_scored_prev", "assists_prev", "clean_sheets_prev", "starts_prev",
                                  "rating_cur", "rating_prev", "role_percentile_cur", "role_percentile_prev",
                                  "news", "ext_rating_2025_26"] + stat_cols)
                     for _, r in squad.sort_values(["minutes_current", "rating"], ascending=False).iterrows()]
        for row in squad_out:
            row["plays"] = plays.get(row["web_name"])
        prev = prev_idx.loc[code] if code in prev_idx.index else None
        clubs.append({
            "code": code, "name": club["name"], "short": club.short_name,
            "summary": row_dict(club, [c for c in table.columns if c not in ("name", "short_name")]),
            "previous": None if prev is None else {
                "position": clean(prev.position), "points_pg": clean(prev.points_pg),
                "xg_for_pg": clean(prev.xg_for_pg), "xg_against_pg": clean(prev.xg_against_pg)},
            "history": (
                [{"season": r.season, "points_pg": clean(r.points_pg)}
                 for _, r in history[history.team == club["name"]].iterrows()]
                + ([{"season": "2025/26", "points_pg": clean(prev.points_pg)}] if prev is not None else [])
                + [{"season": "2026/27", "points_pg": clean(club.points_pg)}]),
            "squad_value_m": clean(squad.market_value_m.sum()),
            "squad_age": clean(np.average(squad.age.dropna(), weights=squad.loc[squad.age.notna(), "minutes_current"] + 1)),
            "value_cap_m": clean(recommend.value_cap(squad)),
            "trend": [row_dict(r, ["gameweek", "opponent_code", "is_home", "goals_for", "goals_against",
                                   "expected_goals_xg", "opp_expected_goals_xg", "points", "xpts"])
                      for _, r in trend[trend.team_code == code].iterrows()],
            "needs": [row_dict(pd.Series(n), ["role", "need_rank", "need_score", "strength", "league_rank",
                                               "role_gap", "team_pressure", "depth_gap", "age_risk", "starters_age", "empty_slots",
                                               "weakest_starter",
                                               "weakest_starter_rating", "starters", "covering", "team_signal_text",
                                               "flags", "priority", "priority_label", "needs_signing", "contested", "starter_detail", "brief"])
                      for n in club_needs.sort_values("need_rank").to_dict("records")],
            "squad": squad_out,
            "targets": target_out,
        })

    data = {
        "generated": date.today().isoformat(),
        "season": CURRENT_SEASON, "previous_season": PREVIOUS_SEASON, "gameweeks": gw_max,
        "role_names": ROLE_NAMES, "role_key_stats": ROLE_KEY_STATS, "stat_labels": STAT_LABELS,
        "role_weights": {k: v for k, v in player_ratings.ROLE_WEIGHTS.items()},
        "score_weights": recommend.SCORE_WEIGHTS,
        "need_weights": squad_needs.NEED_WEIGHTS,
        "role_averages": {r: {k: clean(players[(players.role == r) & (players.minutes_raw_total >= 900)][f"show_{k}"].mean())
                              for k in ROLE_KEY_STATS[r]} for r in ROLE_KEY_STATS},
        "age_points": recommend.AGE_POINTS,
        "market_value_coverage": round(float(players[players.team_code.notna()].market_value_m.notna().mean()), 3),
        "team_names": {int(k): v for k, v in load.teams(CURRENT_SEASON).set_index("code").name.items()},
        "validation": validation,
        "clubs": clubs,
    }
    (OUTPUTS / "dashboard_data.json").write_text(json.dumps(data, ensure_ascii=False))
    pd.DataFrame(report_rows).to_json(OUTPUTS / "scouting_reports.json", orient="records", indent=1, force_ascii=False)
    print(f"{len(clubs)} clubs, {sum(len(c['targets']) for c in clubs)} targets, data through GW{gw_max}")


if __name__ == "__main__":
    main()
