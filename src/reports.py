"""Scouting reports: an LLM writes them when OPENAI_API_KEY is set, otherwise a
template writes them straight from the numbers. Both use the same facts.
"""
import json
import os

from .config import ROLE_NAMES
from .player_ratings import ROLE_KEY_STATS

STAT_LABELS = {
    "goals_prevented_p90": "goals prevented per 90", "save_pct": "save %",
    "sweeper_actions_p90": "sweeper actions per 90", "pass_accuracy": "pass accuracy %",
    "aerial_duels_won_p90": "aerials won per 90", "tackles_int_p90": "tackles + interceptions per 90",
    "duel_win_pct": "duel win %", "accurate_long_balls_p90": "accurate long balls per 90",
    "chances_created_p90": "chances created per 90", "accurate_crosses_p90": "accurate crosses per 90",
    "final_third_passes_p90": "final-third passes per 90", "xa_p90": "xA per 90", "xg_p90": "xG per 90",
    "tackles_won_p90": "tackles won per 90", "interceptions_p90": "interceptions per 90",
    "recoveries_p90": "recoveries per 90", "xg_xa_p90": "xG + xA per 90",
    "successful_dribbles_p90": "successful dribbles per 90",
    "touches_opposition_box_p90": "touches in the box per 90", "goals_p90": "goals per 90",
    "shots_on_target_p90": "shots on target per 90", "finishing_p90": "goals minus xG per 90",
}

SYSTEM_PROMPT = (
    "You are a Premier League recruitment analyst. Write a concise scouting report "
    "(90-130 words) explaining why the target fits the buying club. Use only the numbers "
    "provided. Lead with the squad problem the player solves, compare him directly with "
    "the player he would replace, cover age, market value against the club's value ceiling and "
    "contract length when given, mention one risk, and end with a one-line verdict. "
    "No bullet points, no hype words."
)


def facts(club: dict, need: dict, target: dict, incumbent: dict | None) -> dict:
    """The structured facts both report paths work from."""
    role = need["role"]
    stats = {}
    for s in ROLE_KEY_STATS[role]:
        stats[STAT_LABELS[s]] = {
            "target": float(f"{target.get(s, float('nan')):.2f}"),
            "incumbent": float(f"{incumbent.get(s, float('nan')):.2f}") if incumbent else None,
        }
    return {
        "buying_club": club["name"], "league_position": club["position"],
        "role": ROLE_NAMES[role], "need_rank": need["need_rank"],
        "role_strength_league_rank": need["league_rank"],
        "team_signal": need["team_signal_text"],
        "target": target["web_name"], "target_club": target["team"],
        "target_rating": round(target["rating"], 1), "target_role_percentile": round(target["role_percentile"]),
        "minutes_sample": int(target["minutes_raw_total"]), "minutes_this_season": int(target["minutes_current"]),
        "fpl_price_m": target["price_m"],
        "age": None if target.get("age") != target.get("age") else int(target["age"]),
        "market_value_eur_m": None if target.get("market_value_m") != target.get("market_value_m") else target["market_value_m"],
        "contract_years_left": None if target.get("contract_years_left") != target.get("contract_years_left")
        else round(target["contract_years_left"], 1),
        "club_value_ceiling_eur_m": None if not target.get("value_cap_m") else round(target["value_cap_m"]),
        "replaces": need.get("weakest_starter") if isinstance(need.get("weakest_starter"), str) else None, "replaces_rating": round(need["weakest_starter_rating"], 1),
        "style_fit": round(target["style_fit"] * 100), "stats": stats,
    }


def _template(f: dict) -> str:
    # Only differences of 10% or more are worth a sentence.
    def gap(v):
        t, i = v["target"], v["incumbent"]
        if i is None or t != t or i != i:
            return 0
        return (t - i) / max(abs(i), 1e-6)
    better = [k for k, v in f["stats"].items() if gap(v) >= 0.10]
    worse = [k for k, v in f["stats"].items() if gap(v) <= -0.10]
    r = f["role_strength_league_rank"]
    place = (("the strongest" if r == 1 else f"the {r}{'st' if r == 1 else 'nd' if r == 2 else 'rd' if r == 3 else 'th'} strongest")
             if r <= 10 else ("the weakest" if r == 20 else f"the {21 - r}{'nd' if 21 - r == 2 else 'rd' if 21 - r == 3 else 'th'} weakest"))
    lead = (f"{f['buying_club']}'s {f['role'].lower()} options are {place} of the 20 clubs, "
            f"making it their number {f['need_rank']} priority.")
    if f["replaces"]:
        comp = (f"{f['target']} rates {f['target_rating']} against {f['replaces_rating']} for {f['replaces']}, "
                f"the weakest current starter in the role")
    else:
        comp = (f"{f['target']} rates {f['target_rating']}. {f['buying_club']} have no regular in the role this season, "
                f"so he would fill an empty slot")
    edge = ""
    if better:
        top = better[:3]
        edge = " and is ahead on " + ", ".join(
            f"{k} ({f['stats'][k]['target']:.2f} vs {f['stats'][k]['incumbent']:.2f})" for k in top)
    risk = (f"The risk: he trails on {worse[0]}." if worse else
            f"The risk: the rating rests on {f['minutes_sample']} league minutes, "
            f"{f['minutes_this_season']} of them this season.")
    profile = ""
    if f.get("age") is not None and f.get("market_value_eur_m") is not None:
        profile = (f" At {f['age']} and valued at €{f['market_value_eur_m']:.0f}m"
                   + (f" against a ceiling of about €{f['club_value_ceiling_eur_m']}m for {f['buying_club']}"
                      if f.get("club_value_ceiling_eur_m") else "")
                   + (f", with {f['contract_years_left']} years left on his contract" if f.get("contract_years_left") is not None else "")
                   + ", the deal fits the club's spending pattern.")
    pct = f["target_role_percentile"]
    if pct >= 90 and f["style_fit"] >= 50:
        verdict = "Verdict: priority target."
    elif pct >= 75:
        verdict = "Verdict: strong option to shortlist."
    else:
        verdict = ("Verdict: a clear step up on " + (f["replaces"] or "the current options")
                   + ", but a squad-level signing rather than a long-term starter.")
    standing = (f"He sits in the top {max(1, 100 - pct)}% of {f['role'].lower()}s in the league"
                if pct >= 75 else f"He rates above {pct}% of the league's {f['role'].lower()}s")
    return (f"{lead} {comp}{edge}. {standing} over the two seasons, and his profile is a "
            f"{f['style_fit']}% style match for how {f['buying_club']} use the role.{profile} {risk} {verdict}")


def write_report(f: dict) -> tuple[str, str]:
    """Returns (report, source) where source is 'openai' or 'template'."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": json.dumps(f)}],
            )
            return resp.choices[0].message.content.strip(), "openai"
        except Exception as exc:  # fall back rather than fail the whole run
            print(f"OpenAI call failed for {f['target']}: {exc}")
    return _template(f), "template"
