"""Plain-language briefs for each position, written the way a sporting director
would read them: what is wrong, the evidence, and what to do about it.

Every sentence is built from the pipeline's numbers; nothing is free text.
"""
import pandas as pd

from .config import ROLE_NAMES

# Team measures that bear on each position: (column, sentence, word if good, word if bad, number format).
# Ranks in the team table run 1 = best, so "good" is ranks 1-10.
METRICS = {
    "shots_against_pg": ("concede {v} shots per game", "fewest", "most", "{:.1f}"),
    "xg_against_pg": ("allow {v} xG per game", "least", "most", "{:.2f}"),
    "big_chances_against_pg": ("give up {v} big chances per game", "fewest", "most", "{:.1f}"),
    "box_touches_against_pg": ("let opponents have {v} touches in their box per game", "fewest", "most", "{:.1f}"),
    "set_piece_xg_against": ("have conceded {v} xG from set pieces", "least", "most", "{:.2f}"),
    "aerials_won_pg": ("win {v} aerial duels per game", "most", "fewest", "{:.1f}"),
    "tackles_won_pg": ("win {v} tackles per game", "most", "fewest", "{:.1f}"),
    "interceptions_pg": ("make {v} interceptions per game", "most", "fewest", "{:.1f}"),
    "crosses_pg": ("complete {v} crosses per game", "most", "fewest", "{:.1f}"),
    "big_chances_for_pg": ("create {v} big chances per game", "most", "fewest", "{:.1f}"),
    "box_touches_pg": ("have {v} touches in the opposition box per game", "most", "fewest", "{:.1f}"),
    "xg_for_pg": ("create {v} xG per game", "most", "least", "{:.2f}"),
    "pass_accuracy": ("complete {v}% of their passes", "highest", "lowest", "{:.1f}"),
    "possession": ("average {v}% possession", "highest", "lowest", "{:.0f}"),
    "shots_for_pg": ("take {v} shots per game", "most", "fewest", "{:.1f}"),
}
ROLE_METRICS = {
    "GK": [],
    "CB": ["xg_against_pg", "big_chances_against_pg", "aerials_won_pg", "set_piece_xg_against"],
    "FB": ["box_touches_against_pg", "big_chances_for_pg"],
    "DM": ["shots_against_pg", "big_chances_against_pg", "tackles_won_pg", "interceptions_pg"],
    "CM": ["pass_accuracy", "possession", "big_chances_for_pg"],
    "AM/W": ["big_chances_for_pg", "box_touches_pg", "xg_for_pg"],
    "ST": ["xg_for_pg", "shots_for_pg"],
}


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _place(rank: int, good: str, bad: str) -> str:
    if rank <= 10:
        return f"the {good}" if rank == 1 else f"the {ordinal(rank)} {good}"
    n = 21 - rank
    return f"the {bad}" if n == 1 else f"the {ordinal(n)} {bad}"


def _metric_sentence(club: str, team: pd.Series, col: str) -> tuple[int, str]:
    text, good, bad, fmt = METRICS[col]
    rank = int(team[f"rank_{col}"])
    return rank, f"{club} {text.format(v=fmt.format(team[col]))}, {_place(rank, good, bad)} in the league."


def _special_sentences(club: str, team: pd.Series, role: str) -> list[tuple[int, str]]:
    out = []
    if role == "GK":
        d, rank = team.shot_stopping_delta, int(team.rank_shot_stopping_delta)
        how = f"{abs(d):.1f} fewer goals than" if d >= 0 else f"{abs(d):.1f} more goals than"
        out.append((rank, f"{club} have conceded {how} the quality of shots on target faced would predict, "
                          f"{_place(rank, 'best', 'worst')} shot-stopping record in the league."))
    if role == "ST":
        d, rank = team.finishing_delta, int(team.rank_finishing_delta)
        how = f"{abs(d):.1f} goals more than" if d >= 0 else f"{abs(d):.1f} goals fewer than"
        out.append((rank, f"{club} have scored {how} their chances were worth (xG), "
                          f"{_place(rank, 'best', 'worst')} finishing in the league."))
    return out


def team_evidence(club: str, team: pd.Series, role: str) -> dict:
    """The worst related team measure if any is in the bottom half; otherwise the best one, as reassurance."""
    rows = [_metric_sentence(club, team, c) for c in ROLE_METRICS[role]] + _special_sentences(club, team, role)
    if not rows:
        return {"tone": "neutral", "text": ""}
    worst = max(rows, key=lambda r: r[0])
    if worst[0] >= 14:
        return {"tone": "bad", "text": worst[1]}
    best = min(rows, key=lambda r: r[0])
    return {"tone": "good", "text": best[1]}


def standing(rank: int, n_starters: int) -> str:
    """'the 2nd weakest in the league' style wording for a 1 = strongest rank among 20 clubs."""
    unit = {1: "first choice", 2: "pairing", 3: "trio"}.get(n_starters, "group")
    if rank <= 10:
        place = "the strongest" if rank == 1 else f"the {ordinal(rank)} strongest"
    else:
        place = "the weakest" if rank == 20 else f"the {ordinal(21 - rank)} weakest"
    return f"Measured against the same position at the other 19 clubs, this {unit} is {place} in the league"


def _who(d: dict) -> str:
    age = f"{int(d['age'])}, " if d.get("age") else ""
    listed = (f"; listed as a {ROLE_NAMES[d['own_role']].lower()}, plays this role here" if d.get("pivot") else "")
    return f"{d['name']} ({age}{ordinal(d['pct'])} percentile{listed})"


def _names(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def brief(club: str, need: dict, squad: pd.DataFrame, team: pd.Series, target: dict | None,
          age_rank: int | None, in_xi: set = frozenset()) -> dict:
    role = need["role"]
    rname = ROLE_NAMES[role].lower()
    keys = [f["key"] for f in need["flags"]]
    starters = need["starter_detail"]
    names = {d["name"] for d in starters}
    # Cover = specialists in the role who are not already starting somewhere in the side.
    bench = squad[(squad.role == role) & ~squad.web_name.isin(names | set(in_xi))].sort_values("role_percentile", ascending=False)

    evidence = []
    # What is happening in the position.
    if "empty" in keys:
        def unused(r):
            news = f"; {r.news.rstrip('.')}" if isinstance(r.news, str) and r.news and r.status != "a" else ""
            starts = int(r.starts_cur)
            return (f"{r.web_name} ({ordinal(round(r.role_percentile))} percentile, "
                    f"{starts} start{'' if starts == 1 else 's'} this season{news})")
        listed = [unused(r) for r in bench.itertuples()]
        evidence.append(f"No {rname} has played regularly this season."
                        + (f" The specialists in the squad: {'; '.join(listed)}." if listed
                           else f" There is no {rname} in the squad."))
    for d in starters:
        if d["covering"]:
            evidence.append(f"{d['name']} is normally a {ROLE_NAMES[d['own_role']].lower()} and is filling in here; "
                            f"judged as {'an' if rname[0] in 'aeiou' else 'a'} {rname} he sits in the {ordinal(d['pct'])} percentile.")
    weak = [d for d in starters if d["pct"] < 35 and not d["covering"]]
    for d in weak:
        others = [o for o in starters if o is not d]
        line = f"{d['name']} is the weak link: he rates below {100 - d['pct']}% of the league's regular {rname}s."
        row = squad[squad.web_name == d["name"]]
        if len(row):
            r = row.iloc[0]
            cur, mins = r.get("role_percentile_cur"), r.get("minutes_current", 0)
            if cur == cur and cur is not None and mins >= 270 and cur - d["pct"] >= 25:
                line += (f" He has started this season better ({ordinal(round(cur))} percentile in {int(mins)} minutes),"
                         f" so the case rests mostly on last season. Watch whether the form holds.")
        evidence.append(line)
    if "ageing" in keys:
        old = [d for d in starters if d.get("age") and d["age"] >= (34 if role == "GK" else 31)]
        older = ", ".join(f"{d['name']} is {int(d['age'])}" for d in old)
        rank_txt = (f", the {ordinal(age_rank)} oldest group in the league" if age_rank and age_rank <= 5
                    else "")
        evidence.append(f"Age is catching up: {older}. The starters average {need['starters_age']:.0f}{rank_txt}.")
    if role != "GK" and "empty" not in keys:
        def cover_txt(r):
            injured = {"i": "; injured", "d": "; doubtful", "s": "; suspended"}.get(r.status, "")
            age = f"{int(r.age)}, " if r.age == r.age else ""
            return f"{r.web_name} ({age}{ordinal(round(r.role_percentile))} percentile{injured})"
        cover = [cover_txt(r) for r in bench.head(4).itertuples()]
        good = need["backups_above_median"]
        if not cover:
            evidence.append(f"There is no specialist {rname} behind the {'starters' if len(starters) > 1 else 'starter'}, "
                            f"so an injury means playing someone out of position.")
        else:
          evidence.append(f"Cover: {', '.join(cover)}. "
                        + (("None of the fit ones is above the league average, so one injury weakens the side."
                            if good == 0 else
                            f"Only {good} fit backup{' is' if good == 1 else 's are'} above the league average.")
                           if "thin" in keys else
                           f"{good} fit backup{' is' if good == 1 else 's are'} above the league average, so the position is covered."))
    if not evidence:
        evidence.append(f"{_names([_who(d) for d in starters])} "
                        f"{'hold' if len(starters) > 1 else 'holds'} the position. "
                        f"{standing(need['league_rank'], len(starters))}.")
    elif "empty" not in keys:
        evidence.insert(0, f"{'Starters' if len(starters) > 1 else 'Starter'}: {_names([_who(d) for d in starters])}. "
                           f"{standing(need['league_rank'], len(starters))}.")
    else:
        evidence.append(f"{standing(need['league_rank'], 0).replace('this group', 'with nobody established, the position')}.")

    # The diagnosis in one line.
    if "empty" in keys:
        headline = f"There is no recognised {rname} in the team."
    elif "covering" in keys:
        headline = f"The position is being covered by players from elsewhere in the squad."
    elif weak and "ageing" in keys:
        headline = f"{weak[0]['name']} is the weak link, and the group is getting old."
    elif weak:
        headline = f"{weak[0]['name']} is holding the position back."
    elif "ageing" in keys:
        headline = "Good enough today, but the group is ageing."
    elif "thin" in keys:
        headline = "The starters are fine; the cover behind them is not."
    elif "league" in keys:
        headline = "The position is among the weakest in the league."
    else:
        headline = "No problems: the position is in good hands."

    # The recommendation.
    fit = (f" Best fit in the data: {target['first_name']} {target['second_name']} ({target['team']}, "
           f"{int(target['age']) if target.get('age') == target.get('age') and target.get('age') else '?'}"
           f"{', €%dm' % target['market_value_m'] if target.get('market_value_m') == target.get('market_value_m') and target.get('market_value_m') else ''})."
           if target else "")
    pr = need["priority_label"]
    if pr == "Critical":
        action = f"Sign a starting {rname} in the next window.{fit}"
    elif pr == "High" and weak:
        action = f"Replace {weak[0]['name']} in the starting side with {'an' if rname[0] in 'aeiou' else 'a'} {rname} above the 60th percentile.{fit}"
    elif pr == "High" and "ageing" in keys:
        old = max((d for d in starters if d.get("age")), key=lambda d: d["age"])
        action = f"Plan the succession: sign {'an' if rname[0] in 'aeiou' else 'a'} {rname} aged 27 or under who can start now and take over from {old['name']}.{fit}"
    elif pr == "High":
        action = f"Add {'an' if rname[0] in 'aeiou' else 'a'} {rname} good enough to start, so the team does not drop off when someone is missing.{fit}"
    elif pr == "Moderate" and weak:
        action = (f"Upgrade on {weak[0]['name']} when the right player is available; the rest of the "
                  f"position is sound, so this is not a panic buy.{fit}")
    elif pr == "Moderate" and "league" in keys:
        action = f"Raise the level of the position: the starters are below most of the league's.{fit}"
    elif pr == "Moderate":
        action = ("Not urgent and no signing needed now. Plan the succession for next summer." if "ageing" in keys
                  else "No new starter needed. Add a backup only if the budget allows after the priorities above.")
    else:
        action = "No signing needed."
    te = team_evidence(club, team, role)
    if te["text"]:
        lead = {("bad", True): "The team numbers show it: ", ("good", True): "The team has coped so far: ",
                ("bad", False): "One warning sign: ", ("good", False): "The team numbers back this up: "}
        te["text"] = lead[(te["tone"], pr in ("Critical", "High"))] + te["text"]
    return {"headline": headline, "evidence": evidence, "team": te, "action": action}
