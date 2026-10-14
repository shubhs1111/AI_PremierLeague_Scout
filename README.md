# Premier League Scouting Lab

An AI-assisted recruitment model for all 20 Premier League clubs in 2026/27. For each club it answers four questions, in order:

1. **How has the team played?** Results, expected goals, expected points and league rank on twelve team measures.
2. **How are the current players performing?** Every player is rated 0–100 against others in the same playing role.
3. **Where does the squad fall short?** A need score for each role, compared with the other 19 clubs.
4. **Who would close the gap?** Ranked Premier League targets, each with a scouting report written by an LLM.

Everything is computed from data. Transfer rumours play no part.

**Live dashboard:** https://shubhs1111.github.io/AI_PremierLeague_Scout/ (the same page is in `outputs/premier-league-scouting-lab.html`; open it in any browser).

## Run it

```bash
pip install -r requirements.txt
python -m src.fetch_data        # download FPL Core Insights match + player-match data
python run_pipeline.py          # ratings, needs, targets, reports
python -m src.build_dashboard   # outputs/dashboard.html
```

Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`, default `gpt-4o-mini`) to have OpenAI write the scouting reports. Without a key, a template writes them from the same structured facts, so the pipeline always runs.

## Data

| Source | Used for |
|---|---|
| [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights) 2025/26 and 2026/27 | Match results, team xG, and per-match player stats (passing, duels, xG/xA, goalkeeping) |
| Kaggle: EPL match data 2000–2025 (marcohuiii) | Points-per-game history by season |
| Kaggle: EPL 2025/26 player stats (sananmuzaffarov) | Independent validation of the ratings (provider match rating) |
| Kaggle: Transfermarkt player profiles (davidcariboo) | Date of birth, market value, contract expiry |
| FPL API snapshot | Availability status, FPL price (fallback value) |

Raw data is not stored in this repository; fetch it before running the pipeline.

- `data/raw/`: run `python -m src.fetch_data` (FPL Core Insights, downloaded automatically).
- `data/external/`: download from Kaggle and save under these names:
  - `epl_matches_2000_2025.csv`: [marcohuiii/english-premier-league-epl-match-data-2000-2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025)
  - `pl_player_stats_2025-26_gd35.csv`: [sananmuzaffarov/epl-202526-player-stats-gw131](https://www.kaggle.com/datasets/sananmuzaffarov/epl-202526-player-stats-gw131)
  - `transfermarkt_players.csv`: `players.csv` from [davidcariboo/player-scores](https://www.kaggle.com/datasets/davidcariboo/player-scores)

The dashboard and the files in `outputs/` contain only derived numbers (ratings, needs, targets, reports).

## Method

**Playing roles.** Seven roles: GK, CB, FB, DM, CM, AM/W and ST. Each player takes the role of his listed Transfermarkt position (e.g. Defensive Midfield → DM, Left-Back → FB). The few players without a link get a role from k-means clustering on per-90 style metrics within their FPL position.

**Player rating.** Each role has a weighted set of per-90 metrics (`ROLE_WEIGHTS` in `src/player_ratings.py`).
- Defensive actions (tackles, interceptions, recoveries, blocks, clearances) are possession-adjusted per match, scaled to 50% opposition possession, so defenders on dominant sides are not penalised for having less to do.
- Every outfield player is also rated for every other outfield role, so cover can be judged.
- A player's rating is the weighted average of his percentiles on those metrics against everyone in the same role, scaled to 0–100.
- 2025/26 minutes count at 60% of this season's.
- Ratings are pulled toward the role mean until a player reaches 900 minutes.

**Team analysis.** Expected points come from a Poisson model on each match's xG. Every team measure is ranked 1–20.

**Squad needs** (`src/squad_needs.py`). Each club's shape is read from its minutes this season (back three or four, two or three central midfielders, one or two strikers). The first-choice players in each role are the most-used specialists with 135+ minutes this season. The holding-midfield slot goes to a regular listed DM, or else to the regular central midfielder whose game leans most to the holding role. Roles left short are filled by unused players from neighbouring roles, and each player fills one role only.

Each role is checked for six concrete problems:
- nobody playing there regularly;
- someone filling in out of position;
- a starter in the bottom third of the league for the role;
- a starter aged 31 or over (34 for goalkeepers, who peak later);
- fewer above-average backups than starting slots;
- the role group ranking 15th or lower of 20 clubs.

Priority levels:

| Level | Rule |
|---|---|
| Critical | Empty or out-of-position role plus at least one other problem |
| High | Two or more problems, or an empty or out-of-position role |
| Moderate | One problem |
| Low | No problems |

A position gets transfer targets (marked **Sign**) only when a new player would fix it: every Critical or High position, plus a Moderate one whose problem is a starter's level (weak link or bottom-six group). Age or thin cover alone is marked **Hold**. The number of positions with targets therefore varies by club.

Within a level, roles are ordered by a need score. That score blends:
- 50%: league percentile of role strength (half the starters' average, half the weakest starter);
- 20%: related team measures;
- 15%: depth;
- 15%: age.

**Position briefs** (`src/insights.py`). Each position gets a short brief written from the numbers, in the voice of a sporting director: a one-line diagnosis, the evidence (starters with age and percentile, injuries from FPL news, cover, age rank in the league, form this season vs last), the most relevant team measure phrased with its direction ("concede 10.8 shots per game, the 3rd fewest"), and a recommended action that names the best-fitting target.

**Season views.** Ratings and role percentiles are also computed for each season on its own, so the squad table can show 2026/27, 2025/26 or both blended. The gaps and targets always use the blended rating.

**Target ranking** (`src/recommend.py`). Shortlists are allocated like a draft so the same few bargains are not offered to every club: in each round every Sign position takes its best remaining candidate, most urgent needs first, and no player appears on more than two clubs' shortlists.

Filters:
- 900+ minutes over the two seasons and minutes this season;
- at least +3 rating over the weakest starter;
- affordable: market value no more than the mean of the club's three most valuable players plus 25%;
- not at a clearly stronger club.

Score:

| Component | Weight |
|---|---|
| Rating gain over the weakest starter | 30% |
| Role percentile | 15% |
| Style fit (cosine similarity to the club's incumbents in the role) | 15% |
| Rating gain per √€m of market value | 10% |
| Age (full marks 21–28, 0.1 at 34) | 15% |
| Minutes behind the rating | 10% |
| Realism (relative club strength, contract under 18 months) | 5% |

Realism also scales the total, so a player the club cannot realistically sign does not top the list. Where a role's problem is an ageing group, the age score scales the total as well, so an older signing does not "fix" it.

**Scouting reports** (`src/reports.py`). The model receives only a structured fact sheet: need, team signal, stat comparison with the incumbent, age, value and contract. It is told to use only those numbers.

## Validation

Our 2025/26 ratings are compared with an independent provider's average match rating for 313 regulars (900+ minutes). Switching from clustered roles to listed positions raised the overall correlation from 0.41 to 0.48, and possession-adjusting defensive actions raised it to 0.51.

| Role | Spearman ρ |
|---|---|
| Overall | 0.51 |
| Centre-back | 0.66 |
| Central mid | 0.61 |
| Striker | 0.61 |
| Goalkeeper | 0.60 |
| Defensive mid | 0.56 |
| Attacking mid / winger | 0.50 |
| Full-back | 0.36 |

The match rating also rewards team results, and ours compares players only within their role, so agreement is expected to be partial. Full-back is the weakest role and the first place to improve the weights.

## Limits

- Squad-table counts (starts, goals, assists, clean sheets) can be shown for 2026/27, 2025/26 or both; ratings always blend both seasons.
- Six gameweeks of 2026/27 is a small sample, which is why last season is blended in.
- Transfermarkt values come from the dataset's last refresh in 2025/26. About 93% of current players link by name; the rest fall back to FPL price and a neutral age score.
- Candidates come only from Premier League squads.

## Layout

```
src/            config, fetch_data, load, team_analysis, player_ratings, insights,
                squad_needs, recommend, reports, external, build_dashboard
run_pipeline.py end-to-end run
dashboard/      HTML template (vanilla JS + SVG, no build step)
outputs/        premier-league-scouting-lab.html (standalone dashboard), dashboard_data.json, scouting_reports.json
docs/           index.html for GitHub Pages
data/processed/ player_ratings.csv, team_table.csv, squad_needs.csv
```
