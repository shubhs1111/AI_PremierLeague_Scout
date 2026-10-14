"""Download Premier League data from the FPL Core Insights repository.

Source: https://github.com/olbauday/FPL-Core-Insights (refreshed twice a day).
Run: python -m src.fetch_data
"""
import urllib.parse
import urllib.request

from .config import CURRENT_SEASON, PREVIOUS_SEASON, RAW

BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
MASTER_FILES = ["players", "teams", "playerstats"]
GAMEWEEK_FILES = ["matches", "playermatchstats"]


def _download(url: str, dest) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            body = resp.read()
    except Exception:
        return False
    if body.count(b"\n") < 2:  # header only: gameweek not played yet
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return True


def fetch_season(season: str, max_gw: int = 38) -> int:
    for name in MASTER_FILES:
        _download(f"{BASE}/{season}/{name}.csv", RAW / season / f"{name}.csv")
    played = 0
    for gw in range(1, max_gw + 1):
        folder = urllib.parse.quote(f"By Tournament/Premier League/GW{gw}")
        ok = all(
            _download(f"{BASE}/{season}/{folder}/{name}.csv", RAW / season / "pl" / f"{name}_gw{gw}.csv")
            for name in GAMEWEEK_FILES
        )
        if not ok:
            break
        played = gw
    return played


if __name__ == "__main__":
    for season in (PREVIOUS_SEASON, CURRENT_SEASON):
        print(season, "gameweeks downloaded:", fetch_season(season))
