"""Shared settings for the scouting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

CURRENT_SEASON = "2026-2027"
PREVIOUS_SEASON = "2025-2026"

# Ages and contract lengths are measured on this date.
AS_OF_DATE = "2026-10-14"

# How much a minute from last season counts relative to a minute this season
# when blending per-90 numbers. Recent form matters more, but six gameweeks
# alone are too small a sample to rate players on.
PREVIOUS_SEASON_WEIGHT = 0.6

# Ratings are pulled toward the role average until a player has this many
# blended minutes, so a great 200-minute cameo does not outrank a regular.
SHRINKAGE_MINUTES = 900

# Minimum blended minutes for a player to be considered as a transfer target.
TARGET_MIN_MINUTES = 900
# Minimum minutes this season, so targets are active players right now.
TARGET_MIN_CURRENT_MINUTES = 90

# Starting slots per role in a 4-3-3 / 4-2-3-1 style XI, used to judge depth.
ROLE_SLOTS = {"GK": 1, "CB": 2, "FB": 2, "DM": 1, "CM": 2, "AM/W": 2, "ST": 1}

ROLE_NAMES = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Full-back",
    "DM": "Defensive midfielder",
    "CM": "Central midfielder",
    "AM/W": "Attacking midfielder / winger",
    "ST": "Striker",
}
