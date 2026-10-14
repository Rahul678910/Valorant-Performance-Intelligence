"""Data Processing Module for Valorant Performance Intelligence.

Handles:
- Realistic synthetic Valorant telemetry generation.
- Data validation against game domain constraints.
- Data cleaning, type normalization, and SQLite persistence.
- Export to cleaned CSV and SQLite database.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("valorant.data_processing")

# Domain Constants
AGENT_ROLES: Dict[str, str] = {
    "Jett": "Duelist",
    "Reyna": "Duelist",
    "Raze": "Duelist",
    "Phoenix": "Duelist",
    "Yoru": "Duelist",
    "Neon": "Duelist",
    "Iso": "Duelist",
    "Sova": "Initiator",
    "Breach": "Initiator",
    "Skye": "Initiator",
    "KAY/O": "Initiator",
    "Fade": "Initiator",
    "Gekko": "Initiator",
    "Brimstone": "Controller",
    "Omen": "Controller",
    "Viper": "Controller",
    "Astra": "Controller",
    "Harbor": "Controller",
    "Clove": "Controller",
    "Killjoy": "Sentinel",
    "Cypher": "Sentinel",
    "Sage": "Sentinel",
    "Chamber": "Sentinel",
    "Deadlock": "Sentinel",
}

MAP_POOL: List[str] = [
    "Ascent",
    "Bind",
    "Haven",
    "Split",
    "Lotus",
    "Sunset",
    "Pearl",
    "Breeze",
    "Icebox",
]

RANKS: List[str] = [
    "Iron",
    "Bronze",
    "Silver",
    "Gold",
    "Platinum",
    "Diamond",
    "Ascendant",
    "Immortal",
    "Radiant",
]

RANK_MULTIPLIERS: Dict[str, float] = {
    "Iron": 0.70,
    "Bronze": 0.80,
    "Silver": 0.90,
    "Gold": 1.00,
    "Platinum": 1.10,
    "Diamond": 1.20,
    "Ascendant": 1.30,
    "Immortal": 1.40,
    "Radiant": 1.50,
}


def generate_synthetic_valorant_data(
    n_samples: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic Valorant match telemetry.

    Parameters
    ----------
    n_samples : int
        Number of player-match records to synthesize.
    random_state : int
        RNG seed for determinism.

    Returns
    -------
    pd.DataFrame
        Synthetic raw dataframe matching official Valorant telemetry structure.
    """
    logger.info("Generating %d synthetic match telemetry records (seed=%d)...", n_samples, random_state)
    rng = np.random.default_rng(random_state)

    agents = list(AGENT_ROLES.keys())
    agent_choices = rng.choice(agents, size=n_samples)
    role_choices = [AGENT_ROLES[a] for a in agent_choices]
    map_choices = rng.choice(MAP_POOL, size=n_samples)
    
    # Ranks distribution (bell curve centered around Gold/Platinum)
    rank_probs = np.array([0.05, 0.12, 0.20, 0.25, 0.18, 0.11, 0.05, 0.03, 0.01])
    rank_probs /= rank_probs.sum()
    rank_choices = rng.choice(RANKS, size=n_samples, p=rank_probs)

    # Identifiers
    n_unique_players = max(50, n_samples // 12)
    player_ids = [f"VAL-PLY-{p:04d}" for p in rng.integers(1, n_unique_players + 1, size=n_samples)]
    match_ids = [f"MATCH-{m:06d}" for m in rng.integers(100000, 999999, size=n_samples)]

    # Side preference
    sides = rng.choice(["Attack", "Defense"], size=n_samples, p=[0.5, 0.5])

    # Date range (past 90 days)
    base_date = datetime(2026, 1, 1)
    date_offsets = rng.integers(0, 90, size=n_samples)
    date_choices = [(base_date + timedelta(days=int(d), minutes=int(rng.integers(0, 1440)))).strftime("%Y-%m-%d %H:%M:%S") for d in date_offsets]

    # Gameplay Performance Simulation with domain logic:
    # 1. Total rounds usually 16 to 24 (or up to 28 in overtime)
    is_win = rng.choice([True, False], size=n_samples, p=[0.51, 0.49])
    
    round_wins = np.zeros(n_samples, dtype=int)
    round_losses = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if is_win[i]:
            rw = 13
            rl = rng.integers(3, 12)
        else:
            rl = 13
            rw = rng.integers(3, 12)
        # 4% chance of overtime
        if rng.random() < 0.04:
            extra = rng.integers(1, 4) * 2
            if is_win[i]:
                rw += extra // 2 + 1
                rl += extra // 2 - 1
            else:
                rl += extra // 2 + 1
                rw += extra // 2 - 1
        round_wins[i] = rw
        round_losses[i] = rl

    total_rounds = round_wins + round_losses
    # Match duration roughly 1.8 to 2.2 minutes per round
    match_duration_mins = (total_rounds * rng.uniform(1.7, 2.3)).round(1)

    # Tactical performance metrics correlated with Role, Rank & Outcome
    kills = np.zeros(n_samples, dtype=int)
    deaths = np.zeros(n_samples, dtype=int)
    assists = np.zeros(n_samples, dtype=int)
    acs = np.zeros(n_samples, dtype=float)
    headshot_pct = np.zeros(n_samples, dtype=float)
    first_bloods = np.zeros(n_samples, dtype=int)
    clutches = np.zeros(n_samples, dtype=int)
    economy_rating = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        rank = rank_choices[i]
        role = role_choices[i]
        mult = RANK_MULTIPLIERS[rank]
        won = is_win[i]
        rounds = total_rounds[i]

        # Role base modifiers
        if role == "Duelist":
            base_k_mean = 0.88 * rounds
            base_d_mean = 0.72 * rounds
            base_a_mean = 0.20 * rounds
            fb_lambda = 3.6
        elif role == "Initiator":
            base_k_mean = 0.72 * rounds
            base_d_mean = 0.68 * rounds
            base_a_mean = 0.52 * rounds
            fb_lambda = 2.1
        elif role == "Controller":
            base_k_mean = 0.68 * rounds
            base_d_mean = 0.69 * rounds
            base_a_mean = 0.44 * rounds
            fb_lambda = 1.7
        else:  # Sentinel
            base_k_mean = 0.65 * rounds
            base_d_mean = 0.66 * rounds
            base_a_mean = 0.38 * rounds
            fb_lambda = 1.3

        # Winning teams get higher kill count, fewer deaths
        win_k_bonus = 3.5 if won else -2.5
        win_d_bonus = -3.0 if won else 2.5

        k = int(np.clip(rng.normal(base_k_mean + win_k_bonus, 4.0), 3, 42))
        d = int(np.clip(rng.normal(base_d_mean + win_d_bonus, 3.5), 3, 35))
        a = int(np.clip(rng.normal(base_a_mean, 3.0), 0, 25))

        # First bloods
        fb = int(np.clip(rng.poisson(fb_lambda * (1.15 if won else 0.85)), 0, 9))

        # Clutches (0 to 4)
        clutch_p = 0.12 if won else 0.06
        cl = int(rng.binomial(n=4, p=clutch_p))

        # Headshot % scales with rank
        hs_mean = 14.0 * mult + rng.normal(0, 3.0)
        hs = float(np.clip(hs_mean, 7.0, 48.0).round(1))

        # Average Combat Score (ACS): kills, damage, first bloods contribute
        base_acs = (k * 11.5 + a * 3.8 + fb * 18.0) / (rounds / 20.0) + (15.0 if won else -15.0)
        computed_acs = float(np.clip(rng.normal(base_acs, 25.0), 90.0, 480.0).round(1))

        # Economy Rating (damage dealt per 1000 credits spent, typically 40 to 110)
        econ = float(np.clip(rng.normal(65.0 * (1.12 if won else 0.88), 14.0), 22.0, 135.0).round(1))

        kills[i] = k
        deaths[i] = d
        assists[i] = a
        acs[i] = computed_acs
        headshot_pct[i] = hs
        first_bloods[i] = fb
        clutches[i] = cl
        economy_rating[i] = econ

    results = ["Win" if w else "Loss" for w in is_win]

    df = pd.DataFrame(
        {
            "player_id": player_ids,
            "match_id": match_ids,
            "agent": agent_choices,
            "role": role_choices,
            "map": map_choices,
            "rank": rank_choices,
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "average_combat_score": acs,
            "headshot_percentage": headshot_pct,
            "first_bloods": first_bloods,
            "clutches": clutches,
            "economy_rating": economy_rating,
            "round_wins": round_wins,
            "round_losses": round_losses,
            "attack_defense_side": sides,
            "match_result": results,
            "match_duration": match_duration_mins,
            "date": date_choices,
        }
    )

    logger.info("Generated %d synthetic match rows successfully.", len(df))
    return df


def validate_valorant_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate Valorant dataframe against business and domain rules.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry dataframe to validate.

    Returns
    -------
    Tuple[bool, List[str]]
        Status flag (True if valid) and list of validation issues found.
    """
    issues: List[str] = []
    required_cols = [
        "player_id",
        "match_id",
        "agent",
        "role",
        "map",
        "rank",
        "kills",
        "deaths",
        "assists",
        "average_combat_score",
        "headshot_percentage",
        "first_bloods",
        "clutches",
        "economy_rating",
        "round_wins",
        "round_losses",
        "attack_defense_side",
        "match_result",
        "match_duration",
        "date",
    ]

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing mandatory columns: {missing_cols}")
        return False, issues

    # Null checks
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        issues.append(f"Null values detected: {null_counts[null_counts > 0].to_dict()}")

    # Domain range checks
    if (df["kills"] < 0).any():
        issues.append("Negative kills detected.")
    if (df["deaths"] < 0).any():
        issues.append("Negative deaths detected.")
    if (df["assists"] < 0).any():
        issues.append("Negative assists detected.")
    if ((df["headshot_percentage"] < 0) | (df["headshot_percentage"] > 100)).any():
        issues.append("Headshot percentage out of range [0, 100].")
    if (df["average_combat_score"] < 0).any():
        issues.append("Negative average_combat_score detected.")
    if not set(df["match_result"].unique()).issubset({"Win", "Loss"}):
        issues.append(f"Invalid match results: {df['match_result'].unique()}")
    if not set(df["map"].unique()).issubset(set(MAP_POOL)):
        issues.append("Unrecognized maps present in dataset.")
    if not set(df["rank"].unique()).issubset(set(RANKS)):
        issues.append("Unrecognized ranks present in dataset.")

    is_valid = len(issues) == 0
    if is_valid:
        logger.info("Validation passed: All %d records adhere to domain constraints.", len(df))
    else:
        logger.warning("Validation failed with %d issues: %s", len(issues), issues)

    return is_valid, issues


def clean_valorant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, cast, and normalize Valorant telemetry.

    Parameters
    ----------
    df : pd.DataFrame
        Raw telemetry dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned and normalized dataframe.
    """
    logger.info("Cleaning and normalizing %d telemetry rows...", len(df))
    clean_df = df.copy()

    # Drop exact duplicates if any
    clean_df = clean_df.drop_duplicates()

    # Normalize text fields
    clean_df["agent"] = clean_df["agent"].astype(str).str.strip()
    clean_df["role"] = clean_df["role"].astype(str).str.strip()
    clean_df["map"] = clean_df["map"].astype(str).str.strip()
    clean_df["rank"] = clean_df["rank"].astype(str).str.strip()
    clean_df["match_result"] = clean_df["match_result"].astype(str).str.strip().str.capitalize()
    clean_df["attack_defense_side"] = clean_df["attack_defense_side"].astype(str).str.strip().str.capitalize()

    # Numerical types
    int_cols = ["kills", "deaths", "assists", "first_bloods", "clutches", "round_wins", "round_losses"]
    for c in int_cols:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce").fillna(0).astype(int)

    float_cols = ["average_combat_score", "headshot_percentage", "economy_rating", "match_duration"]
    for c in float_cols:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce").fillna(0.0).astype(float)

    # Date normalization
    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")

    # Domain clipping for extreme outliers
    clean_df["headshot_percentage"] = clean_df["headshot_percentage"].clip(0.0, 100.0)
    clean_df["average_combat_score"] = clean_df["average_combat_score"].clip(0.0, 600.0)

    logger.info("Cleaning complete. Clean dataset size: %d rows.", len(clean_df))
    return clean_df


def save_to_sqlite(df: pd.DataFrame, db_path: Path | str, table_name: str = "matches") -> None:
    """Persist dataframe into SQLite database table.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned telemetry dataframe.
    db_path : Path | str
        Path to SQLite database file.
    table_name : str
        Target table name.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Persisting %d records to SQLite DB at %s (table='%s')...", len(df), db_path, table_name)
    
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()
    logger.info("SQLite persistence complete.")


def load_from_sqlite(db_path: Path | str, table_name: str = "matches") -> pd.DataFrame:
    """Load dataframe from SQLite database table.

    Parameters
    ----------
    db_path : Path | str
        Path to SQLite database.
    table_name : str
        Target table name.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    logger.info("Loading records from SQLite DB at %s (table='%s')...", db_path, table_name)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    
    # Cast date back to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def run_pipeline(
    n_samples: int = 6000,
    raw_csv_path: Optional[Path | str] = None,
    clean_csv_path: Optional[Path | str] = None,
    sqlite_db_path: Optional[Path | str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute end-to-end data ingestion, validation, cleaning, and storage.

    Parameters
    ----------
    n_samples : int
        Number of synthetic records to generate.
    raw_csv_path : Optional[Path | str]
        Path to save raw CSV telemetry.
    clean_csv_path : Optional[Path | str]
        Path to save cleaned CSV.
    sqlite_db_path : Optional[Path | str]
        Path to SQLite database.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Cleaned, validated telemetry dataframe.
    """
    project_root = Path(__file__).resolve().parent.parent
    raw_csv_path = Path(raw_csv_path or project_root / "data" / "raw" / "valorant_matches.csv")
    clean_csv_path = Path(clean_csv_path or project_root / "data" / "processed" / "valorant_clean.csv")
    sqlite_db_path = Path(sqlite_db_path or project_root / "data" / "raw" / "valorant_telemetry.db")

    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    clean_csv_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize
    df_raw = generate_synthetic_valorant_data(n_samples=n_samples, random_state=random_state)
    df_raw.to_csv(raw_csv_path, index=False)
    logger.info("Saved raw dataset to: %s", raw_csv_path)

    # 2. Validate
    is_valid, issues = validate_valorant_data(df_raw)
    if not is_valid:
        raise ValueError(f"Dataset failed domain validation: {issues}")

    # 3. Clean
    df_clean = clean_valorant_data(df_raw)
    df_clean.to_csv(clean_csv_path, index=False)
    logger.info("Saved clean dataset to: %s", clean_csv_path)

    # 4. Save to SQLite
    save_to_sqlite(df_clean, sqlite_db_path, table_name="matches")

    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valorant Data Ingestion & Processing Pipeline")
    parser.add_argument("--n-samples", type=int, default=6000, help="Number of records to synthesize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(n_samples=args.n_samples, random_state=args.seed)
