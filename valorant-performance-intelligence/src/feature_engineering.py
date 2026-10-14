"""Feature Engineering Module for Valorant Performance Intelligence.

Extracts tactical, combat, and efficiency indicators while strictly
preventing data leakage (e.g. excluding post-match round score totals).
Provides modular transformations and a production-grade ColumnTransformer pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("valorant.feature_engineering")

# Categorical and Numerical feature schemas for modeling
NUMERIC_FEATURES: List[str] = [
    "kills",
    "deaths",
    "assists",
    "average_combat_score",
    "headshot_percentage",
    "first_bloods",
    "clutches",
    "economy_rating",
    "match_duration",
    "kd_ratio",
    "kda_ratio",
    "combat_efficiency",
    "first_blood_rate",
    "econ_efficiency",
]

CATEGORICAL_FEATURES: List[str] = [
    "agent",
    "role",
    "map",
    "rank",
    "attack_defense_side",
]

# Explicit list of columns excluded from features to prevent target leakage and ID bias
LEAKAGE_AND_META_COLS: List[str] = [
    "player_id",
    "match_id",
    "date",
    "round_wins",       # Leaks match outcome
    "round_losses",     # Leaks match outcome
    "match_result",     # Target variable
]


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate tactical performance ratios and advanced efficiency metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned telemetry dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with engineered features.
    """
    logger.info("Computing derived tactical features for %d rows...", len(df))
    data = df.copy()

    # 1. K/D and KDA ratios (safe division)
    safe_deaths = np.maximum(data["deaths"].values, 1.0)
    data["kd_ratio"] = (data["kills"].values / safe_deaths).round(2)
    data["kda_ratio"] = ((data["kills"].values + 0.5 * data["assists"].values) / safe_deaths).round(2)

    # 2. Combat Efficiency (ACS per minute played)
    safe_duration = np.maximum(data["match_duration"].values, 10.0)
    data["combat_efficiency"] = (data["average_combat_score"].values / safe_duration).round(2)

    # 3. First Blood Rate (First bloods relative to player kills)
    safe_kills = np.maximum(data["kills"].values, 1.0)
    data["first_blood_rate"] = (data["first_bloods"].values / safe_kills).round(3)

    # 4. Economy Efficiency (Economy rating normalized per kill involvement)
    kill_involvement = np.maximum(data["kills"].values + 0.5 * data["assists"].values, 1.0)
    data["econ_efficiency"] = (data["economy_rating"].values / kill_involvement).round(2)

    logger.info("Successfully engineered %d derived features.", len(NUMERIC_FEATURES) - 9)
    return data


def create_preprocessor() -> ColumnTransformer:
    """Construct a ColumnTransformer that standardizes numerics and one-hot encodes categoricals.

    Returns
    -------
    ColumnTransformer
        Scikit-learn composite preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def extract_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Separate predictor matrix X from target vector y with leakage prevention.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched telemetry dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        X (predictors) and y (binary target: 1 = Win, 0 = Loss).
    """
    if "match_result" not in df.columns:
        raise KeyError("Target column 'match_result' not found in dataframe.")

    # Target: 1 for Win, 0 for Loss
    y = (df["match_result"].astype(str).str.lower() == "win").astype(int).values

    # Select only the sanctioned features
    available_features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[available_features].copy()

    logger.info(
        "Extracted feature matrix X shape: %s, target y shape: %s (Win rate: %.2f%%)",
        X.shape,
        y.shape,
        (y.mean() * 100),
    )
    return X, y


def run_feature_pipeline(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """Execute feature engineering and persist the processed features dataset.

    Parameters
    ----------
    input_path : Path | str | None
        Path to cleaned CSV data.
    output_path : Path | str | None
        Path to output processed features dataset.
    save_parquet : bool
        Whether to export to Parquet in addition to CSV.

    Returns
    -------
    pd.DataFrame
        Enriched feature dataset.
    """
    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(input_path or project_root / "data" / "processed" / "valorant_clean.csv")
    output_path = Path(output_path or project_root / "data" / "processed" / "valorant_features.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {input_path}. Run data_processing first.")

    df_clean = pd.read_csv(input_path)
    df_features = compute_derived_features(df_clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    logger.info("Saved feature-engineered dataset to: %s", output_path)

    if save_parquet:
        parquet_path = output_path.with_suffix(".parquet")
        df_features.to_parquet(parquet_path, index=False)
        logger.info("Saved Parquet dataset to: %s", parquet_path)

    return df_features


if __name__ == "__main__":
    run_feature_pipeline()
