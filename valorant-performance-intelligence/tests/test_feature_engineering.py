"""Unit tests for src.feature_engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import generate_synthetic_valorant_data
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    compute_derived_features,
    create_preprocessor,
    extract_features_and_target,
)


def test_compute_derived_features_math_and_safety():
    """Verify ratio computations and division-by-zero resilience."""
    df_raw = generate_synthetic_valorant_data(n_samples=20, random_state=42)
    # Force zero values to test edge cases
    df_raw.loc[0, "deaths"] = 0
    df_raw.loc[0, "kills"] = 0
    df_raw.loc[0, "match_duration"] = 0.0

    df_feats = compute_derived_features(df_raw)

    assert "kd_ratio" in df_feats.columns
    assert "combat_efficiency" in df_feats.columns
    assert "first_blood_rate" in df_feats.columns
    assert "econ_efficiency" in df_feats.columns

    # Verify no infinite or NaN values
    assert not np.isinf(df_feats["kd_ratio"]).any()
    assert not np.isnan(df_feats["kd_ratio"]).any()
    assert not np.isinf(df_feats["combat_efficiency"]).any()


def test_leakage_prevention_in_feature_extraction():
    """Verify target and score-leaking columns are strictly excluded from X."""
    df_raw = generate_synthetic_valorant_data(n_samples=50, random_state=42)
    df_feats = compute_derived_features(df_raw)
    X, y = extract_features_and_target(df_feats)

    # Assert target is binary vector
    assert set(np.unique(y)).issubset({0, 1})
    assert len(y) == len(df_raw)

    # Critical leakage checks: score totals and target MUST NOT be in X
    assert "round_wins" not in X.columns
    assert "round_losses" not in X.columns
    assert "match_result" not in X.columns
    assert "match_id" not in X.columns
    assert "player_id" not in X.columns


def test_preprocessor_transformation_shape():
    """Verify ColumnTransformer handles numeric scaling and one-hot encoding without errors."""
    df_raw = generate_synthetic_valorant_data(n_samples=40, random_state=42)
    df_feats = compute_derived_features(df_raw)
    X, _ = extract_features_and_target(df_feats)

    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert not np.isnan(X_transformed).any()
