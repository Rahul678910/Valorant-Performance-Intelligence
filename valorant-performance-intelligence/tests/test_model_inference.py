"""Unit tests for trained model loading and live inference."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


def test_model_and_preprocessor_artifacts_exist():
    """Verify serialized model artifacts exist on disk."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    assert (models_dir / "best_model.joblib").exists(), "Champion model missing"
    assert (models_dir / "preprocessor.joblib").exists(), "Preprocessor missing"
    assert (models_dir / "model_metadata.json").exists(), "Metadata missing"


def test_live_inference_prediction_shape_and_range():
    """Verify model inference on synthetic single-row input yields valid probability."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    model = joblib.load(models_dir / "best_model.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")

    # Sample input representing a high-impact winning performance
    win_sample = pd.DataFrame([{
        "agent": "Jett",
        "role": "Duelist",
        "map": "Ascent",
        "rank": "Diamond",
        "attack_defense_side": "Defense",
        "kills": 24,
        "deaths": 8,
        "assists": 5,
        "average_combat_score": 340.0,
        "headshot_percentage": 32.0,
        "first_bloods": 5,
        "clutches": 2,
        "economy_rating": 88.0,
        "match_duration": 42.0,
        "kd_ratio": 3.0,
        "kda_ratio": 3.31,
        "combat_efficiency": 8.1,
        "first_blood_rate": 0.208,
        "econ_efficiency": 3.32,
    }])

    # Sample input representing a struggling losing performance
    loss_sample = pd.DataFrame([{
        "agent": "Jett",
        "role": "Duelist",
        "map": "Ascent",
        "rank": "Diamond",
        "attack_defense_side": "Defense",
        "kills": 7,
        "deaths": 18,
        "assists": 2,
        "average_combat_score": 130.0,
        "headshot_percentage": 14.0,
        "first_bloods": 0,
        "clutches": 0,
        "economy_rating": 42.0,
        "match_duration": 42.0,
        "kd_ratio": 0.39,
        "kda_ratio": 0.44,
        "combat_efficiency": 3.1,
        "first_blood_rate": 0.0,
        "econ_efficiency": 5.25,
    }])

    # Predict Win Sample
    X_win = preprocessor.transform(win_sample)
    assert isinstance(X_win, np.ndarray)
    prob_win = model.predict_proba(X_win)
    assert prob_win.shape == (1, 2)
    win_p = float(prob_win[0, 1])
    assert 0.0 <= win_p <= 1.0
    assert win_p > 0.70, f"Expected high win probability for 24-8 K/D performance, got {win_p}"

    # Predict Loss Sample
    X_loss = preprocessor.transform(loss_sample)
    prob_loss = model.predict_proba(X_loss)
    loss_p = float(prob_loss[0, 1])
    assert 0.0 <= loss_p <= 1.0
    assert loss_p < 0.30, f"Expected low win probability for 7-18 K/D performance, got {loss_p}"
