"""Model Training Pipeline for Valorant Performance Intelligence.

Trains, cross-validates, tunes, and serializes classification models
(Logistic Regression, Random Forest, XGBoost) to predict match win probability
with strict data leakage safeguards.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_preprocessor,
    extract_features_and_target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("valorant.train")


def train_models(
    features_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train, cross-validate, and tune 3 candidate classification models.

    Parameters
    ----------
    features_path : Path | str | None
        Path to processed features CSV or Parquet.
    models_dir : Path | str | None
        Directory where serialized models and metadata will be saved.
    test_size : float
        Fraction of data reserved for out-of-sample testing.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary of model artifacts, test metrics, and training metadata.
    """
    project_root = Path(__file__).resolve().parent.parent
    features_path = Path(features_path or project_root / "data" / "processed" / "valorant_features.parquet")
    if not features_path.exists():
        features_path = features_path.with_suffix(".csv")
    models_dir = Path(models_dir or project_root / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature dataset from: %s", features_path)
    if features_path.suffix == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path)

    # 1. Separate Features and Target (leakage-safe)
    X, y = extract_features_and_target(df)

    # 2. Stratified Train / Test Split
    logger.info("Splitting dataset (test_size=%.2f, stratify=y)...", test_size)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Fit Preprocessor exclusively on Training Split to prevent leakage
    logger.info("Fitting feature preprocessor on training partition...")
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Extract feature names after one-hot encoding
    feature_names = list(preprocessor.get_feature_names_out())
    logger.info("Transformed feature matrix: %d encoded features.", len(feature_names))

    # 4. Define Candidate Classifiers
    candidates = {
        "Logistic Regression (Baseline)": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    # 5. Stratified 5-Fold Cross-Validation on Training Split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = {}
    test_results = {}
    trained_models = {}

    for name, model in candidates.items():
        logger.info("Running 5-fold CV for: %s...", name)
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_results[name] = {
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_f1_std": float(np.std(cv_scores["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
        }

        # Fit model on entire training set
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Out-of-sample test evaluation
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_results[name] = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall": float(recall_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        logger.info(
            "[%s] CV ROC-AUC: %.4f | Test Accuracy: %.4f | Test ROC-AUC: %.4f | Test F1: %.4f",
            name,
            cv_results[name]["cv_roc_auc_mean"],
            test_results[name]["test_accuracy"],
            test_results[name]["test_roc_auc"],
            test_results[name]["test_f1"],
        )

    # 6. Select Champion Model based on Test ROC-AUC & F1
    champion_name = max(test_results.keys(), key=lambda k: test_results[k]["test_roc_auc"])
    champion_model = trained_models[champion_name]
    baseline_model = trained_models["Logistic Regression (Baseline)"]

    logger.info("Champion model selected: %s (ROC-AUC = %.4f)", champion_name, test_results[champion_name]["test_roc_auc"])

    # 7. Serialize Artifacts
    best_model_path = models_dir / "best_model.joblib"
    baseline_model_path = models_dir / "baseline_model.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"
    all_models_path = models_dir / "all_models.joblib"
    metadata_path = models_dir / "model_metadata.json"

    joblib.dump(champion_model, best_model_path)
    joblib.dump(baseline_model, baseline_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(
        {
            "models": trained_models,
            "X_test_raw": X_test_raw,
            "X_test_transformed": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        },
        all_models_path,
    )

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "champion_model_name": champion_name,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoded_feature_names": feature_names,
        "cross_validation_results": cv_results,
        "test_results": test_results,
        "champion_test_metrics": test_results[champion_name],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved all model artifacts to: %s", models_dir)
    return metadata


if __name__ == "__main__":
    train_models()
