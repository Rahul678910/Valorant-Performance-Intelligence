# PRD: Valorant Performance Intelligence

## Objective
Build a complete end-to-end data science and machine learning system for Valorant players and coaches to ingest match data, analyze tactical patterns, predict match win probability with ML models, explain predictions with SHAP, and deliver actionable insights through an interactive Streamlit dashboard.

## Task Backlog
- [x] Task 1: Data Synthesis & Ingestion Engine (`src/data_processing.py`)
  - Realistic Valorant telemetry generation (agents, roles, maps, ranks, kills, deaths, assists, ACS, HS%, clutches, economy, round wins/losses, side, match result).
  - Validation schema, cleaning, SQLite insertion, and CSV/Parquet export.
- [x] Task 2: Feature Engineering Pipeline (`src/feature_engineering.py`)
  - Leakage-safe player & match efficiency metrics (K/D, combat efficiency, first blood rate, clutch conversion, economy impact, side bias).
  - Categorical encoding and scikit-learn compatible transformer.
- [x] Task 3: Exploratory Data Analysis & Visualization Report (`reports/`)
  - Agent, map, role, rank, and side performance analysis.
  - Generate and save publication-grade figures (`reports/figures/`).
  - Generate comprehensive insights report (`reports/eda_insights.md`).
- [x] Task 4: Model Training, Cross-Validation & Hyperparameter Tuning (`src/train_model.py`)
  - Train Logistic Regression, Random Forest, XGBoost.
  - Stratified 5-fold cross-validation and hyperparameter search.
  - Model serialization to `models/`.
- [x] Task 5: Model Evaluation & SHAP Interpretability (`src/evaluate_model.py`)
  - Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
  - SHAP TreeExplainer generation and summary visualization.
  - Generate `reports/model_evaluation.md`.
- [x] Task 6: Streamlit Interactive Dashboard (`src/dashboard.py`)
  - Player overview with KPIs & charts.
  - Agent & map intelligence matrix.
  - Interactive win-probability predictor with real-time SHAP waterfall.
  - Tactical AI coach recommendation engine.
- [x] Task 7: Unit Testing Suite (`tests/`)
  - Test data generator, validation, feature engineering, and model inference with `pytest`.
- [x] Task 8: Verification, Documentation & Final Artifacts
  - Local verification run and dashboard validation.
  - Professional `README.md` and project completion summary.
