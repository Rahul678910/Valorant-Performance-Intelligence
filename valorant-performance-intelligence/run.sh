#!/usr/bin/env bash
# Valorant Performance Intelligence Quick Launcher for Linux/macOS
set -e

echo "========================================================"
echo "  VALORANT PERFORMANCE INTELLIGENCE DASHBOARD"
echo "========================================================"
echo ""

if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

if [ ! -f "models/best_model.joblib" ]; then
    echo "[INFO] Training baseline and champion models..."
    python3 -m src.data_processing --n-samples 6000 --seed 42
    python3 -m src.feature_engineering
    python3 -m src.train_model
    python3 -m src.evaluate_model
fi

echo "[INFO] Launching Streamlit Dashboard on http://localhost:8501..."
streamlit run src/dashboard.py
