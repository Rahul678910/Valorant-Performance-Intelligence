@echo off
REM Valorant Performance Intelligence Quick Launcher for Windows
echo ========================================================
echo   VALORANT PERFORMANCE INTELLIGENCE DASHBOARD
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not detected. Creating .venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
) ELSE (
    call .venv\Scripts\activate.bat
)

IF NOT EXIST "models\best_model.joblib" (
    echo [INFO] Model artifacts missing. Training pipeline...
    python -m src.data_processing --n-samples 6000 --seed 42
    python -m src.feature_engineering
    python -m src.train_model
    python -m src.evaluate_model
)

echo [INFO] Launching Streamlit Dashboard...
streamlit run src\dashboard.py
pause
