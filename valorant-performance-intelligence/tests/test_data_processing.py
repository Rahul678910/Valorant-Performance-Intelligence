"""Unit tests for src.data_processing."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from src.data_processing import (
    clean_valorant_data,
    generate_synthetic_valorant_data,
    load_from_sqlite,
    save_to_sqlite,
    validate_valorant_data,
)


def test_generate_synthetic_data_shape_and_columns():
    """Verify synthetic generator generates expected shape and mandatory fields."""
    n_samples = 150
    df = generate_synthetic_valorant_data(n_samples=n_samples, random_state=123)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_samples
    assert "player_id" in df.columns
    assert "match_id" in df.columns
    assert "average_combat_score" in df.columns
    assert "headshot_percentage" in df.columns
    assert "match_result" in df.columns


def test_validation_logic():
    """Verify validation passes on valid data and fails when domain rules are violated."""
    df_valid = generate_synthetic_valorant_data(n_samples=50, random_state=42)
    is_valid, issues = validate_valorant_data(df_valid)
    assert is_valid is True
    assert len(issues) == 0

    # Inject missing column
    df_invalid = df_valid.drop(columns=["kills"])
    is_valid, issues = validate_valorant_data(df_invalid)
    assert is_valid is False
    assert any("Missing mandatory columns" in s for s in issues)

    # Inject negative value
    df_neg = df_valid.copy()
    df_neg.loc[0, "deaths"] = -5
    is_valid, issues = validate_valorant_data(df_neg)
    assert is_valid is False
    assert any("Negative deaths" in s for s in issues)


def test_clean_valorant_data():
    """Verify data cleaning normalizes whitespace and correctly clips values."""
    df_raw = generate_synthetic_valorant_data(n_samples=30, random_state=42)
    df_raw.loc[0, "agent"] = " Jett  "
    df_raw.loc[0, "headshot_percentage"] = 150.0  # Above 100%

    df_cleaned = clean_valorant_data(df_raw)
    assert df_cleaned.loc[0, "agent"] == "Jett"
    assert df_cleaned.loc[0, "headshot_percentage"] <= 100.0


def test_sqlite_roundtrip():
    """Verify saving to SQLite database and reloading preserves data fidelity."""
    df = generate_synthetic_valorant_data(n_samples=40, random_state=99)
    df_cleaned = clean_valorant_data(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_telemetry.db"
        save_to_sqlite(df_cleaned, db_path, table_name="test_matches")
        assert db_path.exists()

        reloaded = load_from_sqlite(db_path, table_name="test_matches")
        assert len(reloaded) == len(df_cleaned)
        assert set(reloaded.columns) == set(df_cleaned.columns)
