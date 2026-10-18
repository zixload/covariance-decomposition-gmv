"""Tests du pipeline de données."""
import numpy as np
import pandas as pd
import pytest

from src.data import clean_panel, compute_log_returns, descriptive_stats


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Panel de prix synthétiques : 2 tickers complets, 1 incomplet, 1 troué."""
    idx = pd.bdate_range("2000-01-03", periods=300)
    rng = np.random.default_rng(42)
    full1 = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    full2 = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    late = np.full(300, np.nan)
    late[200:] = 30.0                       # coté trop tard -> exclu
    holey = 80 * np.ones(300)
    holey[::10] = np.nan                    # 10% de NaN -> exclu
    return pd.DataFrame({"FULL1": full1, "FULL2": full2, "LATE": late,
                         "HOLEY": holey}, index=idx)


def test_log_returns_exact():
    """r_t = ln(P_t/P_{t-1}) vérifié sur des valeurs connues."""
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    ret = compute_log_returns(prices)
    expected = [np.log(1.10), np.log(99 / 110)]
    assert np.allclose(ret["A"].values, expected)
    assert len(ret) == 2  # la première ligne (NaN) est retirée


def test_clean_drops_incomplete(synthetic_prices):
    """La règle d'exclusion élimine le ticker coté tard et le ticker troué."""
    clean, report = clean_panel(synthetic_prices, start_max="2000-01-31",
                                max_missing_pct=0.02)
    assert set(clean.columns) == {"FULL1", "FULL2"}
    assert not report.loc["LATE", "kept"]
    assert not report.loc["HOLEY", "kept"]


def test_clean_panel_no_nan_monotonic(synthetic_prices):
    """Panel final : aucun NaN, dates strictement croissantes."""
    clean, _ = clean_panel(synthetic_prices, "2000-01-31", 0.02)
    returns = compute_log_returns(clean)
    assert not returns.isna().any().any()
    assert returns.index.is_monotonic_increasing
    assert returns.index.is_unique


def test_descriptive_stats_columns(synthetic_prices):
    """Les stats descriptives ont les bonnes colonnes et sont finies."""
    clean, _ = clean_panel(synthetic_prices, "2000-01-31", 0.02)
    stats = descriptive_stats(compute_log_returns(clean), trading_days=252)
    assert list(stats.columns) == ["mean_ann_pct", "vol_ann_pct", "skew",
                                   "kurtosis", "min_pct", "max_pct"]
    assert np.isfinite(stats.values).all()
