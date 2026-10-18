"""Tests des agrégations multi-N et sous-périodes."""
import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.experiments import cache_path, collect_tables, export_table, slice_result

MODELS = ["(1) emp x emp", "(2) emp x lw", "(3) emp x dcc",
          "(4) garch x emp", "(5) garch x lw", "(6) garch x dcc"]


def fake_result(scale: float) -> BacktestResult:
    """Crée une trajectoire déterministe couvrant deux sous-périodes."""
    index = pd.bdate_range("2019-01-01", "2021-12-31")
    returns = pd.Series(np.sin(np.arange(len(index))) * scale, index=index)
    dates = index[::21]
    turnover = pd.Series(0.2, index=dates)
    concentration = pd.Series(0.3, index=dates)
    predicted = pd.Series(scale ** 2, index=index)
    return BacktestResult(returns, {}, turnover, concentration, predicted)


def experiment_config(tmp_path) -> dict:
    """Configuration minimale des agrégations et exports."""
    return {"backtest": {"trading_days": 252},
            "evaluation": {"dm_lags": 5, "center_returns": True,
                           "subperiods": {"pre": ["2019-01-01", "2019-12-31"],
                                          "post": ["2020-01-01", "2021-12-31"]}},
            "outputs": {"backtest_pattern": str(tmp_path / "bt_n{n}.pkl")}}


def test_slice_result_aligns_all_daily_series():
    """Le découpage conserve des index quotidiens parfaitement alignés."""
    sliced = slice_result(fake_result(0.01), "2020-01-01", "2020-12-31")
    assert sliced.portfolio_returns.index.equals(sliced.predicted_variance.index)
    assert sliced.portfolio_returns.index.min() >= pd.Timestamp("2020-01-01")
    assert sliced.portfolio_returns.index.max() <= pd.Timestamp("2020-12-31")


def test_collect_tables_has_expected_multi_indexes(tmp_path):
    """Les métriques, effets et tests sont indexés par N et sous-période."""
    config = experiment_config(tmp_path)
    results = {50: {name: fake_result(0.006 + i * 0.0005)
                    for i, name in enumerate(MODELS)}}
    summary, effects, tests = collect_tables(config, results)
    assert summary.index.names == ["N", "period", "model"]
    assert effects.index.names == ["N", "period", "effect"]
    assert tests.index.names == ["N", "period", "comparison"]
    assert set(summary.index.get_level_values("period")) == {"full", "pre", "post"}


def test_cache_path_formats_dimension(tmp_path):
    """Chaque dimension possède un fichier de cache distinct."""
    config = experiment_config(tmp_path)
    assert cache_path(config, 200).name == "bt_n200.pkl"


def test_export_table_directory_overrides_config(tmp_path):
    """Un `directory` explicite prend le pas sur outputs.tables_dir, sans
    quoi les tables de robustesse écraseraient celles de la grille principale."""
    config = experiment_config(tmp_path)
    config["outputs"]["tables_dir"] = str(tmp_path / "main")
    other = tmp_path / "robustness"
    export_table(pd.DataFrame({"x": [1]}), "t", config, directory=other)
    assert (other / "t.csv").exists()
    assert not (tmp_path / "main").exists()
