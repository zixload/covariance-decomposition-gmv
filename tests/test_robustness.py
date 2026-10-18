"""Tests des robustesses : robustesses long-only / horizon 21j / diagnostics."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestResult
from src.robustness import (by_weight, collect_robustness_tables,
                            collect_robustness_tables_multi_n,
                            diagnostics_summary, garch_cfg_for_horizon,
                            horizon_dm_table, load_or_run_robustness,
                            robustness_paths, save_robustness)

GARCH_MODELS = ["(4) garch x emp", "(5) garch x lw", "(6) garch x dcc"]
ALL_MODELS = ["(1) emp x emp", "(2) emp x lw", "(3) emp x dcc"] + GARCH_MODELS


def fake_result(scale: float) -> BacktestResult:
    """Trajectoire déterministe couvrant assez de jours pour le test DM."""
    index = pd.bdate_range("2019-01-01", periods=800)
    returns = pd.Series(np.sin(np.arange(len(index))) * scale, index=index)
    dates = index[::21]
    turnover = pd.Series(0.2, index=dates)
    concentration = pd.Series(0.3, index=dates)
    predicted = pd.Series(scale ** 2, index=index)
    return BacktestResult(returns, {}, turnover, concentration, predicted)


def fake_results(scale: float) -> dict[tuple[str, str], BacktestResult]:
    """Grille (modèle, contrainte) synthétique pour les deux contraintes."""
    return {(model, weight): fake_result(scale + 0.0001 * i)
           for i, model in enumerate(ALL_MODELS)
           for weight in ("unconstrained", "long_only")}


def robustness_config(tmp_path) -> dict:
    """Configuration minimale pour robustness_paths/save/load."""
    base = str(tmp_path / "robustness")
    return {"robustness": {
        "garch_horizon_21": 21, "n_assets": 10,
        "cache_pattern": base + "/backtest_n{n}_h{horizon}.pkl",
        "diagnostics_garch_pattern": base + "/garch_n{n}_h{horizon}.csv",
        "diagnostics_dcc_pattern": base + "/dcc_n{n}_h{horizon}.csv",
        "diagnostics_sigma_pattern": base + "/sigma_n{n}_h{horizon}.csv",
        "tables_dir": base + "/tables",
    }, "backtest": {"trading_days": 252},
        "evaluation": {"dm_lags": 5, "center_returns": True,
                      "costs": [0.0, 0.01], "gammas": [1],
                      "charge_initial_turnover": False, "fee_daily_bound": 0.10,
                      "subperiods": {"pre": ["2019-01-01", "2019-12-31"],
                                    "post": ["2020-01-01", "2021-12-31"]}}}


def test_garch_cfg_for_horizon_overrides_only_horizon():
    """Seul forecast_horizon change ; la config source n'est pas mutée."""
    config = {"models": {"garch": {"p": 1, "q": 1, "forecast_horizon": 1}}}
    cfg21 = garch_cfg_for_horizon(config, 21)
    assert cfg21["forecast_horizon"] == 21
    assert cfg21["p"] == 1
    assert config["models"]["garch"]["forecast_horizon"] == 1


def test_robustness_paths_formats_n_and_horizon(tmp_path):
    """Les 4 chemins de robustesse intègrent bien N et l'horizon."""
    config = robustness_config(tmp_path)
    paths = robustness_paths(config, 21)
    assert paths["cache"].name == "backtest_n10_h21.pkl"
    assert paths["garch_log"].name == "garch_n10_h21.csv"
    paths1 = robustness_paths(config, 1)
    assert paths1["cache"] != paths["cache"]  # jamais le meme fichier


def test_save_and_load_or_run_robustness_roundtrip(tmp_path):
    """Le cache écrit par save_robustness est relu à l'identique par
    load_or_run_robustness, sans jamais appeler run_robustness_grid."""
    config = robustness_config(tmp_path)
    results = fake_results(0.01)
    empty_log = pd.DataFrame({"date": [], "x": []}).set_index("date")
    save_robustness(config, 1, results, empty_log, empty_log, empty_log)
    reloaded = load_or_run_robustness(config, 1)
    assert set(reloaded) == set(results)
    pd.testing.assert_series_equal(
        reloaded[("(1) emp x emp", "unconstrained")].portfolio_returns,
        results[("(1) emp x emp", "unconstrained")].portfolio_returns)


def test_horizon_dm_table_compares_only_garch_cells(tmp_path):
    """horizon_dm_table ne compare que les cellules GARCH (D_t change), à
    contrainte de poids fixée, et produit des statistiques finies."""
    config = robustness_config(tmp_path)
    empty_log = pd.DataFrame({"date": [], "x": []}).set_index("date")
    save_robustness(config, 1, fake_results(0.01), empty_log, empty_log, empty_log)
    save_robustness(config, 21, fake_results(0.012), empty_log, empty_log, empty_log)
    table = horizon_dm_table(config, 1, 21)
    assert set(table.index) == set(GARCH_MODELS)
    assert np.isfinite(table[["mean_diff", "DM", "pvalue"]].values).all()


def test_by_weight_extracts_ordered_subgrid():
    """by_weight isole une contrainte de poids en un dict {modele: resultat}
    dans l'ordre canonique des six cellules, consommable par src.evaluation."""
    results = fake_results(0.01)
    subset = by_weight(results, "long_only")
    assert list(subset) == ALL_MODELS
    assert subset["(1) emp x emp"] is results[("(1) emp x emp", "long_only")]


def fake_garch_log() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=3),
                         "fallback_frac": [0.0, 0.1, 0.2],
                         "fit_seconds": [1.0, 2.0, 3.0]}).set_index("date")


def fake_dcc_log() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=3),
                         "persistence": [0.9, 0.95, 0.99],
                         "fit_seconds": [0.5, 0.6, 0.7]}).set_index("date")


def test_diagnostics_summary_averages_logged_columns(tmp_path):
    """diagnostics_summary moyenne les colonnes deja journalisees par
    GarchCache/DccCache (fallback, persistance, temps de calcul)."""
    config = robustness_config(tmp_path)
    empty_log = pd.DataFrame({"date": [], "x": []}).set_index("date")
    save_robustness(config, 1, fake_results(0.01), empty_log,
                    fake_garch_log(), fake_dcc_log())
    row = diagnostics_summary(config, 1)
    assert row["n_windows_fitted"] == 3
    assert row["fallback_frac_mean"] == pytest.approx(0.1)
    assert row["dcc_persistence_mean"] == pytest.approx(0.9466666, rel=1e-4)


def test_collect_robustness_tables_indexes_by_horizon_weight_period(tmp_path):
    """collect_robustness_tables agrege sommaire/attribution/tests/economie
    sur les deux horizons, les deux contraintes de poids et les sous-periodes,
    en reutilisant load_or_run_robustness (aucun recalcul GARCH/DCC)."""
    config = robustness_config(tmp_path)
    empty_log = pd.DataFrame({"date": [], "x": []}).set_index("date")
    save_robustness(config, 1, fake_results(0.01), empty_log,
                    fake_garch_log(), fake_dcc_log())
    save_robustness(config, 21, fake_results(0.012), empty_log,
                    fake_garch_log(), fake_dcc_log())
    summary, effects, tests, economic, diagnostics = collect_robustness_tables(config)
    assert summary.index.names == ["horizon", "weight", "period", "model"]
    assert set(summary.index.get_level_values("horizon")) == {1, 21}
    assert set(summary.index.get_level_values("weight")) == {"unconstrained", "long_only"}
    assert set(summary.index.get_level_values("period")) == {"full", "pre", "post"}
    assert effects.index.names == ["horizon", "weight", "period", "effect"]
    assert tests.index.names == ["horizon", "weight", "period", "comparison"]
    assert economic.index.names == ["horizon", "weight", "period", "comparison",
                                   "cost", "gamma"]
    assert list(diagnostics.index) == [1, 21]


def test_collect_robustness_tables_multi_n_adds_n_index(tmp_path):
    """collect_robustness_tables_multi_n ajoute un niveau d'index N en tete,
    une dimension a la fois, sans muter la config source (meme principe que
    src.experiments.run_experiments pour la grille principale)."""
    base_config = robustness_config(tmp_path)
    empty_log = pd.DataFrame({"date": [], "x": []}).set_index("date")
    for n in (10, 20):
        local = deepcopy(base_config)
        local["robustness"]["n_assets"] = n
        save_robustness(local, 1, fake_results(0.01), empty_log,
                        fake_garch_log(), fake_dcc_log())
        save_robustness(local, 21, fake_results(0.012), empty_log,
                        fake_garch_log(), fake_dcc_log())
    summary, effects, tests, economic, diagnostics = \
        collect_robustness_tables_multi_n(base_config, n_grid=(10, 20))
    assert summary.index.names[0] == "N"
    assert set(summary.index.get_level_values("N")) == {10, 20}
    assert effects.index.names[0] == "N"
    assert tests.index.names[0] == "N"
    assert economic.index.names[0] == "N"
    assert set(diagnostics.index.get_level_values("N")) == {10, 20}
    assert base_config["robustness"]["n_assets"] == 10  # config source non mutee
