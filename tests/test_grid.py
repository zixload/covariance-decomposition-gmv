"""Tests de la grille : reconstruction DRD (règle n°3a) et cohérence des
cellules avec le backtest de référence."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import empirical_cov, run_backtest, run_backtest_multi
from src.grid import (DccCache, GarchCache, LwCache, assemble_sigma,
                      build_grid, build_grid_with_diagnostics, make_estimator)

GARCH_CFG = {"p": 1, "q": 1, "dist": "normal", "rescale": 100,
             "n_jobs": 2, "forecast_horizon": 1}
DCC_CFG = {"alpha0": 0.05, "beta0": 0.90, "bound_sum": 0.999,
           "optimizer": "SLSQP", "maxiter": 100, "ftol": 1e-7,
           "likelihood": "full", "composite_threshold": 30,
           "composite_pairs": "adjacent"}


@pytest.fixture
def panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel synthétique corrélé : 6 actifs, 400 jours."""
    rng = np.random.default_rng(42)
    n, t = 6, 400
    corr = np.full((n, n), 0.4)
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.01, 0.015, 0.02, 0.012, 0.018, 0.025])
    cov = corr * np.outer(vols, vols)
    idx = pd.bdate_range("2015-01-01", periods=t)
    cols = list("ABCDEF")
    returns = pd.DataFrame(rng.multivariate_normal(np.zeros(n), cov, t),
                           idx, cols)
    dvol = pd.DataFrame(rng.uniform(1e6, 1e9, (t, n)), idx, cols)
    return returns, dvol


def test_drd_reconstruction():
    """Règle n°3a : diag(d)·R·diag(d) reconstruit Σ à 1e-10 près."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 5)) * [1, 2, 3, 4, 5]
    sigma = np.cov(a, rowvar=False)
    d = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(d, d)
    np.testing.assert_allclose(assemble_sigma(d, corr), sigma, atol=1e-10)


def test_cell1_equals_empirical_cov(panel):
    """La cellule (1) emp x emp redonne exactement la covariance empirique."""
    returns, _ = panel
    window = returns.iloc[:250]
    est = make_estimator("emp", "emp")
    np.testing.assert_allclose(est(window), empirical_cov(window), atol=1e-10)


def test_multi_matches_single(panel):
    """run_backtest_multi reproduit run_backtest estimateur par estimateur."""
    returns, dvol = panel
    single = run_backtest(returns, dvol, empirical_cov, 250, 21, 4)
    multi = run_backtest_multi(returns, dvol,
                               {"c1": make_estimator("emp", "emp")},
                               250, 21, 4)["c1"]
    pd.testing.assert_series_equal(single.portfolio_returns,
                                   multi.portfolio_returns)
    pd.testing.assert_series_equal(single.turnover, multi.turnover)


def test_garch_cache_shared_between_cells(panel):
    """Les cellules (4) et (5) ne déclenchent qu'UN fit GARCH par fenêtre."""
    returns, dvol = panel
    cache = GarchCache(GARCH_CFG)
    estimators = {"(4)": make_estimator("garch", "emp", cache),
                  "(5)": make_estimator("garch", "lw", cache)}
    run_backtest_multi(returns, dvol, estimators, 250, 21, 3)
    n_rebal = len(range(250, len(returns), 21))
    assert cache.n_fits == n_rebal


def test_grid_cells_produce_valid_sigma(panel):
    """Chaque cellule produit une Σ symétrique définie positive."""
    returns, _ = panel
    window = returns.iloc[:250]
    for name, est in build_grid(GARCH_CFG).items():
        sigma = est(window)
        assert sigma.shape == (6, 6), name
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
        assert np.linalg.eigvalsh(sigma).min() > 0, name


def test_dcc_cells_share_same_correlation(panel):
    """Les cellules (3) et (6) changent D mais partagent exactement R_DCC."""
    returns, _ = panel
    window = returns.iloc[:250, :3]
    garch = GarchCache(GARCH_CFG)
    dcc = DccCache(DCC_CFG, garch)
    sigma3 = make_estimator("emp", "dcc", garch, dcc)(window)
    sigma6 = make_estimator("garch", "dcc", garch, dcc)(window)
    corr3 = sigma3 / np.outer(np.sqrt(np.diag(sigma3)),
                              np.sqrt(np.diag(sigma3)))
    corr6 = sigma6 / np.outer(np.sqrt(np.diag(sigma6)),
                              np.sqrt(np.diag(sigma6)))
    np.testing.assert_allclose(corr3, corr6, atol=1e-12)


def test_garch_cache_logs_one_row_per_fit(panel):
    """Le diagnostic GARCH loggue une ligne par fenêtre fittée, avec les
    bonnes clés et un temps de calcul positif."""
    returns, _ = panel
    window = returns.iloc[:250]
    cache = GarchCache(GARCH_CFG)
    cache.get(window)
    assert len(cache.log) == 1
    row = cache.log[0]
    assert row["n_assets"] == window.shape[1]
    assert 0 <= row["n_fallback"] <= window.shape[1]
    assert 0.0 <= row["fallback_frac"] <= 1.0
    assert row["fit_seconds"] > 0.0


def test_dcc_cache_logs_alpha_beta(panel):
    """Le diagnostic DCC loggue alpha, beta et leur somme sous contrainte."""
    returns, _ = panel
    window = returns.iloc[:250, :3]
    garch = GarchCache(GARCH_CFG)
    dcc = DccCache(DCC_CFG, garch)
    dcc.get(window)
    assert len(dcc.log) == 1
    row = dcc.log[0]
    assert row["alpha"] >= 0
    assert row["beta"] >= 0
    assert row["persistence"] == row["alpha"] + row["beta"]
    assert row["persistence"] < DCC_CFG["bound_sum"]


def test_build_grid_with_diagnostics_exposes_caches(panel):
    """build_grid_with_diagnostics retourne la même grille que build_grid,
    plus des caches qui se remplissent pendant le backtest."""
    returns, dvol = panel
    grid_plain = build_grid(GARCH_CFG, DCC_CFG)
    grid_diag, garch_cache, lw_cache, dcc_cache = build_grid_with_diagnostics(
        GARCH_CFG, DCC_CFG)
    assert set(grid_plain) == set(grid_diag)
    assert garch_cache.log == []
    assert lw_cache.log == []
    run_backtest_multi(returns, dvol, grid_diag, 250, 21, 3)
    n_rebal = len(list(range(250, len(returns), 21)))
    assert len(garch_cache.log) == n_rebal
    assert len(dcc_cache.log) == n_rebal
    # deux cellules LW ((2) emp x lw et (5) garch x lw) journalisent chacune
    assert len(lw_cache.log) == 2 * n_rebal
    d_models = {row["d_model"] for row in lw_cache.log}
    assert d_models == {"emp", "garch"}


def test_lw_cache_logs_rho_separately_per_d_model(panel):
    """LwCache journalise rho séparément pour D=empirique et D=GARCH, car
    les résidus en entrée (donc rho) diffèrent d'une brique D à l'autre."""
    returns, _ = panel
    window = returns.iloc[:250]
    cache = LwCache()
    estimator_emp = make_estimator("emp", "lw", lw_cache=cache)
    garch = GarchCache(GARCH_CFG)
    estimator_garch = make_estimator("garch", "lw", garch, lw_cache=cache)
    estimator_emp(window)
    estimator_garch(window)
    assert len(cache.log) == 2
    assert {row["d_model"] for row in cache.log} == {"emp", "garch"}
    for row in cache.log:
        assert 0.0 <= row["rho"] <= 1.0


def test_lw_cache_none_keeps_estimator_unchanged(panel):
    """Sans lw_cache (par défaut), make_estimator('emp','lw') reproduit
    exactement le comportement d'avant (aucune régression)."""
    returns, _ = panel
    window = returns.iloc[:250]
    plain = make_estimator("emp", "lw")(window)
    with_cache = make_estimator("emp", "lw", lw_cache=LwCache())(window)
    np.testing.assert_allclose(plain, with_cache, atol=1e-14)
