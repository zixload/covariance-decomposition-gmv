"""Tests du backtest GMV — dont le test anti-look-ahead (règle n°1)."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import (empirical_cov, gmv_weights, gmv_weights_long_only,
                          run_backtest, run_backtest_multi_constrained,
                          select_universe, sigma_diagnostics)


@pytest.fixture
def panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Petit panel synthétique : 6 actifs, 400 jours, vols hétérogènes."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2015-01-01", periods=400)
    cols = list("ABCDEF")
    vols = np.array([0.01, 0.015, 0.02, 0.012, 0.018, 0.025])
    returns = pd.DataFrame(rng.normal(0, vols, (400, 6)), idx, cols)
    dvol = pd.DataFrame(rng.uniform(1e6, 1e9, (400, 6)), idx, cols)
    return returns, dvol


def test_gmv_weights_sum_and_variance():
    """Règle n°3b : les poids somment à 1 et Var(w) = 1/(1'Σ⁻¹1)."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 5))
    sigma = np.cov(a, rowvar=False)
    w = gmv_weights(sigma)
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    var_w = w @ sigma @ w
    var_th = 1.0 / (np.ones(5) @ np.linalg.solve(sigma, np.ones(5)))
    assert np.isclose(var_w, var_th, atol=1e-14)
    # le GMV bat tout autre portefeuille pleinement investi
    w_eq = np.full(5, 0.2)
    assert var_w <= w_eq @ sigma @ w_eq + 1e-15


def test_no_look_ahead(panel):
    """Règle n°1 : l'estimateur ne voit JAMAIS de données >= date de rebal."""
    returns, dvol = panel
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy_estimator(window: pd.DataFrame) -> np.ndarray:
        seen.append((window.index[0], window.index[-1]))
        return empirical_cov(window)

    run_backtest(returns, dvol, spy_estimator, window=250, rebalance=21,
                 n_assets=4)
    rebal_positions = range(250, len(returns), 21)
    for (start, last), i in zip(seen, rebal_positions):
        assert last < returns.index[i], "l'estimation voit la date de rebal"
        assert start == returns.index[i - 250]
    assert len(seen) == len(list(rebal_positions))


def test_future_data_irrelevant(panel):
    """Contre-épreuve : modifier les données APRÈS t ne change pas les poids
    décidés en t."""
    returns, dvol = panel
    res1 = run_backtest(returns, dvol, empirical_cov, 250, 21, 4)
    corrupted = returns.copy()
    corrupted.iloc[300:] *= 100.0          # on saccage le futur du 1er rebal
    res2 = run_backtest(corrupted, dvol, empirical_cov, 250, 21, 4)
    d1 = list(res1.weights)[0]
    pd.testing.assert_series_equal(res1.weights[d1], res2.weights[d1])


def test_universe_selection(panel):
    """La sélection retourne bien les N plus gros dollar-volumes de la fenêtre."""
    returns, dvol = panel
    univ = select_universe(dvol, i=250, window=250, n_assets=3)
    expected = dvol.iloc[0:250].mean().nlargest(3).index.tolist()
    assert univ == expected


def test_turnover_is_finite_and_nonnegative(panel):
    """Le turnover est fini et positif, sans borne avec ventes à découvert."""
    returns, dvol = panel
    res = run_backtest(returns, dvol, empirical_cov, 250, 21, 4)
    to = res.turnover.iloc[1:]
    assert np.isfinite(to).all()
    assert (to >= 0).all()


def test_predicted_variance_is_aligned_and_positive(panel):
    """La variance GMV prévue couvre exactement chaque rendement réalisé."""
    returns, dvol = panel
    res = run_backtest(returns, dvol, empirical_cov, 250, 21, 4)
    assert res.predicted_variance.index.equals(res.portfolio_returns.index)
    assert np.isfinite(res.predicted_variance).all()
    assert (res.predicted_variance > 0).all()


def test_gmv_long_only_weights_are_valid_and_dominated():
    """Robustesse : le long-only somme à 1, est non-négatif, et sa variance
    minimale est au moins celle du GMV non contraint (contrainte active)."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 5))
    sigma = np.cov(a, rowvar=False)
    w_lo = gmv_weights_long_only(sigma)
    assert np.isclose(w_lo.sum(), 1.0, atol=1e-8)
    assert (w_lo >= -1e-10).all()
    var_lo = w_lo @ sigma @ w_lo
    w_unc = gmv_weights(sigma)
    var_unc = w_unc @ sigma @ w_unc
    assert var_lo >= var_unc - 1e-12


def test_sigma_diagnostics_matches_eigenvalues():
    """min_eig et cond_number reproduisent numpy sur une matrice connue."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(300, 4))
    sigma = np.cov(a, rowvar=False)
    diag = sigma_diagnostics(sigma)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.isclose(diag["min_eig"], eigvals.min())
    assert np.isclose(diag["cond_number"], eigvals.max() / eigvals.min())


def test_run_backtest_multi_constrained_shares_sigma(panel):
    """Les contraintes partagent Σ_t : le long-only reste dominé par le
    non-contraint sur CHAQUE fenêtre, et les diagnostics couvrent chaque
    (date, modèle)."""
    returns, dvol = panel
    weight_fns = {"unconstrained": gmv_weights,
                  "long_only": gmv_weights_long_only}
    results, diagnostics = run_backtest_multi_constrained(
        returns, dvol, {"c1": empirical_cov}, weight_fns, 250, 21, 4)
    assert set(results) == {("c1", "unconstrained"), ("c1", "long_only")}
    unc = results[("c1", "unconstrained")].predicted_variance
    lo = results[("c1", "long_only")].predicted_variance
    assert (lo >= unc - 1e-10).all()
    n_rebal = len(list(range(250, len(returns), 21)))
    assert len(diagnostics) == n_rebal
    assert (diagnostics["min_eig"] > 0).all()
    assert (diagnostics["cond_number"] >= 1.0).all()
