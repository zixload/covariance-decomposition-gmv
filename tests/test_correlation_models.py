"""Tests de la brique R : corrélation empirique et Ledoit-Wolf."""
import numpy as np
import pandas as pd
import pytest

from src.correlation_models import (dcc_filter, dcc_forecast, dcc_loss,
                                    empirical_corr, fit_dcc,
                                    ledoit_wolf_corr)

DCC_CFG = {"alpha0": 0.05, "beta0": 0.90, "bound_sum": 0.999,
           "optimizer": "SLSQP", "maxiter": 100, "ftol": 1e-7,
           "likelihood": "full", "composite_threshold": 30,
           "composite_pairs": "adjacent"}


@pytest.fixture
def resid() -> pd.DataFrame:
    """Résidus corrélés (corrélation commune 0.3), T=500, N=8."""
    rng = np.random.default_rng(3)
    n = 8
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    z = rng.multivariate_normal(np.zeros(n), corr, size=500)
    idx = pd.bdate_range("2018-01-01", periods=500)
    return pd.DataFrame(z, idx, [f"A{k}" for k in range(n)])


def _check_is_correlation(r: np.ndarray) -> None:
    """Diag = 1, symétrie, valeurs propres > 0 (définie positive)."""
    np.testing.assert_allclose(np.diag(r), 1.0, atol=1e-12)
    np.testing.assert_allclose(r, r.T, atol=1e-12)
    assert np.linalg.eigvalsh(r).min() > 0


def test_empirical_corr_matches_corrcoef(resid):
    r = empirical_corr(resid)
    _check_is_correlation(r)
    np.testing.assert_allclose(r, np.corrcoef(resid.values, rowvar=False),
                               atol=1e-14)


def test_ledoit_wolf_is_correlation(resid):
    _check_is_correlation(ledoit_wolf_corr(resid))


def test_ledoit_wolf_shrinks_towards_identity(resid):
    """Le shrinkage rapproche les hors-diagonales de 0 (cible identité)."""
    off = ~np.eye(8, dtype=bool)
    r_emp = empirical_corr(resid)[off]
    r_lw = ledoit_wolf_corr(resid)[off]
    assert np.abs(r_lw).mean() < np.abs(r_emp).mean()
    # même signe et même structure, juste contractée
    assert np.corrcoef(r_emp, r_lw)[0, 1] > 0.99


def test_ledoit_wolf_positive_definite_when_n_exceeds_t():
    """N > T : la corrélation empirique est singulière, la LW reste DP."""
    rng = np.random.default_rng(11)
    z = pd.DataFrame(rng.standard_normal((40, 60)))
    r_lw = ledoit_wolf_corr(z)
    _check_is_correlation(r_lw)
    r_emp = empirical_corr(z)
    assert np.linalg.eigvalsh(r_emp).min() < 1e-10  # singulière


def test_dcc_path_is_positive_definite_with_unit_diagonal(resid):
    """Règle 3d : chaque R_t DCC est définie positive, avec diag(R_t)=1."""
    path = dcc_filter(resid, alpha=0.04, beta=0.94)
    for corr in path:
        _check_is_correlation(corr)


def test_dcc_estimation_respects_constraints(resid):
    """L'optimisation retourne alpha,beta positifs et alpha+beta<1."""
    alpha, beta = fit_dcc(resid.iloc[:, :3], DCC_CFG)
    assert alpha >= 0
    assert beta >= 0
    assert alpha + beta < DCC_CFG["bound_sum"]
    _check_is_correlation(dcc_forecast(resid.iloc[:, :3], alpha, beta))


def test_composite_loss_matches_filtered_pair_likelihood(resid):
    """Le filtre rapide reproduit la vraisemblance des paires adjacentes."""
    sample = resid.iloc[:50, :4]
    path = dcc_filter(sample, alpha=0.04, beta=0.94)
    rho = path[:, np.arange(3), np.arange(1, 4)]
    x, y = sample.values[:, :-1], sample.values[:, 1:]
    den = 1.0 - rho ** 2
    expected = 0.5 * np.sum(np.log(den) +
                            (x ** 2 - 2 * rho * x * y + y ** 2) / den)
    assert np.isclose(dcc_loss(np.array([0.04, 0.94]), sample, "composite"),
                      expected)
