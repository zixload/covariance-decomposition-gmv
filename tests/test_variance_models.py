"""Tests de la brique D — dont persistance GARCH < 1 (règle n°3c)."""
import numpy as np
import pandas as pd
import pytest

from src.variance_models import empirical_vol, fit_garch_panel

GARCH_CFG = {"p": 1, "q": 1, "dist": "normal", "rescale": 100,
             "n_jobs": 2, "forecast_horizon": 1}


def simulate_garch(t: int, n: int, seed: int = 7) -> pd.DataFrame:
    """Simule n séries GARCH(1,1) (omega=2e-6, alpha=0.08, beta=0.90)."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    z = rng.standard_normal((t, n))
    h = np.full(n, omega / (1 - alpha - beta))
    r = np.empty((t, n))
    for i in range(t):
        r[i] = np.sqrt(h) * z[i]
        h = omega + alpha * r[i] ** 2 + beta * h
    idx = pd.bdate_range("2015-01-01", periods=t)
    return pd.DataFrame(r, idx, [f"A{k}" for k in range(n)])


@pytest.fixture(scope="module")
def garch_fit() -> tuple[pd.DataFrame, "object"]:
    window = simulate_garch(1250, 4)
    return window, fit_garch_panel(window, GARCH_CFG)


def test_empirical_vol_matches_std():
    """empirical_vol == écart-type colonne par colonne (ddof=1)."""
    window = simulate_garch(300, 3)
    np.testing.assert_allclose(empirical_vol(window),
                               window.std(ddof=1).values, atol=1e-15)


def test_garch_persistence_below_one(garch_fit):
    """Règle n°3c : alpha + beta < 1 pour chaque actif après estimation."""
    _, panel = garch_fit
    assert panel.n_fallback == 0
    assert np.isfinite(panel.persistence).all()
    assert (panel.persistence < 1.0).all()
    assert (panel.persistence > 0.5).all()  # données très persistantes


def test_garch_outputs_consistent(garch_fit):
    """Vols positives en unités journalières, résidus ~ N(0,1), shapes OK."""
    window, panel = garch_fit
    assert panel.vol_forecast.shape == (4,)
    assert (panel.vol_forecast > 0).all()
    # la prévision de vol reste dans un ordre de grandeur raisonnable
    ratio = panel.vol_forecast / window.std(ddof=1).values
    assert ((ratio > 0.2) & (ratio < 5.0)).all()
    assert panel.std_resid.shape == window.shape
    np.testing.assert_allclose(panel.std_resid.std(ddof=1), 1.0, atol=0.1)


def test_garch_fallback_on_degenerate_series():
    """Une série dégénérée déclenche le fallback empirique, sans crash."""
    window = simulate_garch(1250, 2)
    window["BAD"] = 0.0
    window.iloc[600, window.columns.get_loc("BAD")] = 0.5  # un seul choc
    panel = fit_garch_panel(window, GARCH_CFG)
    assert panel.vol_forecast.shape == (3,)
    assert np.isfinite(panel.vol_forecast).all()
    assert (panel.vol_forecast > 0).all()
