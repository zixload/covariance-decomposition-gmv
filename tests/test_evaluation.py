"""Tests du protocole Engle-Colacito et de l'attribution D/R."""
import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.evaluation import (compare_models, factorial_attribution,
                            net_returns, performance_fee)


def make_result(returns: pd.Series, variance: float = 0.0001
                ) -> BacktestResult:
    """Construit un résultat minimal aligné pour les tests d'évaluation."""
    dates = returns.index[::21]
    turnover = pd.Series([1.0] + [0.2] * (len(dates) - 1), index=dates)
    predicted = pd.Series(variance, index=returns.index)
    concentration = pd.Series(0.2, index=dates)
    return BacktestResult(returns, {}, turnover, concentration, predicted)


def test_dm_detects_lower_variance_candidate():
    """Un second portefeuille moins volatil donne first-second positif."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2010-01-01", periods=1500)
    shocks = pd.Series(rng.standard_normal(len(index)), index=index)
    high = make_result(0.02 * shocks, variance=0.02 ** 2)
    low = make_result(0.01 * shocks, variance=0.01 ** 2)
    result = compare_models(high, low, maxlags=22)
    assert result.mean_difference > 0
    assert result.statistic > 0
    assert result.pvalue < 0.01


def test_weighted_dm_is_finite():
    """La normalisation Engle-Colacito produit une statistique finie."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2015-01-01", periods=500)
    first = make_result(pd.Series(rng.normal(0, 0.02, 500), index), 0.0004)
    second = make_result(pd.Series(rng.normal(0, 0.015, 500), index), 0.000225)
    result = compare_models(first, second, 10, weighted=True)
    assert np.isfinite([result.mean_difference, result.statistic,
                        result.pvalue]).all()


def test_factorial_attribution_known_values():
    """Les contrastes D, R et interaction reproduisent l'algèbre attendue."""
    losses = {"(1) emp x emp": 10.0, "(2) emp x lw": 8.0,
              "(3) emp x dcc": 9.0, "(4) garch x emp": 7.0,
              "(5) garch x lw": 4.0, "(6) garch x dcc": 5.0}
    effects = factorial_attribution(losses)
    assert effects["D|R=emp"] == -3.0
    assert effects["R=lw|D=emp"] == -2.0
    assert effects["interaction_DxLW"] == -1.0
    assert effects["interaction_DxDCC_emp"] == -1.0


def test_transaction_cost_excludes_initial_trade():
    """Par défaut, le coût n'est déduit qu'après la constitution initiale."""
    index = pd.bdate_range("2020-01-01", periods=50)
    result = make_result(pd.Series(0.001, index=index))
    net = net_returns(result, cost=0.01, charge_initial=False)
    assert net.iloc[0] == 0.001
    assert np.isclose(net.loc[index[21]], 0.001 - 0.01 * 0.2)


def test_performance_fee_is_zero_for_identical_returns():
    """Deux stratégies identiques ont un performance fee nul."""
    index = pd.bdate_range("2020-01-01", periods=250)
    returns = pd.Series(np.sin(np.arange(250)) / 100, index=index)
    fee = performance_fee(returns, returns, gamma=10, trading_days=252,
                          daily_bound=0.10)
    assert np.isclose(fee, 0.0, atol=1e-6)
