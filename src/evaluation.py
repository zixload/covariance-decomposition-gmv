"""Évaluation Engle-Colacito, attribution D/R et valeur économique."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS

from src.backtest import BacktestResult, load_data, run_backtest_multi
from src.grid import build_grid


@dataclass(frozen=True)
class DMResult:
    """Résultat du test de différence moyenne de pertes avec variance HAC."""
    mean_difference: float
    statistic: float
    pvalue: float
    nobs: int


def squared_losses(returns: pd.Series, center: bool = True) -> pd.Series:
    """Construit les pertes quadratiques, centrées pour cibler la variance."""
    clean = returns.dropna()
    values = clean - clean.mean() if center else clean
    return values.pow(2)


def dm_hac(loss_difference: pd.Series, maxlags: int) -> DMResult:
    """Teste E[u_t]=0 par régression sur constante et covariance Newey-West."""
    values = loss_difference.dropna().values
    fit = OLS(values, np.ones((len(values), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    statistic = float(fit.tvalues[0])
    return DMResult(float(fit.params[0]), statistic,
                    float(2.0 * norm.sf(abs(statistic))), len(values))


def compare_models(first: BacktestResult, second: BacktestResult,
                   maxlags: int, center: bool = True,
                   weighted: bool = False) -> DMResult:
    """Compare first-second; un écart positif favorise le second modèle."""
    frame = pd.concat([first.portfolio_returns, second.portfolio_returns,
                       first.predicted_variance, second.predicted_variance],
                      axis=1, join="inner").dropna()
    loss1 = squared_losses(frame.iloc[:, 0], center)
    loss2 = squared_losses(frame.iloc[:, 1], center)
    difference = loss1 - loss2
    if weighted:
        scale = np.sqrt(2.0 * frame.iloc[:, 2] * frame.iloc[:, 3])
        difference = difference / scale.clip(lower=np.finfo(float).eps)
    return dm_hac(difference, maxlags)


def model_losses(results: dict[str, BacktestResult], center: bool = True
                 ) -> dict[str, float]:
    """Retourne la perte quadratique moyenne de chaque cellule de la grille."""
    return {name: float(squared_losses(result.portfolio_returns, center).mean())
            for name, result in results.items()}


def factorial_attribution(loss: dict[str, float]) -> dict[str, float]:
    """Décompose les effets D, R et leurs interactions sur la perte réalisée."""
    c1, c2 = loss["(1) emp x emp"], loss["(2) emp x lw"]
    c3, c4 = loss["(3) emp x dcc"], loss["(4) garch x emp"]
    c5, c6 = loss["(5) garch x lw"], loss["(6) garch x dcc"]
    return {
        "D|R=emp": c4 - c1, "D|R=lw": c5 - c2,
        "D|R=dcc": c6 - c3, "R=lw|D=emp": c2 - c1,
        "R=lw|D=garch": c5 - c4, "R=dcc|D=emp": c3 - c1,
        "R=dcc|D=garch": c6 - c4,
        "interaction_DxLW": c5 - c4 - c2 + c1,
        "interaction_DxDCC_emp": c6 - c4 - c3 + c1,
        "interaction_DxDCC_lw": c6 - c5 - c3 + c2,
    }


def net_returns(result: BacktestResult, cost: float,
                charge_initial: bool = False) -> pd.Series:
    """Déduit le coût proportionnel au turnover aux dates de rebalancement."""
    net = result.portfolio_returns.copy()
    turnover = result.turnover.copy()
    if not charge_initial and not turnover.empty:
        turnover.iloc[0] = 0.0
    common = net.index.intersection(turnover.index)
    net.loc[common] -= cost * turnover.loc[common]
    return net


def quadratic_utility(returns: pd.Series, gamma: float) -> pd.Series:
    """Utilité quadratique de Fleming-Kirby-Ostdiek sur richesse brute."""
    wealth = 1.0 + returns
    return wealth - gamma * wealth.pow(2) / (2.0 * (1.0 + gamma))


def performance_fee(reference: pd.Series, candidate: pd.Series, gamma: float,
                    trading_days: int, daily_bound: float) -> float:
    """Calcule le fee annuel en points de base payé pour passer au candidat."""
    frame = pd.concat([reference, candidate], axis=1, join="inner").dropna()
    target = float(quadratic_utility(frame.iloc[:, 0], gamma).mean())
    objective = lambda fee: float(
        quadratic_utility(frame.iloc[:, 1] - fee, gamma).mean() - target)
    fee = brentq(objective, -daily_bound, daily_bound)
    return float(fee * trading_days * 10_000)


def comparison_pairs() -> dict[str, tuple[str, str]]:
    """Paires identifiant les effets de D et R dans la grille complète."""
    return {
        "D|R=emp": ("(4) garch x emp", "(1) emp x emp"),
        "D|R=lw": ("(5) garch x lw", "(2) emp x lw"),
        "D|R=dcc": ("(6) garch x dcc", "(3) emp x dcc"),
        "R=lw|D=emp": ("(2) emp x lw", "(1) emp x emp"),
        "R=lw|D=garch": ("(5) garch x lw", "(4) garch x emp"),
        "R=dcc|D=emp": ("(3) emp x dcc", "(1) emp x emp"),
        "R=dcc|D=garch": ("(6) garch x dcc", "(4) garch x emp"),
    }


def economic_table(results: dict[str, BacktestResult], config: dict
                   ) -> pd.DataFrame:
    """Calcule SD nette et performance fee pour tous les contrastes D/R."""
    ev, days = config["evaluation"], config["backtest"]["trading_days"]
    rows = []
    for label, (candidate, reference) in comparison_pairs().items():
        for cost in ev["costs"]:
            ref = net_returns(results[reference], cost,
                              ev["charge_initial_turnover"])
            cand = net_returns(results[candidate], cost,
                               ev["charge_initial_turnover"])
            for gamma in ev["gammas"]:
                fee = performance_fee(ref, cand, gamma, days,
                                      ev["fee_daily_bound"])
                rows.append({"comparison": label, "cost": cost,
                             "gamma": gamma, "fee_bps_ann": fee,
                             "candidate_SD_pct": cand.std() * np.sqrt(days) * 100,
                             "reference_SD_pct": ref.std() * np.sqrt(days) * 100})
    return pd.DataFrame(rows).set_index(["comparison", "cost", "gamma"])


def dm_table(results: dict[str, BacktestResult], config: dict) -> pd.DataFrame:
    """Construit les tests EC/DM standard et pondérés des contrastes D/R."""
    ev, rows = config["evaluation"], {}
    for label, (first, second) in comparison_pairs().items():
        standard = compare_models(results[first], results[second],
                                  ev["dm_lags"], ev["center_returns"])
        weighted = compare_models(results[first], results[second],
                                  ev["dm_lags"], ev["center_returns"], True)
        rows[label] = {"mean_diff": standard.mean_difference,
                       "DM": standard.statistic, "pvalue": standard.pvalue,
                       "DM_weighted": weighted.statistic,
                       "pvalue_weighted": weighted.pvalue}
    return pd.DataFrame(rows).T


def run_evaluation(config: dict) -> tuple[dict[str, BacktestResult], pd.DataFrame]:
    """Exécute la grille et retourne les tests EC/DM standard et pondérés."""
    returns, dvol = load_data(config)
    bt, models = config["backtest"], config["models"]
    grid = build_grid(models["garch"], models["dcc"])
    results = run_backtest_multi(returns, dvol, grid, bt["window"],
                                 bt["rebalance"], bt["n_assets"])
    return results, dm_table(results, config)


if __name__ == "__main__":
    from src.data import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_, tests_ = run_evaluation(cfg)
    output = Path(cfg["outputs"]["backtest_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results_, output)
    attribution_ = pd.Series(factorial_attribution(model_losses(results_)),
                             name="loss_difference")
    economic_ = economic_table(results_, cfg)
    for key, table in [("dm_tests_path", tests_),
                       ("attribution_path", attribution_),
                       ("economic_path", economic_)]:
        path = Path(cfg["outputs"][key])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path)
    print(tests_.to_string())
    print(attribution_.to_string())
    print(economic_.to_string())
