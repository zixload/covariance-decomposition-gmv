"""Backtest GMV à fenêtre glissante (squelette commun à toute la grille).

Convention anti-look-ahead : à la date de rebalancement t (position i dans
l'index), la fenêtre d'estimation est returns.iloc[i-window:i], soit
STRICTEMENT les rendements antérieurs à t. Les poids sont appliqués aux
rendements des jours i à i+rebalance-1.

Usage : python -m src.backtest --config config.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CovEstimator = Callable[[pd.DataFrame], np.ndarray]
WeightFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class BacktestResult:
    """Sorties du backtest : rendements simples journaliers bruts, poids,
    turnover et variance GMV prédite."""
    portfolio_returns: pd.Series
    weights: dict = field(repr=False)
    turnover: pd.Series
    concentration: pd.Series
    predicted_variance: pd.Series


def empirical_cov(window: pd.DataFrame) -> np.ndarray:
    """Covariance empirique de la fenêtre (estimateur de référence)."""
    return np.cov(window.values, rowvar=False)


def gmv_weights(sigma: np.ndarray) -> np.ndarray:
    """Poids minimum-variance globaux : w = Σ⁻¹1 / (1'Σ⁻¹1)."""
    ones = np.ones(sigma.shape[0])
    try:
        x = np.linalg.solve(sigma, ones)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(sigma) @ ones
    return x / x.sum()


def gmv_weights_long_only(sigma: np.ndarray) -> np.ndarray:
    """Poids minimum-variance long-only : min w'Σw sous 1'w=1, w>=0 (SLSQP).

    Robustesse au GMV non contraint (résultat principal) : les contraintes
    régularisent implicitement l'optimisation.
    """
    n = sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    constraint = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    result = minimize(lambda w: w @ sigma @ w, w0, method="SLSQP",
                      bounds=[(0.0, None)] * n, constraints=constraint,
                      options={"maxiter": 500, "ftol": 1e-14})
    if not result.success:
        raise RuntimeError(f"echec GMV long-only: {result.message}")
    return result.x / result.x.sum()


def sigma_diagnostics(sigma: np.ndarray) -> dict[str, float]:
    """Valeur propre minimale et nombre de conditionnement de Σ (règle 3.1)."""
    eigvals = np.linalg.eigvalsh(sigma)
    return {"min_eig": float(eigvals.min()),
            "cond_number": float(np.linalg.cond(sigma))}


def select_universe(dollar_volume: pd.DataFrame, i: int, window: int,
                    n_assets: int) -> list[str]:
    """Top-N tickers par dollar-volume moyen sur la fenêtre d'estimation."""
    dv = dollar_volume.iloc[i - window:i].mean()
    return dv.nlargest(n_assets).index.tolist()


def _drift(weights: pd.Series, simple_returns: pd.DataFrame
           ) -> tuple[pd.Series, pd.Series]:
    """Rendements du portefeuille buy-and-hold et poids dérivés en fin de
    période (nombre de titres constant entre deux rebalancements)."""
    gross = (1 + simple_returns[weights.index]).cumprod()
    values = gross.mul(weights, axis=1)
    port_value = values.sum(axis=1)
    port_ret = port_value.pct_change()
    port_ret.iloc[0] = port_value.iloc[0] - 1.0
    return port_ret, values.iloc[-1] / port_value.iloc[-1]


@dataclass
class _CellState:
    """Accumulateurs d'une cellule pendant la boucle de backtest."""
    parts: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    tos: dict = field(default_factory=dict)
    cos: dict = field(default_factory=dict)
    predicted_parts: list = field(default_factory=list)
    prev: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def update(self, w: pd.Series, date: pd.Timestamp,
               hold: pd.DataFrame, predicted_variance: float) -> None:
        """Enregistre poids, turnover, concentration puis fait dériver w."""
        self.weights[date] = w
        both = w.reindex(w.index.union(self.prev.index), fill_value=0.0)
        prev = self.prev.reindex(both.index, fill_value=0.0)
        self.tos[date] = float((both - prev).abs().sum())
        self.cos[date] = float(np.sqrt((w ** 2).sum()))
        port_ret, self.prev = _drift(w, hold)
        self.parts.append(port_ret)
        self.predicted_parts.append(pd.Series(predicted_variance,
                                              index=hold.index))

    def result(self) -> BacktestResult:
        return BacktestResult(pd.concat(self.parts), self.weights,
                              pd.Series(self.tos), pd.Series(self.cos),
                              pd.concat(self.predicted_parts))


def run_backtest_multi_constrained(returns: pd.DataFrame,
                                   dollar_volume: pd.DataFrame,
                                   estimators: dict[str, CovEstimator],
                                   weight_fns: dict[str, WeightFn],
                                   window: int, rebalance: int, n_assets: int
                                   ) -> tuple[dict[tuple[str, str], BacktestResult],
                                             pd.DataFrame]:
    """Backtest GMV multi-estimateurs x multi-contraintes de poids.

    Chaque Σ_t est calculée UNE seule fois par (estimateur, date), puis
    chaque fonction de poids de `weight_fns` (non contraint, long-only, ...)
    lui est appliquée : la comparaison entre contraintes porte donc sur une
    Σ_t strictement identique. Retourne aussi les diagnostics (valeur propre
    minimale, conditionnement) de chaque Σ_t.
    """
    simple = np.exp(returns) - 1.0
    states = {(name, w_name): _CellState()
             for name in estimators for w_name in weight_fns}
    diag_rows = []
    for i in range(window, len(returns), rebalance):
        univ = select_universe(dollar_volume, i, window, n_assets)
        est_window = returns.iloc[i - window:i][univ]
        assert est_window.index[-1] < returns.index[i], "look-ahead detecte"
        date = returns.index[i]
        hold = simple.iloc[i:i + rebalance]
        for name, estimator in estimators.items():
            sigma = estimator(est_window)
            diag_rows.append({"date": date, "model": name,
                              **sigma_diagnostics(sigma)})
            for w_name, w_fn in weight_fns.items():
                w = pd.Series(w_fn(sigma), index=univ)
                predicted_variance = float(w.values @ sigma @ w.values)
                states[(name, w_name)].update(w, date, hold, predicted_variance)
    results = {key: st.result() for key, st in states.items()}
    diagnostics = pd.DataFrame(diag_rows).set_index(["date", "model"])
    return results, diagnostics


def run_backtest_multi(returns: pd.DataFrame, dollar_volume: pd.DataFrame,
                       estimators: dict[str, CovEstimator], window: int,
                       rebalance: int, n_assets: int
                       ) -> dict[str, BacktestResult]:
    """Backtest GMV non contraint de plusieurs estimateurs en UNE passe
    (wrapper de run_backtest_multi_constrained). Même univers et même
    fenêtre pour toutes les cellules à chaque date (les estimateurs GARCH
    partagent ainsi leur cache de fits univariés)."""
    results, _ = run_backtest_multi_constrained(
        returns, dollar_volume, estimators, {"unconstrained": gmv_weights},
        window, rebalance, n_assets)
    return {name: results[(name, "unconstrained")] for name in estimators}


def run_backtest(returns: pd.DataFrame, dollar_volume: pd.DataFrame,
                 estimator: CovEstimator, window: int, rebalance: int,
                 n_assets: int) -> BacktestResult:
    """Backtest GMV d'un seul estimateur (wrapper de run_backtest_multi)."""
    return run_backtest_multi(returns, dollar_volume, {"_": estimator},
                              window, rebalance, n_assets)["_"]


def summarize(result: BacktestResult, trading_days: int) -> dict[str, float]:
    """Métriques annualisées : AV, SD, IR, turnover moyen, concentration."""
    r = result.portfolio_returns
    av = r.mean() * trading_days * 100
    sd = r.std() * np.sqrt(trading_days) * 100
    weights = list(result.weights.values())
    gross = np.mean([w.abs().sum() for w in weights]) if weights else np.nan
    shorts = np.mean([(w < 0).mean() for w in weights]) if weights else np.nan
    return {"AV_pct": round(float(av), 2), "SD_pct": round(float(sd), 2),
            "IR": round(float(av / sd), 3),
            "TO": round(float(result.turnover.iloc[1:].mean()), 3),
            "CO": round(float(result.concentration.mean()), 3),
            "Gross": round(float(gross), 3),
            "Short_frac": round(float(shorts), 3)}


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge rendements et dollar-volume depuis les parquets produits par src.data."""
    returns = pd.read_parquet(config["data"]["processed_path"])
    dvol = pd.read_parquet(config["data"]["dollar_volume_path"])
    return returns, dvol.reindex(returns.index).ffill()


if __name__ == "__main__":
    from src.data import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    returns, dvol = load_data(cfg)
    bt = cfg["backtest"]
    res = run_backtest(returns, dvol, empirical_cov, bt["window"],
                       bt["rebalance"], bt["n_assets"])
    print("GMV | covariance empirique | N =", bt["n_assets"])
    for k, v in summarize(res, bt["trading_days"]).items():
        print(f"  {k}: {v}")
