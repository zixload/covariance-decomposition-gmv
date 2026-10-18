"""Grille D x R : assemblage Σ = D·R·D pour chaque cellule.

Cellules :
  D \\ R      | empirique | Ledoit-Wolf | DCC
  empirique  |    (1)    |     (2)     |     (3)
  GARCH(1,1) |    (4)    |     (5)     |     (6)

Chaque modèle D fournit (d, s) : vols prévues et résidus dévolatilisés ;
le modèle R est estimé sur s, puis Σ = diag(d)·R·diag(d).

Usage : python -m src.grid --config config.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.backtest import CovEstimator, load_data, run_backtest_multi, summarize
from src.correlation_models import (dcc_forecast, empirical_corr, fit_dcc,
                                    ledoit_wolf_corr)
from src.variance_models import GarchPanel, fit_garch_panel

R_MODELS = {"emp": empirical_corr, "lw": ledoit_wolf_corr}


def assemble_sigma(d: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Reconstruction Σ = diag(d) · R · diag(d)."""
    return corr * np.outer(d, d)


def empirical_d(window: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Modèle D empirique : vols = écart-type de la fenêtre, résidus
    dévolatilisés s_t = (r_t - r̄) / d."""
    d = window.std(ddof=1)
    return d.values, (window - window.mean()) / d


class GarchCache:
    """Mémoïse le dernier fit GARCH de panel : à une même date de
    rebalancement, les cellules (4)-(6) partagent les fits univariés.

    `log` accumule un diagnostic par fenêtre effectivement fittée : nombre
    et proportion de fallbacks, persistance moyenne/max, temps de calcul.
    """

    def __init__(self, garch_cfg: dict) -> None:
        self.cfg = garch_cfg
        self._key: tuple | None = None
        self._panel: GarchPanel | None = None
        self.n_fits = 0
        self.log: list[dict] = []

    def get(self, window: pd.DataFrame) -> GarchPanel:
        """Fit du panel, recalculé seulement si la fenêtre a changé."""
        key = (window.index[0], window.index[-1], tuple(window.columns))
        if key != self._key:
            t0 = time.perf_counter()
            self._panel = fit_garch_panel(window, self.cfg)
            elapsed = time.perf_counter() - t0
            n = len(window.columns)
            pers = self._panel.persistence
            self.log.append({
                "date": window.index[-1], "n_assets": n,
                "n_fallback": self._panel.n_fallback,
                "fallback_frac": self._panel.n_fallback / n,
                "persistence_mean": float(np.nanmean(pers)),
                "persistence_max": float(np.nanmax(pers)),
                "fit_seconds": elapsed,
            })
            self._key = key
            self.n_fits += 1
        return self._panel


class LwCache:
    """Journalise l'intensité de shrinkage Ledoit-Wolf à chaque cellule LW
    effectivement calculée, séparément pour D=empirique et D=GARCH (les
    résidus en entrée diffèrent, donc rho aussi).

    `log` accumule (date, d_model, rho) par appel.
    """

    def __init__(self) -> None:
        self.log: list[dict] = []

    def corr(self, resid: pd.DataFrame, d_model: str) -> np.ndarray:
        """Calcule R_LW et journalise rho pour cette fenêtre/ce modèle D."""
        diagnostics: list[float] = []
        corr = ledoit_wolf_corr(resid, diagnostics=diagnostics)
        self.log.append({"date": resid.index[-1], "d_model": d_model,
                         "rho": diagnostics[0]})
        return corr


class DccCache:
    """Mémorise estimation et prévision DCC pour la dernière fenêtre.

    `log` accumule (alpha, beta, alpha+beta, temps de calcul) par fenêtre
    effectivement réestimée.
    """

    def __init__(self, dcc_cfg: dict, garch_cache: GarchCache) -> None:
        self.cfg = dcc_cfg
        self.garch_cache = garch_cache
        self._key: tuple | None = None
        self._corr: np.ndarray | None = None
        self.params: tuple[float, float] | None = None
        self.log: list[dict] = []

    def get(self, window: pd.DataFrame) -> np.ndarray:
        """Retourne la prévision DCC, sans réestimer une même fenêtre."""
        key = (window.index[0], window.index[-1], tuple(window.columns))
        if key != self._key:
            t0 = time.perf_counter()
            resid = self.garch_cache.get(window).std_resid
            self.params = fit_dcc(resid, self.cfg)
            self._corr = dcc_forecast(resid, *self.params)
            elapsed = time.perf_counter() - t0
            alpha, beta = self.params
            self.log.append({"date": window.index[-1], "alpha": alpha,
                             "beta": beta, "persistence": alpha + beta,
                             "fit_seconds": elapsed})
            self._key = key
        return self._corr


def make_estimator(d_name: str, r_name: str,
                   garch_cache: GarchCache | None = None,
                   dcc_cache: DccCache | None = None,
                   lw_cache: LwCache | None = None) -> CovEstimator:
    """Fabrique l'estimateur de covariance d'une cellule (d_name, r_name).

    `lw_cache`, si fourni, journalise l'intensité de shrinkage rho pour les
    cellules LW (diagnostic rho par période) ; par défaut (`None`), le
    calcul est strictement identique à avant (aucun changement de
    comportement pour les appelants existants)."""
    if r_name == "dcc" and dcc_cache is None:
        raise ValueError("cellule DCC : dcc_cache requis")
    r_fn = R_MODELS.get(r_name)
    if d_name == "emp":
        def estimator(window: pd.DataFrame) -> np.ndarray:
            d, s = empirical_d(window)
            if r_name == "dcc":
                corr = dcc_cache.get(window)
            elif r_name == "lw" and lw_cache is not None:
                corr = lw_cache.corr(s, "emp")
            else:
                corr = r_fn(s)
            return assemble_sigma(d, corr)
    elif d_name == "garch":
        if garch_cache is None:
            raise ValueError("cellule GARCH : garch_cache requis")

        def estimator(window: pd.DataFrame) -> np.ndarray:
            panel = garch_cache.get(window)
            if r_name == "dcc":
                corr = dcc_cache.get(window)
            elif r_name == "lw" and lw_cache is not None:
                corr = lw_cache.corr(panel.std_resid, "garch")
            else:
                corr = r_fn(panel.std_resid)
            return assemble_sigma(panel.vol_forecast, corr)
    else:
        raise ValueError(f"modele D inconnu: {d_name}")
    return estimator


def build_grid(garch_cfg: dict, dcc_cfg: dict | None = None
               ) -> dict[str, CovEstimator]:
    """Construit les quatre cellules statiques et, si configuré, les DCC."""
    garch = GarchCache(garch_cfg)
    grid = {
        "(1) emp x emp": make_estimator("emp", "emp"),
        "(2) emp x lw": make_estimator("emp", "lw"),
    }
    if dcc_cfg is not None:
        dcc = DccCache(dcc_cfg, garch)
        grid["(3) emp x dcc"] = make_estimator("emp", "dcc", garch, dcc)
    grid["(4) garch x emp"] = make_estimator("garch", "emp", garch)
    grid["(5) garch x lw"] = make_estimator("garch", "lw", garch)
    if dcc_cfg is not None:
        grid["(6) garch x dcc"] = make_estimator("garch", "dcc", garch, dcc)
    return grid


def build_grid_with_diagnostics(garch_cfg: dict, dcc_cfg: dict | None = None
                                ) -> tuple[dict[str, CovEstimator], GarchCache,
                                          LwCache, DccCache | None]:
    """Comme build_grid, mais expose les caches pour lire leurs logs
    (fallbacks, persistance GARCH, alpha/beta DCC, rho LW, temps de calcul)."""
    garch = GarchCache(garch_cfg)
    lw = LwCache()
    grid = {"(1) emp x emp": make_estimator("emp", "emp"),
            "(2) emp x lw": make_estimator("emp", "lw", lw_cache=lw)}
    dcc = DccCache(dcc_cfg, garch) if dcc_cfg is not None else None
    if dcc is not None:
        grid["(3) emp x dcc"] = make_estimator("emp", "dcc", garch, dcc)
    grid["(4) garch x emp"] = make_estimator("garch", "emp", garch)
    grid["(5) garch x lw"] = make_estimator("garch", "lw", garch, lw_cache=lw)
    if dcc is not None:
        grid["(6) garch x dcc"] = make_estimator("garch", "dcc", garch, dcc)
    return grid, garch, lw, dcc


if __name__ == "__main__":
    from src.data import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])
    returns, dvol = load_data(cfg)
    bt = cfg["backtest"]
    results = run_backtest_multi(returns, dvol, build_grid(
                                 cfg["models"]["garch"], cfg["models"]["dcc"]),
                                 bt["window"], bt["rebalance"], bt["n_assets"])
    table = pd.DataFrame({name: summarize(res, bt["trading_days"])
                          for name, res in results.items()}).T
    print(f"\nGrille D x R | N = {bt['n_assets']} | T = {bt['window']}")
    print(table.to_string())
