"""Robustesses : GMV long-only, GARCH horizon 21j, diagnostics.

Ne touche jamais aux caches/tables principaux (outputs/backtest_n{n}.pkl,
outputs/tables/*). Chaque run est tagué par son horizon GARCH : horizon=1
rejoue le protocole principal en instrumenté (diagnostics Σ_t, fallbacks,
alpha/beta, poids long-only, absents du cache principal) ; horizon=21 est la
robustesse proprement dite. Usage :

    python -m src.robustness --config config.yaml --horizon 1
    python -m src.robustness --config config.yaml --horizon 21
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.backtest import (BacktestResult, gmv_weights, gmv_weights_long_only,
                          load_data, run_backtest_multi_constrained, summarize)
from src.evaluation import (compare_models, dm_table, economic_table,
                            factorial_attribution, model_losses)
from src.experiments import export_table, periods
from src.grid import build_grid_with_diagnostics

WEIGHT_FNS = {"unconstrained": gmv_weights, "long_only": gmv_weights_long_only}
CELL_NAMES = ["(1) emp x emp", "(2) emp x lw", "(3) emp x dcc",
             "(4) garch x emp", "(5) garch x lw", "(6) garch x dcc"]


def garch_cfg_for_horizon(config: dict, horizon: int) -> dict:
    """Copie la config GARCH principale avec un forecast_horizon différent."""
    garch_cfg = deepcopy(config["models"]["garch"])
    garch_cfg["forecast_horizon"] = horizon
    return garch_cfg


def robustness_paths(config: dict, horizon: int) -> dict[str, Path]:
    """Chemins de cache/diagnostics propres à un horizon (jamais les
    fichiers principaux)."""
    rob, n = config["robustness"], config["robustness"]["n_assets"]
    return {
        "cache": Path(rob["cache_pattern"].format(n=n, horizon=horizon)),
        "garch_log": Path(rob["diagnostics_garch_pattern"].format(n=n, horizon=horizon)),
        "dcc_log": Path(rob["diagnostics_dcc_pattern"].format(n=n, horizon=horizon)),
        "sigma_log": Path(rob["diagnostics_sigma_pattern"].format(n=n, horizon=horizon)),
    }


def run_robustness_grid(config: dict, horizon: int
                        ) -> tuple[dict[tuple[str, str], BacktestResult],
                                  pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exécute la grille D x R instrumentée (Σ_t, fallbacks, alpha/beta) avec
    poids non contraint ET long-only, pour l'horizon GARCH demandé."""
    returns, dvol = load_data(config)
    bt = config["backtest"]
    garch_cfg = garch_cfg_for_horizon(config, horizon)
    grid, garch_cache, _lw_cache, dcc_cache = build_grid_with_diagnostics(
        garch_cfg, config["models"]["dcc"])
    results, sigma_log = run_backtest_multi_constrained(
        returns, dvol, grid, WEIGHT_FNS, bt["window"], bt["rebalance"],
        config["robustness"]["n_assets"])
    garch_log = pd.DataFrame(garch_cache.log).set_index("date")
    dcc_log = (pd.DataFrame(dcc_cache.log).set_index("date")
              if dcc_cache is not None else pd.DataFrame())
    return results, sigma_log, garch_log, dcc_log


def save_robustness(config: dict, horizon: int,
                    results: dict[tuple[str, str], BacktestResult],
                    sigma_log: pd.DataFrame, garch_log: pd.DataFrame,
                    dcc_log: pd.DataFrame) -> None:
    """Écrit le cache et les diagnostics sous des noms tagués par l'horizon."""
    paths = robustness_paths(config, horizon)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, paths["cache"])
    sigma_log.to_csv(paths["sigma_log"])
    garch_log.to_csv(paths["garch_log"])
    dcc_log.to_csv(paths["dcc_log"])


def load_or_run_robustness(config: dict, horizon: int
                           ) -> dict[tuple[str, str], BacktestResult]:
    """Charge le cache de robustesse s'il existe, sinon calcule et sauvegarde."""
    paths = robustness_paths(config, horizon)
    if paths["cache"].exists():
        return pd.read_pickle(paths["cache"])
    results, sigma_log, garch_log, dcc_log = run_robustness_grid(config, horizon)
    save_robustness(config, horizon, results, sigma_log, garch_log, dcc_log)
    return results


def horizon_dm_table(config: dict, horizon_a: int, horizon_b: int,
                     weight: str = "unconstrained") -> pd.DataFrame:
    """Compare deux horizons GARCH à modèle et contrainte de poids fixés
    (test EC/DM standard et pondéré, pour chaque cellule GARCH x R)."""
    res_a = load_or_run_robustness(config, horizon_a)
    res_b = load_or_run_robustness(config, horizon_b)
    ev, rows = config["evaluation"], {}
    garch_models = [k for k, w in res_a if w == weight and "garch" in k]
    for model in garch_models:
        standard = compare_models(res_a[(model, weight)], res_b[(model, weight)],
                                  ev["dm_lags"], ev["center_returns"])
        weighted = compare_models(res_a[(model, weight)], res_b[(model, weight)],
                                  ev["dm_lags"], ev["center_returns"], True)
        rows[model] = {"mean_diff": standard.mean_difference,
                       "DM": standard.statistic, "pvalue": standard.pvalue,
                       "DM_weighted": weighted.statistic,
                       "pvalue_weighted": weighted.pvalue}
    return pd.DataFrame(rows).T


def by_weight(results: dict[tuple[str, str], BacktestResult], weight: str
             ) -> dict[str, BacktestResult]:
    """Sous-grille {modèle: BacktestResult} pour une contrainte de poids
    donnée, dans l'ordre canonique des six cellules (réutilisable par toutes
    les fonctions de src.evaluation, qui attendent ce type de dict)."""
    return {model: results[(model, weight)] for model in CELL_NAMES
           if (model, weight) in results}


def diagnostics_summary(config: dict, horizon: int) -> dict[str, float]:
    """Moyennes des diagnostics GARCH/DCC (fallback, persistance, temps de
    calcul) déjà journalisés pour un horizon donné."""
    paths = robustness_paths(config, horizon)
    garch = pd.read_csv(paths["garch_log"])
    row = {"n_windows_fitted": len(garch),
          "fallback_frac_mean": garch["fallback_frac"].mean(),
          "garch_fit_seconds_mean": garch["fit_seconds"].mean()}
    dcc = pd.read_csv(paths["dcc_log"])
    if not dcc.empty:
        row.update({"dcc_persistence_mean": dcc["persistence"].mean(),
                   "dcc_persistence_min": dcc["persistence"].min(),
                   "dcc_persistence_max": dcc["persistence"].max(),
                   "dcc_fit_seconds_mean": dcc["fit_seconds"].mean()})
    return row


def collect_robustness_tables(config: dict, horizons: tuple[int, ...] = (1, 21),
                              weights: tuple[str, ...] = ("unconstrained", "long_only")
                              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame, pd.DataFrame]:
    """Agrège métriques (SD/TO/Gross...), attribution D/R, tests EC/DM et
    valeur économique pour chaque horizon GARCH x contrainte de poids x
    sous-période, en réutilisant exactement l'infrastructure de src.experiments
    (src.experiments.periods, src.evaluation.*). La comparaison unconstrained/long_only et
    horizon 1/21 se lit sur les mêmes tableaux que la grille principale."""
    days = config["backtest"]["trading_days"]
    summary_rows, effect_rows, dm_rows, econ_rows, diag_rows = [], [], [], [], []
    for horizon in horizons:
        results = load_or_run_robustness(config, horizon)
        diag_rows.append({"horizon": horizon, **diagnostics_summary(config, horizon)})
        for weight in weights:
            for period, subset in periods(config, by_weight(results, weight)).items():
                for model, result in subset.items():
                    summary_rows.append({"horizon": horizon, "weight": weight,
                                         "period": period, "model": model,
                                         **summarize(result, days)})
                effects = factorial_attribution(model_losses(subset))
                effect_rows += [{"horizon": horizon, "weight": weight,
                                "period": period, "effect": key,
                                "loss_difference": value}
                               for key, value in effects.items()]
                tests = dm_table(subset, config).reset_index(names="comparison")
                tests.insert(0, "period", period)
                tests.insert(0, "weight", weight)
                tests.insert(0, "horizon", horizon)
                dm_rows.append(tests)
                econ = economic_table(subset, config).reset_index()
                econ.insert(0, "period", period)
                econ.insert(0, "weight", weight)
                econ.insert(0, "horizon", horizon)
                econ_rows.append(econ)
    summary = pd.DataFrame(summary_rows).set_index(
        ["horizon", "weight", "period", "model"])
    effects = pd.DataFrame(effect_rows).set_index(
        ["horizon", "weight", "period", "effect"])
    tests = pd.concat(dm_rows).set_index(
        ["horizon", "weight", "period", "comparison"])
    economic = pd.concat(econ_rows).set_index(
        ["horizon", "weight", "period", "comparison", "cost", "gamma"])
    diagnostics = pd.DataFrame(diag_rows).set_index("horizon")
    return summary, effects, tests, economic, diagnostics


def collect_robustness_tables_multi_n(
        config: dict, n_grid: tuple[int, ...] = (50, 100, 200)
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                  pd.DataFrame, pd.DataFrame]:
    """Étend collect_robustness_tables à plusieurs dimensions N, en ajoutant
    un niveau d'index "N" (même principe que src.experiments : copie de la
    config avec n_assets substitué, jamais de mutation partagée)."""
    per_n = {}
    for n in n_grid:
        local = deepcopy(config)
        local["robustness"]["n_assets"] = n
        per_n[n] = collect_robustness_tables(local)
    summary = pd.concat({n: t[0] for n, t in per_n.items()}, names=["N"])
    effects = pd.concat({n: t[1] for n, t in per_n.items()}, names=["N"])
    tests = pd.concat({n: t[2] for n, t in per_n.items()}, names=["N"])
    economic = pd.concat({n: t[3] for n, t in per_n.items()}, names=["N"])
    diagnostics = pd.concat({n: t[4] for n, t in per_n.items()}, names=["N"])
    return summary, effects, tests, economic, diagnostics


if __name__ == "__main__":
    from src.data import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--horizon", type=int,
                        help="calcule uniquement cet horizon puis quitte "
                             "(omis : rapport complet 1 vs 21, contraintes et "
                             "sous-périodes, à partir des caches existants)")
    parser.add_argument("--multi-n", action="store_true",
                        help="étend le rapport complet à robustness.n_assets_grid "
                             "(50/100/200) au lieu du seul robustness.n_assets")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.multi_n:
        summary_, effects_, tests_, economic_, diag_ = \
            collect_robustness_tables_multi_n(cfg, tuple(cfg["robustness"]["n_assets_grid"]))
        directory = Path(cfg["robustness"]["tables_dir"])
        for name_, table_ in [("summary_robustness_multi_n", summary_),
                              ("attribution_robustness_multi_n", effects_),
                              ("dm_tests_robustness_multi_n", tests_),
                              ("economic_robustness_multi_n", economic_),
                              ("diagnostics_robustness_multi_n", diag_)]:
            export_table(table_, name_, cfg, directory=directory)
        print(summary_.to_string())
    elif args.horizon is not None:
        results_, sigma_, garch_, dcc_ = run_robustness_grid(cfg, args.horizon)
        save_robustness(cfg, args.horizon, results_, sigma_, garch_, dcc_)
        print(f"horizon={args.horizon} | cellules: {sorted(set(k for k, _ in results_))}")
        print(f"fenetres GARCH fittees: {len(garch_)} | fallback moyen: "
             f"{garch_['fallback_frac'].mean():.3f}")
        if not dcc_.empty:
            print(f"persistance DCC (alpha+beta) : min={dcc_['persistence'].min():.4f} "
                 f"max={dcc_['persistence'].max():.4f} mean={dcc_['persistence'].mean():.4f}")
    else:
        summary_, effects_, tests_, economic_, diag_ = collect_robustness_tables(cfg)
        horizon_effect_ = pd.concat(
            {w: horizon_dm_table(cfg, 1, 21, weight=w)
            for w in ("unconstrained", "long_only")}, names=["weight", "model"])
        directory = Path(cfg["robustness"]["tables_dir"])
        for name_, table_ in [("summary_robustness", summary_),
                              ("attribution_robustness", effects_),
                              ("dm_tests_robustness", tests_),
                              ("economic_robustness", economic_),
                              ("diagnostics_robustness", diag_),
                              ("horizon_effect_garch", horizon_effect_)]:
            export_table(table_, name_, cfg, directory=directory)
        print(summary_.to_string())
