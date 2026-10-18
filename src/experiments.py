"""Boucle multi-dimension, sous-périodes et exports des tables."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.backtest import BacktestResult, load_data, run_backtest_multi, summarize
from src.evaluation import (dm_table, economic_table, factorial_attribution,
                            model_losses)
from src.grid import build_grid_with_diagnostics
from src.plots import (plot_attribution, plot_cumulative_wealth,
                       plot_sd_by_dimension)


def cache_path(config: dict, n_assets: int) -> Path:
    """Retourne le fichier de cache du backtest correspondant à N."""
    pattern = config["outputs"]["backtest_pattern"]
    return Path(pattern.format(n=n_assets))


def load_or_run(config: dict, n_assets: int) -> dict[str, BacktestResult]:
    """Charge un backtest en cache ou calcule les six cellules pour N.

    Lors d'un calcul frais (pas de cache), journalise aussi l'intensité de
    shrinkage Ledoit-Wolf (diagnostic rho par période) si
    `outputs.diagnostics_lw_pattern` est configuré."""
    path = cache_path(config, n_assets)
    if path.exists():
        return pd.read_pickle(path)
    returns, dvol = load_data(config)
    bt, models = config["backtest"], config["models"]
    grid, _garch_cache, lw_cache, _dcc_cache = build_grid_with_diagnostics(
        models["garch"], models["dcc"])
    results = run_backtest_multi(returns, dvol, grid, bt["window"],
                                 bt["rebalance"], n_assets)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, path)
    lw_pattern = config.get("outputs", {}).get("diagnostics_lw_pattern")
    if lw_cache.log and lw_pattern:
        lw_path = Path(lw_pattern.format(n=n_assets))
        lw_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(lw_cache.log).to_csv(lw_path, index=False)
    return results


def slice_result(result: BacktestResult, start: str,
                 end: str) -> BacktestResult:
    """Restreint toutes les sorties d'un backtest à une sous-période."""
    interval = slice(start, end)
    weights = {date: value for date, value in result.weights.items()
               if pd.Timestamp(start) <= date <= pd.Timestamp(end)}
    return BacktestResult(result.portfolio_returns.loc[interval], weights,
                          result.turnover.loc[interval],
                          result.concentration.loc[interval],
                          result.predicted_variance.loc[interval])


def periods(config: dict, results: dict[str, BacktestResult]
            ) -> dict[str, dict[str, BacktestResult]]:
    """Construit l'échantillon complet et toutes les sous-périodes configurées."""
    output = {"full": results}
    for name, (start, end) in config["evaluation"]["subperiods"].items():
        output[name] = {model: slice_result(result, start, end)
                        for model, result in results.items()}
    return output


def collect_tables(config: dict, all_results: dict[int, dict[str, BacktestResult]]
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Agrège métriques, attribution et tests DM pour tous N et régimes."""
    summary_rows, effect_rows, dm_rows = [], [], []
    days = config["backtest"]["trading_days"]
    for n_assets, results in all_results.items():
        for period, subset in periods(config, results).items():
            for model, result in subset.items():
                summary_rows.append({"N": n_assets, "period": period,
                                     "model": model, **summarize(result, days)})
            effects = factorial_attribution(model_losses(subset))
            effect_rows += [{"N": n_assets, "period": period, "effect": key,
                             "loss_difference": value}
                            for key, value in effects.items()]
            tests = dm_table(subset, config).reset_index(names="comparison")
            tests.insert(0, "period", period)
            tests.insert(0, "N", n_assets)
            dm_rows.append(tests)
    summary = pd.DataFrame(summary_rows).set_index(["N", "period", "model"])
    effects = pd.DataFrame(effect_rows).set_index(["N", "period", "effect"])
    tests = pd.concat(dm_rows).set_index(["N", "period", "comparison"])
    return summary, effects, tests


def export_table(table: pd.DataFrame, name: str, config: dict,
                 directory: Path | None = None) -> None:
    """Exporte un tableau en CSV et LaTeX booktabs (dossier de `config`
    par défaut, ou `directory` explicite pour les robustesses)."""
    directory = directory or Path(config["outputs"]["tables_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"{name}.csv")
    table.to_latex(directory / f"{name}.tex", float_format="%.6g",
                   multicolumn=True, multirow=True, bold_rows=False)


def collect_economic(config: dict,
                     all_results: dict[int, dict[str, BacktestResult]]
                     ) -> pd.DataFrame:
    """Agrège coûts et performance fees pour chaque N et sous-période."""
    rows = []
    for n_assets, results in all_results.items():
        for period, subset in periods(config, results).items():
            table = economic_table(subset, config).reset_index()
            table.insert(0, "period", period)
            table.insert(0, "N", n_assets)
            rows.append(table)
    return pd.concat(rows).set_index(
        ["N", "period", "comparison", "cost", "gamma"])


def run_experiments(config: dict) -> dict[int, dict[str, BacktestResult]]:
    """Exécute ou recharge toutes les dimensions configurées."""
    output = {}
    for n_assets in config["backtest"]["n_assets_grid"]:
        local = deepcopy(config)
        local["backtest"]["n_assets"] = n_assets
        output[n_assets] = load_or_run(local, n_assets)
    return output


if __name__ == "__main__":
    from src.data import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_ = run_experiments(cfg)
    summary_, attribution_, tests_ = collect_tables(cfg, results_)
    economic_ = collect_economic(cfg, results_)
    for name_, table_ in [("summary_all", summary_),
                          ("attribution_all", attribution_),
                          ("dm_tests_all", tests_),
                          ("economic_all", economic_)]:
        export_table(table_, name_, cfg)
    figures = Path(cfg["outputs"]["figures_dir"])
    plot_sd_by_dimension(summary_, figures / "sd_by_dimension.pdf")
    plot_attribution(attribution_, figures / "attribution_by_dimension.pdf")
    plot_cumulative_wealth(results_[cfg["backtest"]["n_assets"]],
                           figures / "cumulative_wealth_n100.pdf")
