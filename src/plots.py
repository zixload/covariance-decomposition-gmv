"""Figures PDF des résultats de dimension et d'attribution D/R."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import BacktestResult


def _prepare_path(path: str | Path) -> Path:
    """Crée le dossier parent et retourne un chemin normalisé."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_sd_by_dimension(summary: pd.DataFrame, path: str | Path) -> None:
    """Trace la SD annualisée de chaque cellule selon la dimension N."""
    full = summary.xs("full", level="period")["SD_pct"].unstack("model")
    ax = full.plot(marker="o", figsize=(9, 5))
    ax.set(xlabel="Nombre d'actifs N", ylabel="SD réalisée annualisée (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(_prepare_path(path), format="pdf")
    plt.close(ax.figure)


def plot_attribution(attribution: pd.DataFrame, path: str | Path) -> None:
    """Trace les effets de perte D et R selon la dimension de l'univers."""
    effects = ["D|R=emp", "D|R=lw", "R=lw|D=emp", "R=lw|D=garch"]
    full = attribution.xs("full", level="period")["loss_difference"]
    frame = full.unstack("effect")[effects]
    ax = frame.plot(kind="bar", figsize=(9, 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Nombre d'actifs N", ylabel="Différence de perte quadratique")
    ax.legend(fontsize=8, ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(_prepare_path(path), format="pdf")
    plt.close(ax.figure)


def plot_cumulative_wealth(results: dict[str, BacktestResult],
                           path: str | Path) -> None:
    """Trace la richesse cumulée brute des six portefeuilles GMV."""
    wealth = pd.DataFrame({name: (1.0 + result.portfolio_returns).cumprod()
                           for name, result in results.items()})
    ax = wealth.plot(figsize=(9, 5), logy=True)
    ax.set(xlabel="Date", ylabel="Richesse cumulée (échelle log)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(_prepare_path(path), format="pdf")
    plt.close(ax.figure)
