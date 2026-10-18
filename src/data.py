"""Téléchargement, nettoyage et construction du panel de rendements log.

Usage : python -m src.data --config config.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Charge le fichier de configuration YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _download_batch(tickers: list[str], start: str, end: str,
                    max_retries: int) -> pd.DataFrame:
    """Télécharge un lot de tickers via yfinance, avec retries."""
    import yfinance as yf

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                             progress=False, threads=False, group_by="column")
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 — on retente puis on signale
            print(f"  tentative {attempt}/{max_retries} echouee: {exc}")
        time.sleep(2 * attempt)
    return pd.DataFrame()


def download_prices(tickers: list[str], start: str, end: str, raw_dir: str,
                    batch_size: int, max_retries: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Télécharge prix ajustés et volumes par lots, avec cache parquet.

    Retourne (close, dollar_volume), colonnes = tickers.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    closes, dvols = [], []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        cache = raw / f"batch_{i:03d}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            print(f"telechargement lot {i // batch_size + 1} "
                  f"({batch[0]}..{batch[-1]})")
            df = _download_batch(batch, start, end, max_retries)
            if df.empty:
                print(f"  LOT VIDE, ignore: {batch}")
                continue
            df.to_parquet(cache)
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
        vol = df["Volume"]
        closes.append(close)
        dvols.append(close * vol)
    close_all = pd.concat(closes, axis=1)
    dvol_all = pd.concat(dvols, axis=1)
    return close_all.sort_index(), dvol_all.sort_index()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rendements log journaliers : r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def clean_panel(prices: pd.DataFrame, start_max: str, max_missing_pct: float
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclut les tickers à historique incomplet ; retourne (prix, rapport).

    Règles : premier prix valide <= start_max, part de NaN <= max_missing_pct.
    Les NaN résiduels (jours fériés décalés, etc.) sont forward-fillés.
    """
    prices = prices.dropna(axis=1, how="all")
    first_valid = prices.apply(lambda s: s.first_valid_index())
    frac_nan = prices.isna().mean()
    keep = (first_valid <= pd.Timestamp(start_max)) & \
           (frac_nan <= max_missing_pct)
    report = pd.DataFrame({"first_valid": first_valid, "frac_nan": frac_nan,
                           "kept": keep}).sort_values("first_valid")
    clean = prices.loc[:, keep[keep].index].ffill().dropna(axis=0, how="any")
    return clean, report


def descriptive_stats(returns: pd.DataFrame, trading_days: int
                      ) -> pd.DataFrame:
    """Stats descriptives par actif : moyenne/vol annualisées, skew, kurtosis."""
    stats = pd.DataFrame({
        "mean_ann_pct": returns.mean() * trading_days * 100,
        "vol_ann_pct": returns.std() * np.sqrt(trading_days) * 100,
        "skew": returns.skew(),
        "kurtosis": returns.kurt(),
        "min_pct": returns.min() * 100,
        "max_pct": returns.max() * 100,
    })
    return stats.round(2)


def build_dataset(config: dict) -> pd.DataFrame:
    """Pipeline complet : téléchargement -> nettoyage -> rendements -> parquet."""
    cfg = config["data"]
    close, dvol = download_prices(cfg["tickers"], cfg["start"], cfg["end"],
                                  cfg["raw_dir"], cfg["batch_size"],
                                  cfg["max_retries"])
    prices, report = clean_panel(close, cfg["start_max"],
                                 cfg["max_missing_pct"])
    returns = compute_log_returns(prices)
    dvol = dvol.reindex(index=prices.index, columns=prices.columns).ffill()
    for path, df in [(cfg["processed_path"], returns),
                     (cfg["prices_path"], prices),
                     (cfg["dollar_volume_path"], dvol)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    n_excl = int((~report["kept"]).sum())
    print(f"\ntickers demandes: {len(cfg['tickers'])} | retenus: "
          f"{returns.shape[1]} | exclus: {n_excl}")
    print(f"periode effective: {returns.index[0].date()} -> "
          f"{returns.index[-1].date()} ({len(returns)} jours)")
    excluded = report.loc[~report["kept"]]
    if not excluded.empty:
        print("exclus:", ", ".join(excluded.index.astype(str)))
    stats = descriptive_stats(returns, config["backtest"]["trading_days"])
    stats.to_csv(cfg["stats_path"])
    print("\nstats descriptives (moyenne cross-section):")
    print(stats.mean().round(2).to_string())
    avg_corr = returns.corr().values[np.triu_indices(returns.shape[1], 1)]
    print(f"correlation moyenne de paire: {avg_corr.mean():.3f}")
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    build_dataset(load_config(args.config))
