"""Brique D : modèles de variances (volatilités conditionnelles par actif).

Deux modèles : écart-type empirique de la fenêtre, et GARCH(1,1) par actif
(package arch) avec prévision de volatilité à 1 pas. Le fit GARCH fournit
aussi les résidus dévolatilisés s_t = r_t / d_t, inputs du DCC.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd


@dataclass
class GarchPanel:
    """Sorties du fit GARCH(1,1) sur un panel : prévisions de vol à 1 pas
    (unités journalières), résidus standardisés, persistance, nb de fallbacks."""
    vol_forecast: np.ndarray
    std_resid: pd.DataFrame
    persistence: np.ndarray
    n_fallback: int


def empirical_vol(window: pd.DataFrame) -> np.ndarray:
    """Écart-type empirique de chaque actif sur la fenêtre (ddof=1)."""
    return window.std(ddof=1).values


def _fit_one_garch(y: np.ndarray, p: int, q: int, dist: str, horizon: int
                   ) -> tuple[float, np.ndarray, float] | None:
    """GARCH(p,q) zéro-mean sur une série (déjà en %) ; None si échec ou
    persistance >= 1. Retourne (vol_forecast, resid_std, persistance)."""
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            am = arch_model(y, mean="Zero", vol="GARCH", p=p, q=q,
                            dist=dist, rescale=False)
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
            pers = float(sum(res.params[f"alpha[{k+1}]"] for k in range(p)) +
                         sum(res.params[f"beta[{k+1}]"] for k in range(q)))
            if not np.isfinite(pers) or pers >= 1.0:
                return None
            fcast = res.forecast(horizon=horizon, reindex=False)
            vol = float(np.sqrt(fcast.variance.values[-1].mean()))
            return vol, np.asarray(res.std_resid), pers
        except Exception:  # noqa: BLE001 — fallback empirique géré en amont
            return None


def fit_garch_panel(window: pd.DataFrame, garch_cfg: dict) -> GarchPanel:
    """Fit GARCH(1,1) actif par actif sur la fenêtre (rendements démoyennés).

    Fallback : si le fit échoue ou est non stationnaire, vol empirique et
    résidus standardisés naïvement pour cet actif.
    """
    scale = garch_cfg["rescale"]
    demeaned = (window - window.mean()) * scale
    vols, resids, pers = [], {}, []
    n_fallback = 0
    series = [demeaned[col].values for col in demeaned.columns]
    fit = partial(_fit_one_garch, p=garch_cfg["p"], q=garch_cfg["q"],
                  dist=garch_cfg["dist"],
                  horizon=garch_cfg["forecast_horizon"])
    with ThreadPoolExecutor(max_workers=garch_cfg["n_jobs"]) as executor:
        outputs = list(executor.map(fit, series))
    for col, y, out in zip(demeaned.columns, series, outputs):
        if out is None:
            n_fallback += 1
            s = y.std(ddof=1)
            vols.append(s / scale)
            resids[col] = y / s
            pers.append(np.nan)
        else:
            vol, sr, p_ = out
            vols.append(vol / scale)
            resids[col] = sr
            pers.append(p_)
    std_resid = pd.DataFrame(resids, index=window.index)
    return GarchPanel(np.array(vols), std_resid, np.array(pers), n_fallback)
