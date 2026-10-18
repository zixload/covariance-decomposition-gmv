"""Brique R : modèles de la matrice de corrélation.

Chaque modèle prend un panel de résidus dévolatilisés s_t = D_t^{-1} r_t
(DataFrame T x N) et retourne une matrice de corrélation N x N (diag = 1).
Trois modèles : empirique, Ledoit-Wolf (2004) et DCC scalaire (Engle, 2002).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def normalize_correlation(q: np.ndarray) -> np.ndarray:
    """Renormalise une matrice positive en corrélation de diagonale unité."""
    scale = np.sqrt(np.clip(np.diag(q), np.finfo(float).eps, None))
    corr = q / np.outer(scale, scale)
    return (corr + corr.T) / 2


def empirical_corr(resid: pd.DataFrame) -> np.ndarray:
    """Corrélation empirique de la fenêtre de résidus standardisés."""
    return np.corrcoef(resid.values, rowvar=False)


def ledoit_wolf_corr(resid: pd.DataFrame,
                     diagnostics: list | None = None) -> np.ndarray:
    """Corrélation shrinkée Ledoit-Wolf (cible : identité scalaire).

    LedoitWolf est appliqué aux résidus re-standardisés (variance unitaire),
    puis la covariance shrinkée est renormalisée en corrélation exacte.

    Si `diagnostics` est fourni (liste), l'intensité de shrinkage
    (attribut `.shrinkage_` de sklearn, l'estimateur de rho) y est ajoutée ;
    par défaut (`None`), rien n'est journalisé et le calcul est identique à
    avant (aucun changement de comportement pour les appelants existants).
    """
    from sklearn.covariance import LedoitWolf

    z = (resid - resid.mean()) / resid.std(ddof=1)
    lw = LedoitWolf(assume_centered=False).fit(z.values)
    cov = lw.covariance_
    if diagnostics is not None:
        diagnostics.append(float(lw.shrinkage_))
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def dcc_filter(resid: pd.DataFrame, alpha: float, beta: float,
               target: np.ndarray | None = None) -> np.ndarray:
    """Filtre les corrélations DCC conditionnelles pour chaque observation."""
    values = resid.values
    c = empirical_corr(resid) if target is None else target
    q = c.copy()
    path = np.empty((len(resid), resid.shape[1], resid.shape[1]))
    for t in range(len(resid)):
        path[t] = normalize_correlation(q)
        shock = np.outer(values[t], values[t])
        q = (1.0 - alpha - beta) * c + alpha * shock + beta * q
    return path


def _full_dcc_loss(path: np.ndarray, values: np.ndarray) -> float:
    """Log-vraisemblance gaussienne de corrélation, à constante près."""
    loss = 0.0
    for corr, obs in zip(path, values):
        sign, logdet = np.linalg.slogdet(corr)
        if sign <= 0:
            return np.inf
        loss += logdet + obs @ np.linalg.solve(corr, obs)
    return 0.5 * float(loss)


def _composite_dcc_loss(resid: pd.DataFrame, alpha: float,
                        beta: float) -> float:
    """Vraisemblance composite adjacente filtrée en temps linéaire."""
    values = resid.values
    target = empirical_corr(resid)
    q_diag = np.ones(values.shape[1])
    q_pair = np.diag(target, k=1).copy()
    c_pair = q_pair.copy()
    loss = 0.0
    for obs in values:
        rho = q_pair / np.sqrt(q_diag[:-1] * q_diag[1:])
        den = np.clip(1.0 - rho ** 2, np.finfo(float).eps, None)
        pair = obs[:-1] ** 2 - 2.0 * rho * obs[:-1] * obs[1:] + obs[1:] ** 2
        loss += float(np.log(den).sum() + (pair / den).sum())
        q_diag = (1.0 - alpha - beta) + alpha * obs ** 2 + beta * q_diag
        q_pair = ((1.0 - alpha - beta) * c_pair
                  + alpha * obs[:-1] * obs[1:] + beta * q_pair)
    return 0.5 * loss


def dcc_loss(params: np.ndarray, resid: pd.DataFrame,
             likelihood: str = "full") -> float:
    """Calcule la perte DCC complète ou composite en (alpha, beta)."""
    alpha, beta = params
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        return np.inf
    if likelihood == "composite":
        return _composite_dcc_loss(resid, alpha, beta)
    return _full_dcc_loss(dcc_filter(resid, alpha, beta), resid.values)


def fit_dcc(resid: pd.DataFrame, cfg: dict) -> tuple[float, float]:
    """Estime le DCC scalaire sous alpha,beta >= 0 et alpha+beta borné."""
    mode = cfg["likelihood"]
    if mode == "auto":
        mode = "composite" if resid.shape[1] >= cfg["composite_threshold"] else "full"
    width = resid.shape[1] - 1 if mode == "composite" else resid.shape[1]
    scale = len(resid) * width
    objective = lambda x: dcc_loss(x, resid, mode) / scale
    constraint = {"type": "ineq",
                  "fun": lambda x: cfg["bound_sum"] - x[0] - x[1]}
    result = minimize(objective, [cfg["alpha0"], cfg["beta0"]],
                      method=cfg["optimizer"],
                      bounds=[(0, cfg["bound_sum"])] * 2,
                      constraints=constraint,
                      options={"maxiter": cfg["maxiter"], "ftol": cfg["ftol"]})
    if not result.success:
        raise RuntimeError(f"échec estimation DCC: {result.message}")
    return float(result.x[0]), float(result.x[1])


def dcc_forecast(resid: pd.DataFrame, alpha: float, beta: float) -> np.ndarray:
    """Prévoit R au prochain jour après la dernière observation de la fenêtre."""
    c = empirical_corr(resid)
    q = c.copy()
    for obs in resid.values:
        shock = np.outer(obs, obs)
        q = (1.0 - alpha - beta) * c + alpha * shock + beta * q
    return normalize_correlation(q)
