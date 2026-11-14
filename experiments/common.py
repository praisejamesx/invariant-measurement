"""Experiment utilities: IO, metrics, plotting, and reproducibility helpers."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
for _p in (DATA, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# Colourblind-safe palette (Okabe-Ito)
PALETTE = {
    "raw": "#0072B2",
    "log": "#56B4E9",
    "snv": "#009E73",
    "invariant": "#D55E00",
    "calibrated": "#CC79A7",
    "batch_norm": "#E69F00",
    "chance": "#999999",
    "grey": "#666666",
}


def set_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "font.family": "DejaVu Sans",
        }
    )


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_json(obj, name: str, subdir: str = ""):
    path = RESULTS / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_csv(rows, name: str, subdir: str = ""):
    import pandas as pd

    path = RESULTS / subdir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def make_classifier(kind: str = "logreg", seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1, class_weight="balanced_subsample",
        )
    raise ValueError(kind)


def make_regressor(kind: str = "pls", n_components: int = 10):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "pls":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=n_components))
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(kind)


def pca_embed(X: np.ndarray, n_components: int = 2, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    return PCA(n_components=n_components, random_state=seed).fit_transform(Z)


def annotate(ax, text, loc="upper left"):
    ax.text(
        0.02 if "left" in loc else 0.98,
        0.98 if "upper" in loc else 0.02,
        text,
        transform=ax.transAxes,
        ha="left" if "left" in loc else "right",
        va="top" if "upper" in loc else "bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.8),
    )
