"""
Raman case study: real spectra, synthetic instrument transforms.

Data
----
17 real Raman spectra of urine (7 healthy, 10 diseased) from the public
Raman-spectraAnalysis repository.  Each spectrum has 1000 points.  The sample
size is small; this case is a demonstration of the group calculus on real
spectral shapes, not a clinical result.

Instrument model
----------------
    x'(l) = g(l) * x(l) + o(l)
with smooth positive gain g and smooth fluorescence offset o, plus an optional
wavelength shift x'(l) = x(l - delta).  The first two are the per-wavelength
affine group; the shift is outside it.

The spectra are real.  The second instrument is synthetic and is labelled as
such in every figure and table.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (
    DATA, FIGURES, PALETTE, RESULTS, accuracy, make_classifier,
    save_json, set_style,
)
from invmath.invariants import per_wavelength_affine_invariants, snv

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

URL = "https://raw.githubusercontent.com/RimaMansour/Raman-spectraAnalysis/master/Urine/dataset"


def load():
    out = DATA / "raman"
    out.mkdir(parents=True, exist_ok=True)
    files = ["X_raman_healthy.npy", "X_raman_sick.npy",
             "y_labels_healthy.npy", "y_labels_sick.npy"]
    for f in files:
        p = out / f
        if not p.exists():
            import requests
            r = requests.get(f"{URL}/{f}", timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
    Xh = np.load(out / "X_raman_healthy.npy")
    Xs = np.load(out / "X_raman_sick.npy")
    X = np.vstack([Xh, Xs])
    y = np.array([0] * len(Xh) + [1] * len(Xs))
    return X, y


def conditioned_pair(X):
    """Pick the two reference spectra with the best-conditioned difference."""
    std = X.std(axis=0) + 1e-12
    Xz = (X - X.mean(axis=0)) / std
    best, score = (0, 1), -np.inf
    for a, b in itertools.combinations(range(len(X)), 2):
        s = float(np.percentile(np.abs(Xz[a] - Xz[b]), 10))
        if s > score:
            score, best = s, (a, b)
    return best


def apply_instrument(X, group, rng, strength=0.4):
    n, p = X.shape
    lam = np.linspace(0, 1, p)
    out = X.copy()
    if group in {"affine", "affine_shift"}:
        g = np.exp(strength * np.sin(2 * np.pi * lam + rng.uniform(0, 2 * np.pi)))
        o = 0.3 * np.exp(-3 * lam) + 0.03 * rng.normal(size=p)
        out = out * g[None, :] + o[None, :]
    if group == "affine_shift":
        shift = int(rng.integers(-4, 5))
        out = np.array([np.interp(np.arange(p), np.arange(p) - shift, row) for row in out])
    out = out + rng.normal(0, 0.002, size=out.shape)
    return out


def fit_predict(Xtr, ytr, Xte):
    model = PLSRegression(n_components=2, scale=False)
    model.fit(Xtr, ytr.astype(float))
    return (model.predict(Xte).ravel() > 0.5).astype(int)


def feature_sets(Xtr, Xte, anchors):
    a, b = anchors
    return {
        "raw": (Xtr, Xte),
        "snv": (snv(Xtr), snv(Xte)),
        "invariant": (
            per_wavelength_affine_invariants(Xtr, (a, b)),
            per_wavelength_affine_invariants(Xte, (a, b)),
        ),
    }


def run_group(X, y, group, reps=20, seed=0):
    anchors = conditioned_pair(X)
    idx = [i for i in range(len(y)) if i not in anchors]
    Xa, ya = X[idx], y[idx]
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        Xb = apply_instrument(X, group, rng)
        inv_A = per_wavelength_affine_invariants(Xa, anchors)
        inv_B = per_wavelength_affine_invariants(Xb, anchors)[idx]
        for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=rep).split(Xa, ya):
            reps_set = {
                "raw": (Xa[train], Xb[idx][test]),
                "snv": (snv(Xa[train]), snv(Xb[idx][test])),
                "invariant": (inv_A[train], inv_B[test]),
            }
            for name, (A, B) in reps_set.items():
                pred = fit_predict(A, ya[train], B)
                rows.append({"group": group, "representation": name,
                             "rep": rep, "accuracy": accuracy(ya[test], pred)})
    return rows


def synthetic_spectra(n_per_class=120, p=1000, n_classes=5, seed=0):
    """Controlled synthetic Raman spectra with class-specific peaks.

    This is a *synthetic* experiment, used only to show what the group calculus
    predicts when the group holds and the sample size is adequate.  It is not
    evidence about any real instrument.
    """
    rng = np.random.default_rng(seed)
    lam = np.linspace(0, 1, p)
    X, y = [], []
    for c in range(n_classes):
        centres = [0.12 + 0.16 * c, 0.35 + 0.08 * c, 0.62 + 0.05 * c]
        widths = [0.012 + 0.002 * c, 0.02, 0.015]
        amps = [1.0, 0.6 + 0.1 * c, 0.4]
        for _ in range(n_per_class):
            spec = 0.05 * np.exp(-2 * lam)
            for mu, sig, a in zip(centres, widths, amps):
                spec = spec + a * rng.uniform(0.8, 1.2) * np.exp(-0.5 * ((lam - mu) / sig) ** 2)
            spec = spec + rng.normal(0, 0.01, size=p)
            X.append(spec)
            y.append(c)
    return np.asarray(X), np.asarray(y)


def run_synthetic(group="affine", reps=5, seed=0):
    X, y = synthetic_spectra(seed=seed)
    anchors = conditioned_pair(X)
    idx = [i for i in range(len(y)) if i not in anchors]
    Xa, ya = X[idx], y[idx]
    inv_A = per_wavelength_affine_invariants(Xa, anchors)
    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + 100 + rep)
        Xb = apply_instrument(X, group, rng, strength=1.0)
        inv_B = per_wavelength_affine_invariants(Xb, anchors)[idx]
        # single stratified 70/30 split, repeated
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(np.arange(len(ya)), test_size=0.3,
                                  random_state=rep, stratify=ya)
        reps_set = {
            "raw": (Xa[tr], Xb[idx][te]),
            "snv": (snv(Xa[tr]), snv(Xb[idx][te])),
            "invariant": (inv_A[tr], inv_B[te]),
        }
        for name, (A, B) in reps_set.items():
            clf = make_classifier("logreg", seed=rep)
            clf.fit(A, ya[tr])
            rows.append({"group": group, "representation": name,
                         "rep": rep, "accuracy": accuracy(ya[te], clf.predict(B))})
    return rows


def fig_raman(df, path):
    set_style()
    groups = ["identity", "affine", "affine_shift"]
    reps = ["raw", "snv", "invariant"]
    labels = {"raw": "raw", "snv": "SNV (global affine)",
              "invariant": "per-λ affine invariant (2 refs)"}
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(groups))
    w = 0.26
    for i, r in enumerate(reps):
        d = df[df.representation == r].groupby("group").accuracy.agg(["mean", "std"]).reindex(groups)
        ax.bar(x + (i - 1) * w, d["mean"], width=w, yerr=d["std"], capsize=2,
               label=labels[r], color=PALETTE.get(r, None))
    ax.axhline(0.5, color=PALETTE["chance"], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["identity", "per-λ\naffine", "affine +\nwavelength shift"])
    ax.set_ylabel("healthy vs diseased accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Raman: real spectra, synthetic instrument")
    fig.savefig(path)
    plt.close(fig)


def fig_raman_synthetic(df, path):
    set_style()
    reps = ["raw", "snv", "invariant"]
    labels = {"raw": "raw", "snv": "SNV (global affine)",
              "invariant": "per-λ affine invariant"}
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    d = df.groupby("representation").accuracy.agg(["mean", "std"]).reindex(reps)
    ax.bar(range(len(reps)), d["mean"], yerr=d["std"], capsize=3,
           color=[PALETTE.get(r, None) for r in reps])
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([labels[r] for r in reps], rotation=15)
    ax.set_ylabel("5-class accuracy (synthetic)")
    ax.set_ylim(0, 1)
    ax.axhline(0.2, color=PALETTE["chance"], ls="--", lw=1)
    ax.set_title("Controlled Raman group experiment (synthetic)")
    fig.savefig(path)
    plt.close(fig)


def main():
    X, y = load()
    out = RESULTS / "raman"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in ["identity", "affine", "affine_shift"]:
        print("Raman group:", group)
        rows += run_group(X, y, group)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raman_transfer.csv", index=False)

    print("Raman synthetic controlled experiment")
    syn = pd.DataFrame(run_synthetic())
    syn.to_csv(out / "raman_synthetic.csv", index=False)

    fig_raman(df, FIGURES / "raman_transfer.pdf")
    fig_raman_synthetic(syn, FIGURES / "raman_synthetic.pdf")
    summary = df.groupby(["group", "representation"]).accuracy.agg(["mean", "std"]).reset_index()
    save_json({
        "real": summary.to_dict(orient="records"),
        "synthetic": syn.groupby("representation").accuracy.agg(["mean", "std"]).reset_index().to_dict(orient="records"),
    }, "summary.json", "raman")
    print(summary.round(3).to_string(index=False))
    print(syn.groupby("representation").accuracy.agg(["mean", "std"]).round(3).to_string())
    print("note: real Raman spectra for the first table; the controlled experiment is synthetic.")


if __name__ == "__main__":
    main()
