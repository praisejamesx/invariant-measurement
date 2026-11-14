"""
Mass spectrometry case study: controlled synthetic spectra.

Why synthetic
-------------
Public repositories (MassBank, MoNA) contain instrument metadata, but records
are not paired by physical sample across instruments in a form that supports a
cross-instrument train/test split.  We therefore use a controlled synthetic
experiment to test the *group calculus*, and state the data limitation plainly.

Instrument model
----------------
    m' = a*m + b            (m/z calibration: affine)
    I' = g*I                (intensity gain)
plus noise.  Two reference peaks determine (a, b); this is the spectroscopy
analogue of the two-point calibration.  Fragmentation changes (peak intensities
that depend on the instrument's collision energy) are *outside* the affine
group and are included as a boundary condition.

The experiment is synthetic and is labelled as such.  It is not evidence about
any particular mass spectrometer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (
    FIGURES, PALETTE, RESULTS, accuracy, make_classifier, save_json, set_style,
)

import matplotlib.pyplot as plt

N_CLASSES = 5
N_PER_CLASS = 200
P = 1000
MZ = np.linspace(0, 1000, P)


def make_spectrum(c, rng):
    """A synthetic fragment/isotope pattern for class c."""
    spec = np.zeros(P)
    # class-specific fragment positions and intensities
    centres = [
        [120 + 6 * c, 310 + 6 * c, 520 + 6 * c, 760 + 6 * c],
        [90 + 6 * c, 250 + 6 * c, 480 + 6 * c, 700 + 6 * c],
    ][c % 2]
    amps = [1.0, 0.55 + 0.1 * c, 0.35, 0.2]
    for mu, a in zip(centres, amps):
        mu = mu + rng.normal(0, 0.8)
        width = rng.uniform(1.0, 1.8)
        spec += a * rng.uniform(0.85, 1.15) * np.exp(-0.5 * ((MZ - mu) / width) ** 2)
        # isotope envelope
        spec += 0.45 * a * np.exp(-0.5 * ((MZ - (mu + 1.0)) / width) ** 2)
        spec += 0.18 * a * np.exp(-0.5 * ((MZ - (mu + 2.0)) / width) ** 2)
    spec += rng.normal(0, 0.005, size=P)
    return spec


def warp(spec, a, b, gain, rng):
    """Apply m/z affine calibration, intensity gain, and noise."""
    warped = np.interp((MZ - b) / a, MZ, spec, left=0.0, right=0.0)
    return gain * warped + rng.normal(0, 0.005, size=P)


def top_peaks(spec, n=2):
    """Indices of the n strongest local maxima."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(spec, distance=10)
    if len(idx) < n:
        idx = np.argsort(spec)[-n:]
    else:
        idx = idx[np.argsort(spec[idx])[-n:]]
    return np.sort(idx)


def calibrate(spec_target, spec_source, n=2):
    """Estimate the affine m/z map from n matched reference peaks and unwarp."""
    it = top_peaks(spec_target, n)
    is_ = top_peaks(spec_source, n)
    # match by rank (largest to largest)
    order_t = it[np.argsort(spec_target[it])[::-1]]
    order_s = is_[np.argsort(spec_source[is_])[::-1]]
    mt, ms = MZ[order_t], MZ[order_s]
    if np.ptp(ms) < 1e-9:
        return spec_target.copy()
    a, b = np.polyfit(ms, mt, 1)
    # target at m' = a*m+b; sample it on the source grid
    return np.interp(a * MZ + b, MZ, spec_target, left=0.0, right=0.0)


def fragmentation(spec, c, rng):
    """A non-affine change: class-dependent suppression/enhancement of peaks."""
    out = spec.copy()
    for mu in [120 + 6 * c, 310 + 6 * c, 520 + 6 * c, 760 + 6 * c]:
        factor = rng.uniform(0.3, 2.0)
        m = np.abs(MZ - mu) < 12
        out[m] *= factor
    return out


def run():
    rng = np.random.default_rng(0)
    X, y = [], []
    for c in range(N_CLASSES):
        for _ in range(N_PER_CLASS):
            X.append(make_spectrum(c, rng))
            y.append(c)
    X = np.asarray(X)
    y = np.asarray(y)
    # train/test split, stratified
    tr, te = [], []
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        tr.extend(idx[: int(0.7 * len(idx))])
        te.extend(idx[int(0.7 * len(idx)):])
    tr, te = np.asarray(tr), np.asarray(te)

    rows = []
    for group in ["identity", "affine", "affine_fragmentation"]:
        accs = {"raw": [], "calibrated": []}
        for rep in range(5):
            rng = np.random.default_rng(100 + rep)
            if group == "identity":
                a, b, gain = 1.0, 0.0, 1.0
            else:
                a = rng.uniform(0.995, 1.005)
                b = rng.uniform(-15, 15)
                gain = rng.uniform(0.7, 1.4)
            Xt = np.asarray([warp(x, a, b, gain, rng) for x in X])
            if group == "affine_fragmentation":
                Xt = np.asarray([fragmentation(x, c, rng) for x, c in zip(Xt, y)])
            # calibration uses two reference spectra per class?  Here we use
            # two reference peaks within each spectrum (a reference standard).
            Xt_cal = np.asarray([calibrate(t, s) for t, s in zip(Xt, X)])
            for name, (A, B) in {
                "raw": (X[tr], Xt[te]),
                "calibrated": (X[tr], Xt_cal[te]),
            }.items():
                clf = make_classifier("logreg", seed=rep)
                clf.fit(A, y[tr])
                accs[name].append(accuracy(y[te], clf.predict(B)))
        for name, vals in accs.items():
            rows.append({"group": group, "representation": name,
                         "mean_accuracy": float(np.mean(vals)),
                         "std": float(np.std(vals))})
    return pd.DataFrame(rows)


def fig(df, path):
    set_style()
    groups = ["identity", "affine", "affine_fragmentation"]
    reps = ["raw", "calibrated"]
    labels = {"raw": "raw", "calibrated": "2-reference affine calibration"}
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(groups))
    w = 0.34
    for i, r in enumerate(reps):
        d = df[df.representation == r].set_index("group").reindex(groups)
        ax.bar(x + (i - 0.5) * w, d.mean_accuracy, width=w,
               yerr=d["std"], capsize=3, label=labels[r],
               color=PALETTE.get(r, None))
    ax.axhline(1 / N_CLASSES, color=PALETTE["chance"], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["identity", "m/z affine", "affine +\nfragmentation"])
    ax.set_ylabel("5-class accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Mass spec: affine calibration and its boundary (synthetic)")
    fig.savefig(path)
    plt.close(fig)


def main():
    out = RESULTS / "massspec"
    out.mkdir(parents=True, exist_ok=True)
    df = run()
    df.to_csv(out / "massspec_transfer.csv", index=False)
    fig(df, FIGURES / "massspec_transfer.pdf")
    save_json(df.to_dict(orient="records"), "summary.json", "massspec")
    print(df.round(3).to_string(index=False))
    print("note: synthetic spectra; no paired public cross-instrument sample set used.")


if __name__ == "__main__":
    main()
