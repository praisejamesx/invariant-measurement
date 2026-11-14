"""
EEG case study: real EEG, synthetic instrument transforms.

Data
----
PhysioNet EEG Motor Movement/Imagery Dataset (eegmmidb), subject 1, runs 3 and
4 (left vs right fist motor imagery).  64 channels, 160 Hz, 125 s per run.

Instrument model
----------------
A per-channel affine transform in the time domain:
    x'_c(t) = g_c * x_c(t) + o_c
plus an optional common-reference transform x_c -> x_c - x_ref.  g_c and o_c
are the gain and offset of amplifier channel c; the common reference is a
rank-one spatial mixing.  This is a simplified model of a different EEG
headbox / montage.

Invariant
---------
Band power is computed per channel and band.  A per-channel gain adds a
constant to log power, and a per-channel offset is largely removed by the
band-pass filters.  Subtracting a per-channel *baseline* log power (a
reference period, not a label) makes the representation invariant to g_c:
    log P_c(task) - log P_c(baseline) = log(P_c(task)/P_c(baseline)).
This is the EEG analogue of R_s/R_0: a reference measurement, not a label.

The experiment is explicitly a real-signal / synthetic-instrument study.  It
does not claim that the transforms are recordings from a second headbox.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (
    DATA, FIGURES, PALETTE, RESULTS, accuracy, make_classifier,
    save_json, set_style,
)

import matplotlib.pyplot as plt
from scipy.signal import welch

RUNS = {"left": 3, "right": 4}
SFREQ = 160.0
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
WIN_SEC = 2.0
STRIDE_SEC = 2.0  # non-overlapping windows, so train/test are independent


def download_eeg():
    import requests

    out = DATA / "eeg"
    out.mkdir(parents=True, exist_ok=True)
    for name, run in RUNS.items():
        path = out / f"S001R{run:02d}.edf"
        if not path.exists():
            url = f"https://physionet.org/files/eegmmidb/1.0.0/S001/S001R{run:02d}.edf"
            print(f"  downloading {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
    return out


def read_edf(path):
    import pyedflib

    f = pyedflib.EdfReader(str(path))
    n = f.signals_in_file
    x = np.vstack([f.readSignal(i) for i in range(n)])
    f.close()
    return x


def bandpower_features(x, sfreq=SFREQ, win_sec=WIN_SEC, stride_sec=STRIDE_SEC):
    """Return (n_windows, n_channels, n_bands) log band power."""
    n_ch, n_t = x.shape
    win = int(win_sec * sfreq)
    stride = int(stride_sec * sfreq)
    starts = np.arange(0, n_t - win + 1, stride)
    feats = np.empty((len(starts), n_ch, len(BANDS)))
    for i, s in enumerate(starts):
        seg = x[:, s:s + win]
        f, P = welch(seg, fs=sfreq, nperseg=min(256, win), axis=1)
        for j, (lo, hi) in enumerate(BANDS.values()):
            m = (f >= lo) & (f < hi)
            feats[i, :, j] = np.log(np.trapezoid(P[:, m], f[m], axis=1) + 1e-18)
    return feats, starts


def apply_instrument(x, group, rng):
    """Apply a synthetic instrument transform to a (channels, time) array."""
    x = x.copy()
    n_ch = x.shape[0]
    if group == "identity":
        return x
    if group in {"gain", "gain_offset", "gain_offset_reference"}:
        g = np.exp(rng.normal(0, 0.6, size=(n_ch, 1)))
        x = g * x
    if group in {"gain_offset", "gain_offset_reference"}:
        o = rng.normal(0, 30.0, size=(n_ch, 1))
        x = x + o
    if group == "gain_offset_reference":
        ref = x[rng.integers(n_ch)]
        x = x - ref[None, :]
    x = x + rng.normal(0, 2.0, size=x.shape)
    return x


def build_representation(feats, baseline_windows, kind):
    """feats: (n_windows, n_channels, n_bands)."""
    if kind == "raw":
        return feats.reshape(len(feats), -1)
    base = feats[baseline_windows].mean(axis=0)  # (n_ch, n_bands)
    if kind == "baseline":
        return (feats - base[None, :, :]).reshape(len(feats), -1)
    if kind == "baseline_ratio":
        return ((feats - base[None, :, :]) / (np.abs(base)[None, :, :] + 1e-6)).reshape(len(feats), -1)
    if kind == "global_norm":
        z = feats - feats.mean(axis=1, keepdims=True)
        return z.reshape(len(feats), -1)
    raise ValueError(kind)


def run_experiment(group, reps=10, seed=0):
    root = download_eeg()
    data = {}
    for name, run in RUNS.items():
        x = read_edf(root / f"S001R{run:02d}.edf")
        feats, starts = bandpower_features(x)
        data[name] = {"x": x, "feats": feats, "starts": starts}
    # Fixed temporal split: the first 60% of each run is training, the last
    # 40% is held out.  Windows are non-overlapping, so there is no leakage.
    n_train = {}
    for name in RUNS:
        n_train[name] = int(0.6 * len(data[name]["feats"]))
    Xtr_raw = np.concatenate(
        [data[name]["feats"][: n_train[name]] for name in RUNS], axis=0
    )
    ytr = np.concatenate(
        [np.full(n_train[name], i) for i, name in enumerate(RUNS)]
    )
    base_tr = np.arange(min(10, min(n_train.values())))

    rows = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        Xte_raw, yte = [], []
        for i, name in enumerate(RUNS):
            x = apply_instrument(data[name]["x"], group, rng)
            feats, _ = bandpower_features(x)
            test = feats[n_train[name]:]
            Xte_raw.append(test)
            yte.append(np.full(len(test), i))
        Xte_raw = np.concatenate(Xte_raw, axis=0)
        yte = np.concatenate(yte)
        for kind in ["raw", "global_norm", "baseline", "baseline_ratio"]:
            A = build_representation(Xtr_raw, base_tr, kind)
            B = build_representation(Xte_raw, base_tr, kind)
            clf = make_classifier("logreg", seed=rep)
            clf.fit(A, ytr)
            rows.append({"group": group, "representation": kind, "rep": rep,
                         "accuracy": accuracy(yte, clf.predict(B))})
    return rows


def fig_eeg(df, path):
    set_style()
    groups = ["identity", "gain", "gain_offset", "gain_offset_reference"]
    reps = ["raw", "global_norm", "baseline", "baseline_ratio"]
    labels = {"raw": "raw band power", "global_norm": "global-gain invariant",
              "baseline": "baseline-referenced (per-channel gain invariant)",
              "baseline_ratio": "baseline ratio"}
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(groups))
    w = 0.2
    for i, r in enumerate(reps):
        d = df[df.representation == r].groupby("group").accuracy.agg(["mean", "std"]).reindex(groups)
        ax.bar(x + (i - 1.5) * w, d["mean"], width=w, yerr=d["std"], capsize=2,
               label=labels[r], color=PALETTE.get(r, None))
    ax.axhline(0.5, color=PALETTE["chance"], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["identity", "per-channel\ngain", "gain +\noffset", "gain + offset\n+ reference"])
    ax.set_ylabel("left vs right motor imagery accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set_title("EEG: real signals, synthetic instrument transforms")
    fig.savefig(path)
    plt.close(fig)


def main():
    out = RESULTS / "eeg"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group in ["identity", "gain", "gain_offset", "gain_offset_reference"]:
        print("EEG group:", group)
        rows += run_experiment(group, reps=10)
    df = pd.DataFrame(rows)
    df.to_csv(out / "eeg_transfer.csv", index=False)
    fig_eeg(df, FIGURES / "eeg_transfer.pdf")
    summary = df.groupby(["group", "representation"]).accuracy.agg(["mean", "std"]).reset_index()
    save_json(summary.to_dict(orient="records"), "summary.json", "eeg")
    print(summary.round(3).to_string(index=False))
    print("note: real EEG; instrument transforms are synthetic and labelled as such.")


if __name__ == "__main__":
    main()
