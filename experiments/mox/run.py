"""
MOX case study on the UCI Gas Sensor Array Drift dataset.

Experiments
-----------
E1  Cross-batch transfer: train on batch 1, test on batches 2-10.
    Feature sets: raw, log, log-ratio, global-gain invariant, per-batch
    centred (per-sensor gain invariant), per-batch z-scored, calibrated.
E2  Controlled group-stress test: hold the samples fixed, apply a random
    element of a nested group G1 < G2 < G3 to the test half, and measure
    accuracy.  This isolates the effect of the group from the dataset.
E3  Stability of gas-level slope ratios b_i/b_j across batches.

Outputs: results/mox/*.json, results/mox/*.csv, paper/figures/mox_*.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (
    DATA, FIGURES, PALETTE, RESULTS, accuracy, annotate, bootstrap_ci,
    make_classifier, pca_embed, save_csv, save_json, set_style,
)

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(20260707)


# ---------------------------------------------------------------------------
# Loading and feature construction
# ---------------------------------------------------------------------------

def load():
    d = np.load(DATA / "uci_drift.npz")
    R = d["R"].astype(float)
    # A small number of entries are non-positive (fit failures / saturation).
    # Clip to a positive floor before taking logs; this affects <1% of values.
    R = np.clip(R, 1.0, None)
    return {
        "R": R,
        "logR": np.log(R),
        "gas": d["gas"].astype(int),
        "conc": d["concentration"].astype(float),
        "batch": d["batch"].astype(int),
    }


def pairwise_log_ratios(L: np.ndarray) -> np.ndarray:
    """All pairwise log ratios log(R_i/R_j), shape (n, 120)."""
    n, p = L.shape
    iu, ju = np.triu_indices(p, k=1)
    return L[:, iu] - L[:, ju]


def batch_center(L: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Subtract the per-batch, per-sensor mean.

    This is the quotient by the group log R_i -> log R_i + delta_i(batch),
    i.e. an arbitrary per-sensor multiplicative gain that is constant within a
    batch.  It uses the unlabelled test batch, so it is transductive; we label
    it explicitly.
    """
    Z = L.copy()
    for b in np.unique(batch):
        m = batch == b
        Z[m] = Z[m] - Z[m].mean(axis=0, keepdims=True)
    return Z


def batch_zscore(L: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Per-batch, per-sensor z-score.  Removes gain *and* the response scale,
    which also removes the exponent b; included as a negative control."""
    Z = L.copy()
    for b in np.unique(batch):
        m = batch == b
        mu = Z[m].mean(axis=0, keepdims=True)
        sd = Z[m].std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z[m] = (Z[m] - mu) / sd
    return Z


def global_gain_invariant(L: np.ndarray) -> np.ndarray:
    """Subtract the per-sample mean across sensors (invariant to a global gain)."""
    return L - L.mean(axis=1, keepdims=True)


def build_features(data, kind: str) -> np.ndarray:
    L = data["logR"]
    R = data["R"]
    b = data["batch"]
    if kind == "raw":
        return R.copy()
    if kind == "log":
        return L.copy()
    if kind == "log_ratio":
        return pairwise_log_ratios(L)
    if kind == "global_gain":
        return global_gain_invariant(L)
    if kind == "batch_center":
        return batch_center(L, b)
    if kind == "batch_center_ratio":
        return pairwise_log_ratios(batch_center(L, b))
    if kind == "batch_zscore":
        return batch_zscore(L, b)
    if kind == "selectivity":
        return R / R.mean(axis=1, keepdims=True)
    if kind == "rank":
        from scipy.stats import rankdata
        return rankdata(L, axis=1)
    raise ValueError(kind)


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(A)
    return V @ np.diag(np.sqrt(np.maximum(w, 0))) @ V.T


def coral_align(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """CORAL: align the target covariance to the source covariance.

    Sun, Feng & Saenko (2016).  This is an empirical domain-adaptation
    baseline, not a physics-derived invariant.  It uses unlabelled target data.
    """
    ms, mt = Xs.mean(axis=0), Xt.mean(axis=0)
    Cs = np.cov(Xs - ms, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt - mt, rowvar=False) + eps * np.eye(Xt.shape[1])
    A = _sqrtm_psd(Cs) @ np.linalg.inv(_sqrtm_psd(Ct))
    return (Xt - mt) @ A + ms



# ---------------------------------------------------------------------------
# E1: cross-batch transfer
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def e1_transfer(data, feature_kinds, source_batch=1, calibrate_per_class=0):
    results = []
    src = data["batch"] == source_batch
    ys = data["gas"][src]
    for kind in feature_kinds:
        if kind in {"coral", "mean_align"}:
            # These baselines align each target batch to the source batch
            # using unlabelled target data.  Handle them batch by batch.
            L = data["logR"]
            Xs = L[src]
            for b in range(2, 11):
                tgt = data["batch"] == b
                Xt = L[tgt]
                if kind == "coral":
                    Xt = coral_align(Xs, Xt)
                else:
                    Xt = Xt - Xt.mean(axis=0) + Xs.mean(axis=0)
                clf = make_classifier("logreg", seed=0)
                clf.fit(Xs, ys)
                pred = clf.predict(Xt)
                m = evaluate(data["gas"][tgt], pred)
                m.update({"feature": kind, "target_batch": b, "n": int(tgt.sum())})
                results.append(m)
            continue

        X = build_features(data, kind)
        clf = make_classifier("logreg", seed=0)
        Xs = X[src]
        if calibrate_per_class > 0:
            pass
        clf.fit(Xs, ys)
        for b in range(2, 11):
            tgt = data["batch"] == b
            if calibrate_per_class > 0:
                # stratified calibration subset from the target batch
                idx = np.where(tgt)[0]
                rng = np.random.default_rng(1000 + b)
                take = []
                for g in np.unique(data["gas"][tgt]):
                    gi = idx[data["gas"][idx] == g]
                    take.extend(rng.choice(gi, size=min(calibrate_per_class, len(gi)), replace=False))
                cal = np.zeros_like(tgt)
                cal[take] = True
                clf2 = make_classifier("logreg", seed=0)
                clf2.fit(np.vstack([Xs, X[cal]]), np.concatenate([ys, data["gas"][cal]]))
                pred = clf2.predict(X[tgt & ~cal])
                yt = data["gas"][tgt & ~cal]
            else:
                pred = clf.predict(X[tgt])
                yt = data["gas"][tgt]
            m = evaluate(yt, pred)
            m.update({"feature": kind, "target_batch": b, "n": int(len(yt))})
            results.append(m)
        # pooled
        if calibrate_per_class == 0:
            preds, yts = [], []
            for b in range(2, 11):
                tgt = data["batch"] == b
                preds.append(clf.predict(X[tgt]))
                yts.append(data["gas"][tgt])
            m = evaluate(np.concatenate(yts), np.concatenate(preds))
            m.update({"feature": kind, "target_batch": "pooled", "n": int(sum(len(y) for y in yts))})
            results.append(m)
    return results


# ---------------------------------------------------------------------------
# E2: controlled group-stress test
# ---------------------------------------------------------------------------

def apply_group(L: np.ndarray, group: str, rng) -> np.ndarray:
    """Apply a random element of a nested group to log-responses L (n,16)."""
    n, p = L.shape
    if group == "identity":
        return L.copy()
    if group == "global_gain":
        c = rng.uniform(0.5, 2.0, size=(n, 1))
        return L + np.log(c)
    if group == "per_sensor_gain":
        c = rng.uniform(0.2, 5.0, size=(1, p))
        return L + np.log(c)
    if group == "per_sensor_affine":
        k = rng.uniform(0.2, 5.0, size=(1, p))
        d = rng.normal(0, 2.0, size=(1, p))
        return k * L + d
    raise ValueError(group)


def e2_group_stress(data, n_repeats=8):
    """Train and test on the same batch, but transform the test half by G."""
    rows = []
    b = 1
    m = data["batch"] == b
    X = data["logR"][m]
    y = data["gas"][m]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=7, stratify=y)
    feats = {
        "log": lambda Z: Z,
        "global_gain": lambda Z: global_gain_invariant(Z),
        "batch_center": lambda Z: Z - Z.mean(axis=0, keepdims=True),
    }
    for group in ["identity", "global_gain", "per_sensor_gain", "per_sensor_affine"]:
        accs = {k: [] for k in feats}
        for rep in range(n_repeats):
            rng = np.random.default_rng(2000 + rep)
            Xte_g = apply_group(Xte, group, rng)
            # For the batch_center feature, centre both halves with their own
            # means (the transformation is applied per measurement set).
            Xtr_f = Xtr
            Xte_f = Xte_g
            for name, fn in feats.items():
                A = fn(Xtr_f)
                B = fn(Xte_f)
                clf = make_classifier("logreg", seed=rep)
                clf.fit(A, ytr)
                accs[name].append(accuracy(yte, clf.predict(B)))
        for name in feats:
            mu = float(np.mean(accs[name]))
            lo, hi = float(np.percentile(accs[name], 2.5)), float(np.percentile(accs[name], 97.5))
            rows.append({"group": group, "feature": name, "mean_accuracy": mu, "lo": lo, "hi": hi})
    return rows


# ---------------------------------------------------------------------------
# E3: stability of gas-level slope ratios
# ---------------------------------------------------------------------------

def gas_slope_profiles(data):
    """For each (batch, gas) with >=2 concentrations, fit per-sensor slopes.

    Returns a list of dicts with the 16 slopes and the pairwise 120 ratios.
    """
    out = []
    for b in range(1, 11):
        for g in range(1, 7):
            m = (data["batch"] == b) & (data["gas"] == g)
            if m.sum() < 4:
                continue
            cs = np.unique(data["conc"][m])
            if len(cs) < 2:
                continue
            # median log response per concentration level
            prof = []
            for c in cs:
                mc = m & (data["conc"] == c)
                prof.append(np.median(data["logR"][mc], axis=0))
            prof = np.asarray(prof)  # (n_levels, 16)
            x = np.log(cs)
            A = np.column_stack([x, np.ones_like(x)])
            slopes, intercepts = [], []
            for s in range(prof.shape[1]):
                coef, *_ = np.linalg.lstsq(A, prof[:, s], rcond=None)
                slopes.append(coef[0])
                intercepts.append(coef[1])
            slopes = np.asarray(slopes)
            iu, ju = np.triu_indices(16, k=1)
            ratios = slopes[iu] / np.where(np.abs(slopes[ju]) < 1e-9, np.nan, slopes[ju])
            out.append({
                "batch": b, "gas": g, "n_levels": len(cs),
                "slopes": slopes, "intercepts": np.asarray(intercepts), "ratios": ratios,
            })
    return out


def e3_slope_stability(profiles):
    """Coefficient of variation of slope ratios across batches, per gas."""
    rows = []
    for g in range(1, 7):
        ps = [p for p in profiles if p["gas"] == g]
        if len(ps) < 2:
            continue
        R = np.vstack([p["ratios"] for p in ps])
        # ignore pairs that are nan in any batch
        ok = np.all(np.isfinite(R), axis=0)
        R = R[:, ok]
        if R.size == 0:
            continue
        # stability: median absolute deviation relative to median magnitude
        med = np.median(R, axis=0)
        mad = np.median(np.abs(R - med), axis=0)
        cv = np.median(mad / np.where(np.abs(med) < 1e-9, np.nan, np.abs(med)))
        rows.append({"gas": g, "n_batches": len(ps), "n_ratio_pairs": int(R.shape[1]),
                     "median_relative_MAD": float(cv)})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_transfer(df, path):
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    feats = ["raw", "log", "log_ratio", "global_gain", "batch_center", "batch_center_ratio"]
    labels = {
        "raw": "raw R",
        "log": "log R",
        "log_ratio": "pairwise log ratios",
        "global_gain": "global-gain invariant",
        "batch_center": "per-sensor gain invariant",
        "batch_center_ratio": "gain-invariant ratios",
    }
    for f in feats:
        d = df[(df.feature == f) & (df.target_batch != "pooled")]
        if len(d) == 0:
            continue
        ax.plot(d.target_batch.astype(int), d.accuracy, marker="o", ms=3.5,
                lw=1.4, label=labels[f], color=PALETTE.get(f, None))
    ax.axhline(1 / 6, color=PALETTE["chance"], ls="--", lw=1, label="chance (1/6)")
    ax.set_xlabel("target batch (months of aging →)")
    ax.set_ylabel("accuracy (train on batch 1)")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(2, 11))
    ax.legend(ncol=2, frameon=False, loc="lower left")
    ax.set_title("MOX drift: cross-batch transfer without target labels")
    fig.savefig(path)
    plt.close(fig)


def fig_group_stress(df, path):
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    groups = ["identity", "global_gain", "per_sensor_gain", "per_sensor_affine"]
    x = np.arange(len(groups))
    feats = ["log", "global_gain", "batch_center"]
    labels = {"log": "log R (no invariant)", "global_gain": "global-gain invariant",
              "batch_center": "per-sensor gain invariant"}
    for i, f in enumerate(feats):
        d = df[df.feature == f].set_index("group").loc[groups]
        ax.errorbar(x + (i - 1) * 0.08, d.mean_accuracy,
                    yerr=[d.mean_accuracy - d.lo, d.hi - d.mean_accuracy],
                    marker="o", capsize=2, lw=1.4, label=labels[f],
                    color=PALETTE.get(f, None))
    ax.axhline(1 / 6, color=PALETTE["chance"], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["identity", "global\ngain", "per-sensor\ngain", "per-sensor\naffine"])
    ax.set_ylabel("accuracy on transformed test half")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Controlled group stress test: which invariants survive which group")
    fig.savefig(path)
    plt.close(fig)


def fig_pca(data, path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    for ax, kind, title in [
        (axes[0], "log", "log R (batch structure dominates)"),
        (axes[1], "batch_center", "per-sensor gain invariant"),
    ]:
        X = build_features(data, kind)
        Z = pca_embed(X, 2)
        for b in range(1, 11):
            m = data["batch"] == b
            ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.55, label=str(b))
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[1].legend(ncol=2, fontsize=6, frameon=False, title="batch", loc="best")
    fig.suptitle("Batch mixing before and after quotienting the per-sensor gain", y=1.02)
    fig.savefig(path)
    plt.close(fig)


def fig_power_law(data, path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    # pick a sensor and gas with multiple concentrations in several batches
    s, g = 0, 1
    for b in [1, 6, 7, 10]:
        m = (data["batch"] == b) & (data["gas"] == g)
        if m.sum() == 0:
            continue
        cs = np.unique(data["conc"][m])
        means = [np.median(data["R"][m & (data["conc"] == c), s]) for c in cs]
        axes[0].plot(np.log(cs), np.log(means), "o-", ms=3, lw=1, label=f"batch {b}")
    axes[0].set_xlabel("log concentration")
    axes[0].set_ylabel("log resistance")
    axes[0].set_title("Power law is a line in log-log space")
    axes[0].legend(fontsize=7, frameon=False)

    # slope per batch for the same sensor/gas
    slopes = []
    for b in range(1, 11):
        m = (data["batch"] == b) & (data["gas"] == g)
        cs = np.unique(data["conc"][m])
        if len(cs) < 2:
            continue
        means = np.array([np.median(data["R"][m & (data["conc"] == c), s]) for c in cs])
        A = np.column_stack([np.log(cs), np.ones_like(cs)])
        coef, *_ = np.linalg.lstsq(A, np.log(means), rcond=None)
        slopes.append((b, coef[0]))
    if slopes:
        bs, ss = zip(*slopes)
        axes[1].plot(bs, ss, "o-", color=PALETTE["invariant"], ms=4, lw=1.2)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("batch")
    axes[1].set_ylabel("fitted exponent b (sensor 1, gas 1)")
    axes[1].set_title("The exponent is a device/time parameter")
    fig.savefig(path)
    plt.close(fig)


def fig_slope_stability(df, path):
    set_style()
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(df.gas.astype(str), df.median_relative_MAD, color=PALETTE["invariant"], alpha=0.85)
    ax.set_xlabel("gas class")
    ax.set_ylabel("median relative MAD of\nslope ratios across batches")
    ax.set_title("Stability of the restricted-group invariant")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load()
    out = RESULTS / "mox"
    out.mkdir(parents=True, exist_ok=True)

    print("E1: cross-batch transfer")
    feats = ["raw", "log", "log_ratio", "global_gain", "batch_center",
             "batch_center_ratio", "batch_zscore", "selectivity", "rank",
             "coral", "mean_align"]
    rows = e1_transfer(data, feats)
    cal = e1_transfer(data, ["log"], calibrate_per_class=5)
    for r in cal:
        r["feature"] = "calibrated_5pc"
    rows += cal
    df = pd.DataFrame(rows)
    df.to_csv(out / "e1_transfer.csv", index=False)
    save_json(rows, "e1_transfer.json", "mox")

    print("E2: controlled group stress test")
    gs = e2_group_stress(data)
    pd.DataFrame(gs).to_csv(out / "e2_group_stress.csv", index=False)
    save_json(gs, "e2_group_stress.json", "mox")

    print("E3: gas-level slope stability")
    profiles = gas_slope_profiles(data)
    stab = e3_slope_stability(profiles)
    pd.DataFrame(stab).to_csv(out / "e3_slope_stability.csv", index=False)
    save_json(stab, "e3_slope_stability.json", "mox")
    # save raw profiles for downstream plotting
    np.savez_compressed(
        out / "slope_profiles.npz",
        **{f"b{p['batch']}_g{p['gas']}_slopes": p["slopes"] for p in profiles},
    )

    print("Figures")
    fig_transfer(df, FIGURES / "mox_transfer.pdf")
    fig_group_stress(pd.DataFrame(gs), FIGURES / "mox_group_stress.pdf")
    fig_pca(data, FIGURES / "mox_pca.pdf")
    fig_power_law(data, FIGURES / "mox_power_law.pdf")
    if stab:
        fig_slope_stability(pd.DataFrame(stab), FIGURES / "mox_slope_stability.pdf")

    summary = {
        "n": int(len(data["gas"])),
        "n_batches": 10,
        "feature_means": df[df.target_batch == "pooled"].set_index("feature")[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ].to_dict(orient="index"),
        "group_stress": gs,
        "slope_stability": stab,
    }
    save_json(summary, "summary.json", "mox")
    print(json.dumps(summary["feature_means"], indent=2))
    print("done")


if __name__ == "__main__":
    main()
