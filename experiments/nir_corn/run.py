"""
NIR case study on the Corn dataset (80 samples, 3 instruments, 700 wavelengths).

This is the strongest empirical test in the paper because the *same physical
samples* are measured on three genuinely different instruments.

Experiments
-----------
N1  Representation consistency: for matched samples, how well does each
    representation agree across instruments?  Raw vs SNV vs per-wavelength
    affine invariant.
N2  Cross-instrument transfer: train on instrument A, test on B, for protein
    and moisture.  Raw, SNV, per-wavelength invariant (2 references),
    per-wavelength calibration (2 references), PDS (k references).
N3  Reference-count scaling: R2 as a function of the number of reference
    samples used to define the quotient / calibration.

The per-wavelength affine group is
    A'(lambda) = g(lambda) A(lambda) + o(lambda).
Two reference samples measured on both instruments determine g and o at every
wavelength.  The invariant representation is the quotient; the calibrated
representation is the same quotient written in the source frame.  The
experiment therefore tests the group model, not a particular classifier.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (
    DATA, FIGURES, PALETTE, RESULTS, accuracy, make_regressor,
    r2, rmse, save_json, set_style,
)
from invmath.invariants import per_wavelength_affine_invariants, snv

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260707)
INSTRUMENTS = ["m5", "mp5", "mp6"]
PROPERTY_INDEX = {"Moisture": 0, "Oil": 1, "Protein": 2, "Starch": 3}


def load():
    d = np.load(DATA / "corn.npz")
    X = {k: d[k].astype(float) for k in INSTRUMENTS}
    return {
        "X": X,
        "wavelengths": d["wavelengths"].astype(float),
        "props": d["propvals"].astype(float),
        "labels": [str(x).strip() for x in d["property_labels"]],
    }


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def choose_anchors(X: np.ndarray, k: int = 2, seed: int = 0) -> list[int]:
    """Choose well-conditioned reference samples.

    The affine quotient divides by X[a] - X[b] at every wavelength.  A pair
    with a small difference at some wavelength amplifies noise there.  We
    therefore choose the pair that maximises the 10th percentile of the
    standardised absolute difference across wavelengths, then add further
    samples by farthest-point sampling in standardised spectral space.

    This is an a-priori rule: it uses only the source instrument's spectra
    (or any unlabelled spectra) and no labels.
    """
    n = X.shape[0]
    std = X.std(axis=0) + 1e-9
    Xz = (X - X.mean(axis=0)) / std

    best_pair, best_score = (0, 1), -np.inf
    for a in range(n):
        for b in range(a + 1, n):
            score = float(np.percentile(np.abs(Xz[a] - Xz[b]), 10))
            if score > best_score:
                best_score, best_pair = score, (a, b)
    chosen = [best_pair[0], best_pair[1]]
    while len(chosen) < k:
        d = np.min([np.linalg.norm(Xz - Xz[c], axis=1) for c in chosen], axis=0)
        d[chosen] = -np.inf
        chosen.append(int(np.argmax(d)))
    return chosen


def rep_raw(X):
    return X


def rep_snv(X):
    return snv(X)


def rep_log_snv(X):
    return snv(np.log(np.clip(X, 1e-6, None)))


def rep_invariant(X, anchors):
    return per_wavelength_affine_invariants(X, anchor=(anchors[0], anchors[1]))


def calibrate_target(Xs, Xt, anchors):
    """Map target spectra into the source frame using per-wavelength affine
    regression on the reference samples.  With two references this is exact."""
    A = Xs[anchors]
    B = Xt[anchors]
    # solve B = g*A + o per wavelength
    g = np.zeros(Xs.shape[1])
    o = np.zeros(Xs.shape[1])
    for l in range(Xs.shape[1]):
        a = A[:, l]
        b = B[:, l]
        if np.ptp(a) < 1e-12:
            g[l] = 1.0
            o[l] = b.mean() - a.mean()
        else:
            g[l], o[l] = np.polyfit(a, b, 1)
    return (Xt - o) / np.where(np.abs(g) < 1e-9, 1e-9, g)


def pds_target(Xs, Xt, anchors, window: int = 2, alpha: float = 0.01):
    """Piecewise direct standardization with a local window.

    For each source wavelength l, regress the *source* values on the target
    values in [l-window, l+window] using the reference samples, then apply the
    map to the target.  This maps the target into the source frame.  The ridge
    penalty stabilises the local fit when the number of references is small.
    """
    A = Xs[anchors]
    B = Xt[anchors]
    p = Xs.shape[1]
    out = np.zeros_like(Xt)
    for l in range(p):
        lo, hi = max(0, l - window), min(p, l + window + 1)
        model = Ridge(alpha=alpha).fit(B[:, lo:hi], A[:, l])
        out[:, l] = model.predict(Xt[:, lo:hi])
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_pls(X, y, n_components=10):
    model = PLSRegression(n_components=min(n_components, X.shape[1], X.shape[0] - 1), scale=False)
    model.fit(X, y)
    return model


def eval_regression(Xtr, ytr, Xte, yte, n_components=10):
    model = fit_pls(Xtr, ytr, n_components)
    pred = model.predict(Xte).ravel()
    return {"r2": r2(yte, pred), "rmse": rmse(yte, pred)}, pred


def eval_classification(Xtr, ytr, Xte, yte, n_components=10):
    model = fit_pls(Xtr, ytr.astype(float), n_components)
    score = model.predict(Xte).ravel()
    pred = (score > 0.5).astype(int)
    return {"accuracy": accuracy(yte, pred)}, pred


# ---------------------------------------------------------------------------
# N1: model adequacy of the instrument group
# ---------------------------------------------------------------------------

def group_fit(data):
    """How much of the cross-instrument variation does each group explain?

    The three models are nested:
        identity       : Xt = Xs
        global affine  : Xt = g*Xs + o
        per-lambda     : Xt[:,l] = g_l*Xs[:,l] + o_l
    R2 is computed on all entries.  This is an oracle fit (all samples are
    used to estimate the group parameters); it measures whether the group is
    an adequate *model* of instrument variation, not whether it transfers
    without references.
    """
    out = []
    for a, b in itertools.combinations(INSTRUMENTS, 2):
        Xs, Xt = data["X"][a], data["X"][b]
        ss_tot = float(np.sum((Xt - Xt.mean()) ** 2))
        r2_id = 1 - float(np.sum((Xt - Xs) ** 2)) / ss_tot
        g, o = np.polyfit(Xs.ravel(), Xt.ravel(), 1)
        r2_ga = 1 - float(np.sum((Xt - (g * Xs + o)) ** 2)) / ss_tot
        pred = np.empty_like(Xt)
        for l in range(Xt.shape[1]):
            gl, ol = np.polyfit(Xs[:, l], Xt[:, l], 1)
            pred[:, l] = gl * Xs[:, l] + ol
        r2_pl = 1 - float(np.sum((Xt - pred) ** 2)) / ss_tot
        out.append({"pair": f"{a}->{b}", "identity": r2_id,
                    "global_affine": r2_ga, "per_lambda_affine": r2_pl})
    return out


def representation_consistency(data, anchors):
    out = []
    X = data["X"]
    for a, b in itertools.combinations(INSTRUMENTS, 2):
        # matched samples, excluding the anchors themselves
        idx = [i for i in range(X[a].shape[0]) if i not in anchors]
        reps = {
            "raw": (X[a][idx], X[b][idx]),
            "snv": (rep_snv(X[a][idx]), rep_snv(X[b][idx])),
            "invariant": (
                rep_invariant(X[a], anchors)[idx],
                rep_invariant(X[b], anchors)[idx],
            ),
        }
        for name, (U, V) in reps.items():
            # per-wavelength agreement: R2 of V on U across all entries,
            # after removing any residual per-wavelength offset/scale
            u = U.ravel()
            v = V.ravel()
            ss_res = np.sum((v - u) ** 2)
            ss_tot = np.sum((v - v.mean()) ** 2)
            out.append({
                "pair": f"{a}->{b}",
                "representation": name,
                "entrywise_r2": float(1 - ss_res / ss_tot),
                "corr": float(np.corrcoef(u, v)[0, 1]),
                "rmse": float(np.sqrt(np.mean((u - v) ** 2))),
            })
    return out


# ---------------------------------------------------------------------------
# N2: transfer
# ---------------------------------------------------------------------------

def transfer_once(data, source, target, prop, anchors, reps, n_components=10):
    Xs, Xt = data["X"][source], data["X"][target]
    y = data["props"][:, PROPERTY_INDEX[prop]]
    idx = [i for i in range(Xs.shape[0]) if i not in anchors]
    ytr, yte = y[idx], y[idx]
    rows = []
    for name, builder in reps.items():
        if name == "raw":
            Xtr, Xte = Xs[idx], Xt[idx]
        elif name == "snv":
            Xtr, Xte = rep_snv(Xs[idx]), rep_snv(Xt[idx])
        elif name == "log_snv":
            Xtr, Xte = rep_log_snv(Xs[idx]), rep_log_snv(Xt[idx])
        elif name == "invariant":
            Xtr = rep_invariant(Xs, anchors)[idx]
            Xte = rep_invariant(Xt, anchors)[idx]
        elif name == "calibrated":
            Xtr = Xs[idx]
            Xte = calibrate_target(Xs, Xt, anchors)[idx]
        elif name == "pds":
            Xtr = Xs[idx]
            Xte = pds_target(Xs, Xt, anchors, window=2)[idx]
        else:
            raise ValueError(name)
        metrics, _ = eval_regression(Xtr, ytr, Xte, yte, n_components)
        rows.append({"source": source, "target": target, "property": prop,
                     "representation": name, **metrics})
    return rows


def transfer(data, prop="Protein", n_anchor_sets=5, reps=None):
    if reps is None:
        reps = {
            "raw": None, "snv": None, "invariant": None,
            "calibrated": None, "pds": None,
        }
    rows = []
    for source, target in itertools.permutations(INSTRUMENTS, 2):
        for k in range(n_anchor_sets):
            anchors = choose_anchors(data["X"][source], k=5, seed=k)
            # PDS with window 2 needs 5 references; use the first 5.
            for r in transfer_once(data, source, target, prop, anchors, reps):
                r["anchor_set"] = k
                rows.append(r)
    df = pd.DataFrame(rows)
    return df.groupby(
        ["source", "target", "property", "representation"], as_index=False
    ).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
    )


# ---------------------------------------------------------------------------
# N3: reference-count scaling
# ---------------------------------------------------------------------------

def reference_scaling(data, prop="Protein", source="m5", target="mp5", counts=(2, 3, 5, 10, 20)):
    y = data["props"][:, PROPERTY_INDEX[prop]]
    rows = []
    for k in counts:
        for rep in range(5):
            anchors = choose_anchors(data["X"][source], k=k, seed=rep)
            Xs, Xt = data["X"][source], data["X"][target]
            idx = [i for i in range(len(y)) if i not in anchors]
            ytr, yte = y[idx], y[idx]
            for name in ["invariant", "calibrated", "pds"]:
                if name == "invariant":
                    Xtr = rep_invariant(Xs, anchors)[idx]
                    Xte = rep_invariant(Xt, anchors)[idx]
                elif name == "calibrated":
                    Xtr = Xs[idx]
                    Xte = calibrate_target(Xs, Xt, anchors)[idx]
                else:
                    if k < 5:
                        continue
                    Xtr = Xs[idx]
                    Xte = pds_target(Xs, Xt, anchors, window=2)[idx]
                met, _ = eval_regression(Xtr, ytr, Xte, yte, 10)
                rows.append({"n_references": k, "representation": name, "repeat": rep, **met})
    return pd.DataFrame(rows).groupby(
        ["n_references", "representation"], as_index=False
    ).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def anchor_sensitivity(data, prop="Protein", source="m5", target="mp5", n_pairs=30, seed=0):
    """How much does the choice of the two reference samples matter?

    For random pairs we record the conditioning score (10th percentile of the
    standardised absolute anchor difference across wavelengths) and the
    resulting cross-instrument R2.  The conditioned pair chosen by
    choose_anchors is included as the reference point.
    """
    Xs, Xt = data["X"][source], data["X"][target]
    y = data["props"][:, PROPERTY_INDEX[prop]]
    std = Xs.std(axis=0) + 1e-9
    Xz = (Xs - Xs.mean(axis=0)) / std
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(len(y), size=2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    pairs = sorted(pairs)
    conditioned = tuple(choose_anchors(Xs, k=2, seed=0))
    if conditioned not in pairs:
        pairs.append(conditioned)
    rows = []
    for a, b in pairs:
        score = float(np.percentile(np.abs(Xz[a] - Xz[b]), 10))
        idx = [i for i in range(len(y)) if i not in (a, b)]
        A = per_wavelength_affine_invariants(Xs, (a, b))[idx]
        B = per_wavelength_affine_invariants(Xt, (a, b))[idx]
        met, _ = eval_regression(A, y[idx], B, y[idx], 10)
        rows.append({"a": a, "b": b, "conditioning_score": score,
                     "r2": met["r2"], "conditioned": (a, b) == conditioned,
                     "source": source, "target": target, "property": prop})
    return rows


def fig_anchor_sensitivity(df, path):
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.scatter(df.conditioning_score, df.r2, s=18, alpha=0.75,
               color=PALETTE["raw"], label="random reference pair")
    c = df[df.conditioned]
    if len(c):
        ax.scatter(c.conditioning_score, c.r2, s=70, marker="*",
                   color=PALETTE["invariant"], label="conditioned pair")
    ax.set_xlabel("anchor conditioning score\n(10th percentile separation)")
    ax.set_ylabel("cross-instrument $R^2$ (protein)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reference-sample choice controls invariant noise")
    fig.savefig(path)
    plt.close(fig)


def fig_group_fit(df, path):
    set_style()
    pairs = df.pair.tolist()
    x = np.arange(len(pairs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (col, label, color) in enumerate([
        ("identity", "identity", PALETTE["raw"]),
        ("global_affine", "global affine (SNV)", PALETTE["snv"]),
        ("per_lambda_affine", "per-λ affine", PALETTE["invariant"]),
    ]):
        ax.bar(x + (i - 1) * w, df[col], width=w, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("oracle $R^2$ of the group model")
    ax.set_ylim(min(0.9, df[["identity", "global_affine", "per_lambda_affine"]].min().min() - 0.02), 1.001)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Which group explains cross-instrument variation?")
    fig.savefig(path)
    plt.close(fig)


def within_instrument_cv(data, prop="Protein", n_splits=5):
    """Cross-validated prediction on a single instrument.

    This is the reference ceiling: if the representation works within an
    instrument, any cross-instrument loss is attributable to the instrument
    transform, not to the representation destroying the signal.
    """
    anchors = choose_anchors(data["X"]["m5"], k=5, seed=0)
    idx = [i for i in range(data["X"]["m5"].shape[0]) if i not in anchors]
    y = data["props"][:, PROPERTY_INDEX[prop]]
    rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for inst in INSTRUMENTS:
        X = data["X"][inst]
        reps = {
            "raw": X[idx],
            "snv": rep_snv(X[idx]),
            "invariant": rep_invariant(X, anchors)[idx],
        }
        for name, Z in reps.items():
            preds = np.zeros(len(idx))
            for tr, te in kf.split(Z):
                met, pred = eval_regression(Z[tr], y[idx][tr], Z[te], y[idx][te], 10)
                preds[te] = pred
            rows.append({"instrument": inst, "property": prop,
                         "representation": name, "cv_r2": r2(y[idx], preds)})
    return rows


def fig_spectra(data, path):
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = data["wavelengths"]
    for i, inst in enumerate(INSTRUMENTS):
        mu = data["X"][inst].mean(axis=0)
        ax.plot(w, mu + i * 0.15, lw=1.2, label=inst, color=["#0072B2", "#D55E00", "#009E73"][i])
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("mean reflectance (offset)")
    ax.set_title("Same 80 corn samples, three NIR instruments")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def fig_consistency(df, path):
    set_style()
    pairs = sorted(df.pair.unique())
    reps = ["raw", "snv", "invariant"]
    labels = {"raw": "raw", "snv": "SNV (global affine)", "invariant": "per-λ affine invariant"}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = np.arange(len(pairs))
    w = 0.26
    for i, r in enumerate(reps):
        d = df[df.representation == r].set_index("pair").loc[pairs]
        ax.bar(x + (i - 1) * w, d.entrywise_r2, width=w, label=labels[r],
               color=PALETTE.get(r, None))
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("entrywise $R^2$ between instruments")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Representation agreement for the same physical samples")
    fig.savefig(path)
    plt.close(fig)


def fig_transfer(df, path):
    set_style()
    d = df[df.property == "Protein"]
    pairs = sorted(d.source.unique())
    pairs = [f"{s}->{t}" for s in INSTRUMENTS for t in INSTRUMENTS if s != t]
    reps = ["raw", "snv", "invariant", "calibrated", "pds"]
    labels = {"raw": "raw", "snv": "SNV", "invariant": "invariant (2 refs)",
              "calibrated": "affine calibration (5 refs)", "pds": "PDS (5 refs)"}
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(pairs))
    w = 0.15
    for i, r in enumerate(reps):
        vals, errs = [], []
        for p in pairs:
            s, t = p.split("->")
            row = d[(d.source == s) & (d.target == t) & (d.representation == r)]
            vals.append(float(row.r2_mean.iloc[0]) if len(row) else np.nan)
            errs.append(float(row.r2_std.iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - 2) * w, vals, width=w, yerr=errs, capsize=2,
               label=labels[r], color=PALETTE.get(r, None))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=20)
    ax.set_ylabel("test $R^2$ (protein)")
    ax.set_ylim(min(-0.2, np.nanmin(vals) - 0.05), 1.0)
    ax.legend(ncol=2, frameon=False, fontsize=7)
    ax.set_title("Cross-instrument protein prediction without target labels")
    fig.savefig(path)
    plt.close(fig)


def fig_reference_scaling(df, path):
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for name, color in [("invariant", PALETTE["invariant"]),
                        ("calibrated", PALETTE["calibrated"]),
                        ("pds", PALETTE["snv"])]:
        d = df[df.representation == name]
        if len(d) == 0:
            continue
        ax.errorbar(d.n_references, d.r2_mean, yerr=d.r2_std, marker="o", capsize=2,
                    lw=1.4, label=name, color=color)
    ax.set_xlabel("number of reference samples on the target instrument")
    ax.set_ylabel("test $R^2$ (protein, m5→mp5)")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20])
    ax.set_xticklabels(["2", "3", "5", "10", "20"])
    ax.legend(frameon=False)
    ax.set_title("Two references suffice for the per-wavelength affine group")
    fig.savefig(path)
    plt.close(fig)


def fig_scatter(data, anchors, path):
    set_style()
    Xs, Xt = data["X"]["m5"], data["X"]["mp5"]
    idx = [i for i in range(Xs.shape[0]) if i not in anchors]
    U = rep_invariant(Xs, anchors)[idx]
    V = rep_invariant(Xt, anchors)[idx]
    # show one informative wavelength
    l = int(np.argmax(np.std(U, axis=0)))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    axes[0].scatter(Xs[idx, l], Xt[idx, l], s=12, alpha=0.7, color=PALETTE["raw"])
    axes[0].set_xlabel("m5 reflectance")
    axes[0].set_ylabel("mp5 reflectance")
    axes[0].set_title(f"raw, λ={data['wavelengths'][l]:.0f} nm")
    axes[1].scatter(U[:, l], V[:, l], s=12, alpha=0.7, color=PALETTE["invariant"])
    axes[1].set_xlabel("m5 invariant")
    axes[1].set_ylabel("mp5 invariant")
    axes[1].set_title("per-λ affine invariant")
    for ax in axes:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load()
    out = RESULTS / "nir_corn"
    out.mkdir(parents=True, exist_ok=True)

    print("N1: representation consistency")
    anchors = choose_anchors(data["X"]["m5"], k=5, seed=0)
    cons = representation_consistency(data, anchors)
    pd.DataFrame(cons).to_csv(out / "n1_consistency.csv", index=False)
    gf = group_fit(data)
    pd.DataFrame(gf).to_csv(out / "n1_group_fit.csv", index=False)

    print("N1b: within-instrument CV")
    within = within_instrument_cv(data)
    pd.DataFrame(within).to_csv(out / "n1_within_cv.csv", index=False)

    print("N2: transfer")
    tr = transfer(data, prop="Protein", n_anchor_sets=5)
    tr.to_csv(out / "n2_transfer.csv", index=False)
    tr_moist = transfer(data, prop="Moisture", n_anchor_sets=5)
    tr_moist.to_csv(out / "n2_transfer_moisture.csv", index=False)

    print("N3: reference scaling")
    ref = reference_scaling(data)
    ref.to_csv(out / "n3_reference_scaling.csv", index=False)

    print("N4: anchor sensitivity")
    anch = anchor_sensitivity(data)
    pd.DataFrame(anch).to_csv(out / "n4_anchor_sensitivity.csv", index=False)

    print("Figures")
    fig_spectra(data, FIGURES / "nir_spectra.pdf")
    fig_consistency(pd.DataFrame(cons), FIGURES / "nir_consistency.pdf")
    fig_group_fit(pd.DataFrame(gf), FIGURES / "nir_group_fit.pdf")
    fig_transfer(tr, FIGURES / "nir_transfer.pdf")
    fig_reference_scaling(ref, FIGURES / "nir_reference_scaling.pdf")
    fig_scatter(data, anchors, FIGURES / "nir_scatter.pdf")
    fig_anchor_sensitivity(pd.DataFrame(anch), FIGURES / "nir_anchor_sensitivity.pdf")

    summary = {
        "n_samples": int(data["X"]["m5"].shape[0]),
        "n_wavelengths": int(data["X"]["m5"].shape[1]),
        "consistency": cons,
        "group_fit": gf,
        "within_cv": within,
        "protein_transfer": tr.to_dict(orient="records"),
        "moisture_transfer": tr_moist.to_dict(orient="records"),
        "reference_scaling": ref.to_dict(orient="records"),
        "anchor_sensitivity": anch,
    }
    save_json(summary, "summary.json", "nir_corn")
    print(tr.to_string(index=False))
    print("done")


if __name__ == "__main__":
    main()
