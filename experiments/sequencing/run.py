"""
DNA sequencing case study: real reference genome, simulated reads.

Why this modality is different
------------------------------
Sequencing instruments do not apply an affine map to a fixed measurement vector.
They apply a stochastic error channel to a discrete sequence: substitutions,
insertions, deletions, and context-dependent errors.  The composition of these
operations is not a finite-dimensional affine group on a fixed vector space.

Two tasks
---------
coarse : classify reads from five distant genomic regions (100 bp reads, 4-mers)
fine   : classify reads from two haplotypes that differ by 8 SNPs in 800 bp
         (50 bp reads, 5-mers)

Four simulated platform channels:
    A            low substitution error (0.2%)
    B_sub        high substitution error (2%)
    B_indel      substitution + 1% insertions + 1% deletions
    B_context    substitution + context-dependent (CpG-like) errors

Features are k-mer frequency vectors.  We compare:
    raw             no correction
    platform_norm   divide by the platform's mean k-mer spectrum (unlabelled)
A multiplicative/linear channel quotient can remove a platform bias; an
indel/context channel is not affine and no such quotient exists.

This is a simulation.  It uses the real phiX174 genome and published error-rate
ranges, but it is not a substitute for real multi-platform data (GIAB).
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

import matplotlib.pyplot as plt

URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
       "db=nuccore&id=NC_001422.1&rettype=fasta&retmode=text")
BASES = "ACGT"
PLATFORMS = ["A", "B_sub", "B_indel", "B_context"]
PARAMS = {
    "A": dict(sub=0.002),
    "B_sub": dict(sub=0.02),
    "B_indel": dict(sub=0.01, ins=0.01, dele=0.01),
    "B_context": dict(sub=0.01, context=True),
}


def load_genome():
    out = DATA / "sequencing"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phix174.fasta"
    if not path.exists():
        import requests
        r = requests.get(URL, timeout=90)
        r.raise_for_status()
        path.write_text(r.text)
    seq = "".join(line.strip() for line in path.read_text().splitlines()
                  if not line.startswith(">"))
    return seq.upper()


def mutate(read, rng, sub=0.002, ins=0.0, dele=0.0, context=False):
    out = []
    for ch in read:
        if context and ch == "C" and rng.random() < 0.10:
            out.append(rng.choice(list("AT")))
            continue
        r = rng.random()
        if r < dele:
            continue
        if r < dele + ins:
            out.append(rng.choice(list(BASES)))
            out.append(ch)
        elif r < dele + ins + sub:
            out.append(rng.choice([b for b in BASES if b != ch]))
        else:
            out.append(ch)
    return "".join(out)


def kmer_freq(reads, k):
    vocab = {"".join(p): i for i, p in enumerate(itertools.product(BASES, repeat=k))}
    X = np.zeros((len(reads), 4 ** k))
    for i, r in enumerate(reads):
        for j in range(len(r) - k + 1):
            km = r[j:j + k]
            if km in vocab:
                X[i, vocab[km]] += 1
    X = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
    return X


def platform_normalize(X, reference_mean, eps=1e-6):
    Z = X / (reference_mean[None, :] + eps)
    return Z / np.maximum(Z.sum(axis=1, keepdims=True), 1e-12)


def read_from(template, n, platform, read_len, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.integers(0, max(1, len(template) - read_len))
        out.append(mutate(template[s:s + read_len], rng, **PARAMS[platform]))
    return out


def coarse_data(seq, platform, n_per_class, seed):
    bounds = np.linspace(0, len(seq), 6, dtype=int)
    reads, y = [], []
    for c in range(5):
        lo, hi = bounds[c], bounds[c + 1]
        for r in read_from(seq[lo:hi], n_per_class, platform, 100, seed + c):
            reads.append(r)
            y.append(c)
    return reads, np.asarray(y)


def fine_data(seq, platform, n_per_class, seed):
    rng = np.random.default_rng(seed)
    start, length = 500, 800
    ref = seq[start:start + length]
    mut = list(ref)
    for pos in np.linspace(20, length - 20, 8).astype(int):
        mut[pos] = rng.choice([b for b in BASES if b != mut[pos]])
    hap2 = "".join(mut)
    reads, y = [], []
    for label, template in [(0, ref), (1, hap2)]:
        for r in read_from(template, n_per_class, platform, 50, seed + 10 * label):
            reads.append(r)
            y.append(label)
    return reads, np.asarray(y)


def evaluate_task(task, seq):
    rows = []
    for platform in PLATFORMS:
        if task == "coarse":
            n_tr, n_te, k = 120, 120, 4
            # simulate training and test reads for this platform
            tr_reads, ytr = coarse_data(seq, platform, n_tr, seed=100)
            te_reads, yte = coarse_data(seq, platform, n_te, seed=200)
        else:
            n_tr, n_te, k = 300, 300, 5
            tr_reads, ytr = fine_data(seq, platform, n_tr, seed=100)
            te_reads, yte = fine_data(seq, platform, n_te, seed=200)
        Xtr, Xte = kmer_freq(tr_reads, k), kmer_freq(te_reads, k)
        # train on platform A; test on the target platform
        if platform != "A":
            if task == "coarse":
                a_reads, y_a = coarse_data(seq, "A", n_tr, seed=100)
            else:
                a_reads, y_a = fine_data(seq, "A", n_tr, seed=100)
            Xtr, ytr = kmer_freq(a_reads, k), y_a
        reps = {
            "raw": (Xtr, Xte),
            "platform_norm": (platform_normalize(Xtr, Xtr.mean(axis=0)),
                              platform_normalize(Xte, Xte.mean(axis=0))),
        }
        for name, (A, B) in reps.items():
            clf = make_classifier("logreg", seed=0)
            clf.fit(A, ytr)
            rows.append({"task": task, "platform": platform,
                         "representation": name,
                         "accuracy": accuracy(yte, clf.predict(B))})
    return rows


def fig(df, path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, task, chance in [(axes[0], "coarse", 1 / 5), (axes[1], "fine", 1 / 2)]:
        d = df[df.task == task]
        platforms = ["B_sub", "B_indel", "B_context"]
        x = np.arange(len(platforms))
        w = 0.32
        for i, r in enumerate(["raw", "platform_norm"]):
            s = d[d.representation == r].set_index("platform").reindex(platforms)
            ax.bar(x + (i - 0.5) * w, s.accuracy, width=w,
                   label="raw" if r == "raw" else "platform-normalised",
                   color=PALETTE.get(r, None))
        ax.axhline(chance, color=PALETTE["chance"], ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["substitution", "+ indels", "context"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("accuracy (train on A)")
        ax.set_title("coarse regions" if task == "coarse" else "fine haplotypes")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Sequencing: the error channel is not an affine group", y=1.03)
    fig.savefig(path)
    plt.close(fig)


def main():
    out = RESULTS / "sequencing"
    out.mkdir(parents=True, exist_ok=True)
    seq = load_genome()
    rows = evaluate_task("coarse", seq) + evaluate_task("fine", seq)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sequencing_transfer.csv", index=False)
    fig(df, FIGURES / "sequencing_transfer.pdf")
    save_json(df.to_dict(orient="records"), "summary.json", "sequencing")
    print(df.pivot_table(index=["task", "platform"], columns="representation",
                         values="accuracy").round(3).to_string())
    print("note: real phiX174 genome; reads and platform channels are simulated.")


if __name__ == "__main__":
    main()
