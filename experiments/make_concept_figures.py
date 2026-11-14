"""Generate the conceptual figures used in the paper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import FIGURES, set_style


def fig_quotient(path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7))
    # Panel 1: orbits of a one-parameter group on M
    ax = axes[0]
    x = np.linspace(-2.5, 2.5, 200)
    for c in np.linspace(-2, 2, 9):
        ax.plot(x, c + 0.35 * x, color="#0072B2", lw=1.0, alpha=0.65)
    ax.scatter([0.8], [0.3], color="#D55E00", zorder=5, s=35)
    ax.annotate("m", (0.8, 0.3), xytext=(1.0, 1.2),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("measurement space $M$\norbits of $G$", fontsize=9)
    ax.set_xlabel("")

    # Panel 2: quotient
    ax = axes[1]
    ax.axhline(0, color="#0072B2", lw=2.5)
    ax.scatter([0.8], [0], color="#D55E00", zorder=5, s=35)
    ax.annotate("$[m]$", (0.8, 0), xytext=(1.0, 0.35), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("quotient $M/G$\n(invariant representation)", fontsize=9)

    # Panel 3: invariant function
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color="#009E73", lw=2)
    ax.scatter([0.35], [0.35], color="#D55E00", s=35, zorder=5)
    ax.annotate("$I(m)$", (0.35, 0.35), xytext=(0.45, 0.1), fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("invariant $I: M \\to M/G$\n$I(g\\cdot m)=I(m)$", fontsize=9)

    fig.suptitle("The instrument group, its orbits, and the quotient", y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_table(path):
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    # three instruments as boxes
    xs = [0.06, 0.40, 0.74]
    names = ["instrument A", "instrument B", "instrument C"]
    readings = ["2.0", "4.0", "3.0"]
    for x, name, r in zip(xs, names, readings):
        ax.add_patch(Rectangle((x, 0.56), 0.20, 0.24, fc="#EAF2F8", ec="#0072B2", lw=1.2))
        ax.text(x + 0.10, 0.74, name, ha="center", va="center", fontsize=9)
        ax.text(x + 0.10, 0.63, f"reads {r} units", ha="center", va="center", fontsize=9)
    # arrows
    for x in xs:
        ax.annotate("", xy=(x + 0.10, 0.48), xytext=(x + 0.10, 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#666666"))
    # table
    ax.add_patch(Rectangle((0.20, 0.10), 0.60, 0.30, fc="#FDF2E9", ec="#D55E00", lw=1.2))
    ax.text(0.50, 0.34, "the same table", ha="center", fontsize=10, weight="bold")
    ax.text(0.50, 0.24, "affine instrument model:  $y = k\\,x + d$", ha="center", fontsize=9)
    ax.text(0.50, 0.15, "invariant:  $(y_A-y_B)/(y_C-y_B) = (2-4)/(3-4) = 2$", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Three instruments, one table: the ratio is invariant under a common affine change", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_hierarchy(path):
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axis("off")
    rows = [
        ("identity", "no instrument variation", "0", "$d$", "#999999"),
        ("global affine", "$y \\mapsto g y + o$ (whole spectrum)", "2", "$d-2$", "#0072B2"),
        ("per-channel affine", "$y_i \\mapsto g_i y_i + o_i$", "$2p$", "$d-2p$", "#D55E00"),
        ("per-pixel affine", "$y_i \\mapsto g_i y_i + o_i$, all $i$", "$2d$", "0", "#CC79A7"),
    ]
    y = 0.86
    for name, action, gdim, invdim, color in rows:
        ax.add_patch(Rectangle((0.03, y - 0.13), 0.94, 0.15, fc=color, alpha=0.12, ec=color, lw=1.1))
        ax.text(0.05, y - 0.045, name, fontsize=9, weight="bold", va="center")
        ax.text(0.27, y - 0.045, action, fontsize=8.5, va="center")
        ax.text(0.70, y - 0.045, f"group dim {gdim}", fontsize=8.5, va="center")
        ax.text(0.86, y - 0.045, f"invariant dim {invdim}", fontsize=8.5, va="center")
        y -= 0.20
    ax.text(0.5, 0.02, "As the group grows, the quotient shrinks.  At per-pixel affine, the invariant ring is trivial.",
            ha="center", fontsize=8.5, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("The group determines what survives: an observability hierarchy", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_evidence(path):
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    modalities = ["MOX", "NIR", "EEG", "Raman", "Microscopy", "Mass spec", "Sequencing"]
    evidence = ["real cross-\ninstrument", "real cross-\ninstrument", "real signal,\nsynthetic G",
                "real signal,\nsynthetic G", "real image,\nsynthetic G", "synthetic", "real genome,\nsimulated reads"]
    colors = ["#009E73", "#009E73", "#E69F00", "#E69F00", "#E69F00", "#CC79A7", "#CC79A7"]
    y = np.arange(len(modalities))[::-1]
    ax.barh(y, [1] * len(y), color=colors, alpha=0.75, height=0.65)
    for yi, e in zip(y, evidence):
        ax.text(0.03, yi, e, va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(modalities)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Evidence type by modality", fontsize=10)
    ax.spines["bottom"].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_quotient(FIGURES / "concept_quotient.pdf")
    fig_table(FIGURES / "concept_table.pdf")
    fig_hierarchy(FIGURES / "concept_hierarchy.pdf")
    fig_evidence(FIGURES / "concept_evidence.pdf")
    print("wrote conceptual figures")


if __name__ == "__main__":
    main()
