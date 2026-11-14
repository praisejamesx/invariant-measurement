"""
Microscopy case study: real image, synthetic illumination.

Data
----
`skimage.data.cells3d()`: a real fluorescence microscopy volume (60 z-slices,
2 channels, 256x256).  We use one 2-D slice of the nuclei channel.

Instrument model
----------------
A spatially varying multiplicative illumination field and an additive detector
offset:
    I'(x,y) = g(x,y) * I(x,y) + o(x,y)
with g, o smooth (low spatial frequency).  Without a smoothness constraint this
group is per-pixel affine and has no within-image invariant; with smoothness, a
flat-field estimate is the quotient section.

Experiment
----------
Segment nuclei (cell vs background) from the original image (ground truth) and
from a synthetic second instrument.  Compare:
    raw threshold
    flat-field correction (divide by a large-scale illumination estimate)
    reference flat-field (use a blank reference image)
The metric is Dice overlap with the ground-truth mask.

Real image, synthetic instrument: the transform is labelled as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import FIGURES, PALETTE, RESULTS, save_json, set_style

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import data, filters, measure, morphology
from skimage.util import img_as_float


def load_image():
    vol = data.cells3d()  # (z, channel, y, x)
    img = img_as_float(vol[30, 1])  # nuclei channel
    return img


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum() + 1e-12)


def gt_mask(img):
    t = filters.threshold_otsu(img)
    m = img > t
    m = morphology.remove_small_objects(m, 30)
    m = morphology.binary_closing(m, morphology.disk(2))
    return m


def instrument(img, rng, strength=1.0):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # smooth illumination: a tilted Gaussian plus a low-order polynomial
    g = np.exp(strength * (0.7 * (xx / w) - 0.5 * (yy / h)))
    g = gaussian_filter(g, sigma=25)
    o = 0.15 * gaussian_filter(rng.normal(size=(h, w)), sigma=20)
    out = g * img + o
    out = out + rng.normal(0, 0.01, size=out.shape)
    return out, g, o


def estimate_illumination(img, sigma=35):
    # large-scale background: a strong blur of the image itself
    bg = gaussian_filter(img, sigma=sigma)
    bg = np.clip(bg, 1e-6, None)
    return bg / np.median(bg)


def segment(img):
    t = filters.threshold_otsu(img)
    m = img > t
    m = morphology.remove_small_objects(m, 30)
    m = morphology.binary_closing(m, morphology.disk(2))
    return m


def fig_microscopy(img, bad, corrected, masks, path):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("original (real)")
    axes[0, 1].imshow(bad, cmap="gray")
    axes[0, 1].set_title("synthetic instrument")
    axes[0, 2].imshow(corrected, cmap="gray")
    axes[0, 2].set_title("flat-field corrected")
    axes[1, 0].imshow(masks["gt"], cmap="gray")
    axes[1, 0].set_title("ground-truth mask")
    axes[1, 1].imshow(masks["raw"], cmap="gray")
    axes[1, 1].set_title(f"raw threshold (Dice {masks['raw_dice']:.3f})")
    axes[1, 2].imshow(masks["flat"], cmap="gray")
    axes[1, 2].set_title(f"flat-field (Dice {masks['flat_dice']:.3f})")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    out = RESULTS / "microscopy"
    out.mkdir(parents=True, exist_ok=True)
    img = load_image()
    gt = gt_mask(img)
    rows = []
    examples = None
    rng = np.random.default_rng(0)
    for rep in range(10):
        bad, g, o = instrument(img, rng)
        corrected = bad / estimate_illumination(bad)
        m_raw = segment(bad)
        m_flat = segment(corrected)
        # reference flat-field: correct with the known illumination (oracle)
        m_ref = segment((bad - o) / g)
        rows.append({
            "rep": rep,
            "raw_dice": dice(gt, m_raw),
            "flat_dice": dice(gt, m_flat),
            "reference_dice": dice(gt, m_ref),
        })
        if rep == 0:
            examples = (img, bad, corrected,
                        {"gt": gt, "raw": m_raw, "flat": m_flat,
                         "raw_dice": dice(gt, m_raw), "flat_dice": dice(gt, m_flat)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "microscopy_dice.csv", index=False)
    summary = df[["raw_dice", "flat_dice", "reference_dice"]].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["method", "mean", "std"]
    save_json(summary.to_dict(orient="records"), "summary.json", "microscopy")
    fig_microscopy(*examples, FIGURES / "microscopy_flatfield.pdf")
    print(summary.round(3).to_string(index=False))
    print("note: real microscopy image; illumination transform is synthetic.")


if __name__ == "__main__":
    main()
