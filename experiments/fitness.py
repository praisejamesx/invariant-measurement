"""Write the dataset-fitness table (Table 1 in the paper)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import RESULTS

ROWS = [
    dict(
        modality="MOX (metal-oxide gas sensors)",
        dataset="UCI Gas Sensor Array Drift (Vergara et al., 2012)",
        n_instruments="1 array, 10 batches over 36 months",
        same_samples="yes (repeated gases/concentrations)",
        metadata="gas, concentration, batch, 16 sensors",
        group="per-sensor gain / affine in log space",
        n_samples=13910,
        verdict="temporal pseudo-device; tests drift, not distinct hardware",
        evidence="real",
    ),
    dict(
        modality="NIR spectroscopy",
        dataset="Corn (Eigenvector; Cargill)",
        n_instruments="3 (m5, mp5, mp6)",
        same_samples="yes (80 physical samples)",
        metadata="700 wavelengths, moisture/oil/protein/starch",
        group="per-wavelength affine",
        n_samples=80,
        verdict="qualifying cross-instrument benchmark",
        evidence="real cross-instrument",
    ),
    dict(
        modality="Raman spectroscopy",
        dataset="Urine Raman (RimaMansour/Raman-spectraAnalysis)",
        n_instruments="1",
        same_samples="no paired instruments",
        metadata="healthy/diseased labels; 1000-point spectra",
        group="per-wavelength affine + wavelength shift",
        n_samples=17,
        verdict="no qualifying cross-instrument set; real spectra + synthetic instrument",
        evidence="real signal, synthetic group",
    ),
    dict(
        modality="Mass spectrometry",
        dataset="MassBank / MoNA (records inspected); synthetic controlled set",
        n_instruments="many records, not paired by sample",
        same_samples="no stable paired sample set",
        metadata="instrument metadata exists, often incomplete",
        group="m/z affine + intensity gain; fragmentation outside",
        n_samples=1000,
        verdict="no qualifying paired set; controlled synthetic experiment",
        evidence="synthetic",
    ),
    dict(
        modality="EEG",
        dataset="PhysioNet eegmmidb (S001, runs 3-4)",
        n_instruments="1 headbox, 64 channels",
        same_samples="yes across time, not across headboxes",
        metadata="64 channel labels, 160 Hz, 124 two-second windows",
        group="per-channel affine + common reference",
        n_samples=124,
        verdict="no paired headbox set; real signal + synthetic instrument",
        evidence="real signal, synthetic group",
    ),
    dict(
        modality="MEG",
        dataset="MNE sample / public MEG datasets",
        n_instruments="1 scanner per dataset",
        same_samples="simultaneous EEG possible; no paired MEG",
        metadata="sensor geometry, head position",
        group="per-channel gain + reference",
        n_samples=0,
        verdict="derivation only; no paired-instrument public data",
        evidence="derivation",
    ),
    dict(
        modality="Microscopy",
        dataset="scikit-image cells3d (real fluorescence image)",
        n_instruments="1",
        same_samples="no paired scanners",
        metadata="2 channels, 60 z-slices",
        group="smooth illumination field + gain/offset",
        n_samples=1,
        verdict="no paired-scanner set; real image + synthetic illumination",
        evidence="real image, synthetic illumination",
    ),
    dict(
        modality="DNA sequencing",
        dataset="GIAB (real multi-platform); phiX174 (5,386 bp) simulation here",
        n_instruments="multiple platforms (GIAB); 1 simulated pair here",
        same_samples="yes in GIAB; simulated here",
        metadata="platform, read length, quality",
        group="stochastic error channel (not affine)",
        n_samples=1200,
        verdict="qualifying data exist but are large; 1,200 simulated reads demonstrate the boundary",
        evidence="real genome, simulated reads",
    ),
]


def main():
    df = pd.DataFrame(ROWS)
    out = RESULTS / "fitness.csv"
    df.to_csv(out, index=False)
    md = df.to_markdown(index=False)
    (RESULTS / "fitness.md").write_text("# Dataset fitness\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
