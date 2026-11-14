# Instrument Groups and Invariants

A research repository for a physics-derived theory of cross-instrument
measurement, with proofs, real cross-instrument experiments, and honest
negative results.

The central idea: an instrument induces a transformation group on a measurement
space. The device-invariant content of a measurement is the quotient by that
group. The group is derived from the instrument's physics — a power law for MOX
gas sensors, Beer–Lambert for NIR/Raman, a per-channel amplifier for EEG, an
m/z calibration for mass spectrometry, a smooth illumination field for
microscopy, and a stochastic error channel for DNA sequencing. The group
determines what is observable: if the group is as large as the measurement
space, no nonconstant invariant exists.

The framework is a synthesis of measurement theory (Klein, Stevens, Krantz et
al.), invariant theory (Weyl, Mumford, Olver), calibration transfer (Wang &
Kowalski; Fearn; Workman), and domain adaptation (Ben-David et al.; Ganin et
al.; Sun et al.). It does not replace those methods; it derives the group they
assume and states when no invariant can exist.

---

## Read this first

- **Paper (PDF):** [`paper/draft.pdf`](paper/draft.pdf) — compiled 47-page
  draft.
- **Paper (source):** [`paper/draft.md`](paper/draft.md) — formal framework,
  theorems, experiments, dataset-fitness table, limitations.
- **Plain-English explainer:** [`docs/explainer.md`](docs/explainer.md) — the
  two-rulers/one-table intuition, logs, groups, calibration, and what each
  modality shows.
- **Literature review:** [`lit/review.md`](lit/review.md) — what is old, what is
  synthesized, and what this paper does not claim.
- **Dataset fitness:** [`results/fitness.md`](results/fitness.md) — which public
  datasets support cross-instrument claims and which are temporal drift or
  simulation.
- **All key numbers:** [`results/summary.md`](results/summary.md) — generated
  from the experiment CSVs by `experiments/aggregate.py`.
- **Results index:** [`results/README.md`](results/README.md) — maps each paper
  section to the CSV/JSON file that produced it.

---

## Repository layout

```
invmath/                 symbolic derivations, numerical invariants, tests
experiments/
  download_data.py       UCI drift and Corn download + manifest
  common.py              metrics, plotting, classifiers, regressors
  mox/run.py             UCI drift transfer, group-stress, slope stability
  nir_corn/run.py        Corn 3-instrument transfer, group fit, references
  eeg/run.py             real EEG + synthetic gain/offset/reference
  raman/run.py           real Raman + synthetic affine/shift
  microscopy/run.py      real image + synthetic illumination
  massspec/run.py        controlled synthetic m/z calibration
  sequencing/run.py      real phiX174 genome + simulated platforms
  make_concept_figures.py
  aggregate.py           builds results/summary.md from CSVs
  fitness.py             builds results/fitness.csv / .md
  make_references.py     builds paper/references.md from refs.bib
paper/
  draft.md               paper source
  draft.pdf              compiled PDF
  build_pdf.py           PDF build (Pandoc -> DOCX -> LibreOffice)
  refs.bib               bibliography (source of truth)
  references.md          generated author-year list
  figures/               generated figures (PDF)
docs/
  explainer.md           plain-English companion
lit/
  review.md              literature review
data/                    downloaded data + manifest (not committed)
results/                 generated CSV/JSON summaries
```

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python experiments/download_data.py
python experiments/mox/run.py
python experiments/nir_corn/run.py
python experiments/eeg/run.py
python experiments/raman/run.py
python experiments/microscopy/run.py
python experiments/massspec/run.py
python experiments/sequencing/run.py
python experiments/make_concept_figures.py
python experiments/aggregate.py
python experiments/fitness.py
python experiments/make_references.py
python -m pytest invmath/tests -q
```

The experiments download their own data on first run. The download script
records provenance and hashes in `data/manifest.json`. The data are not
committed.

Compile the paper. The repository build works without a TeX installation:

```bash
python paper/build_pdf.py
```

It uses Pandoc for DOCX and LibreOffice for the final PDF, then fixes table
page breaks and duplicate figure captions. The intermediate files go to
`paper/build/` (ignored by git).

If you have a LaTeX toolchain, the direct route is:

```bash
pandoc paper/draft.md --bibliography=paper/refs.bib \
  --resource-path=paper --citeproc --pdf-engine=xelatex -o paper/draft.pdf
```

---

## Key results

### NIR: real cross-instrument transfer

The Corn dataset: 80 physical samples, 3 NIR instruments, 700 wavelengths. The
per-wavelength affine group is an adequate model (oracle R² ≥ 0.9985). A
two-reference invariant transfers at R² 0.834–0.910 across all six ordered
instrument pairs; raw transfer is as low as −9.85 for the distant pair.
Calibration with five references reaches 0.886–0.933; PDS reaches 0.697–0.905.
More references improve the section, not the invariant.

### MOX: temporal drift, not three instruments

The UCI drift data: 13,910 measurements, 16 sensors, 10 batches over 36 months.
Raw pooled accuracy 0.405; log 0.474; normalized selectivity 0.483; exact
per-sensor gain invariance (batch centering) 0.515 pooled but 0.451 per-batch;
five labelled target samples per class 0.711. The full per-sensor affine group
is transitive on a single measurement vector: gas identity is not observable
under arbitrary sensor replacement. The restricted gain group leaves slope
ratios, stable for five of six gases.

### Other modalities

- **EEG:** real PhysioNet signal, synthetic per-channel gain/offset. Baseline
  log ratio 0.834 under gain+offset vs 0.686 raw; a common-reference transform
  breaks the invariant (0.756 baseline, 0.544 baseline ratio).
- **Raman:** 17 real spectra, synthetic affine. Underpowered; invariant 0.533
  vs raw 0.457 under affine, but identity raw is 0.663. Reported as a negative
  result.
- **Microscopy:** real image, synthetic illumination. Dice 0.894 raw → 0.965
  flat-field (oracle 0.987).
- **Mass spec:** synthetic controlled spectra. Raw under m/z affine 0.777 ±
  0.279; two-peak calibration 1.000; fragmentation drops calibration to 0.970.
- **Sequencing:** real phiX174 genome, simulated platforms. Coarse regions are
  robust (~1.0); fine haplotypes are ~0.64–0.67 and platform normalization does
  not help. The error channel is not affine.

---

## Evidence labels

Every empirical claim is labelled:

| label | meaning |
|---|---|
| **real cross-instrument** | same physical samples on at least two real instruments |
| **real signal, synthetic group** | real measurements from one instrument, labelled synthetic transform |
| **synthetic** | both signal and instrument transform simulated |
| **derivation** | mathematical consequence, no experiment |

The dataset-fitness table records the label for each modality. "Multi-device"
is often claimed for temporal drift on one device (MOX), for different
instruments that did not measure the same samples (mass spec), or for one
instrument with a synthetic transform (EEG, Raman, microscopy). Those are
legitimate experiments, but they are not the same as a cross-instrument
benchmark.

---

## Mathematical core

- **Affine group.** \(y \mapsto ky+d\). The three-point ratio
  \((y_1-y_2)/(y_3-y_2)\) is invariant.
- **Polynomial invariants of Aff(1) are trivial.** Translation forces
  dependence on differences; scaling forces homogeneous degree 0; a degree-0
  polynomial is constant. The rational invariant field has degree \(n-2\).
- **MOX.** \(s=aC^b\) gives \(y_B=ky_A+d\) in log space, with
  \(k=b_B/b_A\); the three-point log ratio equals the log-concentration ratio.
- **Full per-sensor affine group.** \(\prod_i \mathrm{Aff}(1)\) is transitive
  on a single vector; the quotient is a point; gas identity is not observable.
- **Per-wavelength affine.** \(A'_{i\lambda}=g_\lambda A_{i\lambda}+o_\lambda\)
  gives the cross-sample invariant
  \((A_{i\lambda}-A_{1\lambda})/(A_{2\lambda}-A_{1\lambda})\); two reference
  samples suffice.
- **Sequencing.** Substitution-only channels are linear on k-mer frequencies;
  indels and context effects are not affine; no general low-dimensional
  invariant exists.

The symbolic checks are in `invmath/groups.py`; the report is
`results/math_report.txt`; the numerical tests are in
`invmath/tests/test_invariants.py` (9 passing).

---

## Relation to the companion OpenSmell paper

The companion paper *Towards Interoperable Digital Olfaction* (Zenodo,
DOI [10.5281/zenodo.21243013](https://doi.org/10.5281/zenodo.21243013)) proves
that raw normalized MOX responses cannot transfer zero-shot across devices with
different power-law exponents. This repository does not contradict that result;
it explains it structurally. The raw response is not invariant under the affine
device map. The invariant objects are lower-dimensional (ratios of log
differences, slope ratios, batch-centered representations). They are useful for
interoperability and drift robustness, not for free molecular identification.

---

## Limitations

- The derived groups are models. NIR is adequate; MOX is partially adequate;
  sequencing is not affine. The group must be validated for a new modality.
- EEG, Raman, microscopy, mass spec, and sequencing use synthetic transforms or
  simulated reads. They test the calculus, not the instrument.
- Samples are small: 17 Raman spectra, 80 NIR samples, one EEG subject.
- The invariant divides by anchor differences; poor anchors amplify noise
  (NIR random-pair median R² −0.073).
- The calibrated MOX baseline uses target labels; it is a section, not an
  invariant. Comparisons state the information used.

---

## Citation

If you use this repository, cite the paper draft and the companion paper:

```bibtex
@misc{james2026invariants,
  author = {James, Praise},
  title  = {Instrument Groups and Invariants: A Physics-Derived Framework for Cross-Instrument Measurement},
  year   = {2026},
  note   = {Manuscript and code: OpenSmell/invariant-measurement}
}

@misc{james2026,
  author       = {James, Praise},
  title        = {Towards Interoperable Digital Olfaction: A Modular Feature Framework for Metal-Oxide Semiconductor Electronic Noses},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.21243013},
  year         = {2026}
}
```

---

## What this repository does not claim

It does not claim that invariance solves interoperability. It does not claim
that the derived groups are exact. It does not claim that synthetic instrument
transforms are recordings from real second instruments. It does not claim
priority for invariance, calibration transfer, or domain adaptation. The
contribution is the synthesis, the physics-derived groups, the observability
boundaries, the dataset-fitness table, and the honest negatives.
