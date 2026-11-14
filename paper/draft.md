# Instrument Groups and Invariants: A Physics-Derived Framework for Cross-Instrument Measurement

**Praise James**  
OpenSmell / CogLab  
Draft — November 2026

---

## Abstract

Two instruments measuring the same sample return different numbers. The
question is not whether the numbers differ, but whether there is a principled
object, computable from the measurements, that does not. We formalize an
instrument as a transformation group acting on a measurement space and define
the device-invariant content of a measurement as the quotient. The group is
derived from the instrument's physics rather than chosen for convenience:
metal-oxide gas sensors give an affine group in log space; Beer–Lambert
spectroscopy gives a per-wavelength affine group; EEG and MEG give a
per-channel affine group plus a reference transform; mass spectrometry gives an
affine map on the m/z axis; microscopy gives a smooth illumination field; DNA
sequencing gives a stochastic error channel that is not affine.

The group determines what is observable. We prove that the full per-sensor
affine group is transitive on a single MOX measurement vector, so the quotient
is a point and no nonconstant invariant exists; gas identity is not observable
under arbitrary sensor replacement. Restricting to per-sensor gain leaves
cross-sensor slope ratios, which are stable for five of six gases in the UCI
drift data. For per-wavelength spectroscopy, two reference samples per
instrument are necessary and sufficient, and the cross-sample ratio is
invariant at every wavelength. For sequencing, a substitution-only channel is
linear on k-mer frequencies, but indels and context-dependent errors make it
nonlinear; no general low-dimensional affine invariant exists.

We test the framework on real cross-instrument data and on labelled
real-signal/synthetic-instrument experiments. NIR transfer on the Corn data (80
samples, 3 instruments) reaches R² 0.83–0.91 with a two-reference invariant,
against raw transfer as low as −9.85. MOX cross-batch accuracy rises from 0.390
(raw) to 0.482 (normalized selectivity) and 0.711 with five labelled target
samples per class; the exact per-sensor gain invariant (batch centering)
reaches 0.515 pooled but 0.451 per-batch, competitive with selectivity. The
common-gain invariant (pairwise log ratios) reaches 0.436, illustrating that
invariance and information trade off. Real EEG, Raman, and microscopy signals
show where the group model holds and where a reference transform breaks it.
The paper includes a dataset-fitness table that separates genuine
cross-instrument evidence from temporal drift and simulation. The framework
does not replace calibration transfer or domain adaptation; it derives the
group those methods assume and states when no invariant can exist.

**Keywords:** instrument invariance, measurement theory, calibration transfer,
domain adaptation, electronic nose, near-infrared spectroscopy, mass
spectrometry, EEG, DNA sequencing.

---

## 1. Introduction

### 1.1 The interoperability problem

A measurement is an interaction between a sample and an instrument. Change the
instrument and the interaction changes. In spectroscopy the change is a
wavelength-dependent gain and offset; in gas sensing it is a change in the
sensor's power-law response; in EEG it is a different amplifier gain, offset,
and reference; in mass spectrometry it is an m/z calibration and a different
fragmentation pattern; in sequencing it is a different error channel. The
numbers move. The sample does not.

The practical response has been calibration transfer: measure a few reference
samples on both instruments, estimate a mapping, and apply it to new samples
[@wang1992; @bouveresse1996; @fearn2001; @workman2018]. The machine-learning
response has been domain adaptation: learn a representation that is empirically
invariant across domains [@bendavid2010; @ganin2016; @sun2016; @gretton2012;
@pan2010]. Both work, and both leave a prior question unanswered: *which
transformations is the instrument entitled to induce?* If the answer is "any
transformation at all," no invariant exists. If the answer is "a common scale,"
a ratio exists. The answer is a scientific claim about the measurement physics,
not a modelling convenience.

This paper takes that question seriously. We define the instrument as a group
acting on measurement space, derive the group from the instrument equations,
and compute the quotient. The quotient is the device-invariant representation.
The size of the group determines what survives.

### 1.2 Three instruments, one table

> **Box 1 — Three instruments, one table.**
>
> Three people measure the same table with three different rulers. Ruler A reads
> 2.0 units, ruler B reads 4.0, ruler C reads 3.0. Each ruler has its own zero
> and its own scale, so the readings are related by an affine map
> \(y \mapsto k y + d\).
>
> The readings differ, but the ratio
> \[
>   \frac{y_A-y_B}{y_C-y_B} = \frac{2-4}{3-4} = 2
> \]
> is the same for all three rulers. It is invariant under a common affine
> change. It is not the table length; it is the shape of the three readings
> relative to one another.
>
> To recover the length in metres, someone must measure a known object on each
> ruler. That is a *section* of the quotient: a choice of one representative
> per orbit. Calibration is a section. The invariant is what remains when the
> section is forgotten.
>
> The rest of the paper is this box, made precise and tested on real
> instruments.

The box is deliberately simple. The difficulty is not the algebra of a single
affine map; it is deciding which group each instrument induces, proving what
the quotient retains, and recognizing when the group is so large that the
quotient is a point.

### 1.3 Contributions

This paper makes five contributions.

1. **A formal instrument-group framework.** We define measurement space,
   instrument group, orbit, quotient, invariant field, and observability in one
   place and use the same definitions across modalities (Section 2).
2. **Physics-derived groups.** We derive the group from the instrument
   equations for MOX, NIR/Raman, mass spectrometry, EEG/MEG, microscopy, and
   sequencing (Section 3). The derivations are collected in
   `invmath/groups.py` and verified symbolically.
3. **Observability theorems.** We prove that the full per-sensor affine group
   is transitive on a single MOX vector, so gas identity is not observable
   under arbitrary sensor replacement; that per-wavelength affine spectroscopy
   has a non-trivial quotient requiring two references; and that the sequencing
   error channel is not affine, so no general low-dimensional invariant exists
   (Section 4).
4. **Cross-instrument experiments.** We test the framework on the UCI MOX
   drift data (13,910 measurements, 10 batches), the Corn NIR data (80 samples,
   3 instruments), real EEG, Raman, and microscopy signals with labelled
   synthetic instrument transforms, controlled synthetic mass spectra, and
   simulated sequencing reads on the real phiX174 genome (Section 5).
5. **A dataset-fitness table and honest negatives.** We state which public
   datasets actually support cross-instrument claims, which are temporal drift
   on one device, and which are simulations (Section 5.9). We report the
   negative results: the full MOX group destroys gas identity, the common-gain
   invariant is not the best classifier, the exact per-sensor gain invariant is
   competitive but not better per-batch, the Raman data are underpowered, and
   platform normalization does not rescue fine-grained sequencing.

### 1.4 Evidence labels

Every empirical claim in this paper carries one of four labels.

| label | meaning |
|---|---|
| **real cross-instrument** | the same physical samples measured on at least two real instruments |
| **real signal, synthetic group** | real measurements from one instrument, with a labelled synthetic transform representing a second instrument |
| **synthetic** | both the signal and the instrument transform are simulated |
| **derivation** | a mathematical consequence of the instrument model, with no experiment |

The labels are not decoration. A synthetic transform can test whether an
algorithm recovers a group that is known to be present; it cannot establish
that a real instrument induces that group. The dataset-fitness table in
Section 5.9 records the distinction.

### 1.5 Relation to prior work

The ingredients of this paper are old. Klein's Erlanger Programm defines a
geometry by its group [@klein1872; @klein1893]. Stevens and the representational
measurement tradition define scale types by admissible transformations
[@stevens1946; @krantz1971; @narens1985; @luce2002]. Invariant theory supplies
the quotient and the invariant field [@weyl1939; @mumford1994; @olver1995].
Calibration transfer supplies the practical sections [@wang1992;
@bouveresse1996; @fearn2001; @workman2018]. Domain adaptation supplies
empirical invariance [@bendavid2010; @ganin2016; @sun2016]. Equivariant
networks supply architectures that respect a chosen group [@cohen2016;
@kondor2018; @bronstein2021].

The contribution is the synthesis and the discipline it imposes: derive the
group from physics, prove what the quotient retains, measure the trade-off
between invariance and information, and state plainly when no invariant exists.
We do not claim priority for invariance, calibration transfer, or domain
adaptation. We do claim that the group-theoretic framing resolves a specific
confusion in the digital-olfaction literature, where "cross-device transfer" is
sometimes claimed for temporal drift on a single device, and where
"normalization" is sometimes treated as if it were a device-invariant
representation without specifying the group.

The companion paper [@james2026] proves that raw normalized MOX responses
cannot transfer zero-shot across devices with different power-law exponents.
That result is consistent with this paper: the raw response is not invariant
under the affine device map. The invariant objects are ratios of log
differences; they are lower-dimensional. The companion theorem says what fails;
this paper says what survives and at what cost.

---

## 2. Framework

### 2.1 Measurement space and instrument group

Let \(M\) be a smooth manifold, the *measurement space*. A particular
measurement is a point \(m \in M\). An instrument induces a Lie group \(G\)
acting smoothly on \(M\),
\[
  \Phi: G \times M \to M, \qquad (g,m) \mapsto g\cdot m.
\]
The orbit of \(m\) is
\[
  G\cdot m = \{g\cdot m : g\in G\}.
\]
Two measurements are *instrument-equivalent* if they lie in the same orbit.
The quotient \(M/G\) is the set of orbits; when the action is free and proper
it is a smooth manifold and the projection \(\pi:M\to M/G\) is a submersion.

**Definition 1 (Instrument group).** An instrument group is a Lie group \(G\)
together with an action on \(M\) such that, for every \(g\in G\), there exists
an admissible instrument state whose measurement map is \(g\). The group is
*derived from physics* if its generators are parameters of the instrument
equations.

The definition is deliberately abstract; the content is in the derivation. For
a MOX sensor the parameters are \(a,b\) in \(s=aC^b\); for a spectrometer they
are \(g(\lambda),o(\lambda)\); for an EEG amplifier they are \(g_c,o_c\); for
a mass spectrometer they are the m/z calibration \((a,b)\) and the intensity
gain.

### 2.2 Invariants, quotient, observability

A function \(I:M\to Q\) is *\(G\)-invariant* if
\[
  I(g\cdot m)=I(m) \quad \text{for all } g\in G,\ m\in M.
\]
The invariant field is
\[
  \mathbb{R}(M)^G = \{f\in \mathbb{R}(M) : f\circ \Phi_g=f \ \forall g\in G\}.
\]
The polynomial invariant ring \(\mathbb{R}[M]^G\) is defined analogously. The
quotient \(M/G\) is the universal invariant: every invariant factors through
\(\pi\).

**Definition 2 (Observability).** A label \(\ell:M\to L\) is
*\(G\)-observable* if \(\ell\) is constant on orbits, equivalently if there
exists \(\bar\ell:M/G\to L\) with \(\ell=\bar\ell\circ\pi\).

Observability is the property that matters for classification. A classifier
trained on instrument A and tested on instrument B can succeed only if the
label is a function of the orbit, or if the classifier has implicitly learned a
section using additional information. This is the precise form of the
statement that "the device signature and the label must be separable."

### 2.3 Dimension counting and the cost of quotienting

For a free action of a Lie group \(G\) on an open dense subset of an
\(d\)-dimensional manifold \(M\),
\[
  \dim(M/G)=d-\dim G.
\]
The rule is a counting rule, not an information measure, but it is decisive.
If \(\dim G\ge d\), generic orbits are open and the quotient is a point (or a
discrete set); no continuous invariant survives. If \(\dim G=d-2\), the
quotient is two-dimensional: two numbers per measurement. Every invariant is
paid for by a dimension.

Table 1 lists the nested groups used in this paper. The first row is the
identity: no instrument variation, all \(d\) dimensions survive. The last row
is the full per-pixel affine group: \(2d\) parameters, no invariant. The
interesting cases are in between.

**Table 1. The observability hierarchy.** \(d\) is the number of measurement
channels; \(p\) is the number of channels that share a group parameter.

| group | action | group dimension | invariant dimension | example |
|---|---|---:|---:|---|
| identity | \(y\) | 0 | \(d\) | same instrument |
| global affine | \(y\mapsto gy+o\) | 2 | \(d-2\) | SNV / MSC |
| per-channel affine | \(y_i\mapsto g_i y_i+o_i\) | \(2p\) | \(d-2p\) | NIR per wavelength, EEG per channel |
| per-pixel affine | \(y_i\mapsto g_i y_i+o_i\), all \(i\) | \(2d\) | 0 | unconstrained illumination |
| full per-sensor affine on one vector | \(y_i\mapsto k_i y_i+d_i\) | \(2p\) | 0 | arbitrary MOX sensor replacement |

![The instrument group, its orbits, and the quotient.](figures/concept_quotient.pdf)

*Figure 1. A group acts on measurement space; the orbit is the set of
measurements an instrument can produce for the same sample; the quotient is
the invariant representation.*

### 2.4 Sections and calibration

A *section* of the quotient is a map \(\sigma:M/G\to M\) with
\(\pi\circ\sigma=\mathrm{id}\). A section chooses one representative per orbit.
Calibration is a section: it maps an invariant back into a chosen instrument's
frame. The invariant is section-independent; the section is not.

This distinction resolves a common confusion. Calibration transfer is often
described as "making the instruments agree." In the group language, it
estimates a section using reference samples. The invariant is the part that
does not need the references. A method can be a good section and a poor
invariant, or vice versa. The NIR experiments in Section 5.3 measure both.

The number of references required is a property of the group, not of the
algorithm. A global affine group has two parameters, so two references suffice
in principle. A per-wavelength affine group has \(2p\) parameters, but the
parameters decouple across wavelengths: two references per wavelength suffice,
and because the same two samples are measured at all wavelengths, two reference
*samples* suffice for the whole spectrum. A per-pixel affine group has \(2d\)
parameters; without a smoothness constraint, \(d\) references are needed, which
is why flat-field correction assumes a smooth illumination field rather than a
free per-pixel map.

### 2.5 What the framework does not do

The framework does not discover the group from data. The group is a modelling
choice justified by physics; the experiments test whether it is adequate. It
does not replace calibration transfer: a section is often more accurate than a
pure invariant when the label depends on absolute scale. It does not promise
that a quotient exists: for sequencing, the relevant channel is not affine and
the quotient construction does not apply. It does not make a weak dataset
strong: 17 Raman spectra remain 17 spectra.

---

## 3. Groups derived from physics

### 3.1 Metal-oxide gas sensors

The conduction model of Barsan and Weimar gives the response of a MOX sensor to
a reducing gas as a power law [@barsan2001; @barsan2003],
\[
  s = a\,C^{b},
\]
where \(C\) is concentration and \(a,b\) depend on the sensor material,
temperature, humidity, and aging. Taking logs,
\[
  y=\log s=\log a+b\log C.
\]
For two devices A and B,
\[
  y_B = k\,y_A+d, \qquad k=\frac{b_B}{b_A}, \qquad
  d=\log a_B-k\log a_A.
\]
The device map is affine in log space. This is the group used in the MOX
experiments. The full per-sensor group acts independently on each sensor,
because each sensor has its own \(a_i,b_i\). The restricted group fixes the
exponent \(b_i\) (same sensor model) and allows only a per-sensor multiplicative
gain in the linear response, which is an additive shift in log space; this
leaves the slope ratios \(b_i/b_j\) invariant.

The UCI drift dataset [@vergara2012; @uciDrift] measures a 16-sensor array over
10 batches and 36 months. It is a temporal drift dataset, not a set of distinct
instruments. We use it as a temporal pseudo-device: the group is the change in
\(a_i,b_i\) over time. The distinction is recorded in the fitness table.

### 3.2 NIR, Raman, and optical spectroscopy

The Beer–Lambert law gives absorbance
\[
  A(\lambda)=\varepsilon(\lambda)\,c\,L+b(\lambda),
\]
where \(\varepsilon\) is the molar absorptivity, \(c\) the concentration, \(L\)
the path length, and \(b\) a baseline [@burns2007; @siesler2002;
@workmanweyer2012]. Instrument differences enter as a wavelength-dependent
gain and offset,
\[
  A'(\lambda)=g(\lambda)A(\lambda)+o(\lambda),
\]
and sometimes as a wavelength shift \(\lambda'=\lambda+\delta\). The first is
an affine group at each wavelength; the second is a translation of the
abscissa and is outside the per-wavelength affine group unless it is estimated
separately.

Scattering adds a smooth multiplicative and additive effect. Standard normal
variate (SNV) and multiplicative scatter correction (MSC) remove a *global*
affine effect [@barnes1989; @geladi1985]. They are the \(p=1\) case of the
per-wavelength group. The Corn data used here contain 80 samples measured on
three NIR instruments (m5, mp5, mp6) at 700 wavelengths [@eigenvector]. The
same physical samples are measured on all three, which makes this the strongest
cross-instrument test in the paper.

### 3.3 Mass spectrometry

A mass spectrum is a set of peaks \((m/z_i,I_i)\). Calibration differences are
approximately affine on the m/z axis,
\[
  m' = a\,m+b,
\]
and detector differences are approximately a gain \(I'=gI\). Two reference
peaks determine \((a,b)\); this is the mass-spectrometry analogue of two-point
calibration. Fragmentation patterns depend on collision energy and ionization
mode and are *not* affine: a change in fragmentation moves intensity between
peaks in a way that no m/z calibration can undo. MassBank [@horai2010] and the
Human Metabolome Database [@wishart2022] are the main public repositories; the
minimum reporting standards [@sumner2007] document the metadata needed for
comparison. The repositories contain many instruments but not, in a form we
could use, the same physical sample measured across them. Section 5.7 therefore
uses a controlled synthetic experiment and labels it.

### 3.4 EEG and MEG

EEG measures scalp potentials; MEG measures magnetic fields. Both are linear
mixtures of underlying sources, but the sensor gains, offsets, and reference
montages differ across headboxes. A simplified instrument model is a
per-channel affine map,
\[
  x'_c(t)=g_c\,x_c(t)+o_c,
\]
plus a common-reference transform \(x_c\mapsto x_c-x_{\mathrm{ref}}\)
[@nunez2006; @hamalainen1993]. Band-power features are positive and scale with
\(g_c^2\), so a per-channel gain adds a constant to log band power. Subtracting
a per-channel baseline log power,
\[
  \log P_c(\text{task})-\log P_c(\text{baseline})
  =\log\frac{P_c(\text{task})}{P_c(\text{baseline})},
\]
cancels the gain exactly. This is the EEG analogue of the MOX \(R_s/R_0\)
ratio: a reference measurement, not a label. The common-reference transform
mixes channels and is not diagonal; it breaks the per-channel invariant, as the
experiment shows (Section 5.4).

MEG has the same per-channel gain and reference structure, with
magnetoencephalography's own sensor geometry and noise. We did not identify a
public dataset with the same physical source measured on two MEG scanners in a
form that supports a cross-instrument split. The MEG content of this paper is
therefore a derivation and a data-status statement, not an experiment.

### 3.5 Microscopy

A fluorescence microscope image is affected by an illumination field, detector
gain and offset, and blur:
\[
  I'(x,y)=g(x,y)\,I(x,y)+o(x,y),
\]
with \(g,o\) smooth. Without smoothness this is a per-pixel affine group and no
within-image invariant exists. With smoothness, a flat-field estimate is the
quotient section: estimate \(g,o\) from a blank reference or from the
large-scale background of the image itself, then divide. Image-based profiling
[@caicedo2017; @ljosa2012; @carpenter2006] and the scikit-image data module
[@vanderwalt2014] supply the real image used in Section 5.6.

### 3.6 DNA sequencing

A sequencing instrument does not apply an affine map to a fixed measurement
vector. It applies a stochastic error channel to a discrete sequence:
substitutions, insertions, deletions, and context-dependent errors
[@goodwin2016; @quince2017]. For a substitution-only, position-independent
channel, the induced map on \(k\)-mer frequency vectors is linear,
\[
  p_B = M p_A,
\]
where \(M\) is a stochastic matrix. If \(M\) is known and invertible, the source
frequencies are recoverable by \(p_A=M^{-1}p_B\). If \(M\) is unknown, it can be
estimated from unlabelled reads, but only a section is identified. Indels and
context effects make the channel nonlinear; there is no finite-dimensional
affine group. Section 4.5 states the boundary precisely; Section 5.8
illustrates it on the real phiX174 genome [@phix174] with simulated reads.

**Table 2. Modality groups and evidence status.**

| modality | physical model | group | invariant | references | evidence |
|---|---|---|---|---|---|
| MOX | \(s=aC^b\) | affine in \(\log s\), per sensor | log ratios, slope ratios | 2–5 per device | real temporal pseudo-device (UCI) |
| NIR | Beer–Lambert | per-wavelength affine | cross-sample ratio | 2 samples | real cross-instrument (Corn) |
| Raman | inelastic scattering | per-wavelength affine + shift | cross-sample ratio | 2 samples | real spectra, synthetic group |
| mass spec | m/z calibration | affine on m/z, gain | calibrated peak positions | 2 peaks | synthetic |
| EEG | volume conduction | per-channel affine + reference | baseline log ratio | baseline period | real signal, synthetic group |
| MEG | volume conduction | per-channel affine + reference | baseline log ratio | baseline period | derivation; no paired data |
| microscopy | illumination field | smooth per-pixel affine | flat-field image | blank / background | real image, synthetic illumination |
| sequencing | stochastic errors | not affine | none in general | error model | real genome, simulated reads |

---

## 4. Theorems and proofs

The proofs are short; their purpose is to make the group choice and its
consequences explicit. The SymPy verification is in `invmath/groups.py` and
`results/math_report.txt`.

### 4.1 The MOX device map

**Theorem 1 (MOX log-affine device map).** Let \(s_A=a_A C^{b_A}\) and
\(s_B=a_B C^{b_B}\) with \(a_A,a_B>0\) and \(b_A,b_B\neq0\). Then
\[
  \log s_B = k\log s_A+d,
  \qquad k=\frac{b_B}{b_A},
  \qquad d=\log a_B-k\log a_A.
\]

*Proof.* Substitute \(\log C=(\log s_A-\log a_A)/b_A\) into
\(\log s_B=\log a_B+b_B\log C\). The result is
\(\log a_B+(b_B/b_A)(\log s_A-\log a_A)\), which is the stated affine map.
\(\square\)

The map is affine with linear part \(k=b_B/b_A\). If \(b_A,b_B>0\) (reducing
gases), \(k>0\) and the map preserves the sign of the log response. If the
exponents have opposite signs, \(k<0\); the group is still affine but not
sign-preserving. The reconstruction error is zero symbolically; the report
records this.

### 4.2 The three-point invariant

**Theorem 2 (Three-point MOX invariant).** For three concentrations
\(C_1,C_2,C_3\) with \(C_3\neq C_2\),
\[
  \frac{y_1-y_2}{y_3-y_2}
  =
  \frac{\log C_1-\log C_2}{\log C_3-\log C_2},
\]
independent of \(a\) and \(b\).

*Proof.* Write \(y_i=\log a+b\log C_i\). Then
\(y_1-y_2=b(\log C_1-\log C_2)\) and
\(y_3-y_2=b(\log C_3-\log C_2)\); the factor \(b\) cancels. \(\square\)

The invariant is the cross-ratio of log concentrations. It characterizes the
*shape* of a sensor's response curve across three concentrations. It does not
by itself identify the gas: two gases can have the same curve shape on one
sensor. Identification requires the cross-sensor profile, which is a different
object and is not invariant under the full per-sensor group.

### 4.3 The full per-sensor group and observability

**Theorem 3 (Full per-sensor affine group is transitive).** Let
\(G=\prod_{i=1}^p \operatorname{Aff}(1)\) act on \(\mathbb{R}^p\) by
\[
  (g\cdot y)_i=k_i y_i+d_i, \qquad k_i>0.
\]
Then \(G\) acts transitively on \(\mathbb{R}^p\). Consequently \(M/G\) is a
point, every invariant is constant, and no nonconstant label is
\(G\)-observable from a single measurement vector.

*Proof.* Given \(y,y'\in\mathbb{R}^p\), choose \(k_i=1\) and
\(d_i=y'_i-y_i\) for every \(i\). Then \(g\cdot y=y'\). Since any two points
are in the same orbit, any invariant takes the same value on all points.
\(\square\)

The theorem is the precise form of the statement that arbitrary per-sensor
affine replacement destroys gas identity. The translation subgroup alone is
already transitive; the positivity of \(k_i\) is not needed for this argument.
The practical consequence is that a single MOX measurement cannot be
device-invariant *and* identify the gas if the device can change each sensor's
offset arbitrarily.

For a *collection* of samples, the group acts on each row of a \(p\times n\)
matrix and row-wise affine cross-ratios survive. Those invariants describe the
configuration of samples, not a device-independent per-sample label. In the
common-latent model \(y_{ik}=\alpha_i+\beta_i x_k\) with
\(x_k=\log C_k\), the row-wise cross-ratios are functions of \(x_k\) only, so
they do not encode gas identity. The distinction between "no invariant at all"
and "invariants that do not carry the label" is important; the theorem proves
the former for a single vector, and the model explains the latter for a
collection.

**Corollary 1 (Restricted groups).** (i) For a common gain
\(G_{\mathrm{common}}=\mathbb{R}_{>0}\) acting by \((g\cdot y)_i=k y_i\),
the response ratios \(y_i/y_j\) are invariant. (ii) For per-sensor additive
shifts \(G_{\mathrm{shift}}=\prod_i\mathbb{R}\) acting by
\((g\cdot y)_i=y_i+c_i\), differences transform as
\((y_i-y_j)\mapsto(y_i-y_j)+(c_i-c_j)\), so they are not invariant in
general; but in the MOX slope model \(y_{ik}=\alpha_i+\beta_i x_k\), the
shifts change only \(\alpha_i\), leaving \(\beta_i\) fixed, so the slope
ratios \(\beta_i/\beta_j\) are invariant. (iii) For per-sensor
multiplicative scalings \(y_i\mapsto k_i y_i\), the slope ratios transform
by \(k_i/k_j\) and are not invariant.

*Proof.* (i) \((k y_i)/(k y_j)=y_i/y_j\). (ii) The shift changes
\(\alpha_i\) by \(c_i\) and leaves \(\beta_i\) unchanged, so
\(\beta_i/\beta_j\) is unchanged. The difference statement is immediate.
(iii) The slope of sensor \(i\) becomes \(k_i\beta_i\), so
\(\beta_i/\beta_j\) becomes \((k_i/k_j)(\beta_i/\beta_j)\).
\(\square\)

The corollary is the mathematical basis of the slope-ratio invariant. It also
explains its limit: it requires the exponent to be stable, which is a physical
assumption, not a theorem about all MOX sensors. The per-batch centering
feature is the empirical form of the per-sensor additive nuisance in log space;
the group-stress experiment tests it directly.

### 4.4 Per-wavelength affine spectroscopy

**Theorem 4 (Two-reference invariant).** Let
\(A'_{i\lambda}=g_\lambda A_{i\lambda}+o_\lambda\) for samples
\(i=1,\dots,n\) and wavelengths \(\lambda=1,\dots,p\), with
\(A_{2\lambda}\neq A_{1\lambda}\). Then for every \(i\) and \(\lambda\),
\[
  r_{i\lambda}
  =
  \frac{A_{i\lambda}-A_{1\lambda}}{A_{2\lambda}-A_{1\lambda}}
\]
is invariant under the group.

*Proof.* Substitute:
\[
  \frac{A'_{i\lambda}-A'_{1\lambda}}{A'_{2\lambda}-A'_{1\lambda}}
  =
  \frac{g_\lambda(A_{i\lambda}-A_{1\lambda})}
       {g_\lambda(A_{2\lambda}-A_{1\lambda})}
  =
  \frac{A_{i\lambda}-A_{1\lambda}}{A_{2\lambda}-A_{1\lambda}}.
  \qquad\square
\]

Two references are necessary and sufficient to determine \(g_\lambda\) and
\(o_\lambda\) for a new instrument:
\[
  g_\lambda=\frac{A'_{2\lambda}-A'_{1\lambda}}{A_{2\lambda}-A_{1\lambda}},
  \qquad
  o_\lambda=A'_{1\lambda}-g_\lambda A_{1\lambda}.
\]
A new target spectrum is mapped into the source frame by
\(A_{i\lambda}=(A'_{i\lambda}-o_\lambda)/g_\lambda\). The denominator is the
anchor conditioning: if \(A_{2\lambda}\approx A_{1\lambda}\) at some
wavelength, the estimate is ill-conditioned and noise is amplified. The NIR
anchor-sensitivity experiment (Section 5.3) measures this directly.

The theorem is the reason two reference samples suffice for a whole spectrum,
not \(2p\) references. The parameters decouple across wavelengths, but the same
two physical samples provide an equation at every wavelength.

![Three instruments, one table.](figures/concept_table.pdf)

*Figure 2. The table example. The readings differ by an affine map; the ratio
is invariant; a known object supplies the section.*

### 4.5 The sequencing boundary

**Proposition 1 (No general affine invariant for sequencing).** Consider a
sequencing channel that maps a source sequence \(S\) to a random read \(R\).
If the channel is a position-independent substitution channel, the induced map
on \(k\)-mer frequency vectors is linear. If the channel includes deletions or
context-dependent errors, the induced map is not affine in general, and there
exist distinct source sequences whose output distributions are identical.

*Proof sketch.* For substitutions, each \(k\)-mer is replaced by another
\(k\)-mer with a probability determined by the substitution matrix; collecting
these probabilities gives a stochastic matrix \(M\) on the \(4^k\)-dimensional
frequency vector. For deletions, a distinguishing \(k\)-mer can be removed
entirely, so two sources that differ only at a deleted position produce the
same read distribution. For a symmetric substitution channel that maps every
base to every other base with equal probability, the output distribution is
uniform for every source, so the map is non-injective and no invariant can
recover the source. A context-dependent channel makes the transition
probabilities depend on neighbouring bases, so the map is not a fixed matrix.
\(\square\)

The proposition is a boundary statement, not a claim that sequencing data are
uninformative. With a known error model and a probabilistic likelihood, one can
still infer the source; the point is that the inference is not a
finite-dimensional affine quotient. The experiment in Section 5.8 shows that a
platform-mean normalization, which is the natural affine-style correction,
does not recover fine-grained haplotype information.

![The group determines what survives.](figures/concept_hierarchy.pdf)

*Figure 3. Nested instrument groups and the dimension of the quotient. As the
group grows, the invariant shrinks.*

---

## 5. Experiments

### 5.1 Protocol and metrics

All experiments follow the same protocol.

1. **Split by instrument, not by sample.** Train on the source instrument;
   test on the target. The target labels are never used to fit the classifier
   or regressor. Where a method uses reference samples, it uses their
   *spectra*, not their labels, unless the table says "labelled calibration."
2. **Report the ceiling.** Within-instrument cross-validation gives the
   performance of the representation when there is no instrument shift. A
   cross-instrument loss is attributed to the shift only relative to this
   ceiling.
3. **Report the nuisance.** For each representation, report the metric on the
   transformed target. A method that removes the nuisance but also removes the
   signal is not a success.
4. **Label the evidence.** Every table and figure states whether the
   instrument transform is real or synthetic.

Metrics are accuracy, balanced accuracy, and macro-F1 for classification; R²
and RMSE for regression; Dice overlap for segmentation. The code is in
`experiments/`; the numbers in this section are generated by
`experiments/aggregate.py` from the CSVs in `results/`.

### 5.2 MOX: UCI drift

**Data.** The UCI Gas Sensor Array Drift dataset contains 13,910 measurements
from a 16-sensor array, 6 gases, and 10 batches collected over 36 months
[@vergara2012; @uciDrift]. The same array is used throughout; the batches are
temporal pseudo-devices, not distinct hardware. We train on batch 1 and test on
batches 2–10.

**Representations.** Raw resistance \(R\); log resistance \(\log R\); pairwise
log ratios \(\log(R_i/R_j)\) (invariant to a common gain); global-gain
invariant \(\log R_i-\overline{\log R}\) (invariant to a common gain);
per-batch centered log resistance (exact invariant to a per-batch per-sensor
multiplicative gain, i.e. an additive shift in log space; not invariant to a
per-sensor scale change; transductive, uses unlabelled target batch moments);
per-batch z-score
(negative control: removes gain and exponent); normalized selectivity
\(R_i/\overline{R}\) (sample-level normalization); rank; CORAL and mean
alignment (domain-adaptation baselines); and a labelled calibration that adds
five target samples per gas class to the training set.

**Result.** Table 3 reports accuracy; rows marked "pooled" aggregate all target
samples, and rows marked "per-batch mean" average the nine batch accuracies.
Raw transfer is 0.405 pooled (0.390 per-batch). Log resistance is 0.474
(0.461). The exact per-sensor gain invariant (per-batch centering) is 0.515
pooled and 0.451 per-batch, the best unsupervised method in the pooled table
but not in the per-batch average. Pairwise log ratios, which remove only a
common gain, are 0.436 (0.427): they remove the nuisance but also remove
signal. Normalized selectivity is 0.483 (0.482); five-sample labelled
calibration is 0.711 per-batch (it has no pooled row). The negative controls
behave as expected: rank loses information (0.303), and the domain-adaptation
baselines are below the log baseline (CORAL 0.303, mean alignment 0.451,
per-batch mean).

**Table 3. MOX accuracy, train batch 1, test batches 2–10.** "Pooled" aggregates
all target batches; "per-batch mean" averages the nine batch accuracies. The
calibrated row uses five labelled target samples per gas class; the other rows
use no target labels.

| representation | evaluation | accuracy | macro-F1 | target labels |
|---|---|---:|---:|---|
| raw | pooled | 0.405 | 0.267 | none |
| log | pooled | 0.474 | 0.359 | none |
| log ratio (common-gain invariant) | pooled | 0.436 | 0.331 | none |
| global-gain invariant | pooled | 0.455 | 0.340 | none |
| normalized selectivity | pooled | 0.483 | 0.392 | none |
| batch centered | pooled | 0.515 | 0.486 | unlabelled target moments |
| batch z-score | pooled | 0.457 | 0.450 | unlabelled target moments |
| calibrated (5 per class) | per-batch mean | 0.711 | 0.668 | 5 labelled per class |

![MOX cross-batch transfer.](figures/mox_transfer.pdf)

*Figure 4. Pooled and per-batch MOX transfer. The calibrated baseline uses
five labelled target samples per class; the invariant features use none.*

The per-batch breakdown matters. Table 4 shows the mean over batches 2–10. Raw
accuracy is 0.390, normalized selectivity 0.482, batch centering 0.451, and
labelled calibration 0.711. The batch-center advantage in the pooled table does
not survive per-batch averaging: it is 0.451 against 0.482 for selectivity, and
the per-batch macro-F1 is essentially tied (0.415 vs 0.414). Batch 8 is difficult for every unsupervised method (raw 0.204, log
0.224, batch center 0.269), while calibration recovers 0.523. Batch 9 is the
clearest case where the log transform helps without labels (0.183 raw, 0.423
log, 0.343 batch center, 0.686 calibrated).

**Table 4. MOX per-batch means over batches 2–10.**

| representation | mean accuracy | mean macro-F1 |
|---|---:|---:|
| raw | 0.390 | 0.251 |
| log | 0.461 | 0.353 |
| normalized selectivity | 0.482 | 0.414 |
| batch centered | 0.451 | 0.415 |
| batch z-score | 0.458 | 0.421 |
| calibrated (5 per class) | 0.711 | 0.668 |

**Controlled group-stress test.** The dataset experiment confounds drift with
sample composition. To isolate the group, we train and test on the *same*
batch, then apply a random element of a nested group to the test half. Table 5
shows the result.

**Table 5. MOX controlled group-stress accuracy.** Train and test on batch 1;
transform the test half by a random element of the stated group. Mean over 8
draws.

| group | log | global-gain invariant | batch centered |
|---|---:|---:|---:|
| identity | 0.987 | 0.987 | 0.987 |
| global gain | 0.982 | 0.987 | 0.984 |
| per-sensor gain | 0.437 | 0.500 | 0.987 |
| per-sensor affine | 0.168 | 0.230 | 0.650 |

The identity and global-gain rows are near ceiling. Per-sensor gain destroys
the log and global-gain features (0.437 and 0.500) but leaves batch centering
at 0.987. Per-sensor affine destroys all three features to different degrees;
batch centering retains 0.650 because it removes the offset but not the
per-sensor gain. This is the exact sense in which "invariance" is group-specific:
each feature is invariant to the group it was designed for, and to no larger
group.

![MOX group stress.](figures/mox_group_stress.pdf)

*Figure 5. Controlled group-stress test. The same samples, transformed by a
random element of each nested group.*

**Slope stability.** The restricted-group invariant is the cross-sensor slope
ratio \(b_i/b_j\). We fit per-sensor slopes for each gas and batch with at
least two concentrations and compute the median relative MAD of the 120
pairwise slope ratios across batches. Five of six gases have median relative
MAD below 0.10 (gas 1: 0.062, gas 2: 0.060, gas 4: 0.093, gas 5: 0.041, gas 6:
0.053). Gas 3 is unstable (0.285). The invariant is therefore useful but not
universal: one gas's slope profile changes enough across batches to break it.

![MOX slope stability.](figures/mox_slope_stability.pdf)

*Figure 6. Stability of the restricted-group slope-ratio invariant across
batches. Gas 3 is the exception.*

**Interpretation.** The exact per-sensor gain invariant (batch centering) is
competitive with normalized selectivity but not better per-batch, and it is
transductive: it uses the unlabelled target batch mean. The common-gain
invariant (pairwise log ratios) is worse than the log baseline because it
removes the absolute response that carries gas information. The labelled
calibration is much better because it uses target labels; it is a section, not
an invariant. This is the central empirical trade-off of the paper: invariance
buys robustness at the cost of information, and the exact invariant is only as
good as the group it assumes.

### 5.3 NIR: Corn, three instruments

**Data.** The Corn dataset contains 80 physical samples measured on three NIR
instruments (m5, mp5, mp6) at 700 wavelengths, with reference values for
moisture, oil, protein, and starch [@eigenvector]. The same samples are measured
on all three instruments. We train on the source instrument and test on the
target; the target property labels are never used. The representation methods
use reference *spectra* only.

**Representations.** Raw reflectance; SNV (global affine); per-wavelength
affine invariant with two anchors; per-wavelength affine calibration with five
reference spectra (maps target into the source frame); and PDS with five
references and a local window of 2 [@wang1992]. The anchors are chosen by a
conditioning rule that maximizes the 10th percentile of the standardized
anchor difference across wavelengths; this uses source spectra only and no
labels.

**Group adequacy.** Before transfer, we ask whether the per-wavelength affine
group is an adequate model of the cross-instrument variation. Table 6 reports
an oracle fit: all samples are used to estimate the group parameters, so this
is a model-adequacy test, not a transfer test. The identity model (no
transform) reaches R² 0.911–0.996 depending on the pair. A global affine
improves to 0.998; a per-wavelength affine improves to 0.9985–0.9998. The
group is adequate: most of the cross-instrument variation is per-wavelength
gain and offset.

**Table 6. Oracle group-model fit for NIR.** R² of the target spectra
reconstructed from the source spectra using the stated group; all samples used
to fit the group.

| pair | identity | global affine | per-wavelength affine |
|---|---:|---:|---:|
| m5 → mp5 | 0.944 | 0.998 | 0.9986 |
| m5 → mp6 | 0.911 | 0.998 | 0.9985 |
| mp5 → mp6 | 0.996 | 1.000 | 0.9998 |

**Within-instrument ceiling.** Five-fold cross-validated protein prediction on
a single instrument is R² 0.915–0.926 for raw, 0.896–0.939 for SNV, and
0.918–0.923 for the invariant. The invariant does not destroy within-instrument
signal; it is a viable representation in the absence of a shift.

**Transfer.** Table 7 reports the six ordered pairs for protein. Raw transfer
is catastrophic for several pairs (m5 → mp5: −4.55; mp5 → m5: −9.85; m5 → mp6:
−5.09), but not for mp5 ↔ mp6 (0.906 and 0.847). The instruments are not
equally different: mp5 and mp6 are close, m5 is far. The per-wavelength
invariant with two references reaches 0.834–0.910 across all six pairs.
Per-wavelength calibration with five references reaches 0.886–0.933. PDS
reaches 0.697–0.905. The invariant has no catastrophic failures and stays in a narrow band
(0.834–0.910); calibration is slightly better on average and has a slightly
narrower range (0.886–0.933); PDS is competitive when the instruments are close
and worse when they are far.

**Table 7. NIR cross-instrument protein prediction (R²).** Train on source,
test on target. "Invariant" uses two reference samples; "calibrated" and "PDS"
use five reference spectra. No target property labels are used.

| pair | raw | SNV | invariant (2 refs) | calibrated (5 refs) | PDS (5 refs) |
|---|---:|---:|---:|---:|---:|
| m5 → mp5 | −4.55 | −5.97 | 0.834 | 0.908 | 0.905 |
| m5 → mp6 | −5.09 | −8.24 | 0.855 | 0.886 | 0.884 |
| mp5 → m5 | −9.85 | −1.87 | 0.864 | 0.903 | 0.724 |
| mp5 → mp6 | 0.906 | 0.940 | 0.851 | 0.916 | 0.697 |
| mp6 → m5 | −4.33 | −2.44 | 0.910 | 0.933 | 0.887 |
| mp6 → mp5 | 0.847 | 0.723 | 0.887 | 0.920 | 0.878 |

![NIR cross-instrument transfer.](figures/nir_transfer.pdf)

*Figure 7. NIR protein prediction across instruments. Raw transfer fails for
the distant pairs; the per-wavelength invariant is stable.*

**Representation agreement.** For matched samples, the entrywise R² between
instruments is 0.912–0.996 for raw, 0.999 for SNV, and 0.878–0.972 for the
invariant. SNV is the best *agreement* because it removes a global affine
effect; the invariant is slightly lower because it quotients a larger group and
amplifies noise at poorly conditioned wavelengths. Agreement is not the same as
transfer: SNV agrees well but transfers poorly for the distant pairs, because a
global affine does not capture per-wavelength differences.

![NIR representation agreement.](figures/nir_consistency.pdf)

*Figure 8. Agreement between instruments for the same samples. SNV maximizes
agreement; the per-wavelength invariant is slightly lower but transfers more
stably.*

**Reference scaling.** Table 8 varies the number of reference samples. The
invariant uses the first two anchors; adding references does not change it
(0.824, 0.834, 0.834, 0.825, 0.801 for 2, 3, 5, 10, 20). Calibration improves
with more references (0.857 at 2, 0.908 at 5, 0.926 at 10, 0.920 at 20). PDS
improves from 0.905 at 5 to 0.941 at 10 and 0.917 at 20. The pure quotient
cannot exploit more than two references without a section; the section can.
This is the cleanest demonstration of the invariant/calibration distinction.

**Table 8. NIR reference-count scaling (m5 → mp5, protein, R²).**

| references | invariant | calibrated | PDS |
|---:|---:|---:|---:|
| 2 | 0.824 | 0.857 | — |
| 3 | 0.834 | 0.861 | — |
| 5 | 0.834 | 0.908 | 0.905 |
| 10 | 0.825 | 0.926 | 0.941 |
| 20 | 0.801 | 0.920 | 0.917 |

![NIR reference scaling.](figures/nir_reference_scaling.pdf)

*Figure 9. More references help the section, not the quotient.*

**Anchor conditioning.** The invariant divides by the anchor difference at
every wavelength. A poorly conditioned pair amplifies noise. Figure 10 shows
30 random anchor pairs and the conditioned pair. The median R² of random pairs
is −0.073; the 75th percentile is 0.831; the conditioned pair is 0.824. The
conditioning score and R² have a Pearson correlation of 0.194 across random
pairs. The distribution is bimodal: most random pairs are bad, a minority are
good, and the conditioning rule finds a good one without labels. Anchor
selection is not a detail; it is part of the invariant algorithm.

![NIR anchor sensitivity.](figures/nir_anchor_sensitivity.pdf)

*Figure 10. The choice of the two reference samples controls the noise of the
invariant.*

**Interpretation.** NIR is the strongest evidence for the framework because the
instrument transform is real and the same samples are measured on three
instruments. The per-wavelength affine group is adequate (oracle R² ≥ 0.9985),
the two-reference invariant transfers at R² 0.83–0.91, and raw transfer fails
catastrophically for the distant pair. The framework predicts the qualitative
pattern: a larger group (per-wavelength) is needed for the distant instruments,
and two references suffice. The residual gap to within-instrument CV (0.92) is
the price of quotienting and noise amplification.

### 5.4 EEG: real signal, synthetic instrument

**Data.** PhysioNet EEG Motor Movement/Imagery dataset, subject 1, runs 3 and 4
(left vs right fist motor imagery), 64 channels at 160 Hz [@goldberger2000;
@schalk2004]. We compute log band power in five bands over non-overlapping
two-second windows (124 windows total; 60% train, 40% test per run). The signal
is real; the second instrument is synthetic.

**Instrument model.** Per-channel gain \(g_c\sim\mathrm{log Normal}(0,0.6)\),
per-channel offset \(o_c\sim N(0,30)\), and an optional common reference
\(x_c\mapsto x_c-x_{\mathrm{ref}}\), plus noise. The gain is the amplifier
gain; the offset is the DC offset; the reference is a montage change.

**Representations.** Raw log band power; global normalization (subtract the
mean across channels); baseline (subtract the per-channel baseline log power);
baseline ratio (divide by the baseline). The baseline uses the first 10 windows
of each run as a reference period, not a label.

**Result.** Table 9 reports accuracy. Under the identity transform, the ceiling
is 0.740 raw, 0.780 baseline, 0.840 global normalization. Under gain, baseline
is 0.720, raw 0.668, baseline ratio 0.648, global normalization 0.622. Under
gain+offset, baseline is 0.834, raw 0.686, baseline ratio 0.734, global
normalization 0.642. Under gain+offset+reference, baseline is 0.756, raw 0.696,
baseline ratio 0.544, global normalization 0.632.

**Table 9. EEG motor-imagery accuracy (left vs right), real signal, synthetic
instrument transforms.** Mean over 10 draws; the identity row has no variance
because the transform is fixed.

| group | raw | global norm | baseline | baseline ratio |
|---|---:|---:|---:|---:|
| identity | 0.740 | 0.840 | 0.780 | 0.780 |
| gain | 0.668 | 0.622 | 0.720 | 0.648 |
| gain + offset | 0.686 | 0.642 | 0.834 | 0.734 |
| gain + offset + reference | 0.696 | 0.632 | 0.756 | 0.544 |

![EEG transfer.](figures/eeg_transfer.pdf)

*Figure 11. EEG transfer under per-channel gain, offset, and common
reference. The baseline invariant is robust to gain and offset; the reference
transform breaks it.*

**Interpretation.** The baseline invariant is the best representation under
gain+offset (0.834 vs 0.686 raw), which is the group it was designed for. The
common-reference transform is not diagonal in the channel basis; it mixes
channels and breaks the per-channel cancellation. Baseline drops to 0.756 and
baseline ratio to 0.544. This is a clean group boundary: the invariant is exact
for per-channel gain, approximate for offset (removed by band-pass filtering),
and wrong for a reference transform. A method that claimed to be
"reference-invariant" would need a different group (a rank-one mixing group),
not a better classifier.

### 5.5 Raman: real spectra, synthetic instrument

**Data.** 17 real Raman spectra of urine (7 healthy, 10 diseased) with 1000
points each [@ramanData]. The sample is small; this is a demonstration of the
group calculus on real spectral shapes, not a clinical result.

**Instrument model.** Smooth per-wavelength gain
\(g(\lambda)=\exp(0.4\sin(2\pi\lambda+\phi))\), smooth fluorescence offset
\(o(\lambda)=0.3e^{-3\lambda}+\text{noise}\), and an optional integer wavelength
shift. The spectra are real; the second instrument is synthetic.

**Result.** Under the identity transform, raw is 0.663, invariant 0.533, SNV
0.540. Under the affine transform, raw 0.457, invariant 0.533, SNV 0.450.
Under affine+shift, raw 0.457, invariant 0.533, SNV 0.430. The invariant is
better than raw under the affine transform, but the identity raw performance is
higher than the invariant. With 17 samples, the confidence intervals are wide;
the result is underpowered. The real-data experiment is reported as a negative
result: it does not establish a Raman transfer advantage.

**Synthetic controlled experiment.** To separate "the group is not the limiting
nuisance" from "the sample is too small," we generate 600 synthetic spectra in
five classes with class-specific peaks, apply the same affine group with a
stronger gain, and compare raw, SNV, and the invariant. The invariant reaches
0.640, raw 0.522, SNV 0.572. The synthetic experiment shows the mechanism; it
is not evidence about a real Raman instrument.

![Raman transfer.](figures/raman_transfer.pdf)

*Figure 12. Raman real spectra with synthetic transforms. The sample is too
small for a stable advantage; the invariant is better than raw under the
affine transform.*

![Raman synthetic.](figures/raman_synthetic.pdf)

*Figure 13. Controlled synthetic Raman experiment. The group is present by
construction; the invariant recovers part of the loss.*

**Interpretation.** The Raman case is a warning about evidence strength. A
real-signal/synthetic-group experiment can show that an algorithm works when
the group is present, but it cannot show that a real instrument induces the
group. The real data are too small to decide. The honest conclusion is that
Raman is a promising modality for the framework, not a demonstrated one.

### 5.6 Microscopy: real image, synthetic illumination

**Data.** A real fluorescence microscopy image from the scikit-image `cells3d`
volume (nuclei channel, one 2-D slice) [@cells3d; @vanderwalt2014]. The image is
real; the illumination field is synthetic.

**Instrument model.** A smooth multiplicative illumination field
\(g(x,y)\) (a tilted Gaussian) and a smooth additive offset \(o(x,y)\), plus
noise. The field is low-frequency, as a real illumination gradient is.

**Task.** Segment nuclei (cell vs background). The ground truth is the Otsu
mask of the original image. We compare three masks: raw threshold on the
transformed image; flat-field correction using a large-scale blur of the image
itself; and an oracle flat-field using the known \(g,o\). The metric is Dice
overlap with the ground truth.

**Result.** Raw Dice is 0.894 ± 0.007; flat-field is 0.965 ± 0.002; oracle
flat-field is 0.987 ± 0.001. The smoothness assumption is doing the work: the
correction estimates the illumination from the image's own large-scale
background, without a blank reference. The oracle shows the ceiling if the
field were known.

![Microscopy flat-field.](figures/microscopy_flatfield.pdf)

*Figure 14. Real microscopy image with synthetic illumination. Flat-field
correction recovers most of the segmentation quality.*

**Interpretation.** Microscopy is the clearest case where the group is too
large without a smoothness constraint. A per-pixel affine group has \(2d\)
parameters and no invariant; a smooth field has a low-dimensional section that
can be estimated from the image. The experiment is a controlled demonstration
of the section, not a claim about a second microscope.

### 5.7 Mass spectrometry: synthetic controlled spectra

**Data.** No paired public cross-instrument sample set was available in a form
that supports a train/test split. We use a controlled synthetic experiment with
five molecular classes, 200 spectra per class, and 1000 m/z bins. The spectra
have class-specific fragment peaks, isotope envelopes, and noise. The
experiment is labelled synthetic.

**Instrument model.** m/z affine calibration \(m'=am+b\) with
\(a\in[0.995,1.005]\), \(b\in[-15,15]\), intensity gain
\(g\in[0.7,1.4]\), and noise. The "calibrated" representation estimates
\((a,b)\) from two reference peaks in each spectrum and unwraps the target.
This is a controlled stand-in for two internal reference peaks or a paired
reference standard. The "fragmentation" condition additionally multiplies
class-specific peak intensities by random factors, which is outside the affine
group.

**Result.** Under identity, raw is 1.000 and calibrated 1.000. Under the m/z
affine group, raw is 0.777 ± 0.279 and calibrated 1.000. The large standard
deviation is the point: raw transfer is unstable across calibration shifts;
calibration removes the instability. Under affine+fragmentation, raw is
0.770 ± 0.288 and calibrated 0.970 ± 0.008. Fragmentation degrades the
calibrated representation, as the group calculus predicts, but the synthetic
fragmentation is mild; a stronger fragmentation model would degrade it further.

**Table 10. Mass-spec synthetic accuracy (5 classes).** Mean ± std over 5
draws.

| group | raw | calibrated (2 peaks) |
|---|---:|---:|
| identity | 1.000 | 1.000 |
| m/z affine | 0.777 ± 0.279 | 1.000 |
| affine + fragmentation | 0.770 ± 0.288 | 0.970 ± 0.008 |

![Mass-spec transfer.](figures/massspec_transfer.pdf)

*Figure 15. Synthetic mass spectra. Affine calibration removes the m/z shift;
fragmentation is outside the group.*

**Interpretation.** The experiment demonstrates the group calculus, not a
mass-spectrometer result. The affine m/z group is real and two reference peaks
suffice; fragmentation is a different group (or no group), and calibration
cannot remove it. The data-status verdict is that public repositories contain
the metadata but not the paired samples needed for a genuine cross-instrument
test.

### 5.8 Sequencing: real genome, simulated reads

**Data.** The real phiX174 genome (5,386 bp) [@phix174]. Reads of length 100
(coarse task) or 50 (fine task) are sampled from the genome. The platform
channels are simulated; the genome is real.

**Tasks.** *Coarse*: classify reads from five distant genomic regions (5
classes). *Fine*: classify reads from two haplotypes that differ by 8 SNPs in
800 bp (2 classes). The fine task is the relevant one for platform transfer:
the signal is a small number of variants, and the platform error channel is
comparable to the signal.

**Platforms.** A: 0.2% substitution. B_sub: 2% substitution. B_indel: 1%
substitution + 1% insertions + 1% deletions. B_context: 1% substitution plus a
context-dependent (CpG-like) error. Features are \(k\)-mer frequencies (4-mers
for coarse, 5-mers for fine). We compare raw features and platform-normalized
features (divide by the platform mean k-mer spectrum estimated from unlabelled
reads).

**Result.** Coarse classification is robust: raw and normalized accuracy are
0.992–1.000 across platforms. Fine classification is near its ceiling:
identity raw 0.662; B_sub raw 0.643; B_indel raw 0.665; B_context raw 0.650.
Platform normalization does not improve it (0.630–0.650), and slightly hurts.
This is the expected boundary: the error channel is not a multiplicative bias,
so dividing by the platform mean does not recover the haplotype signal.

**Table 11. Sequencing accuracy (train on platform A).** Coarse: 5 regions,
100 bp reads, 4-mers. Fine: 2 haplotypes with 8 SNPs, 50 bp reads, 5-mers.

| task | platform | raw | platform-normalized |
|---|---|---:|---:|
| coarse | B_sub | 1.000 | 0.998 |
| coarse | B_indel | 0.997 | 0.997 |
| coarse | B_context | 0.993 | 0.992 |
| fine | B_sub | 0.643 | 0.632 |
| fine | B_indel | 0.665 | 0.637 |
| fine | B_context | 0.650 | 0.630 |

![Sequencing transfer.](figures/sequencing_transfer.pdf)

*Figure 16. Sequencing transfer. Coarse regional classification is robust;
fine haplotype classification is not rescued by a multiplicative
normalization.*

**Interpretation.** Sequencing is the negative case that makes the framework
useful. The group is not affine; a linear quotient does not exist in general.
The correct treatment is a known error model plus probabilistic inference, not
an invariant representation. The experiment also shows that "platform
normalization" can hurt: an affine-style correction applied to a non-affine
channel is the wrong model, and the data say so.

### 5.9 Dataset fitness

Table 12 is the dataset-fitness table. It ranks datasets on five criteria:
multiple physically distinct instruments, overlapping samples, documented
metadata, a group derivable from physics, and enough samples. The full table is
in `results/fitness.csv`.

**Table 12. Dataset fitness for cross-instrument claims.**

| modality | dataset | instruments / samples | group | verdict | evidence |
|---|---|---|---|---|---|
| MOX | UCI Gas Sensor Array Drift | 1 array, 10 batches, 13,910 measurements | per-sensor affine in log space | temporal pseudo-device, not distinct hardware | real |
| NIR | Corn (Eigenvector) | 3 instruments, 80 samples, 700 wavelengths | per-wavelength affine | qualifying cross-instrument benchmark | real cross-instrument |
| Raman | urine Raman | 1 instrument, 17 spectra | per-wavelength affine + shift | no paired instruments; underpowered | real signal, synthetic group |
| mass spec | MassBank / MoNA | many records, not paired by sample | m/z affine + gain | no qualifying paired set | synthetic |
| EEG | PhysioNet eegmmidb | 1 headbox, 64 channels, 124 windows | per-channel affine + reference | no paired headbox | real signal, synthetic group |
| MEG | public MEG datasets | 1 scanner per dataset | per-channel affine + reference | no paired-instrument data | derivation |
| microscopy | scikit-image cells3d | 1 image | smooth illumination field | no paired scanners | real image, synthetic illumination |
| sequencing | GIAB (real); phiX174 simulation | multiple platforms in GIAB; 1 simulated pair | stochastic error channel, not affine | qualifying data exist but are large; simulation shows the boundary | real genome, simulated reads |

The table is a contribution in its own right. "Multi-device" is often claimed
for temporal drift on one device (MOX), for different instruments that did not
measure the same samples (mass spec), or for a single instrument with a
synthetic transform (EEG, Raman, microscopy). Those are legitimate experiments,
but they are not the same as a cross-instrument benchmark. The table makes the
distinction explicit.

![Evidence type by modality.](figures/concept_evidence.pdf)

*Figure 17. Evidence type by modality. Green: real cross-instrument. Orange:
real signal with a synthetic group. Purple: synthetic or derivation.*

---

## 6. Discussion

### 6.1 What invariance buys

The experiments support three claims.

1. **A derived group predicts the qualitative transfer pattern.** In NIR, the
   per-wavelength affine group is adequate (oracle R² ≥ 0.9985), raw transfer
   fails for the distant instruments, and the two-reference invariant is
   stable across all pairs (0.834–0.910). In MOX, the full per-sensor affine
   group destroys the label, and the slope-ratio invariant is stable only under
   the restricted exponent-fixed group. In EEG, the baseline invariant is best under the
   per-channel gain+offset group and fails under a reference transform. The
   group calculus predicts which representation works and when it stops
   working.

2. **The number of references is a group property.** Two references suffice
   for a per-wavelength affine group; five do not improve the invariant
   (0.824 at 2, 0.834 at 5) but do improve the section (0.857 to 0.908).
   The experiment separates the quotient from the section in a way that a
   generic "calibration transfer" comparison does not.

3. **Invariance is not free.** The common-gain invariant in MOX (0.436) is
   worse than the log baseline (0.474); the exact per-sensor gain invariant
   (batch centering) is 0.515 pooled but 0.451 per-batch. The per-wavelength
   invariant in NIR is worse than SNV in *agreement* (0.878–0.972 vs 0.999)
   but better in *transfer* for the distant pairs. The quotient removes nuisance
   and signal together; the right choice depends on whether the label is
   invariant under the group.

### 6.2 The group choice is the scientific claim

The framework forces an explicit choice. "Normalize the data" is not a method;
it is a family of methods indexed by a group. SNV is a global affine group.
Per-wavelength standardization is a per-wavelength affine group. Batch
centering is a translation group. CORAL is a second-moment alignment group.
Each removes a different nuisance and leaves a different invariant.

The choice is falsifiable. The NIR oracle group fit (Table 6) tests whether the
per-wavelength affine group is adequate; it is (R² ≥ 0.9985). The MOX
group-stress test (Table 5) tests which feature is invariant to which group; the
answer is exact and unambiguous. The EEG reference experiment shows that a
group can be *too small*: the per-channel affine group does not include the
common-reference mixing, so the invariant breaks. A paper that reports only
accuracy cannot distinguish "the method is bad" from "the group is wrong."

### 6.3 Calibration versus invariant

Calibration transfer and invariant representation are dual. A section maps the
quotient back into a chosen frame; an invariant forgets the frame. The NIR
reference-scaling experiment is the clearest illustration: the invariant is flat
in the number of references (because two anchors define it), while calibration
improves with more references (because each reference constrains the section).
If the downstream label depends on absolute scale, the section is better. If
the downstream label is invariant under the group, the quotient is better and
more robust.

The MOX result is the same trade-off in a different guise. The labelled
calibration (five target samples per class) reaches 0.711; the best unsupervised
pooled representation reaches 0.515. The calibration uses labels; the invariant does
not. A fair comparison reports both, and states what information each uses.

### 6.4 Reconciliation with the prior zero-shot result

The companion paper [@james2026] proves that raw normalized MOX responses
cannot transfer zero-shot across devices with different power-law exponents.
Theorem 1 of this paper shows why: the device map is affine in log space, and
the raw response is not invariant under it. The invariant objects are ratios of
log differences (Theorem 2), which are lower-dimensional. The prior theorem and
this paper are not in conflict; they describe different objects. The prior
theorem is about the raw response; this paper is about the quotient. The
practical consequence is that "zero-shot raw transfer is impossible" and
"a lower-dimensional invariant transfers" can both be true, and the second does
not rescue the first.

The prior paper also reports a session-invariance result on a single rig and a
failure of zero-shot cross-device transfer. Both are consistent with the
framework: session invariance is a smaller group (temperature, humidity,
aging) than device replacement, and zero-shot raw transfer is the full group.

### 6.5 Negative results

We report four negative or boundary results.

1. **The full MOX per-sensor affine group makes gas identity unobservable**
   (Theorem 3). The common-gain invariant is worse than the log baseline (0.436
   vs 0.474); the exact per-sensor gain invariant is competitive but not
   better per-batch (0.451 vs 0.482 selectivity). The group is too large for
   identification; the useful invariants require a physical restriction (fixed
   exponent).
2. **Raman is underpowered.** With 17 spectra, the invariant is better than raw
   under the affine transform but worse than raw under the identity; the
   confidence intervals are wide. The real-data experiment does not establish a
   transfer advantage.
3. **Sequencing is not affine.** Platform normalization does not improve
   fine-grained haplotype classification (0.630–0.650 vs 0.643–0.665 raw). The
   channel is nonlinear; an affine-style correction is the wrong model.
4. **MOX drift is not a cross-instrument benchmark.** The UCI data are one
   array over 36 months. They test temporal drift, not distinct hardware. The
   fitness table records this.

The negatives are results, not failures of the paper. They delimit the
framework: it applies when the instrument induces a group of the right size,
and it says so when it does not.

### 6.6 Limitations

- **Group misspecification.** The derived groups are models. The NIR group is
  adequate (R² ≥ 0.9985), the MOX group is partially adequate, and the
  sequencing group is wrong. In a new modality, the group must be validated
  before the invariant is trusted.
- **Synthetic transforms.** EEG, Raman, microscopy, mass spec, and sequencing
  use synthetic instrument transforms or simulated reads. They test the
  calculus, not the instrument. The labels are explicit, but a reader should
  not treat them as cross-instrument evidence.
- **Small samples.** Raman has 17 spectra; the NIR data have 80 samples; the
  EEG data are one subject. The confidence intervals are wide, and the
  per-batch MOX variation is large.
- **Noise amplification.** The invariant divides by anchor differences. A
  poorly conditioned anchor pair can produce a large negative R² (NIR random
  pairs median −0.073). Anchor conditioning is part of the algorithm, not an
  optional preprocessing step.
- **Non-stationarity.** The group is assumed fixed per instrument. If the
  instrument drifts during the measurement, the group is time-varying; the
  UCI MOX data are an example. A time-varying group requires a time-varying
  section, which is a different problem.
- **Labels and sections.** The calibrated MOX baseline uses target labels. It
  is a section, not an invariant. Comparisons must state the information used.

### 6.7 A practical recipe

The framework suggests a sequence of decisions.

1. **Write the instrument equation.** Identify the physical parameters that
   change across instruments (gain, offset, exponent, calibration, reference).
2. **Derive the group.** Decide whether it is global affine, per-channel
   affine, per-wavelength affine, a smooth field, or non-affine.
3. **Count dimensions.** If the group dimension is at least the measurement
   dimension, no continuous invariant exists; stop and use a section or a
   different measurement.
4. **Choose the quotient.** Compute the invariant (ratios, cross-ratios,
   baseline differences). Check that the label is not invariant under the
   group; if it is, the group is too large.
5. **Choose the section.** If the label depends on absolute scale, calibrate
   with references. Use the group dimension to decide how many references are
   needed.
6. **Condition the anchors.** Choose reference samples that maximize the
   denominator separation; report the conditioning score.
7. **Validate the group.** Fit the group on paired samples (oracle) and compare
   nested groups (identity, global, per-channel). If the group does not explain
   the variation, the invariant is not trustworthy.
8. **Report the ceiling and the information used.** Within-instrument CV,
   target labels used, reference spectra used.

### 6.8 Open problems

- **Automatic group discovery.** The group is currently derived by hand from
  physics. Can it be inferred from paired data while remaining interpretable?
- **Nonlinear groups.** Sequencing, fragmentation, and saturation are not
  affine. A Lie-groupoid or probabilistic-channel formulation may be the right
  generalization.
- **Optimal reference selection.** The anchor conditioning rule is a heuristic.
  An information-theoretic criterion for reference selection would be better.
- **Time-varying groups.** Drift is a path in the group, not a fixed element.
  A gauge-theoretic formulation with a connection could model it.
- **Information bounds.** How much label information is lost by quotienting?
  The MOX result (common-gain invariant worse than log; exact per-sensor gain
  invariant competitive but not better per-batch) is a warning; a general bound
  would make the trade-off quantitative.
- **Cross-modal invariants.** The same group (affine in log space) appears in
  MOX, spectroscopy, and EEG. Is there a common quotient that spans modalities,
  or only a common formalism?

---

## 7. Conclusion

An instrument is a group. The group is derived from the measurement physics,
and the device-invariant content of a measurement is the quotient. This
framing explains why calibration transfer works, how many references are
needed, and when no invariant can exist. It also explains why some
"normalizations" fail: they quotient by the wrong group.

The experiments support the framework where the group is adequate: NIR
per-wavelength affine transfer reaches R² 0.83–0.91 with two references;
MOX per-sensor gain invariance is exact under the restricted group; EEG
baseline invariance is robust to gain and offset; microscopy flat-field
correction recovers segmentation. They also delimit it: the full MOX group
destroys gas identity, the Raman data are too small, and sequencing is not
affine.

The practical message is not that invariance replaces calibration. It is that
invariance and calibration are two parts of one object: the quotient and its
section. A method that knows which group it is quotienting by can say what it
preserves, what it loses, and what it needs. A method that does not know is
guessing.

---

## Appendix A. Symbolic verification

The derivations are implemented in `invmath/groups.py`; the report is
`results/math_report.txt`. The checks are:

1. The three-point ratio is unchanged under \(y\mapsto ky+d\) (SymPy
   simplifies the difference to 0).
2. The polynomial invariant ring of \(\operatorname{Aff}(1)\) is trivial. The
   symbolic check covers degrees 0–3; the argument is general: translation
   forces dependence on differences, scaling forces homogeneous degree 0, and a
   degree-0 polynomial is constant. The rational invariant field has degree
   \(n-2\), generated by \((y_i-y_1)/(y_2-y_1)\).
3. The MOX device map \(y_B=ky_A+d\) reconstructs \(y_B\) with error 0.
4. The MOX three-point invariant equals the log-concentration ratio with error
   0.
5. The per-wavelength cross-sample ratio is unchanged under
   \(A\mapsto gA+o\) with error 0.
6. The cross-sensor hierarchy: the full per-sensor affine group is transitive
   on a single vector; the restricted per-sensor gain group (an additive shift
   in log space) leaves slope ratios.

The numerical tests in `invmath/tests/test_invariants.py` cover the same
identities with random values. They pass (`9 passed`).

## Appendix B. Reproducibility

**Environment.** Python 3.12, NumPy, pandas, SciPy, scikit-learn, matplotlib,
SymPy, scikit-image, pyedflib, requests, pytest, tabulate, pooch, pybtex. See
`requirements.txt`.

**Data.** `experiments/download_data.py` downloads the UCI drift and Corn
datasets and writes `data/manifest.json` with hashes and provenance. EEG, Raman,
and phiX174 data are downloaded by their experiment scripts. The data are not
committed; the manifest records what was used.

**Run.**

```bash
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

Every number in the paper is produced by one of these scripts and stored in
`results/`; `results/README.md` maps each section to its CSV/JSON file. The
figures are in `paper/figures/`.

**Data and code availability.** The download scripts and the manifest
(`data/manifest.json`) record the provenance and hashes of the UCI and Corn
data. EEG, Raman, and phiX174 data are downloaded by their experiment scripts.
The data are not committed. The code is in this repository; the paper and
explainer are `paper/draft.md` and `docs/explainer.md`.

**Compile.**

```bash
pandoc paper/draft.md --bibliography=paper/refs.bib \
  --resource-path=paper --citeproc --pdf-engine=xelatex -o paper/draft.pdf
```

## References

The BibTeX source is `paper/refs.bib`; a generated author-year list is in
`paper/references.md`. Citation keys are resolved by Pandoc's `--citeproc`.
