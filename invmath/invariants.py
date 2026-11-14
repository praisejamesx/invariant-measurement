"""
invariants.py -- numerical construction of device-invariant representations.

All functions are pure NumPy and operate on arrays.  The docstrings state the
group under which each representation is invariant.  No function estimates
instrument parameters unless explicitly named; the invariants are computed
from the measurements alone.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Affine invariants
# ---------------------------------------------------------------------------

def affine_ratios(y: np.ndarray, anchor: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Return the Aff(1) invariants of a vector y.

    Parameters
    ----------
    y : array, shape (n,)
        Points on the line (e.g. log-responses at n concentrations).
    anchor : (i, j)
        Two indices used as the reference frame.  They must satisfy
        y[i] != y[j].

    Returns
    -------
    r : array, shape (n,)
        r[k] = (y[k] - y[i]) / (y[j] - y[i]).  Invariant under
        y -> k*y + d for any k > 0, d.  The two anchors map to 0 and 1.

    Notes
    -----
    With n=3 this is the single three-point ratio.  With n>3 the n-2
    non-anchor entries are algebraically independent.  The representation is
    a *shape*: it forgets the affine frame (gain and offset).
    """
    y = np.asarray(y, dtype=float)
    i, j = anchor
    denom = y[j] - y[i]
    if abs(denom) < 1e-12:
        raise ValueError("anchor points coincide; affine frame undefined")
    return (y - y[i]) / denom


def affine_ratios_batch(Y: np.ndarray, anchor: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Row-wise affine invariants of a 2-D array, shape (n_samples, n_points)."""
    Y = np.asarray(Y, dtype=float)
    i, j = anchor
    denom = Y[:, j] - Y[:, i]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("some anchor points coincide")
    return (Y - Y[:, [i]]) / denom[:, None]


def cross_ratio_4(y: np.ndarray) -> float:
    """Projective cross-ratio of four points on the line.

    CR = (y1-y2)(y3-y4) / ((y2-y3)(y1-y4)).
    It is invariant under the larger group PGL(2), hence also under Aff(1).
    """
    y = np.asarray(y, dtype=float)
    if y.size != 4:
        raise ValueError("cross-ratio needs exactly four points")
    y1, y2, y3, y4 = y
    return float((y1 - y2) * (y3 - y4) / ((y2 - y3) * (y1 - y4)))


# ---------------------------------------------------------------------------
# MOX: power-law response curves
# ---------------------------------------------------------------------------

def power_law_log_slope(concentrations: np.ndarray, responses: np.ndarray) -> float:
    """Fit b in s = a*C^b by least squares in log-log space."""
    x = np.log(np.asarray(concentrations, dtype=float))
    y = np.log(np.asarray(responses, dtype=float))
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def mox_curve_invariants(
    concentrations: np.ndarray,
    responses: np.ndarray,
    anchor: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Affine invariants of a single sensor's log-response curve.

    The invariant is the affine shape of (log s) as a function of log C.
    For a power law the curve is a line, so the invariant recovers the
    relative spacing of the chosen concentrations, not the gas identity.
    """
    y = np.log(np.asarray(responses, dtype=float))
    return affine_ratios(y, anchor=anchor)


def cross_sensor_slope_ratios(
    concentrations: np.ndarray,
    responses: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Ratios b_i / b_j of per-sensor log-log slopes.

    Parameters
    ----------
    concentrations : array, shape (n_levels,)
    responses : array, shape (n_sensors, n_levels)

    Returns
    -------
    ratios : array, shape (n_sensors*(n_sensors-1)//2,)
        Upper-triangular pairwise slope ratios.  These are invariant under a
        per-sensor multiplicative gain a_i -> c_i a_i (the alpha_i shifts in
        the log-linear model) but NOT under a change of the exponents b_i.
        They are therefore valid for a fixed sensor family, not for arbitrary
        replacement sensors.
    """
    n_sensors = responses.shape[0]
    slopes = np.array(
        [power_law_log_slope(concentrations, responses[i]) for i in range(n_sensors)]
    )
    out = []
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            out.append(slopes[i] / max(abs(slopes[j]), eps))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Spectroscopy: global and per-wavelength invariants
# ---------------------------------------------------------------------------

def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate: row-wise (x - mean) / std.

    This is the invariant of the global affine group x -> g*x + o acting on
    each spectrum (same g, o for all wavelengths).  It is *not* invariant to
    per-wavelength gain and offset.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def per_wavelength_affine_invariants(
    X: np.ndarray,
    anchor: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Cross-sample, per-wavelength affine invariants.

    Parameters
    ----------
    X : array, shape (n_samples, n_wavelengths)
        A matrix of spectra measured on one instrument.
    anchor : (i, j)
        Two *physical samples* measured on every instrument.  These are the
        reference samples; their labels are not needed to compute the
        representation.

    Returns
    -------
    R : array, shape (n_samples, n_wavelengths)
        R[k, l] = (X[k,l] - X[i,l]) / (X[j,l] - X[i,l]).

    Invariance
    ----------
    If another instrument measures the same samples as
    X'[k,l] = g[l]*X[k,l] + o[l], then R' = R exactly (up to noise).
    The representation is therefore invariant to arbitrary per-wavelength
    gain and offset.  It requires the same two physical reference samples on
    both instruments.  This is the spectroscopy analogue of the two-point
    calibration in the MOX paper, but it removes 2*p instrument parameters
    (gain and offset at every wavelength) with only 2 reference samples.
    """
    X = np.asarray(X, dtype=float)
    i, j = anchor
    denom = X[j] - X[i]
    bad = np.abs(denom) < 1e-12
    denom = np.where(bad, 1.0, denom)
    R = (X - X[[i]]) / denom[None, :]
    if np.any(bad):
        R[:, bad] = 0.0
    return R


def global_affine_invariants(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Invariants of x -> g*x + o with a *single* g,o for the whole spectrum.

    Take differences against the first wavelength and divide by a second
    difference.  The result is the affine shape of the spectrum.
    """
    X = np.asarray(X, dtype=float)
    d = X - X[:, [0]]
    denom = d[:, [1]]
    denom = np.where(np.abs(denom) < eps, eps, denom)
    return d / denom


# ---------------------------------------------------------------------------
# Dimension bookkeeping
# ---------------------------------------------------------------------------

def invariant_dimension(space_dim: int, group_dim: int, orbit_generic: bool = True) -> int:
    """Expected dimension of the quotient for a free, generic action.

    For a Lie group G acting freely on an open dense set of M,
        dim(M/G) = dim M - dim G.
    This is the counting rule used to decide whether a group leaves any
    observable content.  If the number is zero, the quotient is a point and
    no nonconstant invariant exists (generically).
    """
    if not orbit_generic:
        raise ValueError("dimension formula requires a generic free orbit")
    return space_dim - group_dim


def observability_ratio(space_dim: int, group_dim: int) -> float:
    """Fraction of the measurement space that survives the quotient.

    This is a heuristic, not a theorem: it counts degrees of freedom, not
    information.  It is useful for comparing instrument groups.
    """
    if space_dim <= 0:
        raise ValueError("space_dim must be positive")
    return max(space_dim - group_dim, 0) / space_dim
