import numpy as np
import pytest

from invmath.invariants import (
    affine_ratios,
    affine_ratios_batch,
    cross_ratio_4,
    cross_sensor_slope_ratios,
    global_affine_invariants,
    invariant_dimension,
    mox_curve_invariants,
    per_wavelength_affine_invariants,
    snv,
)


def test_affine_ratio_invariant():
    rng = np.random.default_rng(0)
    y = rng.normal(size=7)
    r = affine_ratios(y, anchor=(0, 1))
    for _ in range(20):
        k = rng.uniform(0.1, 10)
        d = rng.normal()
        r2 = affine_ratios(k * y + d, anchor=(0, 1))
        np.testing.assert_allclose(r, r2, atol=1e-10)


def test_affine_batch_invariant():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 6))
    R = affine_ratios_batch(Y, anchor=(2, 4))
    for _ in range(10):
        k = rng.uniform(0.1, 10, size=(5, 1))
        d = rng.normal(size=(5, 1))
        R2 = affine_ratios_batch(k * Y + d, anchor=(2, 4))
        np.testing.assert_allclose(R, R2, atol=1e-10)


def test_cross_ratio_invariant_under_projective_map():
    rng = np.random.default_rng(2)
    y = rng.normal(size=4)
    cr = cross_ratio_4(y)
    a, b, c, d = rng.normal(size=4)
    if abs(c * y[0] + d) < 1e-6:
        return
    y2 = (a * y + b) / (c * y + d)
    assert np.isfinite(cross_ratio_4(y2))
    np.testing.assert_allclose(cr, cross_ratio_4(y2), rtol=1e-8)


def test_mox_curve_invariant_under_device_change():
    rng = np.random.default_rng(3)
    C = np.array([10.0, 20.0, 40.0, 80.0])
    y = np.log(C) * 0.7 + 1.3
    r = mox_curve_invariants(C, np.exp(y))
    for _ in range(20):
        a = np.exp(rng.normal())
        b = rng.uniform(0.2, 3.0)
        y2 = np.log(a) + b * np.log(C)
        r2 = mox_curve_invariants(C, np.exp(y2))
        np.testing.assert_allclose(r, r2, atol=1e-10)


def test_per_wavelength_invariant():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 50))
    R = per_wavelength_affine_invariants(X, anchor=(0, 1))
    g = rng.uniform(0.2, 5.0, size=50)
    o = rng.normal(size=50)
    R2 = per_wavelength_affine_invariants(X * g + o, anchor=(0, 1))
    np.testing.assert_allclose(R, R2, atol=1e-9)


def test_snv_invariant_global_affine():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 30))
    S = snv(X)
    g = rng.uniform(0.2, 5.0)
    o = rng.normal()
    S2 = snv(g * X + o)
    np.testing.assert_allclose(S, S2, atol=1e-9)


def test_global_affine_invariants():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(4, 20))
    G = global_affine_invariants(X)
    g = rng.uniform(0.2, 5.0)
    o = rng.normal()
    G2 = global_affine_invariants(g * X + o)
    np.testing.assert_allclose(G, G2, atol=1e-9)


def test_cross_sensor_slope_ratios_gain_invariant():
    rng = np.random.default_rng(7)
    C = np.array([10.0, 20.0, 40.0])
    b = np.array([0.5, 1.0, 1.5])
    a = np.exp(rng.normal(size=3))
    S = a[:, None] * C[None, :] ** b[:, None]
    r1 = cross_sensor_slope_ratios(C, S)
    a2 = np.exp(rng.normal(size=3))
    S2 = a2[:, None] * C[None, :] ** b[:, None]
    r2 = cross_sensor_slope_ratios(C, S2)
    np.testing.assert_allclose(r1, r2, atol=1e-8)


def test_invariant_dimension():
    assert invariant_dimension(10, 2) == 8
    assert invariant_dimension(3, 2) == 1
    with pytest.raises(ValueError):
        invariant_dimension(1, 1, orbit_generic=False)
