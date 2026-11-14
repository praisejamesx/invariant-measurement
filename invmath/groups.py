"""
groups.py -- symbolic derivation of instrument transformation groups.

The central object is a measurement space M and a group G of transformations
induced by the instrument.  The observable content of a measurement is the
field of rational invariants R(M)^G (and, when it exists, the polynomial
invariant ring R[M]^G).  The quotient M/G is the device-invariant
representation.

Run:
    python math/groups.py
"""

from __future__ import annotations

import sympy as sp


# ---------------------------------------------------------------------------
# 1. The affine instrument group Aff(1) and its rational invariants
# ---------------------------------------------------------------------------

def affine_action(y: sp.Symbol, k: sp.Symbol, d: sp.Symbol) -> sp.Expr:
    """A single point transforms as y -> k*y + d."""
    return k * y + d


def three_point_affine_ratio() -> sp.Expr:
    """Prove that (y1-y2)/(y3-y2) is invariant under y -> k*y + d.

    Returns the simplified ratio after the substitution.  The result is the
    original ratio, i.e. k and d cancel identically.
    """
    y1, y2, y3, k, d = sp.symbols("y1 y2 y3 k d", real=True)
    y1p = affine_action(y1, k, d)
    y2p = affine_action(y2, k, d)
    y3p = affine_action(y3, k, d)
    ratio = (y1p - y2p) / (y3p - y2p)
    return sp.simplify(ratio)


def n_point_affine_invariants(n: int) -> list[sp.Expr]:
    """Return a generating set of rational invariants for Aff(1) on n points.

    r_i = (y_i - y_1)/(y_2 - y_1), i = 3..n.  These are invariant under
    y -> k*y + d and are algebraically independent (the invariant field is
    purely transcendental of degree n-2).
    """
    if n < 3:
        raise ValueError("Aff(1) needs at least three points for a ratio invariant")
    y = sp.symbols(f"y1:{n + 1}", real=True)
    out = []
    for i in range(2, n):
        out.append(sp.simplify((y[i] - y[0]) / (y[1] - y[0])))
    return out


def prove_n_point_invariance(n: int = 5) -> list[sp.Expr]:
    """Substitute y_i -> k*y_i + d and verify each ratio is unchanged."""
    k, d = sp.symbols("k d", real=True)
    y = sp.symbols(f"y1:{n + 1}", real=True)
    invs = n_point_affine_invariants(n)
    checks = []
    for inv in invs:
        subs = {y[i]: k * y[i] + d for i in range(n)}
        checks.append(sp.simplify(inv.subs(subs) - inv))
    return checks


def polynomial_invariant_is_trivial_check(max_degree: int = 3) -> dict:
    """Symbolically check that no nonconstant polynomial in the differences is
    invariant under both translation and positive scaling, up to max_degree.

    A translation-invariant polynomial is a polynomial in the differences
    u_i = y_i - y_1.  A scaling y -> k*y makes u_i -> k*u_i.  A homogeneous
    polynomial of degree 0 is constant.  This function verifies the low-degree
    case explicitly, which is the case a reader can check by hand.
    """
    u1, u2, u3 = sp.symbols("u1 u2 u3", real=True)
    k = sp.Symbol("k", positive=True)
    results = {}
    # monomials of total degree d in three differences
    for degree in range(0, max_degree + 1):
        monos = []
        for a in range(degree + 1):
            for b in range(degree - a + 1):
                c = degree - a - b
                monos.append(u1**a * u2**b * u3**c)
        # a general homogeneous polynomial of this degree
        coeffs = sp.symbols(f"c0:{len(monos)}", real=True)
        poly = sum(c * m for c, m in zip(coeffs, monos))
        scaled = sp.expand(poly.subs({u1: k * u1, u2: k * u2, u3: k * u3}))
        diff = sp.expand(scaled - poly)
        eqs = [sp.Eq(sp.expand(diff).coeff(m), 0) for m in monos]
        sol = sp.solve(eqs, coeffs, dict=True)
        free = []
        if sol:
            free = [c for c in coeffs if c not in sol[0]]
        results[degree] = {
            "monomials": len(monos),
            "solution": sol[0] if sol else {},
            "free_coefficients": free,
            "note": "no nonconstant invariant" if degree > 0 and not free else "",
        }
    return results


# ---------------------------------------------------------------------------
# 2. MOX power law: the device group in log space
# ---------------------------------------------------------------------------

def mox_device_map() -> dict:
    """Derive y_B = k*y_A + d from s = a*C^b.

    y = log(s) = log(a) + b*log(C).  For two devices A and B:
        y_B = log(a_B) + b_B*log(C)
            = (b_B/b_A) * (y_A - log(a_A)) + log(a_B)
            = k*y_A + d
    with k = b_B/b_A and d = log(a_B) - k*log(a_A).  Since a_A, a_B > 0,
    k > 0 and the group is Aff(1) restricted to positive linear part.
    """
    C, aA, aB, bA, bB = sp.symbols("C a_A a_B b_A b_B", positive=True)
    yA = sp.log(aA) + bA * sp.log(C)
    yB = sp.log(aB) + bB * sp.log(C)
    k = sp.simplify(bB / bA)
    d = sp.simplify(sp.log(aB) - k * sp.log(aA))
    reconstructed = sp.simplify(k * yA + d)
    return {
        "k": k,
        "d": d,
        "yA": yA,
        "yB": yB,
        "reconstruction_error": sp.simplify(reconstructed - yB),
    }


def mox_three_point_invariant() -> sp.Expr:
    """Prove that the three-point log ratio equals the log-concentration ratio.

    y_i = log(a) + b*log(C_i).  Then
        (y1-y2)/(y3-y2) = (log C1 - log C2)/(log C3 - log C2),
    independent of a and b.
    """
    C1, C2, C3, a, b = sp.symbols("C1 C2 C3 a b", positive=True)
    y1 = sp.log(a) + b * sp.log(C1)
    y2 = sp.log(a) + b * sp.log(C2)
    y3 = sp.log(a) + b * sp.log(C3)
    ratio = (y1 - y2) / (y3 - y2)
    target = (sp.log(C1) - sp.log(C2)) / (sp.log(C3) - sp.log(C2))
    return sp.simplify(ratio - target)


def mox_cross_sensor_matrix() -> dict:
    """Formalise the cross-sensor case and the observability hierarchy.

    For one measurement vector y in R^p, the full device group acts
    coordinatewise:
        y_i -> k_i y_i + d_i,   k_i > 0.
    It is transitive on R^p: for any y, y' choose k_i = 1 and
    d_i = y'_i - y_i.  Hence the quotient is a point and every invariant is
    constant.  A single gas measurement therefore cannot identify the gas
    under arbitrary per-sensor affine replacement.

    For a *collection* of samples (a p x n matrix), the group acts on each
    row, and row-wise affine cross-ratios survive.  Those describe the
    configuration of samples, not a device-independent per-sample label.  In
    the common-latent model
        y_{ik} = alpha_i + beta_i * x_k,   x_k = log(C_k),
    the row-wise cross-ratios are functions of x only, so they do not encode
    gas identity either.

    If the sensitivity exponent is stable across devices (beta_i fixed, the
    "same sensor model" restriction), a per-sensor multiplicative gain in the
    linear response is an additive shift alpha_i -> alpha_i + delta_i.  Then
    beta_i/beta_j is invariant and gas identity becomes observable to the
    extent that the beta profiles differ.
    """
    return {
        "model": "y_{ik} = alpha_i + beta_i * x_k",
        "single_sample_full_group": "prod_i Aff(1): transitive on R^p; quotient is a point; invariant ring is R",
        "multi_sample_full_group": "row-wise affine cross-ratios survive; they describe the sample configuration, not a per-sample gas label",
        "common_latent_model": "row-wise cross-ratios are functions of x_k only, so they do not encode gas identity",
        "restricted_group": "per-sensor additive shift in log space (multiplicative gain in linear response): alpha_i -> alpha_i + delta_i, beta_i fixed",
        "restricted_group_invariants": "x-cross-ratios and beta_i/beta_j",
        "conclusion": (
            "Observability is not a property of the sensor alone; it is a "
            "property of the pair (signal, group). Enlarging the group can "
            "destroy the label."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Beer-Lambert and the spectroscopy hierarchy
# ---------------------------------------------------------------------------

def beer_lambert_groups() -> dict:
    """Three nested instrument groups for absorbance spectroscopy.

    Beer-Lambert: A(lambda) = eps(lambda) * c * L + b(lambda).

    G_scale:      A -> g A                       (global multiplicative gain)
        invariants within one spectrum: A(l1)/A(l2)
    G_affine:     A -> g A + o                   (global gain and offset)
        invariants within one spectrum: (A(l1)-A(l2))/(A(l3)-A(l2))
    G_per_lambda: A(l) -> g(l) A(l) + o(l)      (per-wavelength gain/offset)
        no within-spectrum invariant; with n samples measured on both
        instruments, the cross-sample ratios
            r_{i,l} = (A_{i,l}-A_{1,l})/(A_{2,l}-A_{1,l})
        are invariant for every wavelength l.  Two reference samples per
        instrument (the anchors 1 and 2) therefore align the representation.
    """
    return {
        "G_scale": {
            "action": "A -> g*A, g>0",
            "within_sample_invariants": "A(l1)/A(l2)",
            "dimension": "p-1 for p wavelengths",
        },
        "G_affine": {
            "action": "A -> g*A + o",
            "within_sample_invariants": "(A(l1)-A(l2))/(A(l3)-A(l2))",
            "dimension": "p-2 for p wavelengths",
        },
        "G_per_lambda": {
            "action": "A(l) -> g(l)*A(l) + o(l)",
            "within_sample_invariants": "none",
            "cross_sample_invariants": "(A_i,l - A_1,l)/(A_2,l - A_1,l)",
            "reference_requirement": "2 reference samples per instrument",
        },
    }


def per_wavelength_invariance_proof() -> sp.Expr:
    """Prove the cross-sample per-wavelength ratio is invariant.

    A'_{i,l} = g_l A_{i,l} + o_l.  Then
        (A'_{i,l}-A'_{1,l})/(A'_{2,l}-A'_{1,l})
      = (g_l(A_{i,l}-A_{1,l}))/(g_l(A_{2,l}-A_{1,l}))
      = (A_{i,l}-A_{1,l})/(A_{2,l}-A_{1,l}).
    """
    Ai, A1, A2, g, o = sp.symbols("A_i A_1 A_2 g o", real=True, positive=True)
    lhs = ((g * Ai + o) - (g * A1 + o)) / ((g * A2 + o) - (g * A1 + o))
    rhs = (Ai - A1) / (A2 - A1)
    return sp.simplify(lhs - rhs)


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------

def report() -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("INSTRUMENT GROUPS AND INVARIANTS -- SYMBOLIC REPORT")
    lines.append("=" * 72)

    lines.append("\n[1] Aff(1) on one point: y -> k*y + d")
    lines.append("    Three-point ratio after the action:")
    lines.append(f"      (y1'-y2')/(y3'-y2') = {three_point_affine_ratio()}")
    checks = prove_n_point_invariance(5)
    lines.append(f"    Five-point generating invariants all unchanged: {all(c == 0 for c in checks)}")

    lines.append("\n[2] Polynomial invariants of Aff(1)")
    poly = polynomial_invariant_is_trivial_check(3)
    for deg, info in poly.items():
        lines.append(
            f"    degree {deg}: {info['monomials']} monomials, "
            f"free coefficients = {len(info['free_coefficients'])}, {info['note']}"
        )
    lines.append(
        "    Translation forces dependence on differences; scaling forces\n"
        "    homogeneous degree 0; hence the polynomial invariant ring is R.\n"
        "    The rational invariant field is R(r_3,...,r_n), degree n-2."
    )

    lines.append("\n[3] MOX power law s = a*C^b")
    mox = mox_device_map()
    lines.append(f"    k = {mox['k']}")
    lines.append(f"    d = {mox['d']}")
    lines.append(f"    reconstruction error = {mox['reconstruction_error']}")
    lines.append(f"    three-point invariant error = {mox_three_point_invariant()}")

    lines.append("\n[4] Cross-sensor observability hierarchy")
    cs = mox_cross_sensor_matrix()
    for key, value in cs.items():
        lines.append(f"    {key}: {value}")

    lines.append("\n[5] Beer-Lambert spectroscopy hierarchy")
    bl = beer_lambert_groups()
    for key, value in bl.items():
        lines.append(f"    {key}: {value}")
    lines.append(f"    per-wavelength ratio error = {per_wavelength_invariance_proof()}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
