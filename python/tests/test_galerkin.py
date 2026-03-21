"""Tests for pydme.galerkin — Galerkin weight matrix construction."""

import math

import numpy as np
import pytest

from pydme.galerkin import (
    _dw,
    _w,
    build_disc_weight_matrix,
    build_poly_degree_matrix,
    build_weight_matrix,
)


# ---------------------------------------------------------------------------
# Weight function tests
# ---------------------------------------------------------------------------


class TestWeightFunctions:
    def test_bump_vanishes_at_endpoints(self):
        t = np.array([-1.0, 0.0, 1.0])
        vals = _w(t, p=5)
        assert math.isclose(vals[0], 0.0, abs_tol=1e-12)
        assert math.isclose(vals[2], 0.0, abs_tol=1e-12)
        assert vals[1] > 0

    def test_bump_positive_interior(self):
        t = np.linspace(-0.9, 0.9, 20)
        assert (_w(t, p=5) > 0).all()

    def test_derivative_zero_at_midpoint(self):
        # By symmetry, dw/dt = 0 at t=0 (midpoint of [-1,1])
        dw_mid = _dw(np.array([0.0]), p=5)
        assert abs(float(dw_mid[0])) < 1e-12

    def test_derivative_sign_change(self):
        t_neg = np.array([-0.5])
        t_pos = np.array([0.5])
        # w increases toward centre then decreases → dw > 0 left, < 0 right
        assert float(_dw(t_neg, p=5)[0]) > 0
        assert float(_dw(t_pos, p=5)[0]) < 0


# ---------------------------------------------------------------------------
# build_weight_matrix tests
# ---------------------------------------------------------------------------


class TestBuildWeightMatrix:
    def _uniform_ts(self, n=50, dt=0.1):
        return np.linspace(0.0, (n - 1) * dt, n)

    def test_output_shape_d0(self):
        ts = self._uniform_ts(n=50)
        W = build_weight_matrix(ts, window=10, d=0)
        assert W.shape == (40, 50)

    def test_output_shape_d1(self):
        ts = self._uniform_ts(n=50)
        DW = build_weight_matrix(ts, window=10, d=1)
        assert DW.shape == (40, 50)

    def test_output_shape_d2(self):
        ts = self._uniform_ts(n=50)
        DDW = build_weight_matrix(ts, window=10, d=2)
        assert DDW.shape == (40, 50)

    def test_w_positive(self):
        """W rows (d=0) should have non-negative entries (bump is non-negative)."""
        ts = self._uniform_ts()
        W = build_weight_matrix(ts, window=10, d=0)
        assert (W >= -1e-14).all()

    def test_dw_row_sums_near_zero(self):
        """For a constant function z=1, DW @ z should be ~0 (integrates dw/dt which sums to 0)."""
        ts = self._uniform_ts(n=60)
        DW = build_weight_matrix(ts, window=15, d=1)
        z_const = np.ones(60)
        dz = DW @ z_const
        # The integral of dw/dt over a bump that vanishes at both endpoints
        # is exactly zero by the fundamental theorem of calculus.
        np.testing.assert_allclose(dz, 0.0, atol=1e-10)

    def test_derivative_of_linear_z(self):
        """For z(t) = t, DW @ z should approximate dt/dt = 1 at each window."""
        n, dt = 80, 0.05
        ts = np.linspace(0.0, (n - 1) * dt, n)
        window = 15
        W  = build_weight_matrix(ts, window=window, d=0)
        DW = build_weight_matrix(ts, window=window, d=1)
        z = ts  # linear
        # Galerkin gives: -∫ z dw/dt dt = ∫ dz/dt w dt = 1 * ∫ w dt
        # So DW @ z should equal W @ (ones vector)
        lhs = DW @ z
        rhs = W @ np.ones(n)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)

    def test_invalid_d_raises(self):
        ts = self._uniform_ts()
        with pytest.raises(ValueError):
            build_weight_matrix(ts, window=10, d=3)


# ---------------------------------------------------------------------------
# build_disc_weight_matrix tests
# ---------------------------------------------------------------------------


class TestBuildDiscWeightMatrix:
    def test_shapes(self):
        ts = np.linspace(0, 1, 30)
        DW, W = build_disc_weight_matrix(ts)
        assert DW.shape == (30, 30)
        assert W.shape == (30, 30)

    def test_w_is_identity(self):
        ts = np.linspace(0, 1, 20)
        _, W = build_disc_weight_matrix(ts)
        np.testing.assert_array_equal(W, np.eye(20))

    def test_dw_forward_differences(self):
        ts = np.linspace(0, 1, 5)
        DW, _ = build_disc_weight_matrix(ts)
        z = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        dz = DW @ z
        expected = np.array([2.0, -1.0, 3.0, -1.0, 0.0])
        np.testing.assert_array_equal(dz, expected)


# ---------------------------------------------------------------------------
# build_poly_degree_matrix tests
# ---------------------------------------------------------------------------


class TestBuildPolyDegreeMatrix:
    def test_shape_k1_order1(self):
        N = build_poly_degree_matrix(K=1, order=1)
        # Terms: [0], [1] → 2 terms
        assert N.shape == (2, 1)
        assert set(tuple(row) for row in N.astype(int)) == {(0,), (1,)}

    def test_shape_k2_order2(self):
        N = build_poly_degree_matrix(K=2, order=2)
        # C(2+2, 2) = 6 terms
        assert N.shape == (6, 2)

    def test_all_row_sums_le_order(self):
        for K in (1, 2, 3):
            for order in (1, 2, 3):
                N = build_poly_degree_matrix(K=K, order=order)
                assert (N.sum(axis=1) <= order).all()

    def test_k1_order2_terms(self):
        N = build_poly_degree_matrix(K=1, order=2)
        exponents = sorted(int(row[0]) for row in N)
        assert exponents == [0, 1, 2]

    def test_n_terms_formula(self):
        """Number of terms equals binomial(K+order, order)."""
        from math import comb
        for K in (1, 2, 3):
            for order in (0, 1, 2, 3):
                N = build_poly_degree_matrix(K=K, order=order)
                assert N.shape[0] == comb(K + order, order)

    def test_order_zero(self):
        N = build_poly_degree_matrix(K=3, order=0)
        assert N.shape == (1, 3)
        np.testing.assert_array_equal(N, [[0, 0, 0]])

    def test_dtype_float64(self):
        N = build_poly_degree_matrix(K=2, order=2)
        assert N.dtype == np.float64

    def test_negative_order_raises(self):
        with pytest.raises(ValueError):
            build_poly_degree_matrix(K=2, order=-1)


# ---------------------------------------------------------------------------
# Integration test: Galerkin matrices recover correct dynamics
# ---------------------------------------------------------------------------


class TestGalerkinDynamicsRecovery:
    """Verify that W and DW satisfy the integration-by-parts identity
    for a known smooth signal z(t) = sin(2πt)."""

    def test_ibp_identity_sinusoid(self):
        """DW @ z ≈ W @ dz/dt for a sinusoid."""
        n = 200
        ts = np.linspace(0.0, 1.0, n)
        z = np.sin(2.0 * np.pi * ts)
        dzdt = 2.0 * np.pi * np.cos(2.0 * np.pi * ts)

        window = 20
        W  = build_weight_matrix(ts, window=window, d=0)
        DW = build_weight_matrix(ts, window=window, d=1)

        lhs = DW @ z          # Galerkin estimate of integrated derivative
        rhs = W @ dzdt        # true derivative integrated against w

        # Should agree to within a few percent
        rel_err = np.abs(lhs - rhs) / (np.abs(rhs).mean() + 1e-12)
        assert rel_err.mean() < 0.05, f"mean rel error = {rel_err.mean():.4f}"
