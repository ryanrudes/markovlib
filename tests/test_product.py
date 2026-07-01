"""Tests for the factored (product) state-space support — validated against brute force."""

from __future__ import annotations

import numpy as np
import pytest

import markovlib as mk
from markovlib.product import product_log_emission, product_marginals, product_membership


def test_membership_matches_definition() -> None:
    for n_states in (2, 3):
        for n_factors in (0, 1, 2, 3):
            m = product_membership(n_factors, n_states)
            n = n_states**n_factors
            assert m.shape == (n, n_factors)
            for k in range(n):
                for e in range(n_factors):
                    assert m[k, e] == (k // n_states**e) % n_states


def test_membership_binary_is_bitmask() -> None:
    m = product_membership(4, 2)
    for k in range(16):
        for e in range(4):
            assert m[k, e] == ((k >> e) & 1)


def test_membership_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        product_membership(-1)
    with pytest.raises(ValueError):
        product_membership(3, 0)


def test_log_emission_matches_bruteforce() -> None:
    rng = np.random.default_rng(0)
    for n_states in (2, 3):
        for n_factors in (1, 2, 3):
            n_time = 5
            ev = rng.normal(size=(n_time, n_factors, n_states))
            m = product_membership(n_factors, n_states)
            out = product_log_emission(ev, m)
            n = n_states**n_factors
            assert out.shape == (n_time, n)
            for t in range(n_time):
                for k in range(n):
                    expect = sum(ev[t, e, m[k, e]] for e in range(n_factors))
                    assert out[t, k] == pytest.approx(expect)


def test_log_emission_binary_matches_where_form() -> None:
    # The exact expression contact used: where(active, log_active, log_inactive).sum(axis=2).
    rng = np.random.default_rng(1)
    n_time, n_factors = 6, 3
    log_active = rng.normal(size=(n_time, n_factors))
    log_inactive = rng.normal(size=(n_time, n_factors))
    ev = np.stack([log_inactive, log_active], axis=2)  # (T, E, 2): [...,0]=inactive, [...,1]=active
    m = product_membership(n_factors, 2)
    out = product_log_emission(ev, m)
    ref = np.where(m.astype(bool)[None, :, :], log_active[:, None, :], log_inactive[:, None, :]).sum(axis=2)
    assert np.array_equal(out, ref)  # bit-for-bit


def test_log_emission_rejects_bad_shapes() -> None:
    m = product_membership(2, 2)
    with pytest.raises(ValueError):
        product_log_emission(np.zeros((5, 2)), m)  # 2-D, not 3-D
    with pytest.raises(ValueError):
        product_log_emission(np.zeros((5, 3, 2)), m)  # factor-count mismatch (3 vs 2)
    with pytest.raises(ValueError):
        product_log_emission(np.zeros((5, 2, 2)), np.zeros((4,), dtype=np.intp))  # membership 1-D


def test_marginals_match_bruteforce_and_sum_to_one() -> None:
    rng = np.random.default_rng(2)
    for n_states in (2, 3):
        for n_factors in (1, 2, 3):
            n_time = 4
            n = n_states**n_factors
            joint = rng.random((n_time, n))
            joint /= joint.sum(axis=1, keepdims=True)
            m = product_membership(n_factors, n_states)
            marg = product_marginals(joint, m, n_states)
            assert marg.shape == (n_time, n_factors, n_states)
            for t in range(n_time):
                for e in range(n_factors):
                    for v in range(n_states):
                        expect = sum(joint[t, k] for k in range(n) if m[k, e] == v)
                        assert marg[t, e, v] == pytest.approx(expect)
            assert np.allclose(marg.sum(axis=2), 1.0)  # each factor's marginal is a distribution


def test_marginals_rejects_bad_shapes() -> None:
    m = product_membership(2, 2)
    with pytest.raises(ValueError):
        product_marginals(np.zeros((5,)), m)  # joint 1-D
    with pytest.raises(ValueError):
        product_marginals(np.zeros((5, 4)), np.zeros((4,), dtype=np.intp))  # membership 1-D


def test_exact_product_chain_end_to_end() -> None:
    # factored evidence -> product emission -> DiscreteChain smooth -> product_marginals
    # must equal a brute-force marginalization of the joint gamma.
    rng = np.random.default_rng(3)
    n_factors, n_states, n_time = 3, 2, 8
    n = n_states**n_factors
    ev = rng.normal(size=(n_time, n_factors, n_states))
    m = product_membership(n_factors, n_states)
    log_em = product_log_emission(ev, m)
    log_init = np.log(np.full(n, 1.0 / n))
    trans = rng.random((n, n))
    trans /= trans.sum(axis=1, keepdims=True)
    result = mk.smooth(mk.DiscreteChain(log_init=log_init, log_trans=np.log(trans)), log_em)
    marg = product_marginals(result.gamma, m, n_states)
    for e in range(n_factors):
        for v in range(n_states):
            expect = result.gamma[:, m[:, e] == v].sum(axis=1)
            assert np.allclose(marg[:, e, v], expect)
