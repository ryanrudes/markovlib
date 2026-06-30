"""EM fitting — the defining guarantee (monotonic log-likelihood increase) and the E-step identities.

EM theory guarantees the observed-data log-likelihood never decreases across iterations; that is the
sharpest deterministic correctness test, so it is asserted directly. The expected sufficient statistics
are checked against their marginalization identities (``Σ_j ξ = γ``, ``Σ_ij ξ = 1``, ``loglik`` matches
the smoother).
"""

from __future__ import annotations

import numpy as np

import markovlib as mk
from markovlib.engines.exact_chain import ExactChain


def _random_problem(rng: np.random.Generator, n_states: int, n_steps: int):
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))
    return mk.DiscreteChain(log_init, log_trans), log_em


def test_em_loglik_increases_monotonically():
    rng = np.random.default_rng(0)
    for _ in range(10):
        model, log_em = _random_problem(rng, int(rng.integers(2, 4)), int(rng.integers(5, 12)))
        history = np.array(mk.fit(model, log_em, max_iter=40).loglik_history)
        assert np.all(np.diff(history) >= -1e-9)  # the EM guarantee


def test_em_returns_a_valid_stochastic_model():
    rng = np.random.default_rng(1)
    model, log_em = _random_problem(rng, 3, 10)
    fitted = mk.fit(model, log_em, max_iter=30).model
    assert np.isclose(np.exp(fitted.log_init).sum(), 1.0)
    assert np.allclose(np.exp(fitted.log_trans).sum(axis=1), 1.0)


def test_returned_model_matches_last_recorded_loglik():
    rng = np.random.default_rng(2)
    model, log_em = _random_problem(rng, 3, 10)
    result = mk.fit(model, log_em, max_iter=40)
    assert np.isclose(mk.loglik(result.model, log_em), result.loglik_history[-1])


def test_em_respects_max_iter():
    rng = np.random.default_rng(3)
    model, log_em = _random_problem(rng, 3, 10)
    result = mk.fit(model, log_em, max_iter=4, tol=0.0)  # tol=0 never triggers the convergence break
    assert len(result.loglik_history) == 4


def test_em_stops_early_on_convergence():
    rng = np.random.default_rng(4)
    model, log_em = _random_problem(rng, 2, 8)
    result = mk.fit(model, log_em, max_iter=500, tol=1e-9)
    assert len(result.loglik_history) < 500


def test_expected_stats_marginalization_identities():
    rng = np.random.default_rng(5)
    model, log_em = _random_problem(rng, 3, 7)
    stats = ExactChain().expected_stats(model, log_em)
    assert np.allclose(stats.xi.sum(axis=2), stats.gamma[:-1], atol=1e-9)  # Σ_j ξ[t,i,j] = γ[t,i]
    assert np.allclose(stats.xi.sum(axis=(1, 2)), 1.0, atol=1e-9)  # ξ[t] is a distribution
    assert np.isclose(stats.loglik, mk.smooth(model, log_em).loglik)
