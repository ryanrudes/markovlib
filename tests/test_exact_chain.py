"""ExactChain validated against brute-force path enumeration — the reference discipline in miniature.

For a tiny chain we enumerate every one of the ``S**T`` state paths, score each exactly, and read off
the posterior marginals, the log-likelihood, and the MAP path by definition. ExactChain's message
passes must match that oracle to ~1e-9. This is the same "the obvious-but-slow computation is the
reference the fast engine is tested against" discipline the whole library will lean on.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

import markovlib as mk


def _random_chain(rng: np.random.Generator, n_states: int, n_steps: int):
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))  # unnormalized is fine
    return log_init, log_trans, log_em


def _brute_force(log_init, log_trans, log_em):
    n_steps, n_states = log_em.shape
    paths = np.array(list(product(range(n_states), repeat=n_steps)))
    logps = np.empty(len(paths))
    for k, path in enumerate(paths):
        lp = log_init[path[0]] + log_em[0, path[0]]
        for t in range(1, n_steps):
            lp += log_trans[path[t - 1], path[t]] + log_em[t, path[t]]
        logps[k] = lp
    loglik = float(logsumexp(logps))
    weights = np.exp(logps - loglik)
    gamma = np.zeros((n_steps, n_states))
    for k, path in enumerate(paths):
        for t in range(n_steps):
            gamma[t, path[t]] += weights[k]
    map_path = paths[int(np.argmax(logps))].astype(np.intp)
    return gamma, loglik, map_path


def test_smooth_and_loglik_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n_states, n_steps = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        log_init, log_trans, log_em = _random_chain(rng, n_states, n_steps)
        model = mk.DiscreteChain(log_init, log_trans)
        result = mk.smooth(model, log_em)
        gamma, loglik, _ = _brute_force(log_init, log_trans, log_em)
        assert np.allclose(result.gamma, gamma, atol=1e-9)
        assert np.isclose(result.loglik, loglik, atol=1e-9)
        assert np.isclose(mk.loglik(model, log_em), loglik, atol=1e-9)


def test_decode_matches_brute_force_map():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n_states, n_steps = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        log_init, log_trans, log_em = _random_chain(rng, n_states, n_steps)
        model = mk.DiscreteChain(log_init, log_trans)
        path = mk.decode(model, log_em)
        _, _, map_path = _brute_force(log_init, log_trans, log_em)
        assert np.array_equal(path, map_path)


def test_one_recursion_two_semirings():
    """The thesis: the SAME belief-generic recursion gives α (sum-product) and δ (max-plus)."""
    from markovlib.belief import Categorical
    from markovlib.engines.recursion import categorical_predict, forward, viterbi

    rng = np.random.default_rng(3)
    log_init, log_trans, log_em = _random_chain(rng, 3, 5)
    model = mk.DiscreteChain(log_init, log_trans)
    factors = [mk.Categorical(row) for row in log_em]
    initial = Categorical(log_init)

    alpha = forward(initial, categorical_predict(log_trans, mk.SumProduct()), factors)
    assert np.isclose(alpha[-1].log_mass(), mk.loglik(model, log_em))

    delta = forward(initial, categorical_predict(log_trans, mk.MaxPlus()), factors)
    _, best = viterbi(log_init, log_trans, log_em)
    assert np.isclose(float(delta[-1].log_p.max()), best)


def test_categorical_belief_algebra():
    a = mk.Categorical(np.log([0.2, 0.8]))
    b = mk.Categorical(np.log([0.5, 0.5]))
    combined = a.combine(b)
    assert np.allclose(combined.log_p, np.log([0.1, 0.4]))
    assert np.isclose(combined.log_mass(), np.log(0.5))
    assert np.allclose(combined.normalized().probs(), [0.2, 0.8])
    assert combined.mode() == 1


def test_gamma_rows_are_distributions():
    rng = np.random.default_rng(2)
    log_init, log_trans, log_em = _random_chain(rng, 3, 5)
    result = mk.smooth(mk.DiscreteChain(log_init, log_trans), log_em)
    assert np.allclose(result.gamma.sum(axis=1), 1.0)
    assert np.all(result.gamma >= 0.0)


def test_resolve_engine_is_evidence_not_bool():
    model = mk.DiscreteChain(np.log([0.5, 0.5]), np.log([[0.9, 0.1], [0.1, 0.9]]))
    good = mk.resolve_engine(model, "smooth")
    assert isinstance(good, mk.Exact) and good.ok
    bad = mk.resolve_engine(object(), "smooth")
    assert isinstance(bad, mk.Intractable) and not bad.ok and "no engine" in bad.reason
    unknown = mk.resolve_engine(model, "control")
    assert isinstance(unknown, mk.Intractable)


def test_approximate_resolution_carries_its_character():
    approx = mk.Approximate(engine=object(), method="particle", error_character="O(1/sqrt(N))")
    assert approx.ok and approx.method == "particle" and "sqrt" in approx.error_character


def test_n_states_reports_state_count():
    model = mk.DiscreteChain(np.log([0.2, 0.3, 0.5]), np.log(np.full((3, 3), 1 / 3)))
    assert model.n_states == 3


def test_query_raises_on_intractable_model():
    with pytest.raises(ValueError, match="no engine"):
        mk.smooth(object(), np.zeros((2, 2)))  # type: ignore[arg-type]
