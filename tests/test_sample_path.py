"""FFBS (forward-filter backward-sample) validated against the exact posterior it samples from.

A path drawn by :func:`markovlib.sample_path` is an exact joint sample from ``p(z_{1:T} | x)``. Two
independent oracles pin that down: (1) over many draws the empirical *state marginals* converge to
ExactChain's forward-backward ``gamma`` — the marginals of the very posterior FFBS samples — and
(2) on a near-deterministic chain the single sample collapses onto the Viterbi / MAP path. The
"reify randomness" contract (deterministic given the rng) and homogeneous/time-varying transition
parity are checked too.
"""

from __future__ import annotations

import numpy as np

import markovlib as mk


def _random_chain(rng, n_states, n_steps, *, time_varying=False):
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    if time_varying:
        steps = [np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]) for _ in range(n_steps - 1)]
        log_trans = np.log(np.stack(steps))
    else:
        log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))  # unnormalized is fine
    return mk.DiscreteChain(log_init=log_init, log_trans=log_trans), log_em


def _empirical_marginals(chain, log_em, *, n_draws, seed):
    sampler = np.random.default_rng(seed)
    counts = np.zeros_like(log_em)
    frames = np.arange(log_em.shape[0])
    for _ in range(n_draws):
        counts[frames, mk.sample_path(chain, log_em, rng=sampler)] += 1.0
    return counts / n_draws


def test_empirical_marginals_match_forward_backward():
    rng = np.random.default_rng(0)
    chain, log_em = _random_chain(rng, n_states=3, n_steps=6)
    gamma = mk.smooth(chain, log_em).gamma
    empirical = _empirical_marginals(chain, log_em, n_draws=20000, seed=1)
    assert np.max(np.abs(empirical - gamma)) < 0.02


def test_time_varying_marginals_match_forward_backward():
    rng = np.random.default_rng(2)
    chain, log_em = _random_chain(rng, n_states=3, n_steps=6, time_varying=True)
    assert chain.log_trans.ndim == 3  # exercises the time-varying transition branch
    gamma = mk.smooth(chain, log_em).gamma
    empirical = _empirical_marginals(chain, log_em, n_draws=20000, seed=3)
    assert np.max(np.abs(empirical - gamma)) < 0.02


def test_peaked_posterior_collapses_to_viterbi():
    # Emissions that overwhelmingly favor one planted path: the sample must equal the MAP decode.
    rng = np.random.default_rng(4)
    n_states, n_steps = 4, 12
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    truth = rng.integers(0, n_states, size=n_steps)
    log_em = np.full((n_steps, n_states), -50.0)
    log_em[np.arange(n_steps), truth] = 0.0  # a near-delta emission on `truth`
    chain = mk.DiscreteChain(log_init=log_init, log_trans=log_trans)
    assert np.array_equal(mk.sample_path(chain, log_em, rng=7), mk.decode(chain, log_em))


def test_deterministic_given_rng():
    rng = np.random.default_rng(5)
    chain, log_em = _random_chain(rng, n_states=3, n_steps=8)
    # An int seed => a one-shot draw, reproducible.
    assert np.array_equal(mk.sample_path(chain, log_em, rng=42), mk.sample_path(chain, log_em, rng=42))
    # An explicit Generator => two threads from the same seed agree draw-for-draw.
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(5):
        assert np.array_equal(mk.sample_path(chain, log_em, rng=a), mk.sample_path(chain, log_em, rng=b))
