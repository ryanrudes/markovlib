"""Rao-Blackwellized particle smoother validated against the exact forward-backward smoother.

``particle_smooth`` estimates the *same* smoothing marginals as ``ExactChain`` — the particle method
adds only Monte-Carlo error. So on a small random chain, encoded as a particle model whose states are
one-hot rows (the smoothing-weighted average of one-hot states *is* the marginal distribution), the
smoothed Rao-Blackwellized means converge to ``ExactChain.smooth``'s ``gamma``. With enough particles
the cloud visits every state each frame, so the visited-support backward pass is the exact FFBS up to
sampling error. Determinism-given-seed (the reify-randomness rule) is checked too.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

import markovlib as mk


def _random_chain(rng, n_states, n_steps):
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))  # unnormalized is fine
    return log_init, log_trans, log_em


def _chain_particle_model(log_init, log_trans):
    """A particle model that IS a discrete chain: one-hot particle states, chain transition/emission."""
    n_states = log_init.shape[0]
    eye = np.eye(n_states)
    init_p = softmax(log_init)
    trans_p = softmax(log_trans, axis=1)

    def sample_prior(rng, n):
        return eye[rng.choice(n_states, size=n, p=init_p)]

    def propagate(rng, particles):
        cur = np.argmax(particles, axis=1)
        cdf = np.cumsum(trans_p[cur], axis=1)
        nxt = (cdf > rng.random(cur.shape[0])[:, None]).argmax(axis=1)
        return eye[nxt]

    def log_likelihood(y, particles):  # y = log_em[t] (S,); one-hot pick => particles @ y
        return particles @ y

    def log_transition(states_from, states_to):  # (U_from, U_to) = log P(to | from)
        return log_trans[np.ix_(np.argmax(states_from, axis=1), np.argmax(states_to, axis=1))]

    model = mk.StateSpaceModel(sample_prior=sample_prior, propagate=propagate, log_likelihood=log_likelihood)
    return model, log_transition


def test_particle_smoother_converges_to_exact():
    rng = np.random.default_rng(0)
    n_states, n_steps = 3, 8
    log_init, log_trans, log_em = _random_chain(rng, n_states, n_steps)
    gamma = mk.smooth(mk.DiscreteChain(log_init=log_init, log_trans=log_trans), log_em).gamma
    model, log_transition = _chain_particle_model(log_init, log_trans)
    means = mk.particle_smooth(model, log_em, log_transition, n_particles=4000, seed=1)
    assert means.shape == (n_steps, n_states)
    assert np.max(np.abs(means - gamma)) < 0.05  # converges to the exact smoothing marginal
    assert np.allclose(means.sum(axis=1), 1.0)  # each smoothed row is a distribution


def test_particle_smoother_deterministic_given_seed():
    rng = np.random.default_rng(2)
    log_init, log_trans, log_em = _random_chain(rng, n_states=3, n_steps=5)
    model, log_transition = _chain_particle_model(log_init, log_trans)
    a = mk.particle_smooth(model, log_em, log_transition, n_particles=200, seed=7)
    b = mk.particle_smooth(model, log_em, log_transition, n_particles=200, seed=7)
    assert np.array_equal(a, b)


def test_particle_smoother_raises_on_unsupported_model():
    with pytest.raises(ValueError, match="no engine"):
        mk.particle_smooth(object(), np.zeros((2, 1)), lambda a, b: np.zeros((1, 1)), n_particles=10, seed=0)  # type: ignore[arg-type]
