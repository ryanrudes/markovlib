"""The bootstrap particle filter: deterministic given its seed, and convergent to the exact Kalman filter.

A particle filter is approximate, so it is validated two ways: (1) **reproducibility** — same seed gives
identical output (the reified-randomness property), and (2) **correctness** — on a linear-Gaussian model
(expressed through the generic state-space callables) the filtered means and the marginal log-likelihood
converge, at large ``N``, to the *exact* Kalman filter (markovlib's own ``mk.filter``). It also resolves
as ``Approximate`` — the first engine to do so.
"""

from __future__ import annotations

import numpy as np
import pytest

import markovlib as mk


def _linear_gaussian_as_state_space(model: mk.LinearGaussian) -> mk.StateSpaceModel:
    dim = model.transition.shape[0]
    chol_init = np.linalg.cholesky(model.init_cov)
    chol_proc = np.linalg.cholesky(model.process_noise)
    obs_precision = np.linalg.inv(model.obs_noise)
    _, log_det_r = np.linalg.slogdet(model.obs_noise)
    dim_obs = model.observation.shape[0]

    def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
        return model.init_mean + rng.standard_normal((n, dim)) @ chol_init.T

    def propagate(rng: np.random.Generator, particles: np.ndarray) -> np.ndarray:
        return particles @ model.transition.T + rng.standard_normal((particles.shape[0], dim)) @ chol_proc.T

    def log_likelihood(y: np.ndarray, particles: np.ndarray) -> np.ndarray:
        innovation = y - particles @ model.observation.T  # (N, M)
        quad = np.einsum("ni,ij,nj->n", innovation, obs_precision, innovation)
        return -0.5 * (dim_obs * np.log(2.0 * np.pi) + log_det_r + quad)

    return mk.StateSpaceModel(sample_prior, propagate, log_likelihood)


_LG = mk.LinearGaussian(
    transition=np.array([[0.8]]),
    process_noise=np.array([[0.15]]),
    observation=np.array([[1.0]]),
    obs_noise=np.array([[0.25]]),
    init_mean=np.array([0.0]),
    init_cov=np.array([[1.0]]),
)


def test_particle_filter_is_deterministic_given_seed():
    ssm = _linear_gaussian_as_state_space(_LG)
    ys = np.array([[0.1], [0.5], [-0.3], [0.8], [0.2]])
    first = mk.particle_filter(ssm, ys, n_particles=2000, seed=7)
    second = mk.particle_filter(ssm, ys, n_particles=2000, seed=7)
    assert np.array_equal(first.means, second.means)
    assert first.loglik == second.loglik
    assert np.array_equal(first.ess, second.ess)


def test_particle_filter_converges_to_kalman():
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(10, 1))
    kalman = mk.filter(_LG, observations)
    pf = mk.particle_filter(_linear_gaussian_as_state_space(_LG), observations, n_particles=200_000, seed=1)
    assert np.allclose(pf.means, kalman.means, atol=0.02)  # Monte Carlo, comfortably within tolerance
    assert np.isclose(pf.loglik, kalman.loglik, atol=0.1)


def test_particle_filter_resolves_as_approximate():
    resolution = mk.resolve_engine(_linear_gaussian_as_state_space(_LG), "filter")
    assert isinstance(resolution, mk.Approximate) and resolution.ok
    assert "particle" in resolution.method
    assert "sqrt(N)" in resolution.error_character


def test_particle_filter_ess_in_range():
    ssm = _linear_gaussian_as_state_space(_LG)
    result = mk.particle_filter(ssm, np.array([[0.0], [1.5], [3.0]]), n_particles=1000, seed=3)
    assert result.ess.shape == (3,)
    assert np.all(result.ess >= 1.0) and np.all(result.ess <= 1000.0)


def test_particle_filter_raises_on_unsupported_model():
    with pytest.raises(ValueError, match="no engine"):
        mk.particle_filter(object(), np.zeros((2, 1)), n_particles=10, seed=0)  # type: ignore[arg-type]
