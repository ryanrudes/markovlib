"""04 — Bootstrap particle filter for a nonlinear model; deterministic given the seed.

You supply the dynamics and observation model as three callables (all consuming a seeded RNG), so the
filter is a pure function of its ``seed`` — exactly reproducible.
"""

from __future__ import annotations

import numpy as np

import markovlib as mk


def main() -> None:
    def sample_prior(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, (n, 1))

    def propagate(rng: np.random.Generator, particles: np.ndarray) -> np.ndarray:
        moved: np.ndarray = 0.9 * particles + rng.normal(0.0, 0.3, particles.shape)
        return moved

    def log_likelihood(y: np.ndarray, particles: np.ndarray) -> np.ndarray:
        weights: np.ndarray = -0.5 * ((y - particles[:, 0]) ** 2) / 0.1**2  # N(y; x, 0.1)
        return weights

    ssm = mk.StateSpaceModel(sample_prior, propagate, log_likelihood)
    ys = np.array([[0.1], [0.3], [0.25], [0.4], [0.5]])

    first = mk.particle_filter(ssm, ys, n_particles=4000, seed=0)
    again = mk.particle_filter(ssm, ys, n_particles=4000, seed=0)
    print("filtered means :", first.means.ravel().round(3).tolist())
    print("ESS per step   :", first.ess.round(0).tolist())
    print("same seed -> identical output:", bool(np.array_equal(first.means, again.means)))
    assert np.array_equal(first.means, again.means)  # reified randomness
    assert np.all(first.ess >= 1.0) and np.all(first.ess <= 4000.0)


if __name__ == "__main__":
    main()
