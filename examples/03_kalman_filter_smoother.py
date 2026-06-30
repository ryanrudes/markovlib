"""03 — Kalman filter + RTS smoother: the *same* forward recursion as the HMM, with a Gaussian belief.

``mk.filter`` uses past-only data; ``mk.smooth`` uses the whole sequence (the RTS backward pass).
"""

from __future__ import annotations

import numpy as np

import markovlib as mk


def main() -> None:
    lg = mk.LinearGaussian(
        transition=np.array([[1.0]]),
        process_noise=np.array([[0.01]]),
        observation=np.array([[1.0]]),
        obs_noise=np.array([[0.1]]),
        init_mean=np.array([0.0]),
        init_cov=np.array([[1.0]]),
    )
    ys = np.array([[0.1], [0.3], [0.25], [0.4], [0.5]])

    kf = mk.filter(lg, ys)
    ks = mk.smooth(lg, ys)
    print("filtered means:", kf.means.ravel().round(3).tolist())
    print("smoothed means:", ks.means.ravel().round(3).tolist())
    print("log-likelihood:", round(kf.loglik, 3))
    assert kf.means.shape == (5, 1) and ks.covariances.shape == (5, 1, 1)


if __name__ == "__main__":
    main()
