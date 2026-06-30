"""GaussianChain (the Kalman filter via the shared forward recursion) vs a textbook Kalman filter.

The point: the *same* :func:`~markovlib.engines.recursion.forward` that runs the categorical HMM runs the
Kalman filter once the belief is :class:`~markovlib.belief.GaussianBelief`. The information-form message
passing here is validated against an independent moment-form Kalman filter (predict / innovation / gain),
matching its filtered means, covariances, and data log-likelihood.
"""

from __future__ import annotations

import numpy as np

import markovlib as mk


def _pd(rng: np.random.Generator, dim: int) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    return a @ a.T + dim * np.eye(dim)


def _random_model(rng: np.random.Generator, dim_state: int, dim_obs: int) -> mk.LinearGaussian:
    return mk.LinearGaussian(
        transition=rng.uniform(-0.6, 0.6, size=(dim_state, dim_state)),
        process_noise=_pd(rng, dim_state),
        observation=rng.normal(size=(dim_obs, dim_state)),
        obs_noise=_pd(rng, dim_obs),
        init_mean=rng.normal(size=dim_state),
        init_cov=_pd(rng, dim_state),
    )


def _textbook_kalman(model: mk.LinearGaussian, observations: np.ndarray):
    """The standard moment-form Kalman filter (predict → innovation → gain) — the independent oracle."""
    A, Q, H, R = model.transition, model.process_noise, model.observation, model.obs_noise
    mean, cov = model.init_mean.copy(), model.init_cov.copy()
    means, covs, loglik = [], [], 0.0
    for t, y in enumerate(observations):
        if t > 0:
            mean = A @ mean
            cov = A @ cov @ A.T + Q
        innovation = y - H @ mean
        innovation_cov = H @ cov @ H.T + R
        inv = np.linalg.inv(innovation_cov)
        _, log_det = np.linalg.slogdet(innovation_cov)
        loglik += -0.5 * (len(y) * np.log(2.0 * np.pi) + log_det + innovation @ inv @ innovation)
        gain = cov @ H.T @ inv
        mean = mean + gain @ innovation
        cov = cov - gain @ H @ cov
        means.append(mean)
        covs.append(cov)
    return np.array(means), np.array(covs), float(loglik)


def test_kalman_filter_matches_textbook():
    rng = np.random.default_rng(0)
    for _ in range(40):
        dim_state = int(rng.integers(1, 4))
        dim_obs = int(rng.integers(1, 3))
        n_steps = int(rng.integers(2, 15))
        model = _random_model(rng, dim_state, dim_obs)
        observations = rng.normal(size=(n_steps, dim_obs))

        result = mk.filter(model, observations)
        means, covs, loglik = _textbook_kalman(model, observations)

        assert np.allclose(result.means, means, atol=1e-7)
        assert np.allclose(result.covariances, covs, atol=1e-7)
        assert np.isclose(result.loglik, loglik, atol=1e-7)


def _textbook_rts(model: mk.LinearGaussian, observations: np.ndarray):
    """The standard moment-form Kalman filter + RTS backward pass — the independent smoother oracle."""
    A, Q, H, R = model.transition, model.process_noise, model.observation, model.obs_noise
    mean, cov = model.init_mean.copy(), model.init_cov.copy()
    filt_m, filt_c, pred_m, pred_c = [], [], [], []
    for t, y in enumerate(observations):
        if t > 0:
            mean = A @ mean
            cov = A @ cov @ A.T + Q
        pred_m.append(mean)
        pred_c.append(cov)
        innovation_cov = H @ cov @ H.T + R
        gain = cov @ H.T @ np.linalg.inv(innovation_cov)
        mean = mean + gain @ (y - H @ mean)
        cov = cov - gain @ H @ cov
        filt_m.append(mean)
        filt_c.append(cov)
    n = len(observations)
    sm_m, sm_c = list(filt_m), list(filt_c)
    for t in range(n - 2, -1, -1):
        smoother_gain = filt_c[t] @ A.T @ np.linalg.inv(pred_c[t + 1])
        sm_m[t] = filt_m[t] + smoother_gain @ (sm_m[t + 1] - pred_m[t + 1])
        sm_c[t] = filt_c[t] + smoother_gain @ (sm_c[t + 1] - pred_c[t + 1]) @ smoother_gain.T
    return np.array(sm_m), np.array(sm_c)


def test_kalman_smoother_matches_textbook_rts():
    rng = np.random.default_rng(2)
    for _ in range(30):
        dim_state = int(rng.integers(1, 4))
        dim_obs = int(rng.integers(1, 3))
        n_steps = int(rng.integers(2, 15))
        model = _random_model(rng, dim_state, dim_obs)
        observations = rng.normal(size=(n_steps, dim_obs))

        result = mk.smooth(model, observations)
        means, covs = _textbook_rts(model, observations)

        assert np.allclose(result.means, means, atol=1e-7)
        assert np.allclose(result.covariances, covs, atol=1e-7)
        assert np.isclose(result.loglik, mk.filter(model, observations).loglik, atol=1e-9)


def test_gaussian_belief_combine_adds_canonical_form():
    a = mk.GaussianBelief(np.array([1.0, 2.0]), np.eye(2), 0.5)
    b = mk.GaussianBelief(np.array([0.0, 1.0]), 2.0 * np.eye(2), 1.5)
    combined = a.combine(b)
    assert np.allclose(combined.eta, [1.0, 3.0])
    assert np.allclose(combined.precision, 3.0 * np.eye(2))
    assert np.isclose(combined.log_scale, 2.0)


def test_gaussian_belief_mean_covariance_and_normalization():
    rng = np.random.default_rng(1)
    precision = _pd(rng, 3)
    eta = rng.normal(size=3)
    belief = mk.GaussianBelief(eta, precision, 4.0)
    assert np.allclose(belief.mean(), np.linalg.solve(precision, eta))
    assert np.allclose(belief.covariance(), np.linalg.inv(precision))
    assert np.isclose(belief.normalized().log_mass(), 0.0, atol=1e-9)  # a proper density integrates to 1


def test_filter_raises_on_unsupported_model():
    import pytest

    with pytest.raises(ValueError, match="no engine"):
        mk.filter(object(), np.zeros((2, 1)))  # type: ignore[arg-type]
