"""``GaussianChain`` — the Kalman filter, as the *same* forward recursion with a Gaussian belief.

This is the payoff of the belief abstraction: :func:`~markovlib.engines.recursion.forward` is reused
verbatim — only the belief type changes from :class:`~markovlib.belief.Categorical` to
:class:`~markovlib.belief.GaussianBelief`. The observation at each step enters as a fixed Gaussian
*factor* (the likelihood ``N(y_t | H x, R)`` written as a canonical potential in ``x``), ``combine`` is
the canonical-form product, and the transition push-forward is the linear-Gaussian prediction. The
running message is the unnormalized filtering density ``p(x_t, y_{0:t})``; its mean/covariance are the
Kalman filtered estimate and its ``log_mass`` accumulates the data log-likelihood — so the categorical
HMM and the Kalman filter are literally two instantiations of one recursion.

The RTS smoother (the Gaussian backward pass) and a particle filter (a sibling engine needing reified
randomness) are deliberate next steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from markovlib.belief import GaussianBelief
from markovlib.engines.recursion import forward
from markovlib.model import LinearGaussian

Float = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FilterResult:
    """Kalman filtered estimates: per-step ``means`` ``(T, D)``, ``covariances`` ``(T, D, D)``, and loglik."""

    means: Float
    covariances: Float
    loglik: float


@dataclass(frozen=True)
class GaussianSmoothResult:
    """RTS-smoothed estimates ``p(x_t | y_{0:T-1})``: per-step ``means``, ``covariances``, and the data loglik."""

    means: Float
    covariances: Float
    loglik: float


def _observation_factor(model: LinearGaussian, y: Float) -> GaussianBelief:
    """The likelihood ``N(y | H x, R)`` as a canonical potential in ``x`` (a fixed per-step factor)."""
    obs_precision = np.linalg.inv(model.obs_noise)
    transposed = model.observation.T @ obs_precision
    precision = transposed @ model.observation
    eta = transposed @ y
    dim_obs = int(y.shape[0])
    _, log_det_r = np.linalg.slogdet(model.obs_noise)
    log_scale = -0.5 * (dim_obs * np.log(2.0 * np.pi) + float(log_det_r)) - 0.5 * float(y @ obs_precision @ y)
    return GaussianBelief(eta, precision, log_scale)


def _prior_factor(model: LinearGaussian) -> GaussianBelief:
    """The initial prior ``N(m0, P0)`` as a normalized canonical potential."""
    precision = np.linalg.inv(model.init_cov)
    eta = precision @ model.init_mean
    return GaussianBelief(eta, precision, 0.0).normalized()


def gaussian_predict(transition: Float, process_noise: Float) -> Callable[[GaussianBelief, int], GaussianBelief]:
    """The linear-Gaussian push-forward ``μ ↦ Aμ``, ``Σ ↦ AΣAᵀ + Q`` (mass-preserving)."""

    def predict(belief: GaussianBelief, step: int) -> GaussianBelief:
        log_mass = belief.log_mass()  # prediction is a convolution — it preserves the total mass
        mean = transition @ belief.mean()
        cov = transition @ belief.covariance() @ transition.T + process_noise
        precision = np.linalg.inv(cov)
        eta = precision @ mean
        unscaled = GaussianBelief(eta, precision, 0.0)
        return GaussianBelief(eta, precision, log_mass - unscaled.log_mass())

    return predict


class GaussianChain:
    """Exact Kalman filtering and RTS smoothing for a :class:`~markovlib.model.LinearGaussian` model."""

    def _filtered_beliefs(self, model: LinearGaussian, observations: Float) -> list[GaussianBelief]:
        """The filtered messages — the shared forward recursion with a Gaussian belief."""
        prior = _prior_factor(model)
        factors = [_observation_factor(model, y) for y in observations]
        predict = gaussian_predict(model.transition, model.process_noise)
        return forward(prior, predict, factors)

    def filter(self, model: LinearGaussian, observations: Float) -> FilterResult:
        """The filtered ``p(x_t | y_{0:t})`` mean/covariance per step + the data log-likelihood."""
        beliefs = self._filtered_beliefs(model, observations)
        means = np.stack([belief.mean() for belief in beliefs])
        covariances = np.stack([belief.covariance() for belief in beliefs])
        return FilterResult(means=means, covariances=covariances, loglik=beliefs[-1].log_mass())

    def smooth(self, model: LinearGaussian, observations: Float) -> GaussianSmoothResult:
        """The RTS-smoothed ``p(x_t | y_{0:T-1})`` mean/covariance per step (filter forward + RTS backward)."""
        beliefs = self._filtered_beliefs(model, observations)
        filtered_means = [belief.mean() for belief in beliefs]
        filtered_covs = [belief.covariance() for belief in beliefs]
        transition, process_noise = model.transition, model.process_noise
        smoothed_means = list(filtered_means)  # the last step is already smoothed (= filtered)
        smoothed_covs = list(filtered_covs)
        for t in range(len(beliefs) - 2, -1, -1):
            predicted_mean = transition @ filtered_means[t]
            predicted_cov = transition @ filtered_covs[t] @ transition.T + process_noise
            gain = filtered_covs[t] @ transition.T @ np.linalg.inv(predicted_cov)
            smoothed_means[t] = filtered_means[t] + gain @ (smoothed_means[t + 1] - predicted_mean)
            smoothed_covs[t] = filtered_covs[t] + gain @ (smoothed_covs[t + 1] - predicted_cov) @ gain.T
        return GaussianSmoothResult(np.stack(smoothed_means), np.stack(smoothed_covs), beliefs[-1].log_mass())
