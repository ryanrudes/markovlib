"""``ParticleFilter`` — a bootstrap (sequential-importance-resampling) filter, a *sibling* engine.

Unlike the categorical/Gaussian filters, this does **not** use the shared
:func:`~markovlib.engines.recursion.forward`: the per-step evidence is a *state-dependent* reweighting
(the likelihood evaluated at the particle locations), not a fixed factor to ``combine``, and the
prediction step *samples* the transition. So it stands alongside the segmental DP as an engine that does
not fit the two-sweep mold — and it is the library's first **approximate** engine (Monte Carlo,
``O(1/√N)``), which is why :func:`~markovlib.dispatch.resolve_engine` reports it as ``Approximate``.

Randomness is **reified**: the whole filter is a deterministic function of its ``seed`` (and ``model``,
``observations``, ``n_particles``) — the same "reify the seed as an explicit input" discipline that keeps
the engine referentially transparent. Resampling (systematic, triggered when the effective sample size
falls below ``resample_threshold · N``) and the marginal-likelihood estimate are the standard SIR forms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.model import StateSpaceModel

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.intp]


@dataclass(frozen=True)
class ParticleResult:
    """Filtered (weighted particle) ``means`` ``(T, D)``, the marginal-likelihood estimate, and per-step ESS."""

    means: Float
    loglik: float
    ess: Float


def _systematic_resample(weights: Float, rng: np.random.Generator) -> Int:
    """Systematic resampling: one uniform draw fixes ``N`` evenly-spaced positions on the CDF."""
    n = weights.shape[0]
    positions = (float(rng.random()) + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point drift past the last bin
    indices: Int = np.searchsorted(cumulative, positions).astype(np.intp)
    return indices


class ParticleFilter:
    """A bootstrap particle filter for a :class:`~markovlib.model.StateSpaceModel`."""

    def filter(
        self,
        model: StateSpaceModel,
        observations: Float,
        *,
        n_particles: int,
        seed: int,
        resample_threshold: float = 0.5,
    ) -> ParticleResult:
        """Filter ``observations`` with ``n_particles`` particles; deterministic given ``seed``."""
        rng = np.random.default_rng(seed)
        particles = np.asarray(model.sample_prior(rng, n_particles), dtype=np.float64)
        log_weights = np.full(n_particles, -np.log(n_particles))
        means: list[Float] = []
        ess_per_step: list[float] = []
        loglik = 0.0

        for step, y in enumerate(observations):
            if step > 0:
                particles = np.asarray(model.propagate(rng, particles), dtype=np.float64)
            log_weights = log_weights + np.asarray(model.log_likelihood(y, particles), dtype=np.float64)
            increment = float(logsumexp(log_weights))  # log Σ Wᵢ·p(yₜ|xᵢ) — the marginal-likelihood step
            loglik += increment
            log_weights = log_weights - increment  # renormalize (Σ exp = 1)
            weights = np.exp(log_weights)
            ess = float(1.0 / np.sum(weights**2))
            means.append(weights @ particles)
            ess_per_step.append(ess)
            if ess < resample_threshold * n_particles:
                particles = particles[_systematic_resample(weights, rng)]
                log_weights = np.full(n_particles, -np.log(n_particles))

        return ParticleResult(means=np.array(means), loglik=loglik, ess=np.array(ess_per_step))
