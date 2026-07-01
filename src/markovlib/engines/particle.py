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

from collections.abc import Callable
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

    def smooth(
        self,
        model: StateSpaceModel,
        observations: Float,
        log_transition: Callable[[Float, Float], Float],
        *,
        n_particles: int,
        seed: int,
    ) -> Float:
        """Rao-Blackwellized particle SMOOTHER — the backward-smoothing completion of :meth:`filter`.

        A forward bootstrap pass (as in :meth:`filter`, resampling every step) that additionally *records*,
        per frame, the distinct particle states the cloud occupies and their forward log-weights; then a
        backward Rao-Blackwellized FFBS over only those *visited* states — never the full state space — that
        turns the forward weights into smoothing weights. Returns the *smoothed* Rao-Blackwellized means
        ``(T, D)``: per frame the smoothing-weighted average of the visited state vectors, lower-variance
        than raw particle frequencies by the Rao-Blackwell theorem.

        A smoother needs the transition to be *evaluable* — a bootstrap filter only ever *samples* it —
        so ``log_transition(states_from, states_to)`` must return the ``(U_from, U_to)`` matrix of
        ``log p(state_to | state_from)`` between two small sets of visited states. Deterministic given ``seed``.
        """
        rng = np.random.default_rng(seed)
        observations = np.asarray(observations, dtype=np.float64)
        n_frames = observations.shape[0]
        visited: list[Float] = []  # (U_t, D) distinct states occupied at frame t
        forward_logw: list[Float] = []  # (U_t,) forward log-mass on each (the filtering dist, in log)

        particles = np.asarray(model.sample_prior(rng, n_particles), dtype=np.float64)
        for step in range(n_frames):
            if step > 0:
                particles = np.asarray(model.propagate(rng, particles), dtype=np.float64)
            emit = np.asarray(model.log_likelihood(observations[step], particles), dtype=np.float64)
            uniq, inverse = np.unique(particles, axis=0, return_inverse=True)
            inverse = inverse.ravel()
            visited.append(uniq)
            forward_logw.append(np.array([logsumexp(emit[inverse == k]) for k in range(uniq.shape[0])]))
            weights = np.exp(emit - logsumexp(emit))
            particles = particles[_systematic_resample(weights, rng)]

        # Backward Rao-Blackwellized FFBS over the visited supports only (never the full state space).
        smoothing_logw: list[Float] = [np.zeros(0) for _ in range(n_frames)]
        smoothing_logw[-1] = forward_logw[-1] - logsumexp(forward_logw[-1])
        for t in range(n_frames - 2, -1, -1):
            forward_t = forward_logw[t] - logsumexp(forward_logw[t])
            log_t = np.asarray(log_transition(visited[t], visited[t + 1]), dtype=np.float64)  # (U_t, U_{t+1})
            predictive = logsumexp(forward_t[:, None] + log_t, axis=0)  # (U_{t+1},) one-step predictive
            ratio = smoothing_logw[t + 1] - predictive
            smoothed = forward_t + logsumexp(log_t + ratio[None, :], axis=1)  # (U_t,)
            smoothing_logw[t] = smoothed - logsumexp(smoothed)

        means = np.empty((n_frames, visited[0].shape[1]), dtype=np.float64)
        for t in range(n_frames):
            weights = np.exp(smoothing_logw[t] - logsumexp(smoothing_logw[t]))
            means[t] = weights @ visited[t]
        return means
