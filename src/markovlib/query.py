"""The uniform query surface: ``smooth`` / ``decode`` / ``loglik``.

Each is the same two steps — :func:`~markovlib.dispatch.resolve_engine` to decide *which* engine
resolves the query, then run it — so the caller's interface is independent of the resolved engine. An
:class:`~markovlib.resolution.Intractable` resolution is raised, never silently swallowed. ``decode``
spans both chain kinds (HMM, HSMM); ``smooth`` spans HMM / HSMM / linear-Gaussian (RTS); ``loglik`` is
the plain :class:`~markovlib.model.DiscreteChain` today.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

from markovlib.dispatch import resolve_engine
from markovlib.engines.exact_chain import ExactChain, SmoothResult
from markovlib.engines.gaussian import FilterResult, GaussianChain, GaussianSmoothResult
from markovlib.engines.particle import ParticleFilter, ParticleResult
from markovlib.engines.segmental import SegmentalChain
from markovlib.model import DiscreteChain, LinearGaussian, SemiMarkovChain, StateSpaceModel
from markovlib.resolution import Intractable

Float = npt.NDArray[np.float64]


def _exact_chain(model: DiscreteChain, query: str) -> ExactChain:
    resolution = resolve_engine(model, query)
    if isinstance(resolution, Intractable):
        raise ValueError(resolution.reason)
    engine = resolution.engine
    assert isinstance(engine, ExactChain)
    return engine


@overload
def smooth(model: DiscreteChain, data: Float) -> SmoothResult: ...
@overload
def smooth(model: SemiMarkovChain, data: Float) -> SmoothResult: ...
@overload
def smooth(model: LinearGaussian, data: Float) -> GaussianSmoothResult: ...
def smooth(model: DiscreteChain | SemiMarkovChain | LinearGaussian, data: Float) -> SmoothResult | GaussianSmoothResult:
    """Posterior smoothing — forward–backward marginals (HMM / HSMM) or RTS estimates (linear-Gaussian)."""
    resolution = resolve_engine(model, "smooth")
    if isinstance(resolution, Intractable):
        raise ValueError(resolution.reason)
    engine = resolution.engine
    if isinstance(engine, ExactChain):
        assert isinstance(model, DiscreteChain)
        return engine.smooth(model, data)
    if isinstance(engine, SegmentalChain):
        assert isinstance(model, SemiMarkovChain)
        return engine.smooth(model, data)
    assert isinstance(engine, GaussianChain) and isinstance(model, LinearGaussian)
    return engine.smooth(model, data)


def loglik(model: DiscreteChain, log_emissions: Float) -> float:
    """The data log-likelihood ``log p(observations)``."""
    return _exact_chain(model, "loglik").loglik(model, log_emissions)


def decode(model: DiscreteChain | SemiMarkovChain, log_emissions: Float) -> npt.NDArray[np.intp]:
    """The single most likely state path (Viterbi / MAP) — for an HMM *or* an HSMM model."""
    resolution = resolve_engine(model, "decode")
    if isinstance(resolution, Intractable):
        raise ValueError(resolution.reason)
    engine = resolution.engine
    if isinstance(engine, ExactChain):
        assert isinstance(model, DiscreteChain)
        return engine.decode(model, log_emissions)
    assert isinstance(engine, SegmentalChain) and isinstance(model, SemiMarkovChain)
    return engine.decode(model, log_emissions)


def filter(model: LinearGaussian, observations: Float) -> FilterResult:
    """Kalman filtering for a linear-Gaussian model — the forward recursion with a Gaussian belief."""
    resolution = resolve_engine(model, "filter")
    if isinstance(resolution, Intractable):
        raise ValueError(resolution.reason)
    engine = resolution.engine
    assert isinstance(engine, GaussianChain)
    return engine.filter(model, observations)


def particle_filter(
    model: StateSpaceModel,
    observations: Float,
    *,
    n_particles: int,
    seed: int,
    resample_threshold: float = 0.5,
) -> ParticleResult:
    """Bootstrap particle filtering for a general state-space model (an *approximate* engine)."""
    resolution = resolve_engine(model, "filter")
    if isinstance(resolution, Intractable):
        raise ValueError(resolution.reason)
    engine = resolution.engine
    assert isinstance(engine, ParticleFilter)
    return engine.filter(model, observations, n_particles=n_particles, seed=seed, resample_threshold=resample_threshold)
