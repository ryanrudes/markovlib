"""The uniform query surface: ``smooth`` / ``decode`` / ``loglik``.

Each is the same two steps — :func:`~markovlib.dispatch.resolve_engine` to decide *which* engine
resolves the query, then run it — so the caller's interface is independent of the resolved engine. An
:class:`~markovlib.resolution.Intractable` resolution is raised, never silently swallowed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from markovlib.dispatch import resolve_engine
from markovlib.engines.exact_chain import ExactChain, SmoothResult
from markovlib.model import DiscreteChain
from markovlib.resolution import Intractable

Float = npt.NDArray[np.float64]


def _engine(model: DiscreteChain, query: str) -> ExactChain:
    resolution = resolve_engine(model, query)
    if isinstance(resolution, Intractable):
        raise ValueError(resolution.reason)
    engine = resolution.engine
    assert isinstance(engine, ExactChain)  # the slice's only engine
    return engine


def smooth(model: DiscreteChain, log_emissions: Float) -> SmoothResult:
    """Posterior marginals + log-likelihood (forward–backward)."""
    return _engine(model, "smooth").smooth(model, log_emissions)


def loglik(model: DiscreteChain, log_emissions: Float) -> float:
    """The data log-likelihood ``log p(observations)``."""
    return _engine(model, "loglik").loglik(model, log_emissions)


def decode(model: DiscreteChain, log_emissions: Float) -> npt.NDArray[np.intp]:
    """The single most likely state path (Viterbi / MAP)."""
    return _engine(model, "decode").decode(model, log_emissions)
