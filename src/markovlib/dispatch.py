"""``resolve_engine`` — markovlib's ``decide()``: which engine resolves a query, and how exactly.

The deliberate analog of fungeom's decidable dispatch: given a model and a query, return *evidence*
(:data:`~markovlib.resolution.EngineResolution`) naming the engine and its exactness, never a silent
best-effort. A :class:`~markovlib.model.DiscreteChain` resolves ``smooth`` / ``decode`` / ``loglik``
exactly; a :class:`~markovlib.model.SemiMarkovChain` resolves ``decode`` exactly; a
:class:`~markovlib.model.LinearGaussian` resolves ``filter`` exactly (the Kalman filter). New engines
register here, each declaring ``Exact`` or ``Approximate`` for the queries they resolve.
"""

from __future__ import annotations

from markovlib.engines.exact_chain import ExactChain
from markovlib.engines.gaussian import GaussianChain
from markovlib.engines.particle import ParticleFilter
from markovlib.engines.segmental import SegmentalChain
from markovlib.model import DiscreteChain, LinearGaussian, SemiMarkovChain, StateSpaceModel
from markovlib.resolution import Approximate, EngineResolution, Exact, Intractable

_CHAIN_QUERIES = frozenset({"smooth", "decode", "loglik"})


def resolve_engine(model: object, query: str) -> EngineResolution:
    """Return evidence for which engine resolves ``query`` on ``model``, and whether exactly."""
    if isinstance(model, DiscreteChain) and query in _CHAIN_QUERIES:
        return Exact(ExactChain())
    if isinstance(model, SemiMarkovChain) and query == "decode":
        return Exact(SegmentalChain())
    if isinstance(model, LinearGaussian) and query == "filter":
        return Exact(GaussianChain())
    if isinstance(model, StateSpaceModel) and query == "filter":
        return Approximate(ParticleFilter(), "bootstrap particle filter", "O(1/sqrt(N)) Monte Carlo")
    return Intractable(f"no engine for {type(model).__name__} / query={query!r}")
