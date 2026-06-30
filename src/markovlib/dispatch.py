"""``resolve_engine`` — markovlib's ``decide()``: which engine resolves a query, and how exactly.

The deliberate analog of fungeom's decidable dispatch: given a model and a query, return *evidence*
(:data:`~markovlib.resolution.EngineResolution`) naming the engine and its exactness, never a silent
best-effort. A finite :class:`~markovlib.model.DiscreteChain` resolves every chain query exactly; a
:class:`~markovlib.model.SemiMarkovChain` resolves ``decode`` exactly (its smoother is not built yet, so
``smooth`` / ``loglik`` are honestly :class:`Intractable`). New engines register here, each declaring
``Exact`` or ``Approximate`` for the queries they resolve.
"""

from __future__ import annotations

from markovlib.engines.exact_chain import ExactChain
from markovlib.engines.segmental import SegmentalChain
from markovlib.model import DiscreteChain, SemiMarkovChain
from markovlib.resolution import EngineResolution, Exact, Intractable

_CHAIN_QUERIES = frozenset({"filter", "smooth", "decode", "loglik"})


def resolve_engine(model: object, query: str) -> EngineResolution:
    """Return evidence for which engine resolves ``query`` on ``model``, and whether exactly."""
    if isinstance(model, DiscreteChain) and query in _CHAIN_QUERIES:
        return Exact(ExactChain())
    if isinstance(model, SemiMarkovChain) and query == "decode":
        return Exact(SegmentalChain())
    return Intractable(f"no engine for {type(model).__name__} / query={query!r}")
