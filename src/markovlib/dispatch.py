"""``resolve_engine`` — markovlib's ``decide()``: which engine resolves a query, and how exactly.

The deliberate analog of fungeom's decidable dispatch: given a model and a query, return *evidence*
(:data:`~markovlib.resolution.EngineResolution`) naming the engine and its exactness, never a silent
best-effort. The slice has one engine, so the table is short; the shape is what matters — new engines
(segmental HSMM, Gaussian RTS, particle smoothing) register here, each declaring ``Exact`` or
``Approximate`` for the queries they resolve.
"""

from __future__ import annotations

from markovlib.engines.exact_chain import ExactChain
from markovlib.model import DiscreteChain
from markovlib.resolution import EngineResolution, Exact, Intractable

_EXACT_QUERIES = frozenset({"filter", "smooth", "decode", "loglik"})


def resolve_engine(model: object, query: str) -> EngineResolution:
    """Return evidence for which engine resolves ``query`` on ``model``, and whether exactly.

    A finite :class:`~markovlib.model.DiscreteChain` resolves the chain queries **exactly** via
    :class:`~markovlib.engines.exact_chain.ExactChain`; anything else is :class:`Intractable` (with a
    reason) until an engine claims it.
    """
    if isinstance(model, DiscreteChain) and query in _EXACT_QUERIES:
        return Exact(ExactChain())
    return Intractable(f"no engine for {type(model).__name__} / query={query!r}")
