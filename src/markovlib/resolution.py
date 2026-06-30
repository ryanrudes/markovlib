"""Engine-dispatch resolution: which engine answers a query, and how exactly.

This is markovlib's *own* three-valued decision type — the deliberate analog of fungeom's
``decide()`` (evidence, not a bool; a reason on failure; never silent), but **graded**, because
"which engine, and is it exact or approximate?" is a different judgement than "resolvable or not."
fungeom's lattice is strictly binary (``Resolvable | Unresolvable``); engine dispatch needs a third
arm, so markovlib carries it here rather than routing through fungeom.

* :class:`Exact` — an engine resolves the query with no approximation.
* :class:`Approximate` — an engine resolves it, carrying *how* (method) and the *error character*.
* :class:`Intractable` — no engine resolves it, carrying the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Exact:
    """Evidence that ``engine`` resolves the query exactly."""

    engine: object
    ok: ClassVar[bool] = True


@dataclass(frozen=True)
class Approximate:
    """Evidence that ``engine`` resolves the query approximately, with stated character."""

    engine: object
    method: str
    error_character: str
    ok: ClassVar[bool] = True


@dataclass(frozen=True)
class Intractable:
    """Evidence that no engine resolves the query, and why."""

    reason: str
    ok: ClassVar[bool] = False


type EngineResolution = Exact | Approximate | Intractable
"""The result of :func:`markovlib.dispatch.resolve_engine`."""
