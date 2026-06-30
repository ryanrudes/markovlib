"""markovlib — Markov-like processes as one recursion parameterized by a belief algebra + semiring.

Public surface (vertical slice): the model (:class:`DiscreteChain`), the engine
(:class:`ExactChain`), the semirings (:class:`SumProduct` / :class:`MaxPlus`), the decidable dispatch
(:func:`resolve_engine` → :class:`Exact` / :class:`Approximate` / :class:`Intractable`), and the uniform
queries (:func:`smooth` / :func:`decode` / :func:`loglik`).
"""

from __future__ import annotations

from markovlib.belief import Belief, Categorical
from markovlib.dispatch import resolve_engine
from markovlib.duration import DurationModel, NegBinomDuration
from markovlib.engines.exact_chain import ExactChain, SmoothResult
from markovlib.engines.segmental import SegmentalChain
from markovlib.model import DiscreteChain, SemiMarkovChain
from markovlib.query import decode, loglik, smooth
from markovlib.resolution import Approximate, EngineResolution, Exact, Intractable
from markovlib.semiring import MaxPlus, Semiring, SumProduct

__all__ = [
    "Belief",
    "Categorical",
    "DiscreteChain",
    "SemiMarkovChain",
    "DurationModel",
    "NegBinomDuration",
    "ExactChain",
    "SegmentalChain",
    "SmoothResult",
    "Semiring",
    "SumProduct",
    "MaxPlus",
    "Exact",
    "Approximate",
    "Intractable",
    "EngineResolution",
    "resolve_engine",
    "smooth",
    "decode",
    "loglik",
]
