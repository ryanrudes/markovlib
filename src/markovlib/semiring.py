"""Semirings: the knob that turns one recursion into many algorithms.

A message-passing recursion over a chain needs only two operations: ``⊗`` to combine evidence along a
path, and ``⊕`` to reduce over the alternative predecessors. In the log domain ``⊗`` is ``+`` (shared by
every semiring here), so a semiring is fixed by its ``⊕`` = :meth:`Semiring.reduce`:

* :class:`SumProduct` — ``⊕`` is ``logsumexp`` (marginalization) ⇒ forward–backward, posteriors, likelihood.
* :class:`MaxPlus` — ``⊕`` is ``max`` (optimization) ⇒ Viterbi / MAP.

Swapping the semiring is the whole difference between "what is the posterior?" and "what is the single
best path?" — same recursion (see :func:`markovlib.engines.recursion.forward_messages`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp


@runtime_checkable
class Semiring(Protocol):
    """A log-domain semiring: ``⊗`` is ``+``; ``⊕`` is :meth:`reduce`."""

    @property
    def name(self) -> str:
        """A short label, e.g. ``"sum-product"`` (read-only — frozen implementations satisfy it)."""
        ...

    def reduce(self, a: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
        """The ``⊕`` reduction over ``axis``."""
        ...


@dataclass(frozen=True)
class SumProduct:
    """``⊕`` = ``logsumexp`` — marginalization. Yields forward–backward / posteriors / likelihood."""

    name: str = "sum-product"

    def reduce(self, a: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
        return np.asarray(logsumexp(a, axis=axis), dtype=np.float64)


@dataclass(frozen=True)
class MaxPlus:
    """``⊕`` = ``max`` — optimization. Yields Viterbi / MAP."""

    name: str = "max-plus"

    def reduce(self, a: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
        return np.max(a, axis=axis)
