"""The belief algebra — the message representation the one recursion is generic over.

A :class:`Belief` is a distribution-shaped message. The forward recursion asks only two things of it:
``combine`` (``⊗`` — fold in a local factor / emission) and ``log_mass`` (for the likelihood and for
normalization). The *push-forward* through the dynamics is supplied to the recursion separately as a
``predict`` callable, so the belief type stays ignorant of the transition. Swapping the belief —
:class:`Categorical` (log-vector) now; Gaussian (moment / information form) and particle (samples +
weights) later — turns the *same* recursion into forward–backward, Kalman/RTS, or a particle filter,
exactly as swapping the semiring turns posteriors into MAP. Belief × semiring are the two orthogonal
axes the whole library rides on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

Float = npt.NDArray[np.float64]


class Belief(Protocol):
    """A distribution-shaped message: fold in a like factor, normalize, and report its log-mass."""

    def combine(self, factor: Self, /) -> Self:
        """``⊗`` — fold in another belief / local factor (log-domain for :class:`Categorical`)."""
        ...

    def log_mass(self) -> float:
        """The log of the total (unnormalized) mass — this step's contribution to the likelihood."""
        ...

    def normalized(self) -> Self:
        """This belief rescaled to unit mass."""
        ...


@dataclass(frozen=True)
class Categorical:
    """A categorical belief: an (unnormalized) log-probability vector over ``S`` states."""

    log_p: Float

    def combine(self, factor: Categorical, /) -> Categorical:
        """Pointwise ``⊗`` — add the log-vectors (multiply the unnormalized densities)."""
        return Categorical(self.log_p + factor.log_p)

    def log_mass(self) -> float:
        """``logsumexp`` of the log-vector — the total unnormalized mass."""
        return float(logsumexp(self.log_p))

    def normalized(self) -> Categorical:
        """The belief rescaled to a proper distribution (subtract the log-mass)."""
        return Categorical(self.log_p - self.log_mass())

    def probs(self) -> Float:
        """The normalized probability vector."""
        return np.exp(self.normalized().log_p)

    def mode(self) -> int:
        """The most probable state index (MAP marginal)."""
        return int(np.argmax(self.log_p))
