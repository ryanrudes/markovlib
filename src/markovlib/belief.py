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


@dataclass(frozen=True)
class GaussianBelief:
    """A Gaussian message in information (canonical) form ``φ(x) = exp(-½ xᵀΛx + ηᵀx + g)``.

    ``eta`` is the information vector ``η``, ``precision`` is ``Λ`` (symmetric positive-definite),
    ``log_scale`` is ``g``. The product of two canonical potentials just *adds* ``(η, Λ, g)`` — the
    Gaussian analog of :class:`Categorical`'s log-add — so :meth:`combine` is elementwise addition. And
    ``log_mass`` is ``log ∫ φ``: under the filtering recursion the running message is
    ``φ_t(x_t) = p(x_t, y_{0:t})``, so ``log_mass`` is ``log p(y_{0:t})`` and the final message's
    ``log_mass`` is the data log-likelihood — exactly parallel to Categorical.
    """

    eta: Float
    precision: Float
    log_scale: float = 0.0

    def combine(self, factor: GaussianBelief, /) -> GaussianBelief:
        """``⊗`` — multiply the canonical potentials (add information vectors, precisions, and scales)."""
        return GaussianBelief(
            self.eta + factor.eta, self.precision + factor.precision, self.log_scale + factor.log_scale
        )

    def log_mass(self) -> float:
        """``log ∫ φ(x) dx = g + d/2·log 2π − ½·log|Λ| + ½·ηᵀΛ⁻¹η``."""
        dim = int(self.eta.shape[0])
        _, log_det = np.linalg.slogdet(self.precision)
        quad = float(self.eta @ np.linalg.solve(self.precision, self.eta))
        return float(self.log_scale + 0.5 * dim * np.log(2.0 * np.pi) - 0.5 * float(log_det) + 0.5 * quad)

    def normalized(self) -> GaussianBelief:
        """The belief rescaled to unit mass (a proper density: ``log_mass == 0``)."""
        return GaussianBelief(self.eta, self.precision, self.log_scale - self.log_mass())

    def mean(self) -> Float:
        """The distribution mean ``μ = Λ⁻¹η``."""
        mean: Float = np.linalg.solve(self.precision, self.eta)
        return mean

    def covariance(self) -> Float:
        """The distribution covariance ``Σ = Λ⁻¹``."""
        cov: Float = np.linalg.inv(self.precision)
        return cov
