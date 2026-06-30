"""``ExactChain`` — exact inference for a finite discrete chain.

The reference engine: for a finite-state chain every query (filter / smooth / decode / loglik) is
*exact*, computed by the message passes in :mod:`markovlib.engines.recursion`. Being exact, it is also
the **reference** that future fast/approximate engines are tested against (the "generic recursion is the
oracle" discipline, validated here against brute-force path enumeration in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.engines.recursion import backward_messages, forward_messages, viterbi
from markovlib.model import DiscreteChain
from markovlib.semiring import SumProduct

Float = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SmoothResult:
    """Posterior marginals ``gamma`` ``(T, S)`` (rows sum to 1) and the data log-likelihood."""

    gamma: Float
    loglik: float


class ExactChain:
    """Exact filter / smooth / decode / loglik for a :class:`~markovlib.model.DiscreteChain`."""

    def smooth(self, model: DiscreteChain, log_emissions: Float) -> SmoothResult:
        """Forward–backward: posterior marginals + log-likelihood (sum-product)."""
        alpha = forward_messages(model.log_init, model.log_trans, log_emissions, SumProduct())
        beta = backward_messages(model.log_trans, log_emissions)
        loglik = float(logsumexp(alpha[-1]))
        unnorm = alpha + beta
        gamma = np.exp(unnorm - logsumexp(unnorm, axis=1, keepdims=True))
        return SmoothResult(gamma=gamma, loglik=loglik)

    def loglik(self, model: DiscreteChain, log_emissions: Float) -> float:
        """The data log-likelihood ``log p(observations)`` (forward pass only)."""
        alpha = forward_messages(model.log_init, model.log_trans, log_emissions, SumProduct())
        return float(logsumexp(alpha[-1]))

    def decode(self, model: DiscreteChain, log_emissions: Float) -> npt.NDArray[np.intp]:
        """The single most likely state path (Viterbi / MAP, max-plus)."""
        path, _ = viterbi(model.log_init, model.log_trans, log_emissions)
        return path
