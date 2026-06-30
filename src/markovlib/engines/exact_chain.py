"""``ExactChain`` — exact inference for a finite discrete chain.

The reference engine: for a finite-state chain every query (filter / smooth / decode / loglik) is
*exact*, computed by the belief-generic forward recursion plus the categorical second passes in
:mod:`markovlib.engines.recursion`. Being exact, it is also the **reference** future fast/approximate
engines are tested against (validated here against brute-force path enumeration in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.belief import Categorical
from markovlib.engines.recursion import backward_messages, categorical_predict, forward, viterbi
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

    def _forward(self, model: DiscreteChain, log_emissions: Float) -> list[Categorical]:
        """The sum-product forward beliefs ``α`` — the belief-generic recursion at categorical instance."""
        factors = [Categorical(row) for row in log_emissions]
        predict = categorical_predict(model.log_trans, SumProduct())
        return forward(Categorical(model.log_init), predict, factors)

    def smooth(self, model: DiscreteChain, log_emissions: Float) -> SmoothResult:
        """Forward–backward: posterior marginals + log-likelihood (sum-product)."""
        alpha = self._forward(model, log_emissions)
        beta = backward_messages(model.log_trans, log_emissions)
        loglik = alpha[-1].log_mass()
        log_alpha = np.stack([belief.log_p for belief in alpha])
        unnorm = log_alpha + beta
        gamma = np.exp(unnorm - logsumexp(unnorm, axis=1, keepdims=True))
        return SmoothResult(gamma=gamma, loglik=loglik)

    def loglik(self, model: DiscreteChain, log_emissions: Float) -> float:
        """The data log-likelihood ``log p(observations)`` (forward pass only)."""
        return self._forward(model, log_emissions)[-1].log_mass()

    def decode(self, model: DiscreteChain, log_emissions: Float) -> npt.NDArray[np.intp]:
        """The single most likely state path (Viterbi / MAP, max-plus)."""
        path, _ = viterbi(model.log_init, model.log_trans, log_emissions)
        return path
