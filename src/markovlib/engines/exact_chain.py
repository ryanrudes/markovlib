"""``ExactChain`` — exact inference for a finite discrete chain.

The reference engine: for a finite-state chain every query (filter / smooth / decode / loglik) and the
EM E-step (:meth:`ExactChain.expected_stats`) are *exact*, computed by the belief-generic forward
recursion plus the categorical second passes in :mod:`markovlib.engines.recursion`. Being exact, it is
also the **reference** future fast/approximate engines are tested against (validated against brute-force
path enumeration in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.belief import Categorical
from markovlib.engines.recursion import (
    backward_messages,
    categorical_predict,
    forward,
    forward_filter_backward_sample,
    viterbi,
)
from markovlib.model import DiscreteChain
from markovlib.semiring import SumProduct

Float = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SmoothResult:
    """Posterior marginals ``gamma`` ``(T, S)`` (rows sum to 1) and the data log-likelihood."""

    gamma: Float
    loglik: float


@dataclass(frozen=True)
class ExpectedStats:
    """The EM E-step's expected sufficient statistics.

    ``gamma`` ``(T, S)`` are the state marginals ``P(state_t = s | obs)``; ``xi`` ``(T-1, S, S)`` are the
    pairwise marginals ``P(state_t = i, state_{t+1} = j | obs)`` — together they are everything the
    initial/transition M-step needs. ``loglik`` is the observed-data log-likelihood at these parameters.
    """

    gamma: Float
    xi: Float
    loglik: float


class ExactChain:
    """Exact filter / smooth / decode / loglik + EM E-step for a :class:`~markovlib.model.DiscreteChain`."""

    def _forward(self, model: DiscreteChain, log_emissions: Float) -> list[Categorical]:
        """The sum-product forward beliefs ``α`` — the belief-generic recursion at categorical instance."""
        factors = [Categorical(row) for row in log_emissions]
        predict = categorical_predict(model.log_trans, SumProduct())
        return forward(Categorical(model.log_init), predict, factors)

    def _alpha_beta(self, model: DiscreteChain, log_emissions: Float) -> tuple[Float, Float, float]:
        """The log forward ``α``, the log backward ``β``, and the data log-likelihood."""
        alpha = self._forward(model, log_emissions)
        log_alpha = np.stack([belief.log_p for belief in alpha])
        beta = backward_messages(model.log_trans, log_emissions)
        loglik = float(logsumexp(log_alpha[-1]))
        return log_alpha, beta, loglik

    @staticmethod
    def _gamma(log_alpha: Float, beta: Float) -> Float:
        """Posterior state marginals ``γ[t] = normalize(α[t] + β[t])``."""
        unnorm = log_alpha + beta
        gamma: Float = np.exp(unnorm - logsumexp(unnorm, axis=1, keepdims=True))
        return gamma

    def smooth(self, model: DiscreteChain, log_emissions: Float) -> SmoothResult:
        """Forward–backward: posterior marginals + log-likelihood (sum-product)."""
        log_alpha, beta, loglik = self._alpha_beta(model, log_emissions)
        return SmoothResult(gamma=self._gamma(log_alpha, beta), loglik=loglik)

    def loglik(self, model: DiscreteChain, log_emissions: Float) -> float:
        """The data log-likelihood ``log p(observations)`` (forward pass only)."""
        return self._forward(model, log_emissions)[-1].log_mass()

    def expected_stats(self, model: DiscreteChain, log_emissions: Float) -> ExpectedStats:
        """The EM E-step: state marginals ``γ``, pairwise marginals ``ξ``, and the log-likelihood.

        ``ξ[t, i, j] = normalize( α[t, i] + log A[i, j] + log b_{t+1}(j) + β[t+1, j] )`` (normalized by the
        log-likelihood). It marginalizes to ``γ`` (``Σ_j ξ[t, i, j] = γ[t, i]``) and sums to 1 per ``t``.
        """
        log_alpha, beta, loglik = self._alpha_beta(model, log_emissions)
        gamma = self._gamma(log_alpha, beta)
        transition = model.log_trans[None, :, :] if model.log_trans.ndim == 2 else model.log_trans
        log_xi = log_alpha[:-1, :, None] + transition + (log_emissions[1:] + beta[1:])[:, None, :] - loglik
        return ExpectedStats(gamma=gamma, xi=np.exp(log_xi), loglik=loglik)

    def decode(self, model: DiscreteChain, log_emissions: Float) -> npt.NDArray[np.intp]:
        """The single most likely state path (Viterbi / MAP, max-plus)."""
        path, _ = viterbi(model.log_init, model.log_trans, log_emissions)
        return path

    def sample_path(self, model: DiscreteChain, log_emissions: Float, rng: np.random.Generator) -> npt.NDArray[np.intp]:
        """A posterior state-path *sample* — the stochastic sibling of :meth:`decode` (forward-filter backward-sample)."""
        return forward_filter_backward_sample(model.log_init, model.log_trans, log_emissions, rng)
