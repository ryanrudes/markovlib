"""EM fitting — the third unification: learning is generic via expected sufficient statistics.

The E-step is *exactly* the smoother (:meth:`~markovlib.engines.exact_chain.ExactChain.expected_stats`
→ state + pairwise marginals); the M-step re-estimates each parameter from the expected sufficient
statistics it owns. :func:`fit` runs Baum–Welch for the chain **dynamics** (initial + transition) with
the emission evidence held fixed; the observed-data log-likelihood increases monotonically every
iteration (the EM guarantee, asserted directly in the tests). Learning *emission* parameters needs an
emission model with its own expected-SS M-step — a documented next step (cf. the contact detector's
single-scalar gap-bias EM, which is exactly this pattern over one emission parameter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from markovlib.engines.exact_chain import ExactChain, ExpectedStats
from markovlib.model import DiscreteChain

Float = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FitResult:
    """The fitted model and the (non-decreasing) log-likelihood at each EM iteration."""

    model: DiscreteChain
    loglik_history: tuple[float, ...]


def _m_step(stats: ExpectedStats) -> DiscreteChain:
    """Re-estimate (initial, transition) from the expected sufficient statistics (emissions fixed).

    ``init = γ[0]``; ``A[i, j] = Σ_t ξ[t, i, j] / Σ_t γ[t, i]`` — the standard Baum–Welch updates. Both
    rows are proper distributions by construction (``Σ_j ξ[t, i, j] = γ[t, i]``). Assumes every state has
    positive total posterior (finite emissions), as is the case for any genuine likelihood.
    """
    new_init = np.log(stats.gamma[0])
    xi_sum = stats.xi.sum(axis=0)
    gamma_sum = stats.gamma[:-1].sum(axis=0)
    new_trans = np.log(xi_sum / gamma_sum[:, None])
    return DiscreteChain(log_init=new_init, log_trans=new_trans)


def fit(model: DiscreteChain, log_emissions: Float, *, max_iter: int = 50, tol: float = 1e-6) -> FitResult:
    """Baum–Welch over the chain dynamics; stops when the log-likelihood gain falls below ``tol``.

    The returned model always corresponds to the last recorded log-likelihood
    (``fit(...).loglik_history[-1]``). ``max_iter`` caps the iterations regardless of convergence.
    """
    engine = ExactChain()
    current = model
    history: list[float] = []
    while True:
        stats = engine.expected_stats(current, log_emissions)
        history.append(stats.loglik)
        converged = len(history) >= 2 and history[-1] - history[-2] < tol
        if converged or len(history) >= max_iter:
            break
        current = _m_step(stats)
    return FitResult(model=current, loglik_history=tuple(history))
