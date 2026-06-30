"""The models: a homogeneous Markov chain, and its semi-Markov (explicit-duration) sibling.

Per the design's *engine seam*, the observation model always enters as a precomputed ``(T, S)``
log-emission matrix — the engines are state-agnostic, consuming only that matrix plus the chain's
log-parameters. :class:`DiscreteChain` leaves dwell implicit (geometric, via the self-transition);
:class:`SemiMarkovChain` makes dwell explicit (a per-state :class:`~markovlib.duration.DurationModel`)
and forbids self-transitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from markovlib.duration import DurationModel

Float = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DiscreteChain:
    """A finite-state Markov chain: log-initial ``(S,)`` and a log-transition.

    ``log_trans`` is either **homogeneous** — ``(S, S)`` with ``log_trans[i, j] = log P(state_{t+1} = j |
    state_t = i)`` — or **time-varying** — ``(T-1, S, S)`` with ``log_trans[t]`` the step-``t`` →
    step-``t+1`` transition (e.g. a contact-style gated/input-driven tensor). Every engine dispatches on
    ``log_trans.ndim``, so both run the same code.
    """

    log_init: Float
    log_trans: Float

    @property
    def n_states(self) -> int:
        """The number of discrete states ``S``."""
        return int(self.log_init.shape[0])


@dataclass(frozen=True)
class SemiMarkovChain:
    """A finite semi-Markov chain: explicit dwell + between-state transitions.

    ``log_init`` ``(S,)`` and ``log_trans`` ``(S, S)`` are the segment-entry log-parameters — the diagonal
    of ``log_trans`` is **unused**, since self-transitions are forbidden (two adjacent segments of the same
    state are one segment). Each state has a :class:`~markovlib.duration.DurationModel`; ``max_duration``
    is the right-censoring cap — the longest single segment the decoder considers.
    """

    log_init: Float
    log_trans: Float
    durations: tuple[DurationModel, ...]
    max_duration: int

    @property
    def n_states(self) -> int:
        """The number of discrete states ``S``."""
        return int(self.log_init.shape[0])


@dataclass(frozen=True)
class LinearGaussian:
    """A linear-Gaussian state-space model — the continuous-state sibling of :class:`DiscreteChain`.

    ``x_{t+1} = A x_t + N(0, Q)``; ``y_t = H x_t + N(0, R)``; ``x_0 ~ N(m0, P0)``. Shapes: ``transition``
    ``A`` and ``process_noise`` ``Q`` are ``(D, D)``; ``observation`` ``H`` is ``(M, D)``; ``obs_noise``
    ``R`` is ``(M, M)``; ``init_mean`` ``m0`` is ``(D,)``; ``init_cov`` ``P0`` is ``(D, D)``. Decoded by
    the Kalman filter, which is the *same* forward recursion as the HMM with a Gaussian belief.
    """

    transition: Float
    process_noise: Float
    observation: Float
    obs_noise: Float
    init_mean: Float
    init_cov: Float
