"""The model: a homogeneous finite-state Markov chain.

The slice's one model. Per the design's *engine seam*, the observation model enters as a precomputed
``(T, S)`` log-emission matrix — the engine is state-agnostic, consuming only that matrix plus the
chain's initial/transition log-parameters. Richer models (semi-Markov durations, input-driven
transitions, continuous/Gaussian state) slot in later behind the same seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class DiscreteChain:
    """A homogeneous finite-state Markov chain: log-initial ``(S,)`` and log-transition ``(S, S)``.

    ``log_trans[i, j] = log P(state_t = j | state_{t-1} = i)``.
    """

    log_init: npt.NDArray[np.float64]
    log_trans: npt.NDArray[np.float64]

    @property
    def n_states(self) -> int:
        """The number of discrete states ``S``."""
        return int(self.log_init.shape[0])
