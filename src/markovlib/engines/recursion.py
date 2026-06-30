"""The one forward recursion, plus the two second passes.

:func:`forward_messages` is *the* load-bearing primitive: a single recursion whose only free choice is
the semiring. With :class:`~markovlib.semiring.SumProduct` it is the forward pass ``α`` of
forward–backward; with :class:`~markovlib.semiring.MaxPlus` it is the Viterbi score table ``δ``. The
sum-product smoother adds a :func:`backward_messages` pass; the max-plus decoder adds argmax
backpointers and a backtrace (:func:`viterbi`). That shared shape *is* the unification the library is
built on — inference and decoding are one recursion in two semirings.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.semiring import Semiring

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.intp]


def forward_messages(log_init: Float, log_trans: Float, log_em: Float, semiring: Semiring) -> Float:
    """The forward recursion ``msg[t, j] = em[t, j] ⊗ ⊕_i ( msg[t-1, i] ⊗ trans[i, j] )``.

    ``⊗`` is ``+`` (log domain); ``⊕`` is ``semiring.reduce``. SumProduct → forward ``α``;
    MaxPlus → Viterbi ``δ`` (scores only — see :func:`viterbi` for the backpointers).
    """
    n_steps, n_states = log_em.shape
    msg = np.empty((n_steps, n_states), dtype=np.float64)
    msg[0] = log_init + log_em[0]
    for t in range(1, n_steps):
        msg[t] = log_em[t] + semiring.reduce(msg[t - 1][:, None] + log_trans, axis=0)
    return msg


def backward_messages(log_trans: Float, log_em: Float) -> Float:
    """The sum-product backward pass ``β[t, i] = logsumexp_j ( trans[i, j] + em[t+1, j] + β[t+1, j] )``."""
    n_steps, n_states = log_em.shape
    beta = np.zeros((n_steps, n_states), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        beta[t] = logsumexp(log_trans + (log_em[t + 1] + beta[t + 1])[None, :], axis=1)
    return beta


def viterbi(log_init: Float, log_trans: Float, log_em: Float) -> tuple[Int, float]:
    """Max-plus forward with argmax backpointers + backtrace → ``(map_path, best_logscore)``.

    Shares the recursion *shape* of :func:`forward_messages` (semiring = MaxPlus) but additionally
    records, at each step, which predecessor achieved the max — the backpointers a MAP path needs.
    """
    n_steps, n_states = log_em.shape
    delta = np.empty((n_steps, n_states), dtype=np.float64)
    psi = np.empty((n_steps, n_states), dtype=np.intp)
    delta[0] = log_init + log_em[0]
    for t in range(1, n_steps):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_em[t] + np.max(scores, axis=0)
    path = np.empty(n_steps, dtype=np.intp)
    path[-1] = np.intp(np.argmax(delta[-1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(delta[-1].max())
