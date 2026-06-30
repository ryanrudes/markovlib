"""The one forward recursion (belief-generic), plus the categorical second passes.

:func:`forward` is *the* load-bearing primitive: a single recursion generic over the belief type and a
``predict`` push-forward, touching the belief only through :meth:`~markovlib.belief.Belief.combine`. The
categorical chain supplies :func:`categorical_predict` (a semiring matrix-vector push): with
:class:`~markovlib.semiring.SumProduct` the messages are the forward ``α`` of forward–backward; with
:class:`~markovlib.semiring.MaxPlus` they are the Viterbi score table ``δ``. The sum-product smoother
adds :func:`backward_messages`; the max-plus decoder adds argmax backpointers + a backtrace
(:func:`viterbi`). Same recursion shape, two semirings, any belief — the unification the library is
built on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.belief import Belief, Categorical
from markovlib.semiring import Semiring

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.intp]


def forward[B: Belief](initial: B, predict: Callable[[B], B], factors: Sequence[B]) -> list[B]:
    """The forward recursion ``msg[t] = predict(msg[t-1]) ⊗ factor[t]`` (``msg[0] = initial ⊗ factor[0]``).

    Generic over the belief type ``B``: it touches the belief only through ``combine`` and the supplied
    ``predict`` push-forward, so categorical / Gaussian / particle beliefs all run this one loop.
    """
    messages = [initial.combine(factors[0])]
    for t in range(1, len(factors)):
        messages.append(predict(messages[t - 1]).combine(factors[t]))
    return messages


def categorical_predict(log_trans: Float, semiring: Semiring) -> Callable[[Categorical], Categorical]:
    """The chain push-forward for categorical beliefs: ``b'[j] = ⊕_i ( b[i] ⊗ trans[i, j] )``.

    ``⊗`` is ``+`` (log domain); ``⊕`` is ``semiring.reduce``. The closure captures the transition and
    the semiring, so :func:`forward` stays oblivious to both.
    """

    def predict(belief: Categorical) -> Categorical:
        return Categorical(semiring.reduce(belief.log_p[:, None] + log_trans, axis=0))

    return predict


def backward_messages(log_trans: Float, log_em: Float) -> Float:
    """The sum-product backward pass ``β[t, i] = logsumexp_j ( trans[i, j] + em[t+1, j] + β[t+1, j] )``."""
    n_steps, n_states = log_em.shape
    beta = np.zeros((n_steps, n_states), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        beta[t] = logsumexp(log_trans + (log_em[t + 1] + beta[t + 1])[None, :], axis=1)
    return beta


def viterbi(log_init: Float, log_trans: Float, log_em: Float) -> tuple[Int, float]:
    """Max-plus forward with argmax backpointers + backtrace → ``(map_path, best_logscore)``.

    Shares the recursion *shape* of :func:`forward` (semiring = MaxPlus) but additionally records, at
    each step, which predecessor achieved the max — the backpointers a single MAP path needs.
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
