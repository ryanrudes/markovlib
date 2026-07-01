"""The one forward recursion (belief-generic), plus the categorical second passes.

:func:`forward` is *the* load-bearing primitive: a single recursion generic over the belief type and a
per-step ``predict`` push-forward, touching the belief only through :meth:`~markovlib.belief.Belief.combine`.
The categorical chain supplies :func:`categorical_predict` (a semiring matrix-vector push): with
:class:`~markovlib.semiring.SumProduct` the messages are the forward ``α`` of forward–backward; with
:class:`~markovlib.semiring.MaxPlus` they are the Viterbi score table ``δ``. The sum-product smoother adds
:func:`backward_messages`; the max-plus decoder adds argmax backpointers + a backtrace (:func:`viterbi`).

Transitions may be **homogeneous** (a single ``(S, S)`` ``log_trans``) or **time-varying** (a
``(T-1, S, S)`` tensor, where ``log_trans[k]`` is the step-``k`` → step-``k+1`` transition). Every pass
dispatches on ``log_trans.ndim``, so the gated/inhomogeneous case (a contact-style per-frame transition)
runs the same code as the plain chain.
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


def forward[B: Belief](initial: B, predict: Callable[[B, int], B], factors: Sequence[B]) -> list[B]:
    """The forward recursion ``msg[t] = predict(msg[t-1], t) ⊗ factor[t]`` (``msg[0] = initial ⊗ factor[0]``).

    Generic over the belief type ``B``: it touches the belief only through ``combine`` and the supplied
    ``predict`` push-forward (given the previous belief and the target step ``t``), so categorical /
    Gaussian / particle beliefs — homogeneous or time-varying — all run this one loop.
    """
    messages = [initial.combine(factors[0])]
    for t in range(1, len(factors)):
        messages.append(predict(messages[t - 1], t).combine(factors[t]))
    return messages


def categorical_predict(log_trans: Float, semiring: Semiring) -> Callable[[Categorical, int], Categorical]:
    """The chain push-forward for categorical beliefs: ``b'[j] = ⊕_i ( b[i] ⊗ trans[i, j] )``.

    ``⊗`` is ``+`` (log domain); ``⊕`` is ``semiring.reduce``. ``log_trans`` is ``(S, S)`` (homogeneous)
    or ``(T-1, S, S)`` (time-varying); entering step ``t`` uses the step-``(t-1)`` → step-``t`` transition.
    """
    homogeneous = log_trans.ndim == 2

    def predict(belief: Categorical, step: int) -> Categorical:
        transition = log_trans if homogeneous else log_trans[step - 1]
        return Categorical(semiring.reduce(belief.log_p[:, None] + transition, axis=0))

    return predict


def backward_messages(log_trans: Float, log_em: Float) -> Float:
    """The sum-product backward pass ``β[t, i] = logsumexp_j ( trans_t[i, j] + em[t+1, j] + β[t+1, j] )``.

    ``trans_t`` is the step-``t`` → step-``t+1`` transition (``log_trans`` itself if homogeneous, else
    ``log_trans[t]``).
    """
    n_steps, n_states = log_em.shape
    homogeneous = log_trans.ndim == 2
    beta = np.zeros((n_steps, n_states), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        transition = log_trans if homogeneous else log_trans[t]
        beta[t] = logsumexp(transition + (log_em[t + 1] + beta[t + 1])[None, :], axis=1)
    return beta


def viterbi(log_init: Float, log_trans: Float, log_em: Float) -> tuple[Int, float]:
    """Max-plus forward with argmax backpointers + backtrace → ``(map_path, best_logscore)``.

    Shares the recursion *shape* of :func:`forward` (semiring = MaxPlus) but additionally records, at each
    step, which predecessor achieved the max. Supports homogeneous and time-varying ``log_trans``.
    """
    n_steps, n_states = log_em.shape
    homogeneous = log_trans.ndim == 2
    delta = np.empty((n_steps, n_states), dtype=np.float64)
    psi = np.empty((n_steps, n_states), dtype=np.intp)
    delta[0] = log_init + log_em[0]
    for t in range(1, n_steps):
        transition = log_trans if homogeneous else log_trans[t - 1]
        scores = delta[t - 1][:, None] + transition
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_em[t] + np.max(scores, axis=0)
    path = np.empty(n_steps, dtype=np.intp)
    path[-1] = np.intp(np.argmax(delta[-1]))
    for t in range(n_steps - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(delta[-1].max())


def forward_filter_backward_sample(log_init: Float, log_trans: Float, log_em: Float, rng: np.random.Generator) -> Int:
    """Draw a state path ``z_{1:T} ~ p(z | observations)`` by forward-filter, backward-sample (FFBS).

    The **stochastic** sibling of :func:`viterbi`: the same forward pass — here the sum-product filter
    ``α[t] = em[t] + logsumexp_i(α[t-1, i] + trans_{t-1}[i, :])`` — then a backward pass that *samples*
    each state from ``p(z_t | z_{t+1}, x_{1:t}) ∝ α[t] · trans_t[:, z_{t+1}]`` rather than taking the
    argmax. The drawn path is an exact *joint* sample from the posterior over state sequences (not
    independent per-frame draws). Randomness is reified as the explicit ``rng`` — thread one
    ``Generator`` to make a *sequence* of draws (e.g. the label step of a blocked-Gibbs sweep)
    reproducible. Supports homogeneous ``(S, S)`` and time-varying ``(T-1, S, S)`` ``log_trans``.
    """
    n_steps, n_states = log_em.shape
    homogeneous = log_trans.ndim == 2
    log_alpha = np.empty((n_steps, n_states), dtype=np.float64)
    log_alpha[0] = log_init + log_em[0]
    for t in range(1, n_steps):
        transition = log_trans if homogeneous else log_trans[t - 1]
        log_alpha[t] = log_em[t] + logsumexp(log_alpha[t - 1][:, None] + transition, axis=0)
    path = np.empty(n_steps, dtype=np.intp)
    path[-1] = _sample_categorical(log_alpha[-1], rng)
    for t in range(n_steps - 2, -1, -1):
        transition = log_trans if homogeneous else log_trans[t]
        path[t] = _sample_categorical(log_alpha[t] + transition[:, path[t + 1]], rng)
    return path


def _sample_categorical(log_weights: Float, rng: np.random.Generator) -> np.intp:
    """Sample one category ``∝ exp(log_weights)`` via the Gumbel-max trick (log-space, no normalization)."""
    gumbel = rng.gumbel(size=log_weights.shape)
    return np.intp(np.argmax(log_weights + gumbel))
