"""``SegmentalChain`` — exact MAP decoding for a semi-Markov chain (right-censored explicit-duration Viterbi).

The semi-Markov analog of :class:`~markovlib.engines.exact_chain.ExactChain`'s ``decode``: a dynamic
program over *segments* (maximal runs of one state) rather than frames. A segment of state ``s`` over
frames ``[a, b]`` (length ``d``) scores ``Σ emit[a..b, s] + dwell_s(d) + log P(prev → s)``; the first
segment uses ``log_init[s]``. Three refinements make this the **standard right-censored EDHMM**:

* the inter-segment transition is the off-diagonal of ``log_trans`` **renormalized** to a distribution
  over the other states (self-transitions are forbidden — persistence is the dwell's job);
* ``max_duration`` is a tractability cap, not a hard ceiling: a segment that runs the full cap is
  **right-censored** (scored by ``log P(d ≥ cap)``, the survival) and may *continue in the same state*
  at zero cost, so a genuine bout longer than the cap is represented exactly rather than truncated;
* the DP therefore tracks two flavours per ``(t, s)`` — a naturally-ended segment (``d < cap``) and a
  censored one (``d == cap``) — with per-flavour segment backpointers for the backtrace.

This matches ``contact/hmm``'s `hsmm.py` decoder bit-for-bit (see ``verify_markovlib.py``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from markovlib.duration import DurationModel
from markovlib.engines.exact_chain import SmoothResult
from markovlib.model import SemiMarkovChain

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.intp]

_NEG = -1e30  # a finite stand-in for log 0 (matches the reference decoder's arithmetic exactly)


def inter_segment_logtrans(log_trans: Float) -> Float:
    """The between-segment switch: zero the diagonal and renormalize each row over the other states.

    An HSMM "transition" links two *naturally ended* segments, so it must change state; the result is a
    proper distribution over ``s' ≠ s``. A single-state row (all ``-inf``) stays so (S=1 never switches).
    """
    switch = np.array(log_trans, dtype=np.float64)
    np.fill_diagonal(switch, _NEG)
    row = logsumexp(switch, axis=1)
    safe = np.where(np.isfinite(row) & (row > _NEG / 2), row, 0.0)
    return switch - safe[:, None]


def duration_table(durations: tuple[DurationModel, ...], max_duration: int) -> Float:
    """The censored dwell table ``(S, cap)``: pmf for ``d = 1..cap-1``, survival ``P(d ≥ cap)`` at ``d = cap``."""
    table = np.empty((len(durations), max_duration), dtype=np.float64)
    for s, dur in enumerate(durations):
        for d in range(1, max_duration):
            table[s, d - 1] = dur.log_pmf(d)
        table[s, max_duration - 1] = dur.log_survival(max_duration)
    return table


def segmental_viterbi(log_init: Float, switch: Float, log_dur: Float, log_em: Float, max_duration: int) -> Int:
    """Right-censored explicit-duration Viterbi → the MAP state path ``(T,)``.

    ``switch`` is the renormalized inter-segment transition (see :func:`inter_segment_logtrans`);
    ``log_dur`` is the censored dwell table (see :func:`duration_table`).
    """
    n_steps, n_states = log_em.shape
    cap = max_duration
    prefix = np.zeros((n_steps + 1, n_states), dtype=np.float64)
    np.cumsum(log_em, axis=0, out=prefix[1:])

    # Two flavours of "a segment of state s ends at frame t-1": natural (d < cap) and censored (d == cap).
    v_end = np.full((n_steps + 1, n_states), _NEG, dtype=np.float64)
    v_cens = np.full((n_steps + 1, n_states), _NEG, dtype=np.float64)
    bd_end = np.zeros((n_steps + 1, n_states), dtype=np.intp)
    bp_end = np.full((n_steps + 1, n_states), -1, dtype=np.intp)
    bf_end = np.zeros((n_steps + 1, n_states), dtype=np.intp)
    bd_cens = np.zeros((n_steps + 1, n_states), dtype=np.intp)
    bp_cens = np.full((n_steps + 1, n_states), -1, dtype=np.intp)
    bf_cens = np.zeros((n_steps + 1, n_states), dtype=np.intp)
    states = np.arange(n_states)

    for t in range(1, n_steps + 1):
        d_max = min(t, cap)
        seg_emit = prefix[t][None, :] - prefix[t - d_max : t][::-1, :]  # (d_max, S): emission mass of each segment
        dur_term = log_dur[:, :d_max].T  # (d_max, S)

        best_prev = np.maximum(v_end, v_cens)
        tot_prev = best_prev[t - d_max : t][::-1, :]  # (d_max, S')
        cens_prev = v_cens[t - d_max : t][::-1, :]  # (d_max, S) same-state continuation

        trans_score = tot_prev[:, :, None] + switch[None, :, :]  # (d_max, S', S)
        switch_in = np.max(trans_score, axis=1)  # (d_max, S)
        switch_idx = np.argmax(trans_score, axis=1)  # (d_max, S) -> predecessor state
        cont_in = cens_prev  # (d_max, S)
        entry = np.maximum(switch_in, cont_in)

        if d_max == t:  # the first segment of the record may start here
            entry = entry.copy()
            start_vals = np.maximum(log_init, entry[t - 1])
            entry[t - 1] = start_vals

        cand = entry + seg_emit + dur_term  # (d_max, S)

        use_cont = cont_in >= switch_in
        idx = np.arange(d_max)[:, None]
        t_pred = t - (idx + 1)
        src_end = v_end[t_pred, switch_idx]
        src_cens = v_cens[t_pred, switch_idx]
        switch_pflav = (src_cens > src_end).astype(np.intp)
        pstate = np.where(use_cont, states[None, :], switch_idx)
        pflav = np.where(use_cont, 1, switch_pflav)
        if d_max == t:
            is_start = start_vals >= np.maximum(switch_in[t - 1], cont_in[t - 1])
            pstate[t - 1] = np.where(is_start, -1, pstate[t - 1])
            pflav[t - 1] = np.where(is_start, 0, pflav[t - 1])

        is_cap = d_max == cap
        natural = cand[: cap - 1] if is_cap else cand
        if natural.shape[0] > 0:
            best = np.argmax(natural, axis=0)
            v_end[t] = natural[best, states]
            bd_end[t] = best + 1
            bp_end[t] = pstate[best, states]
            bf_end[t] = pflav[best, states]
        if is_cap:
            v_cens[t] = cand[cap - 1]
            bd_cens[t] = cap
            bp_cens[t] = pstate[cap - 1]
            bf_cens[t] = pflav[cap - 1]

    state = int(np.argmax(np.maximum(v_end[n_steps], v_cens[n_steps])))
    flavour = 1 if v_cens[n_steps, state] > v_end[n_steps, state] else 0
    path = np.empty(n_steps, dtype=np.intp)
    t = n_steps
    while t > 0:
        bd, bp, bf = (bd_cens, bp_cens, bf_cens) if flavour == 1 else (bd_end, bp_end, bf_end)
        d = int(bd[t, state])
        prev_state = int(bp[t, state])
        prev_flavour = int(bf[t, state])
        path[t - d : t] = state
        t -= d
        if prev_state < 0:
            break
        state, flavour = prev_state, prev_flavour
    return path


def segmental_posteriors(
    log_init: Float, switch: Float, log_dur: Float, log_em: Float, max_duration: int
) -> tuple[Float, float]:
    """Right-censored explicit-duration forward–backward → ``(per-frame posteriors (T, S), loglik)``.

    The sum-product analog of :func:`segmental_viterbi`: the two-flavour (natural-end / censored)
    forward and backward passes, then each segment scatters its (normalized) probability uniformly
    across the frames it covers. Same censoring semantics (survival mass at ``d == cap``, same-state
    continuation across a censored boundary). ``switch`` and ``log_dur`` are as for the decoder.
    """
    n_steps, n_states = log_em.shape
    cap = max_duration
    prefix = np.zeros((n_steps + 1, n_states), dtype=np.float64)
    np.cumsum(log_em, axis=0, out=prefix[1:])

    # --- Forward: alpha_star (a segment of s begins at t), alpha_end (natural end), alpha_cens (censored end).
    alpha_star = np.full((n_steps + 1, n_states), _NEG, dtype=np.float64)
    alpha_end = np.full((n_steps + 1, n_states), _NEG, dtype=np.float64)
    alpha_cens = np.full((n_steps + 1, n_states), _NEG, dtype=np.float64)
    alpha_star[0] = log_init
    for t in range(1, n_steps + 1):
        natural = min(t, cap - 1)
        if natural >= 1:
            seg = prefix[t][None, :] - prefix[t - natural : t][::-1, :]
            alpha_end[t] = logsumexp(alpha_star[t - natural : t][::-1, :] + log_dur[:, :natural].T + seg, axis=0)
        if t >= cap:
            alpha_cens[t] = alpha_star[t - cap] + log_dur[:, cap - 1] + (prefix[t] - prefix[t - cap])
        if t < n_steps:
            tot_end = np.logaddexp(alpha_end[t], alpha_cens[t])
            switched = logsumexp(tot_end[:, None] + switch, axis=0)
            alpha_star[t] = np.logaddexp(switched, alpha_cens[t])
    loglik = float(logsumexp(np.logaddexp(alpha_end[n_steps], alpha_cens[n_steps])))

    # --- Backward: beta[t, s] = log p(o_t.. | a segment of s begins at t). cont_* are the post-end values.
    beta = np.full((n_steps + 1, n_states), _NEG, dtype=np.float64)

    def cont_nat(u: int) -> Float:  # value following a *natural* end at u (a switch to another state)
        if u == n_steps:
            return np.zeros(n_states, dtype=np.float64)
        following: Float = logsumexp(switch + beta[u][None, :], axis=1)
        return following

    def cont_cens(u: int) -> Float:  # following a *censored* end at u (switch OR same-state continue)
        if u == n_steps:
            return np.zeros(n_states, dtype=np.float64)
        following: Float = np.logaddexp(logsumexp(switch + beta[u][None, :], axis=1), beta[u])
        return following

    for t in range(n_steps - 1, -1, -1):
        terms = []
        natural = min(n_steps - t, cap - 1)
        if natural >= 1:
            seg = prefix[t + 1 : t + natural + 1, :] - prefix[t][None, :]
            cont = np.stack([cont_nat(t + d) for d in range(1, natural + 1)], axis=0)
            terms.append(logsumexp(log_dur[:, :natural].T + seg + cont, axis=0))
        if n_steps - t >= cap:
            seg = prefix[t + cap, :] - prefix[t, :]
            terms.append(log_dur[:, cap - 1] + seg + cont_cens(t + cap))
        if terms:
            beta[t] = logsumexp(np.stack(terms, axis=0), axis=0)

    # --- Per-frame occupancy: each segment scatters its (normalized) probability across its frames.
    cont_nat_tab = np.stack([cont_nat(u) for u in range(n_steps + 1)], axis=0)
    cont_cens_tab = np.stack([cont_cens(u) for u in range(n_steps + 1)], axis=0)
    gamma = np.zeros((n_steps, n_states), dtype=np.float64)
    for s in range(n_states):
        for start in range(n_steps):
            d_max = min(n_steps - start, cap)
            seg_emit = prefix[start + 1 : start + d_max + 1, s] - prefix[start, s]
            dwell = log_dur[s, :d_max].copy()
            cont = cont_nat_tab[start + 1 : start + d_max + 1, s].copy()
            if d_max == cap:
                cont[cap - 1] = cont_cens_tab[start + cap, s]
            seg_logp = alpha_star[start, s] + dwell + seg_emit + cont - loglik
            seg_p = np.exp(np.minimum(seg_logp, 0.0))
            gamma[start : start + d_max, s] += np.cumsum(seg_p[::-1])[::-1]  # suffix scatter
    row = gamma.sum(axis=1, keepdims=True)
    gamma = gamma / np.where(row > 0.0, row, 1.0)
    return gamma, loglik


class SegmentalChain:
    """Exact right-censored EDHMM decoding (MAP path) and smoothing (per-frame posteriors)."""

    def decode(self, model: SemiMarkovChain, log_emissions: Float) -> Int:
        """The most likely state path under the explicit-duration model."""
        switch = inter_segment_logtrans(model.log_trans)
        log_dur = duration_table(model.durations, model.max_duration)
        return segmental_viterbi(model.log_init, switch, log_dur, log_emissions, model.max_duration)

    def smooth(self, model: SemiMarkovChain, log_emissions: Float) -> SmoothResult:
        """Per-frame posteriors + log-likelihood under the explicit-duration model (segmental FB)."""
        switch = inter_segment_logtrans(model.log_trans)
        log_dur = duration_table(model.durations, model.max_duration)
        gamma, loglik = segmental_posteriors(model.log_init, switch, log_dur, log_emissions, model.max_duration)
        return SmoothResult(gamma=gamma, loglik=loglik)
