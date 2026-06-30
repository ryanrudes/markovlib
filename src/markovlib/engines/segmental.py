"""``SegmentalChain`` — exact MAP decoding for a semi-Markov chain (explicit-duration Viterbi).

The semi-Markov analog of :class:`~markovlib.engines.exact_chain.ExactChain`'s ``decode``: a dynamic
program over *segments* rather than frames. ``δ[t, j]`` is the best score of a labelling of ``o[1..t]``
whose final segment is state ``j`` ending exactly at ``t``; each segment contributes its dwell log-pmf,
its summed emissions, and the between-state transition into it (self-transitions forbidden). Backpointers
over ``(duration, predecessor)`` reconstruct the MAP segmentation. This is the piece a plain HMM cannot
express — dwell is *modelled*, not left implicitly geometric.

Note: every segment (including the trailing one) is scored by the dwell pmf — i.e. full observation.
Right-censoring the final segment by its survival function, and the duration-aware smoother
(``hsmm_posteriors``), are deliberate next refinements.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from markovlib.model import SemiMarkovChain

Float = npt.NDArray[np.float64]
Int = npt.NDArray[np.intp]


def segmental_viterbi(
    log_init: Float, log_trans: Float, dur_logpmf: Float, log_em: Float, max_duration: int
) -> tuple[Int, float]:
    """Explicit-duration Viterbi → ``(map_path, best_logscore)``.

    ``dur_logpmf`` is ``(S, max_duration)`` with ``dur_logpmf[s, d-1] = log P(dwell_s = d)``. Raises if no
    valid segmentation exists under the cap (e.g. a single-state chain longer than ``max_duration``).
    """
    n_steps, n_states = log_em.shape
    cum = np.vstack([np.zeros((1, n_states)), np.cumsum(log_em, axis=0)])  # cum[t] = Σ_{τ≤t} log_em[τ]
    delta = np.full((n_steps + 1, n_states), -np.inf)
    bp_dur = np.zeros((n_steps + 1, n_states), dtype=np.intp)
    bp_prev = np.full((n_steps + 1, n_states), -1, dtype=np.intp)
    for t in range(1, n_steps + 1):
        for j in range(n_states):
            for d in range(1, min(max_duration, t) + 1):
                start = t - d
                segment = cum[t, j] - cum[start, j] + dur_logpmf[j, d - 1]
                if start == 0:
                    cand, prev = log_init[j] + segment, np.intp(-1)
                else:
                    scores = delta[start] + log_trans[:, j]
                    scores[j] = -np.inf  # no self-transition between adjacent segments
                    prev = np.intp(np.argmax(scores))
                    cand = scores[prev] + segment
                if cand > delta[t, j]:
                    delta[t, j] = cand
                    bp_dur[t, j] = d
                    bp_prev[t, j] = prev
    last = int(np.argmax(delta[n_steps]))
    if not np.isfinite(delta[n_steps, last]):
        raise ValueError("no valid segmentation under the duration cap")
    path = np.empty(n_steps, dtype=np.intp)
    t, j = n_steps, last
    while t > 0:
        d = int(bp_dur[t, j])
        path[t - d : t] = j
        j = int(bp_prev[t, j])
        t -= d
    return path, float(delta[n_steps, last])


class SegmentalChain:
    """Exact MAP decoding for a :class:`~markovlib.model.SemiMarkovChain`."""

    def decode(self, model: SemiMarkovChain, log_emissions: Float) -> Int:
        """The most likely state path under the explicit-duration model (segmental Viterbi)."""
        cap = model.max_duration
        dur_logpmf = np.array(
            [[model.durations[s].log_pmf(d) for d in range(1, cap + 1)] for s in range(model.n_states)]
        )
        path, _ = segmental_viterbi(model.log_init, model.log_trans, dur_logpmf, log_emissions, cap)
        return path
