"""SegmentalChain (right-censored EDHMM Viterbi) vs brute-force enumeration — the oracle, with censoring.

The brute force enumerates every valid labelled segmentation under the *same* EDHMM rules the engine
encodes: segments of length ``≤ cap`` scored by the dwell pmf (or the survival at ``d == cap``), adjacent
segments of *different* states pay the renormalized inter-segment switch, and a same-state continuation
is allowed *only* across a censored (``d == cap``) boundary at zero cost. The DP must match that oracle's
MAP exactly. The duration model (geometric limit, survival, normalization, mean) is checked directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import markovlib as mk
from markovlib.duration import NegBinomDuration
from markovlib.engines.segmental import inter_segment_logtrans


def _random_hsmm(rng: np.random.Generator, n_states: int, max_duration: int) -> mk.SemiMarkovChain:
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    concentration = float(rng.uniform(0.5, 3.0))  # shared across states
    durations = tuple(
        NegBinomDuration(mean=float(rng.uniform(1.5, 4.0)), concentration=concentration) for _ in range(n_states)
    )
    return mk.SemiMarkovChain(log_init, log_trans, durations, max_duration)


def _brute_force_edhmm(model: mk.SemiMarkovChain, log_em):
    switch = inter_segment_logtrans(model.log_trans)
    n_steps, n_states = log_em.shape
    cap = model.max_duration
    cum = np.vstack([np.zeros(n_states), np.cumsum(log_em, axis=0)])
    best = {"score": -np.inf, "path": np.empty(0, dtype=np.intp)}

    def dwell(state: int, d: int) -> float:
        return model.durations[state].log_survival(cap) if d == cap else model.durations[state].log_pmf(d)

    def recurse(start, prev_state, prev_censored, score, frames):
        if start == n_steps:
            if score > best["score"]:
                best["score"] = score
                best["path"] = np.array(frames, dtype=np.intp)
            return
        for d in range(1, min(cap, n_steps - start) + 1):
            seg_em = cum[start + d] - cum[start]
            for s in range(n_states):
                if prev_state is None:
                    entry = float(model.log_init[s])
                elif s == prev_state:
                    if not prev_censored:  # same-state adjacency only across a censored boundary
                        continue
                    entry = 0.0
                else:
                    entry = float(switch[prev_state, s])
                recurse(start + d, s, d == cap, score + entry + dwell(s, d) + float(seg_em[s]), [*frames, *([s] * d)])

    recurse(0, None, False, 0.0, [])
    return best["path"], best["score"]


def test_segmental_viterbi_matches_brute_force_with_censoring():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n_states = int(rng.integers(2, 4))
        n_steps = int(rng.integers(2, 7))
        max_duration = int(rng.integers(2, 4))  # small cap -> censoring is active
        model = _random_hsmm(rng, n_states, max_duration)
        log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))
        ref_path, _ = _brute_force_edhmm(model, log_em)
        assert np.array_equal(mk.decode(model, log_em), ref_path)


def test_single_state_persists_via_censoring():
    # S=1 cannot switch, but censored continuation lets the lone state span the whole record.
    durations = (NegBinomDuration(mean=2.0, concentration=1.0),)
    model = mk.SemiMarkovChain(np.array([0.0]), np.array([[0.0]]), durations, max_duration=3)
    assert np.array_equal(mk.decode(model, np.zeros((7, 1))), np.zeros(7, dtype=np.intp))


def test_duration_cap_of_one():
    # cap == 1: every segment is a length-1 censored segment (the empty-natural-branch edge).
    rng = np.random.default_rng(5)
    model = _random_hsmm(rng, 2, max_duration=1)
    log_em = np.log(rng.uniform(0.05, 1.0, size=(5, 2)))
    path = mk.decode(model, log_em)
    assert path.shape == (5,) and set(path.tolist()) <= {0, 1}


def test_negbinom_geometric_limit():
    dur = NegBinomDuration(mean=4.0, concentration=1.0)
    q = 1.0 / 4.0
    for d in range(1, 8):
        assert math.isclose(dur.log_pmf(d), math.log(q) + (d - 1) * math.log(1 - q), rel_tol=1e-9)


def test_negbinom_survival_is_the_pmf_tail():
    dur = NegBinomDuration(mean=3.0, concentration=2.0)
    assert math.isclose(dur.log_survival(1), 0.0, abs_tol=1e-12)  # P(d >= 1) = 1
    for d in (2, 4, 7):
        tail = sum(math.exp(dur.log_pmf(k)) for k in range(d, 2000))
        assert math.isclose(math.exp(dur.log_survival(d)), tail, abs_tol=1e-9)


def test_negbinom_is_a_distribution_with_the_right_mean():
    dur = NegBinomDuration(mean=3.5, concentration=2.0)
    mass = sum(math.exp(dur.log_pmf(d)) for d in range(1, 600))
    mean = sum(d * math.exp(dur.log_pmf(d)) for d in range(1, 600))
    assert math.isclose(mass, 1.0, abs_tol=1e-9)
    assert math.isclose(mean, 3.5, abs_tol=1e-6)


def test_negbinom_validates_its_parameters():
    with pytest.raises(ValueError, match="mean dwell"):
        NegBinomDuration(mean=1.0, concentration=1.0)
    with pytest.raises(ValueError, match="concentration"):
        NegBinomDuration(mean=2.0, concentration=0.0)
    with pytest.raises(ValueError, match="duration"):
        NegBinomDuration(mean=2.0, concentration=1.0).log_pmf(0)


def test_resolve_engine_hsmm_decode_exact_smooth_intractable():
    model = _random_hsmm(np.random.default_rng(1), 2, 3)
    assert model.n_states == 2
    assert isinstance(mk.resolve_engine(model, "decode"), mk.Exact)
    assert isinstance(mk.resolve_engine(model, "smooth"), mk.Intractable)


def test_decode_raises_on_unsupported_model():
    with pytest.raises(ValueError, match="no engine"):
        mk.decode(object(), np.zeros((2, 2)))  # type: ignore[arg-type]
