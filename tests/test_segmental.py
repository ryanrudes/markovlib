"""SegmentalChain (explicit-duration Viterbi) vs brute-force segmentation enumeration — the oracle again.

For a tiny chain we enumerate *every* valid labelled segmentation (segments ≤ the duration cap, adjacent
segments of distinct states), score each by definition, and take the best. The DP must match that oracle
exactly. Plus: the duration model's geometric limit, normalization, and mean are checked directly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import markovlib as mk
from markovlib.duration import NegBinomDuration


def _random_hsmm(rng: np.random.Generator, n_states: int, max_duration: int) -> mk.SemiMarkovChain:
    log_init = np.log(rng.dirichlet(np.ones(n_states)))
    log_trans = np.log(np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)]))
    durations = tuple(
        NegBinomDuration(mean=float(rng.uniform(1.5, 3.5)), concentration=float(rng.uniform(0.5, 3.0)))
        for _ in range(n_states)
    )
    return mk.SemiMarkovChain(log_init, log_trans, durations, max_duration)


def _brute_force_hsmm_map(model, log_em):
    n_steps, n_states = log_em.shape
    cum = np.vstack([np.zeros(n_states), np.cumsum(log_em, axis=0)])
    best = {"score": -np.inf, "path": np.empty(0, dtype=np.intp)}

    def recurse(start, prev, score, segments):
        if start == n_steps:
            if score > best["score"]:
                path: list[int] = []
                for state, length in segments:
                    path.extend([state] * length)
                best["score"] = score
                best["path"] = np.array(path, dtype=np.intp)
            return
        for d in range(1, min(model.max_duration, n_steps - start) + 1):
            for s in range(n_states):
                if prev is not None and s == prev:
                    continue
                entry = model.log_init[s] if prev is None else model.log_trans[prev, s]
                seg_em = cum[start + d, s] - cum[start, s]
                recurse(start + d, s, score + entry + model.durations[s].log_pmf(d) + seg_em, [*segments, (s, d)])

    recurse(0, None, 0.0, [])
    return best["path"], best["score"]


def test_segmental_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n_states = int(rng.integers(2, 4))
        n_steps = int(rng.integers(2, 7))
        max_duration = int(rng.integers(2, 4))
        model = _random_hsmm(rng, n_states, max_duration)
        log_em = np.log(rng.uniform(0.05, 1.0, size=(n_steps, n_states)))
        ref_path, _ = _brute_force_hsmm_map(model, log_em)
        assert np.array_equal(mk.decode(model, log_em), ref_path)


def test_negbinom_geometric_limit():
    dur = NegBinomDuration(mean=4.0, concentration=1.0)
    q = 1.0 / 4.0
    for d in range(1, 8):
        assert math.isclose(dur.log_pmf(d), math.log(q) + (d - 1) * math.log(1 - q), rel_tol=1e-12)


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
    assert isinstance(mk.resolve_engine(model, "decode"), mk.Exact)
    assert isinstance(mk.resolve_engine(model, "smooth"), mk.Intractable)


def test_infeasible_segmentation_raises():
    # A single-state chain cannot alternate, so a sequence longer than the cap has no valid segmentation.
    durations = (NegBinomDuration(mean=2.0, concentration=1.0),)
    model = mk.SemiMarkovChain(np.array([0.0]), np.array([[0.0]]), durations, max_duration=3)
    with pytest.raises(ValueError, match="no valid segmentation"):
        mk.decode(model, np.zeros((5, 1)))


def test_decode_raises_on_unsupported_model():
    with pytest.raises(ValueError, match="no engine"):
        mk.decode(object(), np.zeros((2, 2)))  # type: ignore[arg-type]
