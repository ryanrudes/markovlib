# Changelog

All notable changes to **marmo** (the `markovlib` inference library) are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version is derived from git tags
(`vX.Y.Z`); see [RELEASING.md](RELEASING.md).

## [Unreleased]

## [0.1.0] - 2026-06-30

The initial release: the full Markov-like-process inference family behind a decidable engine dispatch.

### Added

- **The core abstraction** — a `Belief` protocol (`Categorical`, `GaussianBelief`) and a `Semiring`
  (`SumProduct`, `MaxPlus`); one belief-generic forward recursion (`engines.recursion.forward`) that,
  parameterized by belief × semiring, is forward–backward, the Kalman filter, or Viterbi.
- **Discrete HMM** — `DiscreteChain` with `smooth` (forward–backward posteriors + log-likelihood),
  `decode` (Viterbi MAP), `loglik`, and `fit` (Baum–Welch over the dynamics; monotone log-likelihood).
  Transitions may be homogeneous `(S, S)` or time-varying `(T-1, S, S)`.
- **Semi-Markov HMM** — `SemiMarkovChain` + `NegBinomDuration` (shifted negative-binomial dwell;
  `concentration → 1` recovers the geometric dwell): the standard right-censored explicit-duration
  Viterbi (`decode`) and forward–backward posteriors (`smooth`), with same-state continuation past the
  duration cap.
- **Linear-Gaussian state space** — `LinearGaussian` with the Kalman `filter` and the RTS `smooth`
  smoother, expressed through the shared forward recursion with a Gaussian (information-form) belief.
- **Particle filter** — `StateSpaceModel` + `particle_filter`: a bootstrap (SIR) filter for general
  nonlinear / non-Gaussian models, with systematic resampling on low ESS. Randomness is reified — the
  filter is a deterministic function of its `seed`. The first **approximate** engine.
- **Decidable dispatch** — `resolve_engine(model, query)` returns `Exact | Approximate | Intractable`
  evidence (mirroring fungeom's `decide()`); the uniform queries raise on `Intractable` rather than
  guess.
- **Optional `markovlib.fungeom` bridge** (Python 3.13+) — `observations_from_signal` carries a gappy
  fungeom time-`Signal` into markovlib as `(observations, present mask)`, so a dropout becomes
  `present[t] = False` — the "missing observation ⇒ predict-only" hook. Excluded from the 3.12 core
  gate; verified separately (see `examples/verify_fungeom_bridge.py`).

Every engine is validated against an independent oracle (brute-force enumeration or a textbook
reference) under `mypy --strict` and a 100% coverage gate.

[Unreleased]: https://github.com/ryanrudes/markovlib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ryanrudes/markovlib/releases/tag/v0.1.0
