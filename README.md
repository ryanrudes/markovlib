<h1 align="center">markovlib</h1>

<p align="center"><em>Markov-like processes as one recursion over a belief algebra and a semiring — HMM, HSMM, Kalman, and particle inference behind a decidable engine dispatch.</em></p>

<p align="center">
  <a href="https://github.com/ryanrudes/markovlib/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ryanrudes/markovlib/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="Coverage 100%" src="https://img.shields.io/badge/coverage-100%25-brightgreen">
  <img alt="Typed" src="https://img.shields.io/badge/typed-mypy%20strict-blue">
  <img alt="License Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-lightgrey">
</p>

---

markovlib is a small, typed library for **Markov-like processes** — hidden Markov models, semi-Markov
(explicit-duration) models, linear-Gaussian state-space models, and general nonlinear ones. Its one
idea: each of these is **a factor graph with repeated temporal structure**, and every algorithm —
filter, smooth, decode, learn — is a *query* that resolves to a message-passing engine over a **belief
algebra**. Swap the belief (categorical → Gaussian → particle) or the semiring (sum-product → max-plus)
and the *same* recursion becomes forward–backward, the Kalman filter, or Viterbi.

```python
import numpy as np
import markovlib as mk

model  = mk.DiscreteChain(np.log([0.6, 0.4]), np.log([[0.7, 0.3], [0.2, 0.8]]))
log_em = np.log([[.9, .1], [.8, .2], [.1, .9], [.2, .8], [.6, .4]])  # (T, S) log p(obs | state) — you supply this

post = mk.smooth(model, log_em)   # forward–backward: posterior marginals + log-likelihood
path = mk.decode(model, log_em)   # Viterbi: the MAP state path  ->  [0, 0, 1, 1, 1]
mk.fit(model, log_em)             # Baum–Welch: learn the dynamics (loglik never decreases)
```

## Why markovlib

- **One recursion, many algorithms.** A single belief-generic forward recursion is *the* primitive.
  With a categorical belief it is forward–backward; with a Gaussian belief it is the Kalman filter.
  Swap the semiring from sum-product to max-plus and posteriors become the MAP path. Inference,
  decoding, and learning are three queries over one model, not three codebases.
- **Decidable dispatch — never a silent approximation.** `resolve_engine(model, query)` returns
  *evidence*: `Exact`, `Approximate` (carrying the method and error character, e.g. *O(1/√N) Monte
  Carlo*), or `Intractable` (carrying a reason). The exact engines (HMM, HSMM, Kalman) say so; the
  particle filter honestly says it is approximate. Nothing guesses behind your back.
- **Exact where it can be.** Finite chains get exact forward–backward / Viterbi / EM; linear-Gaussian
  models get the exact Kalman filter and RTS smoother; explicit-duration models get the standard
  right-censored EDHMM. Approximation (the particle filter) is opt-in and labelled.
- **Reproducible randomness.** The particle filter is a *deterministic function of its seed* — the
  seed is an explicit input, so a run is exactly reproducible.
- **Typed, tested, tiny.** numpy + scipy only, `mypy --strict`, and a **100% coverage** gate. Every
  engine is validated against an independent oracle (brute-force enumeration or a textbook reference).

## Install

```bash
pip install markovlib
```

Requires Python 3.12+; `numpy` and `scipy` come with it.

**From a checkout, with the dev extras** (`ruff`, `mypy`, `pytest`):

```bash
git clone https://github.com/ryanrudes/markovlib && cd markovlib
uv pip install -e '.[dev]'
```

## The engines, at a glance

You build a **model**, supply your per-step evidence, then run a **query**. For discrete chains the
evidence is a `(T, S)` log-emission matrix — `log p(observation_t | state = s)` — which *you* compute
from your own observation model. That matrix is the seam: markovlib never sees your raw data.

| Model | Belief | Queries | Engine | Exactness |
| --- | --- | --- | --- | --- |
| `DiscreteChain` | categorical | `smooth` · `decode` · `loglik` · `fit` | forward–backward / Viterbi / Baum–Welch | **exact** |
| `SemiMarkovChain` | categorical + duration | `decode` · `smooth` | right-censored explicit-duration (EDHMM) | **exact** |
| `LinearGaussian` | Gaussian | `filter` · `smooth` | Kalman filter / RTS smoother | **exact** |
| `StateSpaceModel` | particle | `particle_filter` | bootstrap (SIR) particle filter | **approximate** (`O(1/√N)`) |

Transitions may be homogeneous `(S, S)` or time-varying `(T-1, S, S)`. Durations are
`NegBinomDuration(mean, concentration)` (the `concentration → 1` limit is a plain HMM's geometric dwell).

## A taste — each model in two lines

```python
# Semi-Markov: dwell is modelled, so 1-frame blips are intrinsically improbable.
hsmm = mk.SemiMarkovChain(log_init, log_trans,
                          durations=(mk.NegBinomDuration(5, 4), mk.NegBinomDuration(3, 4)), max_duration=12)
mk.decode(hsmm, log_em)        # explicit-duration MAP path

# Kalman filter + RTS smoother over a linear-Gaussian model.
lg = mk.LinearGaussian(A, Q, H, R, m0, P0)
mk.filter(lg, observations)    # FilterResult(means, covariances, loglik)
mk.smooth(lg, observations)    # RTS-smoothed estimates

# Particle filter over a general nonlinear model (deterministic given the seed).
ssm = mk.StateSpaceModel(sample_prior, propagate, log_likelihood)
mk.particle_filter(ssm, observations, n_particles=5000, seed=0)
```

## Examples

Runnable, commented scripts in [`examples/`](examples/) — each is exercised by the test suite, so they
stay current:

| Script | Shows |
| --- | --- |
| [`01_hmm_quickstart`](examples/01_hmm_quickstart.py) | a discrete HMM: `smooth` / `decode` / `loglik`, then `fit` to learn the dynamics |
| [`02_semi_markov_durations`](examples/02_semi_markov_durations.py) | explicit dwell models — a 1-frame blip absorbed, a genuine bout kept; time-varying transitions |
| [`03_kalman_filter_smoother`](examples/03_kalman_filter_smoother.py) | the Kalman filter and the RTS smoother as the *same* recursion with a Gaussian belief |
| [`04_particle_filter`](examples/04_particle_filter.py) | a bootstrap particle filter for a nonlinear model; reproducibility from the seed; ESS |
| [`05_decidable_dispatch`](examples/05_decidable_dispatch.py) | `resolve_engine` → `Exact` / `Approximate` / `Intractable`; the belief × semiring core |

```bash
python examples/01_hmm_quickstart.py
```

## Learn more

- **[`docs/design.md`](docs/design.md)** — the design: a Markov-like process as a factor graph with
  repeated temporal structure; the three unifications (one recursion × semiring; the engine family;
  learning via expected sufficient statistics); and the decidable engine dispatch.
- **The API** is small and discoverable from `markovlib.__all__`; every public symbol is typed.

## Development

```bash
uv pip install -e '.[dev]'
pytest --cov=markovlib   # tests + 100% coverage gate
ruff check . && ruff format --check .
mypy                     # strict
```

CI runs all four on every push (Python 3.12 and 3.13) and validates that the package builds; the same
checks are available as pre-commit hooks (`pre-commit install && pre-commit install --hook-type
pre-push`). Releases are tag-driven and publish to PyPI via Trusted Publishing — see
[`RELEASING.md`](RELEASING.md).

## License

[Apache-2.0](LICENSE).
