# AGENTS.md

Operating guide for an agent (or human) working in this repo. Read this first, then
[`README.md`](README.md) (what the library *is* and the public surface) and
[`docs/design.md`](docs/design.md) (the design — the three unifications and the engine dispatch).
`CLAUDE.md` is a symlink to this file.

## What this is

markovlib is a typed library for **Markov-like processes**. The thesis: a Markov-like process is a
*factor graph with repeated temporal structure*, and every algorithm — filter / smooth / decode /
learn — is a **query** that resolves (decidably) to a message-passing engine over a **belief algebra**.
One belief-generic forward recursion, parameterized by *belief × semiring*, is forward–backward (HMM),
the Kalman filter (linear-Gaussian), or Viterbi (max-plus). Engines that don't fit that two-sweep mold
(the explicit-duration HSMM DP, the particle filter) are *sibling* engines.

It is **numpy/scipy only, Python 3.12+**, with `mypy --strict` and a **100% coverage** gate. The one
optional exception is `markovlib.fungeom` — a Python-3.13 bridge to the fungeom substrate, kept out of
the core gate (excluded from mypy + coverage) and verified separately.

## Commands

`uv`-managed. From a checkout:

```bash
uv pip install -e '.[dev]'
uv run --extra dev pytest --cov=markovlib   # tests + 100% coverage gate
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
uv run --extra dev mypy                     # strict
uv build                                     # sdist + wheel (hatch-vcs version from git tags)
```

**A change is only done when ruff, ruff format, mypy --strict, and pytest (at 100% coverage) all
pass.** These run in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) on Python 3.12 and
3.13, and are wired as pre-commit hooks (coverage gate on pre-push). Releases are tag-driven and
publish to PyPI via Trusted Publishing — see [`RELEASING.md`](RELEASING.md).

## Architecture & conventions

- **`belief.py`** — the `Belief` protocol + `Categorical` / `GaussianBelief`. **`semiring.py`** —
  `SumProduct` / `MaxPlus`. **`engines/recursion.py`** — the one belief-generic `forward`. Keep the
  belief/semiring boundary: new *message representations* go in beliefs, the *reduce knob* in semirings.
- **`engines/`** — one engine per file: `exact_chain` (HMM FB/Viterbi/EM), `segmental` (EDHMM
  Viterbi + posteriors), `gaussian` (Kalman + RTS), `particle` (bootstrap SIR). The **emission/evidence
  is the seam**: engines consume a precomputed `(T, S)` log-emission matrix (or model-specific
  evidence), never raw data.
- **`resolution.py` + `dispatch.py`** — `resolve_engine(model, query) → Exact | Approximate |
  Intractable`, the decidable dispatch (the deliberate analog of fungeom's `decide()`). Register new
  engines here; declare `Exact` or `Approximate` honestly. **`query.py`** is the uniform surface.
- **Validation philosophy.** Every engine is checked against an *independent oracle* — a brute-force
  enumeration (HMM/HSMM) or a textbook reference (Kalman/RTS) or convergence to an exact engine
  (particle → Kalman). Don't weaken a test to make it pass; fix the code.
- **Reify randomness.** Anything stochastic (the particle filter) takes a `seed` as an explicit input
  and is a deterministic function of it.

## The constellation

markovlib is the research home of the Markov machinery that `standalone_contact_detection`'s
`contact/` package now delegates to (bit-for-bit). It is vendored there as a git submodule. The
optional `markovlib.fungeom` bridge connects to **fungeom** (a general decidability substrate), the
geometry/time library that `retarget` builds on. Each repo has its own `AGENTS.md`; defer to each for
its conventions.
