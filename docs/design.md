# The design of markovlib

A design note, in the spirit of the library's siblings: read [`README.md`](../README.md) first for the
public surface, then this for *why* the internals look the way they do.

## The thesis

> A **Markov-like process** is a factor graph with repeated temporal structure. Every *algorithm* —
> filter, smooth, decode, learn — is a **query** that resolves (decidably) to a **message-passing
> engine** over a **belief algebra**. Inference and learning are the same machinery in different
> dressings.

Concretely: you describe a *model* (an initial, a transition, an evidence/observation channel). markovlib
turns a query over that model into a recursion over *messages* (beliefs). The trick is that almost all of
the apparent variety — HMM vs Kalman vs Viterbi — is **two orthogonal choices**, not different codebases.

## Unification 1 — one recursion, parameterized by belief × semiring

The load-bearing primitive is a single forward recursion ([`engines/recursion.py`](../src/markovlib/engines/recursion.py)):

```
msg[0] = initial ⊗ factor[0]
msg[t] = predict(msg[t-1]) ⊗ factor[t]
```

It touches a belief only through `combine` (`⊗`) and a `predict` push-forward. Two axes select the
algorithm:

- **Belief** — the message representation. `Categorical` (a log-probability vector) makes the recursion
  forward–backward; `GaussianBelief` (information / canonical form, where `combine` is *addition* — the
  Gaussian analog of categorical log-add) makes it the **Kalman filter**. A particle cloud would make it
  a particle filter — except its update is *state-dependent* (reweighting at the particle locations), so
  it doesn't fit the fixed-factor `⊗` mold and is a sibling engine instead.
- **Semiring** — the reduce knob inside `predict`. `SumProduct` (`logsumexp`) gives marginal posteriors
  and likelihood; `MaxPlus` (`max`) turns the *same* forward pass into the Viterbi score table.

So the categorical HMM's forward–backward and the linear-Gaussian Kalman filter are literally two
instantiations of one function; posteriors and the MAP path are one function with the semiring swapped.
The unnormalized-message convention makes `log_mass` of the final message *be* the data log-likelihood in
both the categorical and Gaussian cases — they stay parallel down to the likelihood.

## Unification 2 — the engine family, exact and approximate

Not everything fits the two-sweep mold; those are **sibling engines**, registered alongside the shared
recursion:

| engine | model | what it is |
| --- | --- | --- |
| `ExactChain` | `DiscreteChain` | forward–backward, Viterbi, EM — exact for a finite chain |
| `SegmentalChain` | `SemiMarkovChain` | the standard **right-censored explicit-duration** (EDHMM) Viterbi + forward–backward; dwell is modelled (a shifted negative-binomial), and a censored segment may continue past the cap, so long bouts are exact rather than truncated |
| `GaussianChain` | `LinearGaussian` | the Kalman filter (shared recursion) + the RTS smoother (a Gaussian backward pass) |
| `ParticleFilter` | `StateSpaceModel` | a bootstrap (SIR) filter for nonlinear / non-Gaussian models — **approximate** (`O(1/√N)`) |

## Unification 3 — learning is generic via expected sufficient statistics

EM is not a separate algorithm: its **E-step is the smoother** (`ExactChain.expected_stats` extends
forward–backward with the pairwise marginals ξ), and its **M-step re-estimates each parameter from the
expected sufficient statistics it owns**. `fit` runs Baum–Welch over the chain dynamics with the emission
evidence held fixed; the observed-data log-likelihood is non-decreasing every iteration (the EM
guarantee, asserted directly in the tests).

## The decidable dispatch

A query is resolved by `resolve_engine(model, query)` ([`dispatch.py`](../src/markovlib/dispatch.py)),
which returns **evidence**, never a silent best-effort — the deliberate analog of fungeom's `decide()`:

- `Exact(engine)` — an engine resolves it with no approximation (HMM, HSMM, Kalman).
- `Approximate(engine, method, error_character)` — it resolves it *approximately*, and says how (the
  particle filter: *bootstrap particle filter, O(1/√N) Monte Carlo*).
- `Intractable(reason)` — nothing resolves it; the reason is carried.

The uniform queries (`smooth` / `decode` / `loglik` / `filter` / `particle_filter`) raise on
`Intractable` rather than guess. This three-valued shape is why "is this answer exact?" is a first-class,
inspectable property of the library rather than tribal knowledge.

## Reified randomness

The particle filter is the one stochastic engine, and it is a **deterministic function of its seed** —
the seed is an explicit input, threaded through a single `numpy` generator. Same `(model, observations,
n_particles, seed)` ⇒ identical output. Honesty about approximation (it resolves as `Approximate`) and
honesty about randomness (reproducible from the seed) are the same discipline: nothing hidden.

## Validation

Every engine is checked against an **independent oracle**, not just internal consistency:
forward–backward / Viterbi / EDHMM against brute-force enumeration of all paths or labelled
segmentations; the Kalman filter and RTS smoother against a textbook moment-form implementation; EM
against the monotone-likelihood guarantee; the particle filter against convergence to the *exact* Kalman
filter on a linear-Gaussian model. The whole suite runs under `mypy --strict` and a 100% coverage gate.
