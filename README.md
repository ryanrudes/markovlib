# markovlib

*Markov-like processes as a factor graph with repeated temporal structure: one recursion, parameterized
by a belief algebra + semiring.*

markovlib is a general library for Markov-like models (HMM/HSMM, Kalman, particle, MDP/POMDP). Its thesis:
**every inference algorithm is a query that resolves to a message-passing engine over a belief algebra;
swap the semiring (sum-product ↔ max-plus) and the *same* recursion yields posteriors ↔ MAP.**
`resolve_engine(model, query)` returns evidence — *which* engine resolves the query and whether
**exactly** (`Exact | Approximate | Intractable`), deliberately mirroring fungeom's `decide()`
(evidence, not a bool; a reason on failure; never silent).

**Status: vertical slice.** `DiscreteChain → ExactChain` — forward–backward smoothing + Viterbi decoding
from one semiring-parameterized forward recursion — validated **bit-for-bit against brute-force path
enumeration** (the reference the fast paths are tested against). numpy/scipy only, Python 3.12,
**fungeom-free** (so it can back the contact repo's package bit-for-bit). A geometry-aware
`markovlib.fungeom` bridge (pose-valued state, `Signal`-carried gappy observations) is a later,
optional, Python-3.13 layer.

```python
import numpy as np, markovlib as mk

model = mk.DiscreteChain(log_init, log_trans)   # (S,) log-initial + (S,S) log-transition
post  = mk.smooth(model, log_emissions)         # post.gamma (T,S) posteriors, post.loglik
path  = mk.decode(model, log_emissions)         # MAP state path (T,)
```

Design + rationale (the depth-A three-layer architecture, and the reframed fungeom "membership" rule —
*admit by honest referential transparency, exactness is a grade not a gate*) live in the contact repo's
memory and in `functional_api/docs/substrate-membership.md`.
