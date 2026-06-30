"""Duration models for semi-Markov chains — explicit dwell-time distributions.

In a plain Markov chain the time spent in a state is *implicitly* geometric (set by the self-transition
probability). A semi-Markov chain instead gives each state an **explicit** dwell distribution and forbids
self-transitions — persistence is owned by the duration, not the transition. :class:`NegBinomDuration`
is the shifted negative-binomial dwell (a mean and a dispersion); its ``concentration → 1`` limit is the
geometric dwell, so a semi-Markov chain with geometric durations degrades exactly to a plain HMM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from scipy.special import gammaln


class DurationModel(Protocol):
    """A dwell-time distribution over integer durations ``d ≥ 1``."""

    def log_pmf(self, d: int) -> float:
        """``log P(dwell = d)`` for ``d ≥ 1``."""
        ...


@dataclass(frozen=True)
class NegBinomDuration:
    """Shifted negative-binomial dwell over ``d ≥ 1`` with given ``mean`` (> 1) and ``concentration`` (> 0).

    Parameterized so ``E[d] = mean`` exactly. The ``concentration → 1`` limit is the geometric dwell
    ``P(d) = (1 - q)^{d-1} q`` with ``q = 1 / mean`` — a plain HMM's implicit dwell — and larger
    ``concentration`` concentrates the distribution around the mean (less dispersed than geometric).
    """

    mean: float
    concentration: float

    def __post_init__(self) -> None:
        if self.mean <= 1.0:
            raise ValueError(f"mean dwell must be > 1 (got {self.mean})")
        if self.concentration <= 0.0:
            raise ValueError(f"concentration must be > 0 (got {self.concentration})")

    def log_pmf(self, d: int) -> float:
        """``log P(dwell = d)`` — the shifted negative-binomial log-pmf at integer ``d ≥ 1``."""
        if d < 1:
            raise ValueError(f"duration must be >= 1 (got {d})")
        r = self.concentration
        p = r / (r + self.mean - 1.0)
        k = d - 1
        return float(gammaln(k + r) - gammaln(k + 1.0) - gammaln(r) + r * math.log(p) + k * math.log1p(-p))
