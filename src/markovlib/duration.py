"""Duration models for semi-Markov chains — explicit dwell-time distributions.

In a plain Markov chain the time spent in a state is *implicitly* geometric (set by the self-transition
probability). A semi-Markov chain instead gives each state an **explicit** dwell distribution and forbids
self-transitions — persistence is owned by the duration, not the transition. :class:`NegBinomDuration`
is the shifted negative-binomial dwell (a mean and a dispersion); its ``concentration → 1`` limit is the
geometric dwell, so a semi-Markov chain with geometric durations degrades exactly to a plain HMM.

Both the point mass (:meth:`~NegBinomDuration.log_pmf`) and the right tail
(:meth:`~NegBinomDuration.log_survival`, ``log P(dwell ≥ d)``) are exposed: the segmental decoder scores
a naturally-ended segment by the pmf and a segment that hits the duration cap by the survival (the
standard explicit-duration **right-censoring**, which lets a state persist past the cap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scipy.stats import nbinom


class DurationModel(Protocol):
    """A dwell-time distribution over integer durations ``d ≥ 1``."""

    def log_pmf(self, d: int) -> float:
        """``log P(dwell = d)`` for ``d ≥ 1``."""
        ...

    def log_survival(self, d: int) -> float:
        """``log P(dwell ≥ d)`` — the right-censored mass a capped segment scores."""
        ...


@dataclass(frozen=True)
class NegBinomDuration:
    """Shifted negative-binomial dwell over ``d ≥ 1`` with given ``mean`` (> 1) and ``concentration`` (> 0).

    Parameterized so ``E[d] = mean`` exactly: with ``k = d - 1 ~ NB(r = concentration, p)`` and
    ``p = r / (r + mean - 1)``. The ``concentration → 1`` limit is the geometric dwell
    ``P(d) = (1 - q)^{d-1} q`` with ``q = 1 / mean`` (a plain HMM's implicit dwell); larger
    ``concentration`` concentrates the mass around the mean. Backed by :data:`scipy.stats.nbinom`.
    """

    mean: float
    concentration: float

    def __post_init__(self) -> None:
        if self.mean <= 1.0:
            raise ValueError(f"mean dwell must be > 1 (got {self.mean})")
        if self.concentration <= 0.0:
            raise ValueError(f"concentration must be > 0 (got {self.concentration})")

    def _params(self) -> tuple[float, float]:
        r = self.concentration
        p = r / (r + self.mean - 1.0)
        return r, p

    def log_pmf(self, d: int) -> float:
        """``log P(dwell = d)`` — the shifted negative-binomial log-pmf at integer ``d ≥ 1``."""
        if d < 1:
            raise ValueError(f"duration must be >= 1 (got {d})")
        r, p = self._params()
        return float(nbinom.logpmf(d - 1, r, p))

    def log_survival(self, d: int) -> float:
        """``log P(dwell ≥ d)``. Since ``d = k + 1``, ``P(d ≥ D) = P(k > D - 2) = nbinom.sf(D - 2)``."""
        r, p = self._params()
        return float(nbinom.logsf(d - 2, r, p))
