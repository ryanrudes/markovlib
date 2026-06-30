"""05 — The decidable dispatch (Exact / Approximate / Intractable) and the belief × semiring core."""

from __future__ import annotations

import numpy as np

import markovlib as mk
from markovlib.engines.recursion import categorical_predict, forward


def main() -> None:
    model = mk.DiscreteChain(np.log([0.6, 0.4]), np.log([[0.7, 0.3], [0.2, 0.8]]))
    log_em = np.log([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])

    # resolve_engine returns *evidence* — never a silent best-effort.
    print("HMM   smooth  ->", type(mk.resolve_engine(model, "smooth")).__name__)  # Exact
    unsupported = mk.resolve_engine(model, "control")
    assert isinstance(unsupported, mk.Intractable)
    print("HMM   control ->", unsupported.reason)

    # The belief × semiring core: the SAME forward recursion, two semirings.
    initial = mk.Categorical(np.log([0.6, 0.4]))
    factors = [mk.Categorical(row) for row in log_em]
    alpha = forward(initial, categorical_predict(model.log_trans, mk.SumProduct()), factors)  # forward–backward α
    delta = forward(initial, categorical_predict(model.log_trans, mk.MaxPlus()), factors)  # Viterbi δ
    print("sum-product final log-mass:", round(alpha[-1].log_mass(), 3), "(= data log-likelihood)")
    print("max-plus    final best    :", round(float(delta[-1].log_p.max()), 3), "(= best-path score)")
    assert np.isclose(alpha[-1].log_mass(), mk.loglik(model, log_em))


if __name__ == "__main__":
    main()
