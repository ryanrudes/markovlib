"""01 — Discrete HMM: the four things you do with a chain (smooth, decode, loglik, fit).

You bring the model (initial + transition) and a ``(T, S)`` log-emission matrix you computed yourself;
markovlib does the temporal inference.
"""

from __future__ import annotations

import numpy as np

import markovlib as mk


def main() -> None:
    log_init = np.log([0.6, 0.4])
    log_trans = np.log([[0.7, 0.3], [0.2, 0.8]])
    log_em = np.log([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.6, 0.4]])  # log p(obs | state)

    model = mk.DiscreteChain(log_init, log_trans)

    post = mk.smooth(model, log_em)  # forward–backward
    path = mk.decode(model, log_em)  # Viterbi MAP
    print("posterior gamma[0]:", post.gamma[0].round(3).tolist(), " loglik:", round(post.loglik, 3))
    print("Viterbi MAP path  :", path.tolist())
    print("loglik (forward)  :", round(mk.loglik(model, log_em), 3))

    fitted = mk.fit(model, log_em, max_iter=50)  # Baum–Welch (learns the dynamics)
    history = fitted.loglik_history
    print("Baum–Welch loglik :", round(history[0], 3), "->", round(history[-1], 3), f"({len(history)} iters)")
    assert all(b >= a - 1e-9 for a, b in zip(history, history[1:], strict=False))  # never decreases


if __name__ == "__main__":
    main()
