"""02 — Semi-Markov (explicit-duration) HMM: a 1-frame blip is absorbed, a real bout is kept.

Give each state an explicit dwell prior — here "free" likes long dwells and "contact" is short — and
short spurious segments become intrinsically improbable. The model suppresses them, not a post-hoc
``drop_short_runs``.
"""

from __future__ import annotations

import numpy as np

import markovlib as mk


def main() -> None:
    log_init = np.log([0.5, 0.5])
    log_trans = np.log([[0.5, 0.5], [0.5, 0.5]])
    # state 0 = "free" (long dwells), state 1 = "contact" (~12-frame dwells); sharpness 8.
    durations = (mk.NegBinomDuration(mean=60.0, concentration=8.0), mk.NegBinomDuration(mean=12.0, concentration=8.0))
    hsmm = mk.SemiMarkovChain(log_init, log_trans, durations, max_duration=80)

    # a mild free preference everywhere, with one strong 1-frame contact blip at t=30.
    blip = np.full((60, 2), 0.0)
    blip[:, 0], blip[:, 1] = np.log(0.6), np.log(0.4)
    blip[30] = [np.log(0.02), np.log(0.98)]
    blip_path = mk.decode(hsmm, blip)
    print("1-frame blip -> all free:", bool(np.all(blip_path == 0)), "(absorbed by the dwell prior)")

    # a genuine 20-frame contact bout is recovered, free elsewhere.
    bout = np.full((60, 2), np.log(0.5))
    bout[20:40, 0], bout[20:40, 1] = np.log(0.2), np.log(0.8)
    bout_path = mk.decode(hsmm, bout)
    print("20-frame bout -> contact frames in [20,40):", int(bout_path[20:40].sum()), "/ 20")
    assert np.all(blip_path == 0)
    assert bout_path[20:40].mean() > 0.9

    # time-varying transitions: pass a (T-1, S, S) tensor instead of (S, S) — nothing else changes.
    tv = np.stack([log_trans for _ in range(59)])
    post = mk.smooth(mk.DiscreteChain(log_init, tv), bout)
    print("time-varying HMM smooth rows sum to 1:", bool(np.allclose(post.gamma.sum(1), 1.0)))


if __name__ == "__main__":
    main()
