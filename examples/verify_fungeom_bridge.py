"""Verify ``markovlib.fungeom`` — the optional bridge to fungeom's time-``Signal`` substrate.

This is **not** part of markovlib's core 3.12 test suite: it requires **fungeom (Python 3.13+)** and is
run in a separate environment. It demonstrates the observation-carrier seam: a gappy fungeom ``Signal``
(a dropout it honestly refuses to invent across) resolves onto a sampling grid as observations **plus a
present mask**, where a gap ⇒ ``present == False`` — the "missing observation ⇒ predict-only" hook a
filter consumes.

Run (in a 3.13 venv with fungeom + markovlib installed)::

    uv venv --python 3.13 .venv-bridge
    uv pip install --python .venv-bridge -e <path/to/fungeom> -e .
    .venv-bridge/bin/python examples/verify_fungeom_bridge.py
"""

import numpy as np
from fungeom import ScalarSignal, Sampling

from markovlib.fungeom import observations_from_signal


def main() -> None:
    # A scalar signal whose value equals time, with a dropout: samples at 0,1,2 then 5,6 (gap 2→5 > max_gap).
    signal = ScalarSignal.from_samples(np.array([0.0, 1.0, 2.0, 5.0, 6.0]), [0.0, 1.0, 2.0, 5.0, 6.0], max_gap=1.5)
    grid = Sampling.at_times(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    observations, present = observations_from_signal(signal, grid)
    assert present.tolist() == [True, True, True, False, False, True, True], present.tolist()
    assert np.allclose(observations[present].ravel(), [0.0, 1.0, 2.0, 5.0, 6.0]), observations.ravel()
    assert np.all(np.isnan(observations[~present])), observations.ravel()

    print("BRIDGE OK: fungeom gappy Signal -> markovlib observations + present mask (gap => predict-only)")
    print("  present:", present.tolist())
    print("  obs    :", [None if np.isnan(v) else v for v in observations.ravel().tolist()])


if __name__ == "__main__":
    main()
