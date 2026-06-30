"""Optional bridge: carry observations from a fungeom time-``Signal`` into markovlib.

This is the ``markovlib.fungeom`` seam — the consumer-facing edge where the general inference library
meets fungeom's geometry/time substrate (for geometry-heavy consumers like *retarget*). It is **not**
imported by ``markovlib`` itself: markovlib's core stays fungeom-free and Python-3.12, while this module
requires **fungeom (Python 3.13+)**. Import it explicitly: ``from markovlib.fungeom import …``.

The one capability here is the **observation carrier**: a fungeom ``Signal`` is a *partial* function of
time — defined on its support, honestly ``Unresolvable`` in a dropout/gap — so resolving it onto a
sampling grid yields not just observations but a **present mask**. A gap ⇒ ``present[t] == False`` is
exactly the "missing/irregular observation ⇒ predict-only" signal a filter needs; the occlusion-aware
gappiness of motion-capture data flows through for free, instead of being silently interpolated.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from fungeom import Resolvable  # requires fungeom — Python 3.13+

Float = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]


def observations_from_signal(signal: object, sampling: object) -> tuple[Float, Bool]:
    """Resolve a scalar fungeom ``Signal`` onto a ``Sampling`` grid → ``(observations (T, 1), present (T,))``.

    Where the signal is defined, ``observations[t]`` is its (reconstructed) value and ``present[t]`` is
    ``True``; in a gap — a dropout the signal honestly refuses to invent across — ``observations[t]`` is
    ``nan`` and ``present[t]`` is ``False``. That ``present`` mask is the predict-only hook: a filter
    skips the measurement update wherever an observation is genuinely missing.
    """
    times = np.asarray(sampling.resolve().times, dtype=float)  # type: ignore[attr-defined]
    observations = np.full((times.shape[0], 1), np.nan, dtype=np.float64)
    present = np.zeros(times.shape[0], dtype=np.bool_)
    for index, instant in enumerate(times):
        decision = signal.at(float(instant)).decide()  # type: ignore[attr-defined]
        if isinstance(decision, Resolvable):
            observations[index, 0] = float(decision.value)
            present[index] = True
    return observations, present
