"""Each numbered example runs cleanly — so the examples in the README stay current and correct.

The ``markovlib.fungeom`` example is excluded (it needs Python 3.13 + fungeom; it is verified
separately in a 3.13 environment).
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

_EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("0*.py"))


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path: Path) -> None:
    runpy.run_path(str(path), run_name="__main__")
