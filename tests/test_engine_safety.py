"""Portable regression tests for vector dimension safety."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
spec = importlib.util.spec_from_file_location("algebra_engine", MODULE / "engine.py")
assert spec and spec.loader
engine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)


class DimensionSafetyTests(unittest.TestCase):
    def test_exact_dimension_is_accepted(self):
        vector = np.ones(256, dtype=np.float32)
        self.assertIs(engine._align(vector, 256), vector)

    def test_short_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            engine._align(np.ones(128, dtype=np.float32), 256)

    def test_long_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            engine._align(np.ones(768, dtype=np.float32), 256)


if __name__ == "__main__":
    unittest.main()
