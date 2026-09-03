"""Deterministic proof that support restriction cannot cross a feedback barrier."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sqlite3
import sys
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
sys.path.insert(0, str(MODULE))

from contract import Contract  # noqa: E402
from planner import run_chain  # noqa: E402
from scope import Ctx  # noqa: E402

spec = importlib.util.spec_from_file_location("algebra_math_barrier", MODULE / "math.py")
assert spec and spec.loader
math_surface = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = math_surface
spec.loader.exec_module(math_surface)


class FeedbackBarrierTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE support(id TEXT PRIMARY KEY)")
        self.db.executemany("INSERT INTO support VALUES (?)", [("b",), ("c",)])
        self.ids = ["a", "b", "c"]
        angles = [10.0, -20.0, 30.0]
        self.matrix = np.array(
            [
                [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
                for angle in angles
            ],
            dtype=np.float32,
        )
        self.contract = Contract.from_mapping({
            "m": {
                "kind": "mask",
                "relation": "support",
                "key": "id",
                "expression": 'mask("m")',
            },
            "q": {
                "kind": "query",
                "relation": "support",
                "key": "id",
                "value": "unused",
                "columns": ["id"],
                "expression": 'similar("q")',
            },
            "q2": {
                "kind": "query",
                "relation": "support",
                "key": "id",
                "value": "unused",
                "columns": ["id"],
                "expression": 'similar("q2")',
            },
        })

    def tearDown(self):
        self.db.close()

    @staticmethod
    def embed_query(name):
        if name == "q":
            return np.array([1.0, 0.0], dtype=np.float32)
        if name == "q2":
            return np.array([0.0, 1.0], dtype=np.float32)
        raise KeyError(name)

    def execute(self, expression):
        top = math_surface.parse_math(expression, self.contract.bindings())
        ctx = Ctx(
            embed_fn=self.embed_query,
            E=self.matrix,
            ids=self.ids,
            db=self.db,
            contract=self.contract,
        )
        rows = run_chain(top, ctx)
        return rows, ctx.log

    def test_restriction_closes_over_a_fused_relation(self):
        rows, plan = self.execute("τ₂(m ▷ ((E @ q) ⊕ (E @ q2)))")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row[0] for row in rows}, {"b", "c"})
        self.assertTrue(any(line.startswith("fuse[rrf]") for line in plan))
        self.assertTrue(any(line.startswith("restrict[sql]") for line in plan))

    def test_restriction_does_not_commute_with_centroid_feedback(self):
        restrict_after, plan_after = self.execute(
            "τ₂(m ▷ (E @ centroid(τ₁(E @ q))))"
        )
        restrict_seed, plan_seed = self.execute(
            "τ₂(m ▷ (E @ centroid(τ₁(m ▷ (E @ q)))))"
        )

        self.assertEqual([row[0] for row in restrict_after], ["c", "b"])
        self.assertEqual([row[0] for row in restrict_seed], ["b", "c"])
        self.assertAlmostEqual(restrict_after[0][1], math.cos(math.radians(20)), places=6)
        self.assertAlmostEqual(restrict_seed[1][1], math.cos(math.radians(50)), places=6)
        self.assertTrue(plan_after[0].startswith("BARRIER: PRF inner plan"))
        self.assertTrue(plan_seed[0].startswith("BARRIER: PRF inner plan"))
        self.assertNotIn("restrict[sql]", plan_after[1])
        self.assertIn("restrict[sql]", plan_seed[1])


if __name__ == "__main__":
    unittest.main()
