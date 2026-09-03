"""Score-device contract: raw inner product, linear collapse, and thresholds."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
sys.path.insert(0, str(MODULE))

import engine  # noqa: E402
import lower  # noqa: E402
import parse  # noqa: E402
from score_semantics import CONTRACT, stable_top_indices  # noqa: E402


class ScoreDeviceContractTests(unittest.TestCase):
    def test_contract_declares_raw_dot_and_single_query_partition(self):
        self.assertEqual(CONTRACT.score_unit, "inner_product:f32")
        self.assertEqual(CONTRACT.composed_query_normalization, "none")
        self.assertEqual(CONTRACT.partitioning, "one query per execution")
        self.assertEqual(CONTRACT.tie_order, ("score:desc", "id:asc"))
        self.assertEqual(CONTRACT.absolute_tolerance, 1e-6)

    def test_tie_order_is_score_desc_then_id_asc(self):
        ids = ["z", "a", "m"]
        scores = np.array([0.5, 0.5, 0.4], dtype=np.float32)
        order = stable_top_indices(ids, scores, 2)
        self.assertEqual([ids[int(index)] for index in order], ["a", "z"])


class ScoreSemanticsTests(unittest.TestCase):
    def setUp(self):
        # Every E row and primitive query is unit length.
        self.E = np.array(
            [[1.0, 0.0], [0.8, -0.6], [0.6, -0.8]], dtype=np.float32
        )
        self.ids = ["a", "b", "c"]
        self.queries = {
            "q1": np.array([1.0, 0.0], dtype=np.float32),
            "q2": np.array([0.0, 1.0], dtype=np.float32),
        }
        self.db = sqlite3.connect(":memory:")

    def tearDown(self):
        self.db.close()

    def embed(self, name):
        return self.queries[name]

    def test_primitive_unit_query_equals_cosine(self):
        q = engine.compose([(1.0, "q1")], self.embed)
        np.testing.assert_array_equal(q, self.queries["q1"])
        np.testing.assert_allclose(self.E @ q, [1.0, 0.8, 0.6], atol=1e-7)

    def test_composed_query_retains_magnitude(self):
        q = engine.compose([(1.0, "q1"), (-0.5, "q2")], self.embed)
        np.testing.assert_array_equal(q, np.array([1.0, -0.5], dtype=np.float32))

    def test_collapse_preserves_threshold_relation(self):
        top = parse.parse(
            "top(10, threshold(0.95, (E @ similar(q1)) - 0.5 * (E @ similar(q2))))"
        )
        folded, _fold_log, fold_mm = lower.run(
            top, self.embed, self.E, self.ids, self.db, fold=True
        )
        naive, _naive_log, naive_mm = lower.run(
            top, self.embed, self.E, self.ids, self.db, fold=False
        )
        self.assertEqual(fold_mm, 1)
        self.assertEqual(naive_mm, 2)
        self.assertEqual([identity for identity, _ in folded], ["b", "a", "c"])
        self.assertEqual([identity for identity, _ in folded], [identity for identity, _ in naive])
        np.testing.assert_allclose(
            [score for _, score in folded],
            [score for _, score in naive],
            atol=1e-6,
            rtol=0.0,
        )

    def test_normalization_mutation_is_detected(self):
        raw = self.queries["q1"] - 0.5 * self.queries["q2"]
        correct = self.E @ raw
        mutated = self.E @ (raw / np.linalg.norm(raw))
        correct_support = {
            identity for identity, score in zip(self.ids, correct) if score >= 0.95
        }
        mutated_support = {
            identity for identity, score in zip(self.ids, mutated) if score >= 0.95
        }
        self.assertEqual(correct_support, {"a", "b", "c"})
        self.assertEqual(mutated_support, {"b"})
        self.assertNotEqual(correct_support, mutated_support)


if __name__ == "__main__":
    unittest.main()
