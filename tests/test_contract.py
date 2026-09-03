"""Portable regression tests for explicit SQLite operand contracts."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import sys
import unittest


MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
sys.path.insert(0, str(MODULE))
from contract import Contract  # noqa: E402


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE documents(id TEXT PRIMARY KEY, text TEXT)")
        self.db.execute("CREATE TABLE scores(id TEXT PRIMARY KEY, weight REAL)")
        self.db.executemany(
            "INSERT INTO scores VALUES (?, ?)",
            [("a", 1.0), ("b", 0.5)],
        )

    def tearDown(self):
        self.db.close()

    def test_explicit_mask_and_weight_relations_resolve(self):
        contract = Contract.from_mapping({
            "m": {"kind": "mask", "relation": "scores", "key": "id"},
            "w": {
                "kind": "weight",
                "relation": "scores",
                "key": "id",
                "columns": ["weight"],
                "normalize": "max",
            },
        })
        self.assertEqual(contract.mask_ids(self.db, "m"), ["a", "b"])
        weights = contract.weight_values(self.db, "w", ["a", "b", "c"])
        self.assertEqual(weights.tolist(), [1.0, 0.5, 0.0])
        tokens = {row[1] for row in contract.orient_rows(self.db)}
        self.assertTrue({"m ▷ S", "w ⊙ S", "S₁ ⊕ S₂", "τₖ(S)"} <= tokens)
        self.assertNotIn("m ⊙ w ⊙ S", tokens)

    def test_unknown_operand_fails_closed(self):
        contract = Contract.from_mapping({})
        with self.assertRaises(KeyError):
            contract.mask_ids(self.db, "missing")


if __name__ == "__main__":
    unittest.main()
