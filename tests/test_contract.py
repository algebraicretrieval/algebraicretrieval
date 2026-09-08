"""Portable regression tests for explicit SQLite operand contracts."""

from __future__ import annotations

import importlib.util
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

spec = importlib.util.spec_from_file_location("algebra_math_contract", MODULE / "math.py")
assert spec and spec.loader
math_surface = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = math_surface
spec.loader.exec_module(math_surface)


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


class FourRowWitnessTests(unittest.TestCase):
    """The PEM four-row data through the existing Algebra compiler and scorer."""

    # Fixture from PEM proof f407f61; expected values are stated independently.
    FIXTURE = (
        ("a", [1, 0], 0.25),
        ("b", [0.8, 0.6], 1.0),
        ("c", [0, 1], 0.25),
        ("d", [-1, 0], 0.125),
    )
    MASK = ("a", "c", "d")
    QUERIES = (("q1", [1, 0]), ("q2", [0, 1]))
    SCORED = "w ⊙ (m ▷ ((E @ q1) - 0.5·(E @ q2)))"
    EXPECTED = [("a", 0.25), ("c", -0.125), ("d", -0.125)]

    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE embeddings(id TEXT PRIMARY KEY, vector BLOB NOT NULL);
            CREATE TABLE queries(id TEXT PRIMARY KEY, vector BLOB NOT NULL);
            CREATE TABLE mask(id TEXT PRIMARY KEY);
            CREATE TABLE weights(id TEXT PRIMARY KEY, weight REAL NOT NULL);
        """)
        self.db.executemany("INSERT INTO embeddings VALUES (?, ?)", [
            (identity, np.asarray(vector, dtype=np.float32).tobytes())
            for identity, vector, _ in self.FIXTURE
        ])
        self.db.executemany("INSERT INTO queries VALUES (?, ?)", [
            (identity, np.asarray(vector, dtype=np.float32).tobytes())
            for identity, vector in self.QUERIES
        ])
        self.db.executemany("INSERT INTO mask VALUES (?)", [(i,) for i in self.MASK])
        self.db.executemany("INSERT INTO weights VALUES (?, ?)", [
            (identity, weight) for identity, _, weight in self.FIXTURE
        ])
        self.contract = Contract.from_mapping({
            "E": {"kind": "matrix", "relation": "embeddings", "key": "id",
                  "columns": ["vector"], "dtype": "float32"},
            "m": {"kind": "mask", "relation": "mask", "key": "id",
                  "expression": 'mask("m")'},
            "w": {"kind": "weight", "relation": "weights", "key": "id",
                  "columns": ["weight"], "normalize": "none", "expression": 'weight("w")'},
            **{
                name: {"kind": "query", "relation": "queries", "key": "id",
                       "columns": ["vector"], "value": name,
                       "expression": f'similar("{name}")'}
                for name, _ in self.QUERIES
            },
        })

    def execute(self, k=2, *, reverse_matrix=False):
        embed_fn, matrix, ids = self.contract.runtime(self.db)
        if reverse_matrix:
            matrix, ids = matrix[::-1], ids[::-1]
        selection = {2: "τ₂", 4: "τ₄"}[k]
        top = math_surface.parse_math(f"{selection}({self.SCORED})", self.contract.bindings())
        ctx = Ctx(embed_fn=embed_fn, E=matrix, ids=ids, db=self.db, contract=self.contract)
        return run_chain(top, ctx), ctx

    def test_compiled_mask_weight_and_signed_scores(self):
        rows, ctx = self.execute()
        self.assertEqual(rows, self.EXPECTED[:2])
        self.assertEqual(ctx.matmul_rows, len(self.MASK))

    def test_top_k_retains_negative_support_without_padding(self):
        rows, _ = self.execute(k=4)
        self.assertEqual(rows, self.EXPECTED)
        self.assertEqual(rows[1][1], rows[2][1])  # Exact tie at the top-2 boundary.

    def test_key_alignment_and_cutoff_tie_ignore_input_order(self):
        self.db.execute("DELETE FROM mask")
        self.db.executemany("INSERT INTO mask VALUES (?)", [(self.MASK[i],) for i in (1, 0, 2)])
        self.db.execute("DELETE FROM weights")
        self.db.executemany("INSERT INTO weights VALUES (?, ?)", [
            (self.FIXTURE[i][0], self.FIXTURE[i][2]) for i in (3, 1, 0, 2)
        ])
        rows, _ = self.execute(reverse_matrix=True)
        self.assertEqual(rows, self.EXPECTED[:2])


if __name__ == "__main__":
    unittest.main()
