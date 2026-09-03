"""Typed operator regressions: restriction, modulation, fusion, and barriers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
sys.path.insert(0, str(MODULE))

import parse  # noqa: E402
from ir import Fuse, Rescore, Restrict  # noqa: E402


class TypedOperatorTests(unittest.TestCase):
    def test_restriction_and_weight_are_distinct(self):
        top = parse.parse(
            "top(5, type(user_prompt) ▷ (decay(7) ⊙ (E @ similar(auth))))"
        )
        self.assertIsNotNone(top.expr.mask)
        self.assertEqual([w.kind for w in top.expr.weights], ["decay"])

    def test_ascii_restrict_and_modulate(self):
        top = parse.parse(
            "top(5, restrict(type(user_prompt), modulate(decay(7), E @ similar(auth))))"
        )
        self.assertIsNotNone(top.expr.mask)
        self.assertEqual([w.kind for w in top.expr.weights], ["decay"])

    def test_mask_odot_is_rejected(self):
        with self.assertRaisesRegex(parse.ParseError, "never 'm ⊙ S'"):
            parse.parse("top(5, type(user_prompt) ⊙ (E @ similar(auth)))")

    def test_mask_star_is_rejected(self):
        with self.assertRaisesRegex(parse.ParseError, "scalar multiplication only"):
            parse.parse("top(5, type(user_prompt) * (E @ similar(auth)))")

    def test_threshold_must_wrap_its_input(self):
        parse.parse("top(5, threshold(0.2, E @ similar(auth)))")
        with self.assertRaisesRegex(parse.ParseError, "changes selection"):
            parse.parse("top(5, threshold(0.2) ⊙ (E @ similar(auth)))")

    def test_score_arithmetic_requires_one_unit(self):
        left = parse.parse_score("E @ similar(a)")
        right = parse.parse_score("E @ similar(b)")
        right.unit = "bm25:minmax"
        with self.assertRaisesRegex(parse.ParseError, "not commensurable"):
            parse._combine("+", ("score", left), ("score", right))

    def test_score_arithmetic_requires_one_support_context(self):
        parse.parse(
            "top(5, type(user_prompt) ▷ ((E @ similar(a)) - (E @ similar(b))))"
        )
        with self.assertRaisesRegex(parse.ParseError, "identical support"):
            parse.parse(
                "top(5, (type(user_prompt) ▷ (E @ similar(a))) + (E @ similar(b)))"
            )

    def test_restriction_closes_over_relation_valued_plans(self):
        restricted = parse.parse(
            "top(5, type(user_prompt) ▷ expand(1, E @ similar(auth)))"
        )
        self.assertIsInstance(restricted.expr, Restrict)

    def test_rank_fusion_is_explicit(self):
        fused = parse.parse("top(5, kw(auth) ⊕ (E @ similar(auth)))")
        self.assertIsInstance(fused.expr, Fuse)
        with self.assertRaisesRegex(parse.ParseError, "rank fusion"):
            parse.parse("top(5, kw(auth) + (E @ similar(auth)))")

    def test_rescore_is_explicit(self):
        rescored = parse.parse(
            "top(5, rescore(similar(auth), expand(1, E @ similar(seed))))"
        )
        self.assertIsInstance(rescored.expr, Rescore)
        with self.assertRaisesRegex(parse.ParseError, "use rescore"):
            parse.parse("top(5, similar(auth) ⊙ expand(1, E @ similar(seed)))")

    def test_mask_is_support_not_zero_weight(self):
        scores = {"a": -0.1, "b": -0.2, "c": 0.9, "d": 0.8}
        support = {"a", "b"}
        restricted = sorted(
            ((key, score) for key, score in scores.items() if key in support),
            key=lambda item: (-item[1], item[0]),
        )[:2]
        zero_multiplied = sorted(
            ((key, score if key in support else 0.0) for key, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )[:2]
        self.assertEqual(restricted, [("a", -0.1), ("b", -0.2)])
        self.assertEqual(zero_multiplied, [("c", 0.0), ("d", 0.0)])


if __name__ == "__main__":
    unittest.main()
