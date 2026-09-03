"""Portable regression tests for the mathematical notation seam."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE = Path(__file__).resolve().parents[1] / "module" / "algebra"
sys.path.insert(0, str(MODULE))
spec = importlib.util.spec_from_file_location("algebra_math", MODULE / "math.py")
assert spec and spec.loader
math_surface = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = math_surface
spec.loader.exec_module(math_surface)


class MathematicalSurfaceTests(unittest.TestCase):
    def test_parenthesized_expansion_preserves_precedence(self):
        top = math_surface.parse_math(
            "a = (E @ similar(q1)) - (E @ similar(q2))\nτ₁₀(0.5·a)"
        )
        self.assertEqual([term.coef for term in top.expr.terms], [0.5, -0.5])

    def test_assignments_are_top_down_and_single_assignment(self):
        top = math_surface.parse_math(
            "a = E @ similar(q1)\nb = a - (E @ similar(q2))\nτ₁₀(b)"
        )
        self.assertEqual(len(top.expr.terms), 2)
        with self.assertRaisesRegex(math_surface.MathError, "cannot rebind"):
            math_surface.parse_math(
                "a = E @ similar(q1)\na = E @ similar(q2)\nτ₁₀(a)"
            )

    def test_forward_and_self_references_fail(self):
        with self.assertRaisesRegex(math_surface.MathError, "invalid assignment a"):
            math_surface.parse_math("a = b\nb = E @ similar(q2)\nτ₁₀(a)")
        with self.assertRaisesRegex(math_surface.MathError, "invalid assignment a"):
            math_surface.parse_math("a = a\nτ₁₀(a)")

    def test_repeated_relation_macro_builds_distinct_subplans(self):
        top = math_surface.parse_math(
            "seed = E @ similar(seed)\n"
            "τ₆₀(seed ⊕ window(12, expand(1, seed)))"
        )
        left = top.expr.parts[0]
        nested = top.expr.parts[1].inner.inner
        self.assertEqual(left, nested)
        self.assertIsNot(left, nested)

    def test_named_assignment_matches_expanded_expression(self):
        named = math_surface.parse_math(
            "q = similar(algebraic retrieval)\nτ₅(E @ q)"
        )
        expanded = math_surface.parse_math(
            "τ₅(E @ similar(algebraic retrieval))"
        )
        self.assertEqual(named, expanded)

    def test_unused_assignment_does_not_rewrite_retrieval_text(self):
        baseline = math_surface.transpile("τ₅(E @ similar(q planning))")
        assigned = math_surface.transpile(
            "q = similar(alpha)\nτ₅(E @ similar(q planning))"
        )
        self.assertEqual(assigned, baseline)

    def test_bound_name_inside_retrieval_text_remains_literal(self):
        expression = math_surface.transpile(
            "m = type(user_prompt)\nτ₅(E @ similar(mask m behavior))"
        )
        self.assertIn("similar(mask m behavior)", expression)
        self.assertNotIn("similar(mask (type(user_prompt)) behavior)", expression)

    def test_restriction_and_explicit_rrf_survive_normalization(self):
        result = math_surface.transpile(
            "τ₅(type(user_prompt) ▷ (kw(auth) ⊕RRF (E @ similar(auth))))"
        )
        self.assertEqual(
            result,
            "top(5, type(user_prompt) ▷ (kw(auth) ⊕ (E @ similar(auth))))",
        )

    def test_quoted_symbols_remain_literal(self):
        source = 'τ₁₀(E @ similar("H₂O; C× compiler; τ₁(x)"))'
        result = math_surface.transpile(source)
        self.assertEqual(
            result,
            'top(10, E @ similar("H₂O; C× compiler; τ₁(x)"))',
        )


if __name__ == "__main__":
    unittest.main()
