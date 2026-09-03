"""The heart — algebraic laws as rewrite rules. v1: linear collapse.

LINEAR COLLAPSE (distributivity of raw inner product over +,− and scalar mult):
    Σ_i coef_i · (E @ q_i)  =  E @ ( Σ_i coef_i · q_i )

Primitive q_i are normalized individually. The merged query is not normalized;
its magnitude is observable by scores and threshold. Float32 implementations
compare the two evaluation orders with the score-device absolute tolerance.

A ScoreExpr's `terms` (each coef·Score(q_i)) collapse into ONE ScoreTerm whose query
is the merged linear combination of all the q_i — provided every term is data-independent
(no qterm references a nested Top). K matmuls over the N×d matrix become one.

This is what makes `E @ q1 - 0.5*E @ q2` compile to one matmul over `q1 - 0.5*q2`.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir import ScoreExpr, ScoreTerm, QTerm, Top  # noqa: E402


def _data_dependent(q) -> bool:
    return any(isinstance(t.arg, Top) for t in q)


def collapse(expr: ScoreExpr) -> ScoreExpr:
    """Fold the linear combination of Score terms into a single Score(merged q).

    Leaves data-dependent terms (centroid(nested Top)) as their own terms — they
    cannot pre-compose (a barrier; lower.py resolves them first). Returns a new
    ScoreExpr; idempotent.
    """
    foldable, blocked = [], []
    for st in expr.terms:
        (blocked if _data_dependent(st.q) else foldable).append(st)

    merged_terms = list(blocked)
    if foldable:
        merged_q: list[QTerm] = []
        for st in foldable:
            for qt in st.q:
                merged_q.append(QTerm(coef=st.coef * qt.coef, kind=qt.kind, arg=qt.arg))
        merged_terms.insert(0, ScoreTerm(coef=1.0, q=merged_q))

    return ScoreExpr(terms=merged_terms, mask=expr.mask, weights=expr.weights, selectors=expr.selectors, unit=expr.unit)


def matmul_count(expr: ScoreExpr) -> int:
    """How many E@q matmuls this ScoreExpr will execute as-is (one per Score term,
    plus one per nested Top — counted by lower at run time for nested). Top-level only."""
    return len(expr.terms)
