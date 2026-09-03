"""IR for the algebraic query compiler.

Two spaces:
  query-space (dim d):  a query vector is Σ coef·qterm, qterm ∈ {similar(text), centroid(arg)}
  score-space (support A, unit u): Score(q) = raw float32 inner product E @ q.

Canonical normalized form:   Top(n, ScoreExpr)
  ScoreExpr = mask ▷ ((Π weights) ⊙ (Σ_i coef_i · Score(q_i)))

Masks restrict support. Weights modulate finite scores. Selection barriers such
as threshold, diverse, and dedup are explicit wrappers and never masquerade as
weights.

The linear-collapse law (rewrite.py) merges Σ coef_i·Score(q_i) into Score(Σ coef_i·q_i)
— K matmuls over the N×d matrix become ONE. That fold is the planner's core move and
the thing `E @ q1 - 0.5*E @ q2` exists to demonstrate.

A qterm whose arg is a nested Top (e.g. centroid(top(...))) is DATA-DEPENDENT: its
vector isn't known until the inner query runs. That forces a barrier (sequence) —
lower.py evaluates the inner Top first, substitutes the resulting ids, then folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QTerm:
    coef: float
    kind: str            # 'similar' | 'centroid'
    arg: Any             # similar: text str | centroid: 'prev' | list[str] ids | Top (nested → sequence)


@dataclass
class ScoreTerm:
    coef: float
    q: list              # list[QTerm] — the query vector for this term (E @ q)


@dataclass
class Mask:
    preds: list          # list of predicate tuples → compiled to a SQL pre-filter in lower


@dataclass
class Weight:
    kind: str            # 'decay' | 'contract'
    arg: Any


@dataclass
class Selector:
    kind: str            # 'threshold' | 'diverse' | 'dedup'
    arg: Any


@dataclass
class ScoreExpr:
    terms: list                       # list[ScoreTerm]  — the Σ coef·Score(q)
    mask: Mask | None = None
    weights: list = field(default_factory=list)   # list[Weight]
    selectors: list = field(default_factory=list) # list[Selector] barriers
    unit: str = "inner_product:f32"            # score provenance/commensurability tag


@dataclass
class Op:
    """A scope-transformer (Class B) — the thing a single backend cannot do:
    enlarge / re-shape the candidate SET mid-pipeline. `inner` is the sub-plan
    whose scored result it transforms (a ScoreExpr or a nested Op).

    True scalar `weights` may modulate the transformed scores through `⊙`.
    Support-changing or sequential `selectors` remain explicit wrappers."""
    kind: str          # 'expand' | 'coedited' | 'surprise' | 'flavor'
    args: dict
    inner: object      # ScoreExpr | Op
    weights: list = field(default_factory=list)   # post-transform Weight list
    selectors: list = field(default_factory=list) # post-transform Selector barriers


@dataclass
class Restrict:
    """Support restriction around a relation-valued plan that cannot be pushed
    into its inner score expression without crossing a semantic barrier."""
    mask: Mask
    inner: object
    weights: list = field(default_factory=list)
    selectors: list = field(default_factory=list)


@dataclass
class Keyword:
    """A score-role primitive over the FTS5/BM25 substrate (not embedding-E).
    Produces a scored set just like a ScoreExpr, so it composes through explicit
    RRF fusion and carries an optional pre-filter mask, scalar weights, and
    selector barriers — a set-valued plan node alongside Op/Restrict/Fuse/Rescore."""
    text: str
    mask: object = None        # optional Mask (FTS pre-filter scope)
    weights: list = field(default_factory=list)
    selectors: list = field(default_factory=list)
    unit: str = "bm25:minmax"


@dataclass
class Fuse:
    """Explicit RRF set-union of scored sub-plans (`left ⊕ right`)."""
    parts: list                # list of sub-plans (ScoreExpr | Op | Fuse | Rescore)
    method: str = "rrf"
    weights: list = field(default_factory=list)
    selectors: list = field(default_factory=list)
    unit: str = "rrf:k60:rank0+1"


@dataclass
class Rescore:
    """Re-score a transformed set by a new query through `rescore(query, set)`."""
    query: object              # ScoreExpr — the new query vector
    inner: object              # the scope-transformer whose set is rescored
    weights: list = field(default_factory=list)
    selectors: list = field(default_factory=list)
    unit: str = "inner_product:f32"


@dataclass
class Top:
    n: int
    expr: object       # ScoreExpr | Op | Restrict | Fuse | Rescore
    by: str = "score"  # selection key: 'score' (top-n) | 'time:asc' (earliest) | 'time:desc' (latest)
