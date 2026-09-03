"""M0 — single-instruction lowering.

The degenerate case of the codegen: when an expression is Class-A only
(masks + one composed similarity score + top), it lowers to ONE numpy backend
call against the masked subset. This is the "fast path" — really the compiler
recognizing a single-instruction program — and it gives faithfulness for free
(single-stage IS vec_ops's own score_candidates).

Later phases add the ordered-chain engine for multi-stage / scope-transform
expressions; M0 proves the spine: Scope cable, pushdown, faithfulness.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scope import Scope, Ctx          # noqa: E402
from materializers import (           # noqa: E402
    mask_core, score_core, score_all_core, decay_core, contract_weight_core, select_core,
    mmr_core, dedup_core, time_select_core)
from engine import compose            # noqa: E402
from score_semantics import normalize_primitive  # noqa: E402
from rewrite import collapse          # noqa: E402
from ir import Top, Op, Restrict, Fuse, Rescore, Keyword, Mask  # noqa: E402
from operators import (               # noqa: E402
    expand_core, coedited_core, same_repo_core, window_core, surprise_core)
from materializers import keyword_core, _where, _mask_idsql  # noqa: E402

_SEED_K = {"expand": 60, "coedited": 60, "same_repo": 60, "window": 60, "surprise": 300}
_TRANSFORMS = {
    "expand": expand_core, "coedited": coedited_core, "same_repo": same_repo_core,
    "window": window_core, "surprise": surprise_core,
}


_ZERO_Q = ("degenerate query vector — terms cancelled to ~zero "
           "(check suppression coefficients, e.g. similar(x) - similar(x))")


def _check_q(q):
    import numpy as _np
    if float(_np.linalg.norm(q)) < 1e-8:
        raise ValueError(_ZERO_Q)
    return q


def _compose_q(expr, embed_fn):
    """Collapse the linear similarity terms into one query vector (M0: similar only)."""
    folded = collapse(expr)
    terms = []
    for st in folded.terms:
        for qt in st.q:
            if qt.kind != "similar":
                raise ValueError(
                    "M0 single-instruction lowering handles only similar() terms; "
                    f"got {qt.kind}() — needs the multi-stage planner (Phase 3)")
            terms.append((float(st.coef) * float(qt.coef), qt.arg))
    return _check_q(compose(terms, embed_fn))


def _final_select(scope, top, ctx):
    """The terminal selection barrier. Time-ordered (earliest/latest) when the
    Top carries a time discriminant, else plain top-n by score. Used at BOTH
    select sites in run_chain (the set-valued-plan path and the ScoreExpr path)
    so `earliest(15, kw(x) ⊕ similar(y))` — whose inner is a Fuse — is honored."""
    if top.by == "time:asc":
        return time_select_core(scope, top.n, ctx, ascending=True)
    if top.by == "time:desc":
        return time_select_core(scope, top.n, ctx, ascending=False)
    return select_core(scope, top.n, ctx)


def is_single_instruction(top) -> bool:
    """True only for one unit primitive that vec_ops may normalize harmlessly.

    Signed, scaled, or multi-term expressions must use raw inner-product
    execution because vec_ops normalizes the final query vector.
    """
    if top.by != "score":      # time-ordered selection can't lower to vec_ops
        return False
    expr = top.expr
    if expr.weights or expr.selectors or len(expr.terms) != 1:
        return False
    term = expr.terms[0]
    if float(term.coef) != 1.0 or len(term.q) != 1:
        return False
    query = term.q[0]
    return query.kind == "similar" and float(query.coef) == 1.0


def run_m0(top, embed_fn, E, ids, db, contract=None):
    """Lower a single-instruction expression: mask (SQL) -> score (numpy, pushdown).

    Returns (pairs, ctx). pairs = [(id, score)] top-n.
    """
    ctx = Ctx(embed_fn=embed_fn, E=E, ids=ids, db=db, contract=contract)
    if not is_single_instruction(top):
        return run_chain(top, ctx), ctx
    expr = top.expr

    scope = Scope(ids=None)                       # full corpus
    if expr.mask:
        scope = mask_core(scope, expr.mask.preds, ctx)   # SQL backend
        if scope.ids == []:
            return [], ctx

    q = _compose_q(expr, embed_fn)
    out = score_core(scope, q, top.n, ctx)         # numpy backend, scores the subset
    ctx.materializations = 1
    _stash_signal(out, ctx)
    return list(zip(out.ids, out.scores.tolist())), ctx


# ── Phase 1: the general ordered-chain planner ─────────────────────────────────
# Codegen for any expression. Lowers in canonical substrate order (ordering
# semantics): mask (SQL, earliest = pushdown) → score (numpy, full subset, no
# selection) → weights (numpy, modulate) → selection (numpy barrier, last). A
# data-dependent query term (centroid(top(...)) = PRF) is a barrier: its inner
# plan runs first, its ids feed the outer query vector. Single-instruction
# expressions go through M0's fused fast path; everything else uses this.

def _centroid_vec(arg, ctx: Ctx) -> np.ndarray:
    """Resolve a centroid query term to a vector. Nested Top = PRF barrier:
    evaluate the inner plan first, then mean its result vectors."""
    if isinstance(arg, Top):
        ctx.log.append(f"BARRIER: PRF inner plan -> top {arg.n}")
        inner = run_chain(arg, ctx)               # recurse (the barrier)
        ids = [i for i, _ in inner]
    elif isinstance(arg, Mask):                   # centroid over a masked SET
        ids = [r[0] for r in ctx.db.execute(_mask_idsql(arg.preds, ctx.db)).fetchall()]
        ctx.log.append(f"centroid over mask -> {len(ids)} chunks")
    elif isinstance(arg, list):
        ids = arg
    else:
        raise ValueError("centroid(prev) is only valid inside a '>' sequence")
    rows = [ctx.id_to_idx[i] for i in ids if i in ctx.id_to_idx]
    if not rows:
        return np.zeros(ctx.E.shape[1], dtype=np.float32)
    c = ctx.E[rows].mean(axis=0)
    n = np.linalg.norm(c)
    return (c / n if n else c).astype(np.float32)


def _resolve_q(expr, ctx: Ctx) -> np.ndarray:
    """Compose normalized primitives into one unnormalized linear query."""
    folded = collapse(expr)
    q = None
    for st in folded.terms:
        for qt in st.q:
            coef = float(st.coef) * float(qt.coef)
            if qt.kind == "similar":
                v = normalize_primitive(ctx.embed_fn(qt.arg))
            elif qt.kind == "centroid":
                v = normalize_primitive(_centroid_vec(qt.arg, ctx))
            else:
                raise ValueError(f"unknown query term {qt.kind}")
            v = v * coef
            q = v if q is None else q + v
    return np.asarray(_check_q(q), dtype=np.float32)


def _threshold(scope: Scope, t: float) -> Scope:
    """Selection filter: keep scores >= t (cutoff, not top-k)."""
    if not scope.scored:
        return scope
    keep = np.where(scope.scores >= t)[0]
    sig = scope.signal[keep] if scope.signal is not None else None
    return Scope(ids=[scope.ids[int(i)] for i in keep], scores=scope.scores[keep],
                 signal=sig)


def _apply_post_stages(scope: Scope, weights, selectors, n: int, ctx: Ctx):
    """Apply scalar weights, then explicit support-changing selector barriers."""
    for weight in weights:
        if weight.kind == "decay":
            scope = decay_core(scope, weight.arg, ctx)
        elif weight.kind == "contract":
            scope = contract_weight_core(scope, weight.arg, ctx)
        else:
            raise ValueError(f"unknown weight {weight.kind}")
    selected = False
    for selector in selectors:
        if selector.kind == "threshold":
            scope = _threshold(scope, selector.arg)
        elif selector.kind == "dedup":
            scope = dedup_core(scope, n, selector.arg, ctx)
            selected = True
        elif selector.kind == "diverse":
            scope = mmr_core(scope, n, selector.arg, ctx)
            selected = True
        else:
            raise ValueError(f"unknown selector {selector.kind}")
    return scope, selected


def _seed_scope(inner, ctx: Ctx, n: int) -> Scope:
    """Score a sub-plan into a seed pool (a ScoreExpr scores; a plan node runs)."""
    if isinstance(inner, (Keyword, Op, Restrict, Fuse, Rescore)):
        scope, _ = run_plan(inner, ctx, n)
        return scope
    pairs = run_chain(Top(n, inner), ctx)
    return Scope(ids=[i for i, _ in pairs],
                 scores=np.array([s for _, s in pairs], dtype=np.float64))


def _transform(op: Op, ctx: Ctx) -> Scope:
    """The set transform: seed the inner sub-plan, then enlarge/re-shape."""
    seed = _seed_scope(op.inner, ctx, _SEED_K.get(op.kind, 60))
    core = _TRANSFORMS.get(op.kind)
    if core is None:
        raise ValueError(f"unknown scope-transformer {op.kind}")
    return core(seed, op.args, ctx)


def _run_fuse(node: Fuse, ctx: Ctx, n: int) -> Scope:
    """Set-union by reciprocal-rank fusion: run each sub-plan to a pool, merge by
    rank (scale-free), return the fused scored set."""
    pool = max(n * 5, 100)
    ranked = [[i for i, _ in run_chain(Top(pool, p), ctx)] for p in node.parts]
    K = 60
    agg: dict = {}
    for lst in ranked:
        for rank, i in enumerate(lst):
            agg[i] = agg.get(i, 0.0) + 1.0 / (K + rank + 1)
    order = sorted(agg, key=lambda i: (-agg[i], str(i)))
    ctx.log.append(f"fuse[rrf]: {len(node.parts)} sub-plans -> {len(order)} merged")
    return Scope(ids=order, scores=np.array([agg[i] for i in order], dtype=np.float64))


def _run_rescore(node: Rescore, ctx: Ctx, n: int) -> Scope:
    """Re-score a transformed SET by a new query vector (the dream sentence)."""
    setscope = _seed_scope(node.inner, ctx, max(n * 10, 200))
    q = _resolve_q(node.query, ctx)
    ctx.log.append(f"rescore: {len(setscope.ids)}-chunk set by new query")
    return score_all_core(Scope(ids=setscope.ids, scores=None), q, ctx)


def _run_keyword(node: Keyword, ctx: Ctx, n: int) -> Scope:
    preds = node.mask.preds if node.mask else None
    return keyword_core(node.text, preds, n, ctx)


def run_plan(node, ctx: Ctx, n: int):
    """Run any set-valued plan node (Keyword | Op | Restrict | Fuse | Rescore), apply its
    post-weights (decay/diverse/dedup), and return (scope, had_selection)."""
    if isinstance(node, Keyword):
        base = _run_keyword(node, ctx, n)
    elif isinstance(node, Op):
        base = _transform(node, ctx)
    elif isinstance(node, Restrict):
        inner_scope, _ = run_plan(node.inner, ctx, max(n * 10, 200))
        base = mask_core(inner_scope, node.mask.preds, ctx)
    elif isinstance(node, Fuse):
        base = _run_fuse(node, ctx, n)
    elif isinstance(node, Rescore):
        base = _run_rescore(node, ctx, n)
    else:
        raise ValueError(f"run_plan: not a plan node ({type(node).__name__})")
    return _apply_post_stages(base, node.weights, node.selectors, n, ctx)


run_op = run_plan   # back-compat alias (Op is a plan node)


def _stash_signal(scope, ctx):
    """Expose the final scope's per-id corpus-null z on ctx for the surface
    (run_records), so run_chain's RETURN stays a 2-tuple (existing consumers
    unaffected). Empty when no cosine scorer ran (kw/fuse/pure-transform)."""
    if getattr(scope, "signal", None) is not None:
        ctx.signal_by_id = {i: float(z)
                            for i, z in zip(scope.ids, scope.signal.tolist())}
    else:
        ctx.signal_by_id = {}


def run_chain(top, ctx: Ctx):
    """General codegen. Returns [(id, score)] top-n; stashes per-id signal on ctx. Threads one Scope through
    mask → score_all → weights → selection. matmul_rows reflects the scored
    subset (pushdown holds even with weights — the thing the old path couldn't do).
    A scope-transformer (Op) at the root runs its set transform, then top-n."""
    expr = top.expr

    if isinstance(expr, (Keyword, Op, Restrict, Fuse, Rescore)):   # set-valued plan node
        scope, had_selection = run_plan(expr, ctx, top.n)
        if not had_selection:                        # no diverse/dedup attached → top-n / time
            scope = _final_select(scope, top, ctx)
        ctx.materializations = 1
        _stash_signal(scope, ctx)
        return list(zip(scope.ids, scope.scores.tolist()))

    # query first (PRF barriers may run inner plans), then mask (pushdown), then
    # score the subset, then weights, then select.
    q = _resolve_q(expr, ctx)

    scope = Scope(ids=None)
    if expr.mask:
        scope = mask_core(scope, expr.mask.preds, ctx)
        if scope.ids == []:
            return []

    scope = score_all_core(scope, q, ctx)          # full scored subset, no selection

    scope, had_selection = _apply_post_stages(
        scope, expr.weights, expr.selectors, top.n, ctx
    )
    if not had_selection:
        scope = _final_select(scope, top, ctx)       # top-n by score, or earliest/latest by time
    ctx.materializations = 1
    _stash_signal(scope, ctx)
    return list(zip(scope.ids, scope.scores.tolist()))


def run(top, embed_fn, E, ids, db, contract=None):
    """Convenience entry for the plugin: build a Ctx and run the chain.
    Returns [(id, score)]. Raises NotImplementedError for diverse/dedup (Phase 3),
    which the caller routes to the legacy lower.run path."""
    ctx = Ctx(embed_fn=embed_fn, E=E, ids=ids, db=db, contract=contract)
    return run_chain(top, ctx)


def run_records(top, embed_fn, E, ids, db, contract=None):
    """Surface entry: run the chain, return (pairs, signal_by_id). pairs are the
    usual (id, score) 2-tuples; signal_by_id maps id -> corpus-null z over the FINAL
    result set (empty when no cosine scorer ran — kw/fuse/pure-transform)."""
    ctx = Ctx(embed_fn=embed_fn, E=E, ids=ids, db=db, contract=contract)
    pairs = run_chain(top, ctx)
    return pairs, (ctx.signal_by_id or {})
