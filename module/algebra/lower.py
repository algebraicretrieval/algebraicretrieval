"""Lower IR → execution. The backend: compose query vectors (numpy), build masks
(SQL), run the folded matmul(s), apply weights, select. Reuses engine primitives.

Emits a plan log and a matmul count — the evidence that the planner did its job
(linear collapse: K score terms → 1 matmul; data-dependent terms → a barrier).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir import Top, ScoreExpr, ScoreTerm, QTerm  # noqa: E402
from rewrite import collapse  # noqa: E402
from score_semantics import stable_top_indices  # noqa: E402


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


_TS_CACHE: dict = {}


def _timestamps(db, ids):
    """Epoch-second array aligned with `ids` (0 where missing). Memoized per cell."""
    path = None
    for r in db.execute("PRAGMA database_list").fetchall():
        if r[1] == "main":
            path = r[2]
            break
    key = path or id(ids)
    if key not in _TS_CACHE:
        tmap = dict(db.execute(
            "SELECT id, timestamp FROM _raw_chunks WHERE timestamp IS NOT NULL").fetchall())
        _TS_CACHE[key] = np.array([float(tmap.get(i, 0) or 0) for i in ids], dtype=np.float64)
    return _TS_CACHE[key]


def _apply_weights(scores, weights, ctx):
    """Element-wise weight modulation. decay(n): 0.5^(age_days/n) half-life."""
    now = time.time()
    for w in weights:
        if w.kind == "decay":
            ts = _timestamps(ctx.db, ctx.ids)
            age = np.maximum(0.0, (now - ts) / 86400.0)
            wv = np.where(ts > 0, 0.5 ** (age / max(1, w.arg)), 1.0)
            scores = scores * wv
            ctx.log.append(f"weight: decay({w.arg}d half-life)")
    return scores


def _mask_ids(db, preds):
    """Compile mask predicates → SQL pre-filter → set of eligible ids."""
    where = []
    for p in preds:
        if p[0] == "recent":
            where.append(f"timestamp >= CAST(strftime('%s','now','-{int(p[1])} days') AS INTEGER)")
        elif p[0] == "type":
            v = str(p[1]).replace("'", "''")
            where.append(f"type = '{v}'")
        elif p[0] == "minlen":
            where.append(f"length(content) >= {int(p[1])}")
    if not where:
        return None
    sql = "SELECT id FROM chunks WHERE " + " AND ".join(where)
    return {r[0] for r in db.execute(sql).fetchall()}


class _Ctx:
    def __init__(self, embed_fn, E, ids, db):
        self.embed_fn, self.E, self.ids, self.db = embed_fn, E, ids, db
        self.idx = {x: i for i, x in enumerate(ids)}
        self.matmuls = 0
        self.log = []


def _resolve_qterm(qt: QTerm, ctx: _Ctx):
    """Return a concrete query vector for one qterm; recurse (barrier) on nested Top."""
    if qt.kind == "similar":
        return _norm(ctx.embed_fn(qt.arg)) * qt.coef
    # centroid
    arg = qt.arg
    if isinstance(arg, Top):
        ctx.log.append(f"BARRIER: evaluate nested {_fmt(arg.expr)} -> top {arg.n} (data-dependent)")
        inner = _eval(arg.expr, ctx, fold=True, n=arg.n)
        arg = [i for i, _ in inner]
    elif arg == "prev":
        raise ValueError("centroid(prev) only valid inside a '>' sequence; use centroid(top(...))")
    rows = [ctx.idx[i] for i in arg if i in ctx.idx]
    if not rows:
        return np.zeros(ctx.E.shape[1], dtype=np.float32)
    return _norm(ctx.E[rows].mean(axis=0)) * qt.coef


def _score_term(st: ScoreTerm, ctx: _Ctx):
    """One score term → (coef, score_array). One matmul."""
    q = None
    for qt in st.q:
        v = _resolve_qterm(qt, ctx)
        q = v if q is None else q + v
    # Primitive qterms are normalized individually in _resolve_qterm. The
    # composed vector retains magnitude so collapse preserves scores/thresholds.
    ctx.matmuls += 1
    return st.coef, ctx.E @ q


def _eval(expr: ScoreExpr, ctx: _Ctx, fold: bool, n: int):
    e = collapse(expr) if fold else expr
    scores = None
    for st in e.terms:
        c, s = _score_term(st, ctx)
        scores = c * s if scores is None else scores + c * s

    # Element-wise weights modulate scores. Selectors are explicit barriers.
    div = next((s for s in e.selectors if s.kind == "diverse"), None)
    ddp = next((s for s in e.selectors if s.kind == "dedup"), None)
    threshold = next((s for s in e.selectors if s.kind == "threshold"), None)
    if e.weights:
        scores = _apply_weights(scores, e.weights, ctx)
    if threshold is not None:
        scores = np.where(scores >= threshold.arg, scores, -np.inf)
    # mask
    if e.mask:
        keep = _mask_ids(ctx.db, e.mask.preds)
        if keep is not None:
            m = np.array([ctx.ids[i] in keep for i in range(len(ctx.ids))])
            scores = np.where(m, scores, -np.inf)
    if ddp is not None:
        ctx.log.append(f"select: dedup(>{ddp.arg} cosine) — drop near-identical content")
        return _dedup_select(scores, ctx.E, ctx.ids, n, ddp.arg)
    if div is not None:
        ctx.log.append(f"select: MMR diverse(λ={div.arg}) — sequential, runs last")
        return _mmr_select(scores, ctx.E, ctx.ids, n, div.arg)
    order = stable_top_indices(ctx.ids, scores, n)
    return [(ctx.ids[int(i)], float(scores[int(i)])) for i in order]


def _dedup_select(scores, E, ids, n, thresh):
    """Top-N after dropping near-identical-content results (cosine > thresh with an
    already-kept item). The fix for clone-collapse that MMR can't handle —
    you can't diversify clones, but you can remove them."""
    P = min(len(ids), max(n * 40, 500))
    pool = stable_top_indices(ids, scores, P)
    kept = []
    for j in pool:
        v = E[int(j)]
        if any(float(v @ E[int(k)]) > thresh for k in kept):
            continue
        kept.append(int(j))
        if len(kept) >= n:
            break
    return [(ids[i], float(scores[i])) for i in kept]


def _mmr_select(scores, E, ids, n, lam):
    """Maximal Marginal Relevance: greedily pick high-score items that are NOT
    redundant with already-picked ones. The sequential op the laws can't fold —
    and the antidote to PRF collapse (returning N near-identical chunks)."""
    P = min(len(ids), max(n * 20, 200))
    pool = stable_top_indices(ids, scores, P)
    if len(pool) == 0:
        return []
    V = E[pool]
    sims = V @ V.T
    rel = scores[pool]
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)   # normalize for the tradeoff
    chosen, picked = [], np.zeros(len(pool), bool)
    for _ in range(min(n, len(pool))):
        best, bestval = -1, -1e18
        red = sims[:, picked].max(axis=1) if picked.any() else np.zeros(len(pool))
        for j in range(len(pool)):
            if picked[j]:
                continue
            val = lam * rel[j] - (1 - lam) * red[j]
            if val > bestval:
                bestval, best = val, j
        picked[best] = True
        chosen.append(pool[best])
    return [(ids[int(i)], float(scores[int(i)])) for i in chosen]


def _fmt(expr: ScoreExpr) -> str:
    parts = []
    for st in expr.terms:
        qs = "+".join(f"{qt.coef:g}·{qt.kind}({qt.arg if not hasattr(qt.arg,'expr') else '…'})" for qt in st.q)
        parts.append(f"{st.coef:g}·E@({qs})")
    return " + ".join(parts)


def run(top: Top, embed_fn, E, ids, db, fold: bool = True):
    """Execute. Returns (results, plan_log, matmuls)."""
    ctx = _Ctx(embed_fn, E, ids, db)
    folded = collapse(top.expr) if fold else top.expr
    ctx.log.insert(0, f"plan: {_fmt(folded)}  | top {top.n}  | "
                      f"{len(folded.terms)} matmul(s){' [collapsed]' if fold else ' [unfolded]'}")
    res = _eval(top.expr, ctx, fold=fold, n=top.n)
    return res, ctx.log, ctx.matmuls
