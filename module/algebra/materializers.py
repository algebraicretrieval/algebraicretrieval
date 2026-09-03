"""M0 — operator cores + descriptors.

The cores are thin: a scorer that WRAPS flex's score_candidates (inheriting its
matrix-slice pushdown — the bug fix), and a SQL mask that runs a WHERE. Each
core is Scope -> Scope. OpDesc.substrate is a CODEGEN TARGET (which backend
emits the opcode), not a domain boundary.

M0 ships the scorer + mask only. Weights (decay), selection (diverse/dedup), and
structural enrichments (pagerank via compute_on_candidates) arrive in later
phases; they reuse the same Scope contract.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scope import Scope, Ctx  # noqa: E402
from score_semantics import score_inner_product, stable_top_indices  # noqa: E402

from flex.retrieve.score import score_candidates  # the backend ISA (numpy)


@dataclass
class OpDesc:
    substrate: str            # codegen target: 'sql' | 'numpy' | 'graph'
    barrier: bool = False     # selection/data-dependent — planner can't fuse across it
    consumes_score: bool = False


# ── SQL backend ───────────────────────────────────────────────────────────────

def _where(preds) -> str:
    parts = []
    for p in preds:
        if p[0] == "type":
            v = str(p[1]).replace("'", "''")
            parts.append(f"type = '{v}'")
        elif p[0] == "recent":
            parts.append(
                f"timestamp >= CAST(strftime('%s','now','-{int(p[1])} days') AS INTEGER)")
        elif p[0] == "minlen":
            parts.append(f"length(content) >= {int(p[1])}")
        elif p[0] == "community":          # graph: source community (join enrichment)
            parts.append(
                f"id IN (SELECT es.chunk_id FROM _edges_source es "
                f"JOIN _enrich_source_graph g ON es.source_id=g.source_id "
                f"WHERE g.community_id={int(p[1])})")
        elif p[0] == "hub":                # graph: chunks from hub sessions
            parts.append(
                "id IN (SELECT es.chunk_id FROM _edges_source es "
                "JOIN _enrich_source_graph g ON es.source_id=g.source_id "
                "WHERE g.is_hub=1)")
        elif p[0] == "repo":               # identity: chunks touching a repo
            v = str(p[1]).replace("'", "''")
            parts.append(f"id IN (SELECT chunk_id FROM _edges_repo_identity "
                         f"WHERE repo_root LIKE '%{v}%')")
        elif p[0] == "eq":                 # generic col=val — column resolved from @orient,
            col = str(p[1])                # not a hardcoded vocabulary (doc_type/section/etc)
            if not col.isidentifier():     # guard the identifier against injection
                raise ValueError(f"eq: unsafe column name {col!r}")
            v = str(p[2]).replace("'", "''")
            parts.append(f"{col} = '{v}'")
        elif p[0] == "raw":                # escape: a raw predicate (authorizer-guarded on live path)
            parts.append(f"({p[1]})")
        elif p[0] == "contract":
            parts.append(f"mask({p[1]})")
        else:
            raise ValueError(f"mask: unknown predicate {p!r}")
    return " AND ".join(parts) if parts else "1=1"


def _has_table(db, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _pred_idsql(p, db) -> str:
    """One predicate → a `SELECT id …` id-set from the CHEAPEST source.

    Resolving a mask through the `chunks` view costs ~640ms on a large cell
    (the view CASE-computes `type` and materializes the full 7-way join, so
    `WHERE type=…` can't use idx_types_message_type). Routing to the base table
    is ~4ms. Each branch returns the SAME id set the view-predicate would —
    `type` mirrors the view's file-body-precedence CASE — so the result is
    faithful (verified in test_mask_pushdown.py). Falls back to the view for
    predicates with no cheap base form (`eq` on arbitrary columns, `raw`, and
    `type` on cells without `_types_message`, e.g. docpac)."""
    # Bare base-table id-sets (index seek, no per-row join). These can include a
    # few "orphan" ids present in _types_*/_edges_* but not in _raw_chunks, which
    # the view (anchored `FROM _raw_chunks`) omits — HARMLESS: every consumer
    # (score_core's pre_filter ∩ matrix, keyword's BM25 pool, centroid's
    # id_to_idx) intersects against the embedding matrix (= _raw_chunks ids), so
    # orphans are dropped at scoring. Verified in test_mask_pushdown.py that the
    # scored universe is identical to the view and the extras are exactly orphans.
    # (Message vs file-body ids are disjoint by construction — `:fb:` suffix — so
    # no file-body-precedence guard is needed.)
    k = p[0]
    if k == "type" and _has_table(db, "_types_message"):
        v = str(p[1]).replace("'", "''")
        if v == "file":                       # view: WHEN fb NOT NULL THEN 'file'
            return "SELECT chunk_id AS id FROM _types_file_body"
        return f"SELECT chunk_id AS id FROM _types_message WHERE type='{v}'"
    if k == "recent":                         # _raw_chunks column — no orphans
        return (f"SELECT id FROM _raw_chunks WHERE timestamp >= "
                f"CAST(strftime('%s','now','-{int(p[1])} days') AS INTEGER)")
    if k == "minlen":
        return f"SELECT id FROM _raw_chunks WHERE length(content) >= {int(p[1])}"
    if k == "community":
        return (f"SELECT es.chunk_id AS id FROM _edges_source es "
                f"JOIN _enrich_source_graph g ON es.source_id=g.source_id "
                f"WHERE g.community_id={int(p[1])}")
    if k == "hub":
        return ("SELECT es.chunk_id AS id FROM _edges_source es "
                "JOIN _enrich_source_graph g ON es.source_id=g.source_id WHERE g.is_hub=1")
    if k == "repo":
        v = str(p[1]).replace("'", "''")
        return f"SELECT chunk_id AS id FROM _edges_repo_identity WHERE repo_root LIKE '%{v}%'"
    # eq / raw / type-without-_types_message → faithful view predicate
    return "SELECT id FROM chunks WHERE " + _where([p])


def _mask_idsql(preds, db) -> str:
    """Full mask id-query: INTERSECT the per-predicate id-sets (= AND of preds,
    faithful to `SELECT id FROM chunks WHERE <preds ANDed>`), each from its
    cheapest base table. No preds → all chunks."""
    if not preds:
        return "SELECT id FROM _raw_chunks"
    return " INTERSECT ".join(f"SELECT id FROM ({_pred_idsql(p, db)})" for p in preds)


def mask_core(scope: Scope, preds, ctx: Ctx) -> Scope:
    """SQL backend: native and contract id-sets, intersected with incoming scope."""
    contract_preds = [predicate for predicate in preds if predicate[0] == "contract"]
    native_preds = [predicate for predicate in preds if predicate[0] != "contract"]
    sets = []
    if native_preds:
        sql = _mask_idsql(native_preds, ctx.db)
        sets.append({str(row[0]) for row in ctx.db.execute(sql).fetchall()})
    for predicate in contract_preds:
        if ctx.contract is None:
            raise ValueError("contract mask used without a contract runtime")
        sets.append(set(ctx.contract.mask_ids(ctx.db, predicate[1])))
    allowed = set(ctx.ids) if not sets else set.intersection(*sets)
    source_ids = ctx.ids if scope.ids is None else scope.ids
    keep = [index for index, identity in enumerate(source_ids) if identity in allowed]
    ids = [source_ids[index] for index in keep]
    scores = scope.scores[keep] if scope.scores is not None else None
    signal = scope.signal[keep] if scope.signal is not None else None
    ctx.log.append(f"restrict[sql]: {_where(preds)} -> {len(ids)} ids")
    return Scope(ids=ids, scores=scores, signal=signal)


MASK_DESC = OpDesc(substrate="sql", barrier=False, consumes_score=False)


# ── numpy backend ───────────────────────────────────────────────────────────────

def score_core(scope: Scope, q: np.ndarray, n: int, ctx: Ctx) -> Scope:
    """numpy backend: wrap score_candidates with the incoming scope as pre_filter.

    The matmul scores ONLY the masked subset (pushdown). matmul_rows records the
    rows that actually entered the multiply — the bug-fix evidence.
    """
    pf = set(scope.ids) if scope.ids is not None else None
    if pf is not None:
        ctx.matmul_rows = sum(1 for i in pf if i in ctx.id_to_idx)
    else:
        ctx.matmul_rows = len(ctx.ids)

    res = score_candidates(
        matrix=ctx.E,
        ids=ctx.ids,
        id_to_idx=ctx.id_to_idx,
        query_vec=q,
        pre_filter_ids=pf,
        limit=n,
        oversample=max(n * 3, 200),
    )
    ctx.log.append(f"score[numpy]: {ctx.matmul_rows} rows @ q -> top {len(res)}")
    return Scope(
        ids=[r["id"] for r in res],
        scores=np.array([r["score"] for r in res], dtype=np.float64),
        # score_candidates already computes _z (corpus-null z over the pre-filtered
        # pool) per result and we surface it here (M0 path; run_m0 is currently unused).
        signal=np.array([(r.get("_z") if r.get("_z") is not None else 0.0)
                         for r in res], dtype=np.float64),
    )


SCORE_DESC = OpDesc(substrate="numpy", barrier=False, consumes_score=False)


# ── multi-stage numpy cores (Phase 1) ──────────────────────────────────────────
# Single-instruction lowering (M0) fuses score+select into one score_candidates
# call. A multi-stage chain needs them SEPARATE: score the full subset (no
# selection) → modulate with weights → select last. This is the ordered chain.

def score_all_core(scope: Scope, q: np.ndarray, ctx: Ctx) -> Scope:
    """numpy backend: E_sub @ q over the masked subset, NO selection. Returns the
    full scored pool so a downstream weight can modulate before top-n. Pushdown:
    only the subset rows enter the matmul."""
    if scope.ids is not None:
        idx = np.fromiter(
            (ctx.id_to_idx[i] for i in scope.ids if i in ctx.id_to_idx),
            dtype=np.int64)
    else:
        idx = np.arange(len(ctx.ids), dtype=np.int64)
    ctx.matmul_rows = int(idx.size)
    if idx.size == 0:
        return Scope(ids=[], scores=np.array([], dtype=np.float64),
                     signal=np.array([], dtype=np.float64))
    s = score_inner_product(ctx.E[idx], q).astype(np.float64)
    # signal = corpus-null z: σ above what a random doc in THIS scored pool scores
    # under q. The pool IS the masked subset (pushdown), so its moments are the
    # subset-null for free — the right frame for homogeneous cells, no precompute.
    # Distinctiveness/presence, NOT relevance. Carried pre-decay through selection.
    mu = float(s.mean())
    sd = float(s.std())
    sig = (s - mu) / sd if sd > 1e-9 else np.zeros_like(s)
    sub_ids = [ctx.ids[int(i)] for i in idx]
    ctx.log.append(f"score_all[numpy]: {idx.size} rows @ q raw inner product (no selection)")
    return Scope(ids=sub_ids, scores=s, signal=sig)


SCORE_ALL_DESC = OpDesc(substrate="numpy", barrier=False, consumes_score=False)


_TS_CACHE: dict = {}


def _timestamps(ctx: Ctx) -> dict:
    """{id: epoch_seconds} from _raw_chunks, memoized per cell connection."""
    path = None
    for r in ctx.db.execute("PRAGMA database_list").fetchall():
        if r[1] == "main":
            path = r[2]
            break
    key = path or id(ctx.ids)
    if key not in _TS_CACHE:
        tmap = dict(ctx.db.execute(
            "SELECT id, timestamp FROM _raw_chunks WHERE timestamp IS NOT NULL").fetchall())
        _TS_CACHE[key] = {i: float(tmap.get(i, 0) or 0) for i in ctx.ids}
    return _TS_CACHE[key]


def decay_core(scope: Scope, days: int, ctx: Ctx) -> Scope:
    """numpy weight: scores *= 0.5^(age_days / N), element-wise. The algebra's
    decay semantics (NOT vec_ops's 1/(1+x) — kept consistent with the module's
    existing test_weights). Reads scores; does not wrap score_candidates."""
    if not scope.scored:
        raise ValueError("decay() needs a scored scope (no scorer upstream)")
    import time as _time
    ts = _timestamps(ctx)
    now = _time.time()
    out = scope.scores.astype(np.float64).copy()
    for k, _id in enumerate(scope.ids):
        t = ts.get(_id, 0.0)
        if t > 0:
            age = max(0.0, (now - t) / 86400.0)
            out[k] *= 0.5 ** (age / max(1, days))
    ctx.log.append(f"decay[numpy]: 0.5^(age/{days})")
    # signal is pre-decay (semantic distinctiveness); decay reweights rank, not presence
    return Scope(ids=scope.ids, scores=out, signal=scope.signal)


DECAY_DESC = OpDesc(substrate="numpy", barrier=False, consumes_score=True)


def contract_weight_core(scope: Scope, symbol: str, ctx: Ctx) -> Scope:
    """Multiply scores by a contract-declared scalar relation aligned on identity."""
    if not scope.scored:
        raise ValueError("contract weight needs a scored scope")
    if ctx.contract is None:
        raise ValueError("contract weight used without a contract runtime")
    values = ctx.contract.weight_values(ctx.db, symbol, scope.ids)
    ctx.log.append(f"weight[sql→numpy]: {symbol} over {len(scope.ids)} ids")
    return Scope(ids=scope.ids, scores=scope.scores * values, signal=scope.signal)


def select_core(scope: Scope, n: int, ctx: Ctx) -> Scope:
    """numpy selection barrier: top-n by score. Runs last."""
    if not scope.scored:
        raise ValueError("top() needs a scored scope")
    order = stable_top_indices(scope.ids, scope.scores, n)
    sig = scope.signal[order] if scope.signal is not None else None
    ctx.log.append(f"select[numpy]: top {n}")
    return Scope(ids=[scope.ids[int(i)] for i in order], scores=scope.scores[order],
                 signal=sig)


SELECT_DESC = OpDesc(substrate="numpy", barrier=True, consumes_score=True)


def keyword_core(text: str, mask_preds, n: int, ctx: Ctx) -> Scope:
    """Score-role primitive over FTS5/BM25 (not embedding-E). Returns a scored
    Scope (min-max normalized BM25 to [0,1], matching flex keyword()). Optional
    mask_preds restrict the FTS scope (pushed into the FTS JOIN). Results are
    filtered to embedded ids so the set composes with vector ops downstream."""
    import re
    import sqlite3
    import uuid
    from flex.retrieve.keyword import sanitize_fts5   # flex's public sanitizer

    db = ctx.db
    sanitized = sanitize_fts5(text)

    # Scope is applied as a Python post-filter on an enlarged FTS pool — no temp
    # table (the live search connection runs a read-only authorizer that forbids
    # CREATE/INSERT). A masked keyword pulls a bigger global BM25 pool then keeps
    # the in-scope hits; very small masks may under-fill (documented tradeoff).
    mask_ids = None
    if mask_preds:
        mask_ids = {r[0] for r in db.execute(_mask_idsql(mask_preds, db)).fetchall()}
        if not mask_ids:
            return Scope(ids=[], scores=np.array([], dtype=np.float64))
    limit = max(n * 50, 3000) if mask_ids is not None else max(n, 200)

    fts_sql = ("SELECT c.id, -bm25(chunks_fts) AS rank FROM chunks_fts "
               "JOIN _raw_chunks c ON chunks_fts.rowid = c.rowid "
               "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?")

    try:
        rows = db.execute(fts_sql, (sanitized, limit)).fetchall()
        if not rows and " " in sanitized and "OR" not in sanitized:
            rows = db.execute(fts_sql, (" OR ".join(sanitized.split()), limit)).fetchall()
    except sqlite3.OperationalError:
        words = [w for w in re.sub(r"[^\w\s]", "", text).split() if len(w) > 1]
        esc = " OR ".join(f'"{w}"' for w in words) or '""'
        rows = db.execute(fts_sql, (esc, limit)).fetchall()

    rows = [(r[0], float(r[1])) for r in rows
            if r[0] in ctx.id_to_idx and (mask_ids is None or r[0] in mask_ids)]
    if not rows:
        ctx.log.append(f"keyword[fts5]: '{text}' -> 0 hits")
        return Scope(ids=[], scores=np.array([], dtype=np.float64))
    ranks = np.array([r[1] for r in rows], dtype=np.float64)
    lo, hi = ranks.min(), ranks.max()
    norm = (ranks - lo) / (hi - lo) if hi > lo else np.full(len(ranks), 0.5)
    ctx.log.append(f"keyword[fts5]: '{text}' -> {len(rows)} hits (bm25 normalized)")
    return Scope(ids=[r[0] for r in rows], scores=norm)


def mmr_core(scope: Scope, n: int, lam: float, ctx: Ctx) -> Scope:
    """numpy selection barrier: MMR diversity. Pools the top by score, then
    greedily picks relevance minus redundancy. Mirrors lower._mmr_select."""
    if not scope.scored:
        raise ValueError("diverse() needs a scored scope")
    P = min(len(scope.ids), max(n * 20, 200))
    order = stable_top_indices(scope.ids, scope.scores, P)
    pool_ids = [scope.ids[int(i)] for i in order]
    rel = scope.scores[order].astype(np.float64)
    sig_pool = scope.signal[order] if scope.signal is not None else None
    idx = np.array([ctx.id_to_idx[i] for i in pool_ids], dtype=np.int64)
    V = ctx.E[idx]
    sims = V @ V.T
    m = len(pool_ids)
    rn = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
    chosen, picked = [], np.zeros(m, dtype=bool)
    for _ in range(min(n, m)):
        red = sims[:, picked].max(axis=1) if picked.any() else np.zeros(m)
        best, bestval = -1, -1e18
        for j in range(m):
            if picked[j]:
                continue
            val = lam * rn[j] - (1.0 - lam) * red[j]
            if val > bestval:
                bestval, best = val, j
        picked[best] = True
        chosen.append(best)
    ctx.log.append(f"select[numpy]: MMR diverse(λ={lam}) over pool {m}")
    return Scope(ids=[pool_ids[c] for c in chosen],
                 scores=np.array([rel[c] for c in chosen], dtype=np.float64),
                 signal=(np.array([sig_pool[c] for c in chosen], dtype=np.float64)
                         if sig_pool is not None else None))


def dedup_core(scope: Scope, n: int, thresh: float, ctx: Ctx) -> Scope:
    """numpy selection barrier: drop near-identical content (cosine > thresh with
    a kept item). Mirrors lower._dedup_select — removes clones MMR can't spread."""
    if not scope.scored:
        raise ValueError("dedup() needs a scored scope")
    P = min(len(scope.ids), max(n * 40, 500))
    order = stable_top_indices(scope.ids, scope.scores, P)
    kept_idx, kept_vecs = [], []
    for o in order:
        gi = ctx.id_to_idx[scope.ids[int(o)]]
        v = ctx.E[gi]
        if any(float(v @ kv) > thresh for kv in kept_vecs):
            continue
        kept_idx.append(int(o))
        kept_vecs.append(v)
        if len(kept_idx) >= n:
            break
    ctx.log.append(f"select[numpy]: dedup(>{thresh}) over pool {len(order)}")
    return Scope(ids=[scope.ids[k] for k in kept_idx],
                 scores=scope.scores[kept_idx],
                 signal=scope.signal[kept_idx] if scope.signal is not None else None)


def time_select_core(scope: Scope, n: int, ctx: Ctx, *, ascending: bool) -> Scope:
    """numpy selection barrier: re-order a relevance-scored Scope by SOURCE TIME.

    The time axis's missing selection corner (it has a mask `recent` and a weight
    `decay`, but no selection until now). Mirrors dedup_core's shape: pool the
    top-P by RELEVANCE first, then re-select within that pool by time — ascending
    = earliest (oldest), descending = latest (newest). The relevance pool is what
    makes earliest(similar(genesis)) mean "the oldest RELEVANT chunk", not the
    oldest chunk in the cell. Reuses the memoized _timestamps map; ids with no
    usable timestamp (<=0) are dropped so epoch-0 nulls can't fake-win 'oldest'.
    Scores stay attached (the resolvable-array invariant — this only re-orders)."""
    if not scope.scored:
        raise ValueError("earliest()/latest() needs a scored scope (no scorer upstream)")
    ts = _timestamps(ctx)
    P = min(len(scope.ids), max(n * 40, 500))
    order = stable_top_indices(scope.ids, scope.scores, P)      # top-P by relevance
    pool = [(int(i), ts.get(scope.ids[int(i)], 0.0)) for i in order]
    pool = [(i, t) for i, t in pool if t > 0]                   # drop null timestamps
    pool.sort(key=(
        (lambda item: (item[1], str(scope.ids[item[0]])))
        if ascending else
        (lambda item: (-item[1], str(scope.ids[item[0]])))
    ))
    keep = [i for i, _ in pool[:n]]
    ctx.log.append(
        f"select[numpy]: {'earliest' if ascending else 'latest'} {n} by source time "
        f"(relevance pool {P}, {len(pool)} timestamped)")
    return Scope(ids=[scope.ids[k] for k in keep],
                 scores=scope.scores[keep],
                 signal=scope.signal[keep] if scope.signal is not None else None)
