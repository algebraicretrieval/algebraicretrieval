"""Class B operators — scope-transformers. The reason the planner exists.

These do what a single backend (score_candidates / compute_on_candidates / one
vec_ops call) structurally cannot: enlarge or re-shape the candidate SET
mid-pipeline. Each takes a scored seed Scope and returns a transformed Scope.

Data-backed against the claude_code substrate:
  expand   — graph: seed chunks -> their sessions -> sibling chunks (the
             surrounding conversation), score propagated by alpha.
  coedited — identity: seed chunks -> file_uuids they touch -> chunks touching
             the same files (provenance), score propagated.
  surprise — numpy: re-rank by productive mismatch — demote chunks aligned with
             a community's centroid (KL/contrastive intuition).

flavor_by(cell) needs another cell's materialized attractors (cross-cell); it is
a forward edge until attractor packs exist — see specs/operator-catalog.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scope import Scope, Ctx  # noqa: E402


def _fetch_pairs(db, sql_tmpl, keys, chunk=900):
    """Run `sql_tmpl % placeholders` over keys in chunks (SQLite var limit)."""
    out, keys = [], list(keys)
    for i in range(0, len(keys), chunk):
        part = keys[i:i + chunk]
        ph = ",".join("?" * len(part))
        out += db.execute(sql_tmpl % ph, part).fetchall()
    return out


# ── the general key-traverse: chunk -> shared key -> chunks ──────────────────
# expand / coedited / same_repo are the SAME walk over different (chunk_id, key)
# edge tables. One core; they are named instances (the decomposition principle).
# edge_table/key_col are internal constants, never user input — no injection.

_TRAVERSE = {
    "expand":   ("_edges_source",        "source_id", "sessions"),
    "coedited": ("_edges_file_identity", "file_uuid", "files"),
    "same_repo": ("_edges_repo_identity", "repo_root", "repos"),
}


def _traverse_key(seed: Scope, edge_table: str, key_col: str, label: str, args: dict,
                  ctx: Ctx, cap: int = 3000, max_keys: int = 300) -> Scope:
    if not seed.ids:
        return seed
    alpha = float(args.get("alpha", 0.5))
    seed_score = {i: float(s) for i, s in zip(seed.ids, seed.scores)}

    key_best: dict = {}
    for c, k in _fetch_pairs(
            ctx.db, f"SELECT chunk_id, {key_col} FROM {edge_table} WHERE chunk_id IN (%s)",
            seed.ids):
        sc = seed_score.get(c, 0.0)
        if sc > key_best.get(k, -1e9):
            key_best[k] = sc
    if not key_best:
        return seed

    keys = sorted(key_best, key=lambda k: -key_best[k])[:max_keys]
    out = dict(seed_score)
    added = 0
    for c, k in _fetch_pairs(
            ctx.db, f"SELECT chunk_id, {key_col} FROM {edge_table} WHERE {key_col} IN (%s)",
            keys):
        if c in out or c not in ctx.id_to_idx:
            continue
        out[c] = alpha * key_best.get(k, 0.0)
        added += 1
        if added >= cap:
            ctx.log.append(f"{label}: capped neighbors at {cap}")
            break
    ids = list(out)
    ctx.log.append(f"traverse[{label}]: {len(seed.ids)} seeds -> {len(ids)} chunks "
                   f"across {len(keys)} {label.split('->')[-1] if '->' in label else 'keys'}")
    return Scope(ids=ids, scores=np.array([out[i] for i in ids], dtype=np.float64))


def expand_core(seed, args, ctx):
    t, k, _ = _TRAVERSE["expand"]
    return _traverse_key(seed, t, k, "expand", args, ctx)


def coedited_core(seed, args, ctx):
    t, k, _ = _TRAVERSE["coedited"]
    return _traverse_key(seed, t, k, "coedited", args, ctx)


def same_repo_core(seed, args, ctx):
    t, k, _ = _TRAVERSE["same_repo"]
    return _traverse_key(seed, t, k, "same_repo", args, ctx)


def window_core(seed: Scope, args: dict, ctx: Ctx, cap: int = 3000) -> Scope:
    """Sequential scope-transform: pull chunks within ±W positions of each seed in
    the SAME source — the surrounding turns (sessions) or adjacent sections
    (docpac). Score decays by distance (alpha^|Δposition|). Uses the UNIVERSAL
    `_edges_source` (chunk→source, position), not a cell-specific column, so it
    works on every cell family."""
    if not seed.ids:
        return seed
    w = int(args.get("w", 3))
    alpha = float(args.get("alpha", 0.6))
    seed_score = {i: float(s) for i, s in zip(seed.ids, seed.scores)}

    meta = {r[0]: (r[1], r[2]) for r in _fetch_pairs(
        ctx.db, "SELECT chunk_id, source_id, position FROM _edges_source WHERE chunk_id IN (%s)",
        seed.ids)}
    out = dict(seed_score)
    added = 0
    for sid, (src, pos) in meta.items():
        if src is None or pos is None:
            continue
        rows = ctx.db.execute(
            "SELECT chunk_id, position FROM _edges_source WHERE source_id=? AND position BETWEEN ? AND ?",
            (src, pos - w, pos + w)).fetchall()
        base = seed_score.get(sid, 0.0)
        for cid, cpos in rows:
            if cid in out or cid not in ctx.id_to_idx:
                continue
            out[cid] = alpha ** abs(int(cpos) - int(pos)) * base
            added += 1
            if added >= cap:
                break
        if added >= cap:
            ctx.log.append(f"window: capped at {cap}")
            break
    ids = list(out)
    ctx.log.append(f"window[seq ±{w}]: {len(seed.ids)} seeds -> {len(ids)} chunks")
    return Scope(ids=ids, scores=np.array([out[i] for i in ids], dtype=np.float64))


def surprise_core(seed: Scope, args: dict, ctx: Ctx, beta: float = 0.7) -> Scope:
    """numpy re-rank by productive mismatch: demote seeds aligned with a
    community's centroid. score' = score - beta * (E[id] @ C_community). High
    score = relevant to the query but NOT typical of the community."""
    community = str(args.get("community", "")).strip()
    if not seed.ids:
        return seed
    srcs = [r[0] for r in ctx.db.execute(
        "SELECT source_id FROM _enrich_source_graph "
        "WHERE CAST(community_id AS TEXT)=? OR community_label=?",
        (community, community)).fetchall()]
    if not srcs:
        raise ValueError(f"surprise: no community '{community}' in _enrich_source_graph")
    chunk_rows = _fetch_pairs(
        ctx.db, "SELECT chunk_id FROM _edges_source WHERE source_id IN (%s)", srcs)
    cidx = [ctx.id_to_idx[c[0]] for c in chunk_rows if c[0] in ctx.id_to_idx]
    if not cidx:
        raise ValueError(f"surprise: community '{community}' has no embedded chunks")
    C = ctx.E[np.array(cidx, dtype=np.int64)].mean(axis=0)
    n = np.linalg.norm(C)
    if n:
        C = C / n
    sidx = np.array([ctx.id_to_idx[i] for i in seed.ids], dtype=np.int64)
    align = (ctx.E[sidx] @ C.astype(np.float32)).astype(np.float64)
    ctx.log.append(f"surprise[numpy]: demote alignment to community '{community}' "
                   f"({len(cidx)} chunks)")
    return Scope(ids=list(seed.ids), scores=seed.scores - beta * align)
