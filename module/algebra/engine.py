"""Phase 0 keystone — the scoring primitive every later phase needs.

Reach the embedder and the cell's normalized embedding matrix `E` from outside
the vec_ops closure, compose an ARBITRARY query vector `q`, and score `E @ q`.

We reuse vec_ops's warm, row-normalized VectorCache and identical query embedder.
A single normalized primitive therefore produces the same cosine ranking. A
linear composition retains its magnitude and is scored by raw inner product;
that is the score-preserving semantics vec_ops's always-normalized query path
cannot provide.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_semantics import compose_query, normalize_primitive, score_inner_product, stable_top_indices

_QUERY_PREFIX = "search_query: "   # must match register_vec_ops's embed_query lambda


def cell_engine(name: str):
    """Return (db, embed_fn, E, ids) for a cell, reusing vec_ops's warm cache.

    - db: open connection with vec_ops registered (the oracle for round-trip tests)
    - embed_fn(text) -> np.ndarray: normalized query vector, vec_ops's prefix
    - E: (N, d) normalized matrix — the same object vec_ops scores against
    - ids: list[str] aligned with E rows
    """
    from flex.retrieve.execute import open_cell_for_query, _cache_state
    from flex.engine import _query_embedder_for

    db = open_cell_for_query(name)            # warms _cache_state[name], registers UDF
    state = _cache_state.get(name)
    if not state or "_raw_chunks" not in state.get("caches", {}):
        raise RuntimeError(
            f"no warm VectorCache for '{name}' — cell has no embeddings? "
            f"(algebra engine needs an embedded cell)")
    cache = state["caches"]["_raw_chunks"]
    if cache.matrix is None or not cache.ids:
        raise RuntimeError(f"VectorCache for '{name}' is empty")

    embed_query, _ = _query_embedder_for(
        state.get("model"), state.get("serve_dim")
    )

    def embed_fn(text: str) -> np.ndarray:
        # Resolve through the cell's model tag and serve dimension. The global
        # default can be a different vector space for legacy or int8-tagged cells.
        v = np.asarray(embed_query(text), dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v / n if n else v

    return db, embed_fn, cache.matrix, cache.ids


_db_cache: dict = {}   # db file path -> (E, ids), built once per process


def cell_engine_from_db(db):
    """(embed_fn, E, ids) from a LIVE connection — for use inside a materializer,
    which gets `db` but not a cell name. Builds E from `_raw_chunks` blobs once,
    memoized per db path (first call ~1-2s on a large cell, then free)."""
    path = None
    for row in db.execute("PRAGMA database_list").fetchall():
        if row[1] == "main":
            path = row[2]
            break

    if path and path in _db_cache:
        E, ids = _db_cache[path]
    else:
        from flex.retrieve.embeddings import _serve_dim
        from flex.retrieve.vec_ops import VectorCache
        serve_dim = int(_serve_dim(db) or 128)
        c = VectorCache().load_from_db(
            db, "_raw_chunks", "embedding", "id", serve_dim=serve_dim
        )
        if c.matrix is None or not c.ids:
            raise RuntimeError("no embeddings in this cell (algebra engine needs an embedded cell)")
        E, ids = c.matrix, c.ids
        if path:
            _db_cache[path] = (E, ids)

    from flex.engine import _query_embedder_for
    from flex.retrieve.embeddings import _serve_dim, active_model

    embed_query, _ = _query_embedder_for(
        active_model(db), int(_serve_dim(db) or E.shape[1])
    )

    def embed_fn(text: str) -> np.ndarray:
        return normalize_primitive(embed_query(text))

    return embed_fn, E, ids


def compose(terms, embed_fn) -> np.ndarray:
    """Compose normalized primitives without renormalizing their signed sum."""
    return compose_query((float(coef), embed_fn(text)) for coef, text in terms)


def _align(q: np.ndarray, d: int) -> np.ndarray:
    """Require the query vector and matrix to inhabit exactly the same space."""
    if q.shape[0] != d:
        raise ValueError(
            f"query vector dimension mismatch: q={q.shape[0]}, E={d}; "
            "refusing implicit truncation"
        )
    return q


def score(E: np.ndarray, ids, q: np.ndarray, top: int = 10, mask_ids=None):
    """Raw inner-product E @ q, top-N. Unit primitive q is cosine-equivalent."""
    q = _align(q, E.shape[1])
    s = score_inner_product(E, q)
    if mask_ids is not None:
        keep = np.fromiter((i for i, _id in enumerate(ids) if _id in mask_ids),
                           dtype=np.int64)
        if keep.size == 0:
            return []
        local = stable_top_indices([ids[int(i)] for i in keep], s[keep], top)
        order = keep[local]
    else:
        order = stable_top_indices(ids, s, top)
    return [(ids[int(i)], float(s[int(i)])) for i in order]
