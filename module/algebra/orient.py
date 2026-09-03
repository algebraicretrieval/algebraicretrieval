"""algebra('orient') — the cell projected into the algebra's OPERAND SPACE.

Generic @orient describes the cell as tables (schema/views/presets). This is the
in-dialect orient: the masks / scores / transforms / weights you can compose on
THIS cell, each with its live value domain and an availability flag — so you
never write an operator that silently no-ops (e.g. coedited on a cell with no
file identity). Pure SQL introspection; no dependency on the scoring engine, so
it loads/runs standalone via the plugin's _sibling loader.

Returns rows: (section, token, domain, status).
"""
from __future__ import annotations


def _has(db, name) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (name,)
    ).fetchone() is not None


def _count(db, sql, args=()) -> int:
    try:
        r = db.execute(sql, args).fetchone()
        return int(r[0]) if r and r[0] is not None else 0
    except Exception:
        return 0


def _cols(db, table) -> list:
    """Column names via a 0-row SELECT (authorizer-safe — PRAGMA and
    pragma_table_info() are both blocked on the read-only query path)."""
    try:
        cur = db.execute(f"SELECT * FROM {table} LIMIT 0")
        return [c[0] for c in cur.description]
    except Exception:
        return []


def _meta_value(db, key):
    for table in ("_metadata", "_meta"):
        if not _has(db, table):
            continue
        row = db.execute(f"SELECT value FROM {table} WHERE key=?", (key,)).fetchone()
        if row and row[0] not in (None, ""):
            return row[0]
    return None


def _content_lengths(db, sample_size=128):
    """Return bounded length evidence; never sort an entire large corpus."""
    count, low, high = db.execute(
        "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM _raw_chunks "
        "WHERE content IS NOT NULL"
    ).fetchone()
    if not count:
        return [], True
    if count <= sample_size:
        rows = db.execute(
            "SELECT length(content) FROM _raw_chunks WHERE content IS NOT NULL"
        ).fetchall()
        return sorted(row[0] for row in rows), True

    points = sorted({
        int(low + (high - low) * offset / (sample_size - 1))
        for offset in range(sample_size)
    })
    placeholders = ",".join("?" for _ in points)
    rows = db.execute(
        f"SELECT length(content) FROM _raw_chunks "
        f"WHERE content IS NOT NULL AND rowid IN ({placeholders})",
        points,
    ).fetchall()
    return sorted(row[0] for row in rows), False


def orient_rows(db) -> list:
    rows: list = []
    add = lambda *a: rows.append(tuple(a))

    # ── masks (the operands, grounded in live values) ──
    if _has(db, "_types_message"):
        vals = db.execute(
            "SELECT type, COUNT(*) FROM _types_message WHERE type IS NOT NULL "
            "GROUP BY type ORDER BY 2 DESC LIMIT 8").fetchall()
        add("mask", "type(X)", ", ".join(f"{t}:{n}" for t, n in vals), "ok")
    if _has(db, "_types_docpac"):
        vals = db.execute(
            "SELECT doc_type, COUNT(*) FROM _types_docpac WHERE doc_type IS NOT NULL "
            "GROUP BY doc_type ORDER BY 2 DESC LIMIT 10").fetchall()
        add("mask", "eq(doc_type, X)", ", ".join(f"{t}:{n}" for t, n in vals), "ok")

    if _has(db, "_enrich_source_graph"):
        gcols = _cols(db, "_enrich_source_graph")
        if "community_id" in gcols:
            ncomm = _count(db, "SELECT COUNT(DISTINCT community_id) FROM _enrich_source_graph "
                               "WHERE community_id IS NOT NULL")
            if ncomm:
                lbl_expr = "COALESCE(community_label,'')" if "community_label" in gcols else "''"
                top = db.execute(
                    f"SELECT community_id, {lbl_expr}, COUNT(*) FROM _enrich_source_graph "
                    "WHERE community_id IS NOT NULL GROUP BY community_id "
                    "ORDER BY 3 DESC LIMIT 5").fetchall()
                dom = "; ".join(
                    (f"{cid}:{(lbl.split(' ·')[0] or '?')[:18]}({n})" if lbl else f"{cid}({n})")
                    for cid, lbl, n in top)
                add("mask", "community(N)", f"{ncomm} communities — {dom}", "ok")
        if "is_hub" in gcols:
            nhub = _count(db, "SELECT COUNT(*) FROM _enrich_source_graph WHERE is_hub=1")
            add("mask", "hub()", f"{nhub} hub sources", "ok" if nhub else "inert (no hubs)")

    nrepo = _count(db, "SELECT COUNT(*) FROM _edges_repo_identity") if _has(db, "_edges_repo_identity") else 0
    add("mask", "repo(h)", f"{nrepo} repo edges", "ok" if nrepo else "inert (no repo identity)")

    span = db.execute("SELECT MIN(timestamp), MAX(timestamp) FROM _raw_chunks "
                      "WHERE timestamp IS NOT NULL").fetchone()
    if span and span[0] and span[1] and span[1] > span[0]:
        add("mask", "recent(N)", f"cell spans ~{int((span[1]-span[0])/86400)}d (N = days back)", "ok")
    else:
        add("mask", "recent(N)", "no usable timestamps", "inert")

    lens, exact_lengths = _content_lengths(db)
    if lens:
        n = len(lens)
        mark = "" if exact_lengths else "~"
        add("mask", "minlen(N)", f"{mark}p50={lens[n//2]} {mark}p90={lens[min(n-1,int(n*0.9))]} chars "
                                 f"(minlen(120) trims stubs)", "ok")

    # ── scores (which scoring substrates are populated) ──
    serve_dim = _meta_value(db, "vec:serve_dim")
    dim_label = f", {serve_dim}d" if serve_dim is not None else ""
    add("score", "similar(text)", f"unit primitive → cosine-equivalent inner product; composed q keeps magnitude (warm VectorCache{dim_label})", "ok")
    add("score", "kw(text)", "FTS5/BM25", "ok" if _has(db, "chunks_fts") else "inert (no chunks_fts)")
    add("score", "centroid(top(..)|<mask>)", "PRF / set centroid", "ok")
    if _has(db, "_types_resolved"):
        cols = [c for c in _cols(db, "_types_resolved") if c != "chunk_id"]
        add("score", "near_type(T)", f"T ∈ {{{', '.join(cols[:10])}}}",
            "token pending — use type_resolve('T > p') + JOIN")
    else:
        add("score", "near_type(T)", "needs _types_projected (run types.fit + enrich)", "inert")

    # ── scope-transforms (will they actually enlarge on this cell?) ──
    add("transform", "expand(N,·)", "session/doc siblings (_edges_source)",
        "ok" if _has(db, "_edges_source") else "inert")
    nfi = _count(db, "SELECT COUNT(*) FROM _edges_file_identity") if _has(db, "_edges_file_identity") else 0
    add("transform", "coedited(·)", f"{nfi} file-identity edges",
        "ok" if nfi else "inert (no file identity → returns only seeds)")
    add("transform", "same_repo(·)", f"{nrepo} repo edges", "ok" if nrepo else "inert")
    add("transform", "window(W,·)", "±W positions in same source",
        "ok" if _has(db, "_edges_source") else "inert")

    # ── typed composition operators ──
    add("operator", "m ▷ S / restrict(m,S)", "Mask[K] restricts relational support", "ok")
    add("operator", "w ⊙ S / modulate(w,S)", "Weight[K] modulates finite scores", "ok")
    add("operator", "S₁ ⊕ S₂ / rrf(S₁,S₂)", "explicit reciprocal-rank fusion", "ok")
    add("operator", "rescore(q,S)", "rank a transformed set by a new query", "ok")

    # ── true weights ──
    add("weight", "decay(N)", "recency half-life of N days (score *= 0.5^(age/N))", "ok")

    # ── support-changing and sequential selectors ──
    add("selection", "threshold(t,·)",
        "score cutoff; calibrate t per substrate", "ok")
    add("selection", "diverse(λ,·) / dedup(t,·)",
        "MMR diversity / near-duplicate removal", "ok")
    ts_ok = bool(span and span[0] and span[1] and span[1] > span[0])
    add("selection", "earliest(N,·) / latest(N,·)",
        "oldest / newest N by source time (lineage / recency, no ORDER BY)",
        "ok" if ts_ok else "inert (no usable timestamps)")
    return rows
