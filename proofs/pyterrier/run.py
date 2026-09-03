#!/usr/bin/env python3
"""Differential and independent score-device validation for Algebraic Retrieval."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

import faiss
import numpy as np
import pandas as pd
import pyterrier as pt


ROOT = Path(__file__).resolve().parents[2]
ALGEBRA = Path(os.environ.get("FLEX_ALGEBRA_ROOT", ROOT / "module" / "algebra"))
VASWANI = Path(os.environ.get("VASWANI_DB", Path(__file__).parent / "fixtures" / "vaswani.db"))
if source := os.environ.get("FLEX_SOURCE_ROOT"):
    sys.path.insert(0, source)
sys.path.insert(0, str(ALGEBRA))

from contract import Contract  # noqa: E402
import planner  # noqa: E402
import rewrite  # noqa: E402
from score_semantics import CONTRACT, SCORE_ATOL, stable_top_indices  # noqa: E402


def _math_surface():
    spec = importlib.util.spec_from_file_location("_algebra_math", ALGEBRA / "math.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Algebra mathematical notation seam")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(db):
    rows = db.execute("SELECT docno, vector FROM embeddings ORDER BY docno").fetchall()
    ids = [str(row[0]) for row in rows]
    matrix = np.ascontiguousarray(
        np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows]),
        dtype=np.float32,
    )
    source = {
        "q1": db.execute("SELECT vector FROM query_vectors WHERE qid='1'").fetchone(),
        "q2": db.execute("SELECT vector FROM embeddings WHERE docno='2008'").fetchone(),
    }
    queries = {}
    for name, row in source.items():
        if row is None:
            raise ValueError(f"missing fixture vector: {name}")
        query = np.frombuffer(row[0], dtype=np.float32)
        if query.shape != (matrix.shape[1],):
            raise ValueError(f"fixture vector dimension mismatch: {name}")
        norm = float(np.linalg.norm(query))
        if norm < 1e-8:
            raise ValueError(f"degenerate fixture vector: {name}")
        queries[name] = np.ascontiguousarray(query / norm, dtype=np.float32)
    return ids, matrix, queries


def _faiss_scores(index, query, count):
    """Return IndexFlatIP scores aligned back to matrix row order."""
    distances, labels = index.search(
        np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1), count
    )
    aligned = np.empty(count, dtype=np.float32)
    aligned[labels[0]] = distances[0]
    return aligned


def _relation(ids, scores, *, support=None, weights=None, cutoff=None, top=10):
    """Build a PyTerrier relation from independently supplied aligned scores."""
    score_by_id = {identity: float(score) for identity, score in zip(ids, scores)}
    candidates = list(ids if support is None else support)
    if weights is not None:
        score_by_id = {
            identity: score_by_id[identity] * float(weights[identity])
            for identity in candidates
        }

    def score(frame):
        result = frame.copy()
        result["score"] = result["docno"].map(score_by_id)
        if cutoff is not None:
            result = result[result["score"] >= float(cutoff)].copy()
        result = result.sort_values(
            ["score", "docno"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        result["rank"] = np.arange(len(result), dtype=np.int64)
        return result

    frame = pd.DataFrame({
        "qid": "1",
        "query": "What are chemical reactions?",
        "docno": candidates,
    })
    return (pt.apply.generic(score) % top).transform(frame)


def _compare(algebra_rows, reference_rows, *, atol=SCORE_ATOL):
    algebra = [
        (rank, str(identity), float(score))
        for rank, (identity, score) in enumerate(algebra_rows)
    ]
    reference = [
        (int(row.rank), str(row.docno), float(row.score))
        for row in reference_rows.itertuples()
    ]
    algebra_keys = {(rank, identity) for rank, identity, _ in algebra}
    reference_keys = {(rank, identity) for rank, identity, _ in reference}
    rank_id_differences = sorted(algebra_keys ^ reference_keys)
    reference_scores = {(rank, identity): score for rank, identity, score in reference}
    deltas = {
        (rank, identity): abs(score - reference_scores[(rank, identity)])
        for rank, identity, score in algebra
        if (rank, identity) in reference_scores
    }
    score_differences = [
        [rank, identity, delta]
        for (rank, identity), delta in sorted(deltas.items())
        if delta > atol
    ]
    differences = [
        ["rank_or_id", rank, identity] for rank, identity in rank_id_differences
    ] + [["score", *row] for row in score_differences]
    return {
        "differences": differences,
        "rank_id_differences": [list(row) for row in rank_id_differences],
        "score_differences": score_differences,
        "max_score_delta": max(deltas.values(), default=0.0),
        "absolute_tolerance": atol,
    }


def _rows(rows):
    return [
        {"rank": rank, "id": str(identity), "score": float(score)}
        for rank, (identity, score) in enumerate(rows)
    ]


def _reference_rows(rows):
    return [
        {"rank": int(row.rank), "id": str(row.docno), "score": float(row.score)}
        for row in rows.itertuples()
    ]


def _bundle_hash():
    aggregate = hashlib.sha256()
    files = {}
    for path in sorted(ALGEBRA.glob("*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.name] = digest
        aggregate.update(path.name.encode())
        aggregate.update(bytes.fromhex(digest))
    return aggregate.hexdigest(), files


def _case(expression, algebra_rows, reference_rows, **extra):
    comparison = _compare(algebra_rows, reference_rows)
    return {
        "expression": expression,
        **extra,
        "algebra": _rows(algebra_rows),
        "faiss_pyterrier": _reference_rows(reference_rows),
        # Compatibility field retained for existing receipt readers. PyTerrier
        # shapes the relation; FAISS supplies the independent numerical scores.
        "pyterrier": _reference_rows(reference_rows),
        **comparison,
    }


def _mutation_checks(ids, faiss_index, queries, correct_linear_cutoff):
    raw_query = queries["q1"] - np.float32(0.5) * queries["q2"]
    normalized_query = raw_query / np.linalg.norm(raw_query)
    mutated_scores = _faiss_scores(faiss_index, normalized_query, len(ids))
    mutated_relation = _relation(ids, mutated_scores, cutoff=0.20, top=500)
    mutated_ids = [str(row.docno) for row in mutated_relation.itertuples()]
    correct_ids = [str(row.docno) for row in correct_linear_cutoff.itertuples()]

    tie_ids = ["z", "a", "m"]
    tie_scores = np.array([0.5, 0.5, 0.4], dtype=np.float32)
    stable = [tie_ids[int(i)] for i in stable_top_indices(tie_ids, tie_scores, 2)]
    input_order_mutation = tie_ids[:2]

    base_scores = {"a": -0.1, "b": -0.2, "c": 0.9, "d": 0.8}
    support = {"a", "b"}
    restricted = sorted(
        ((identity, score) for identity, score in base_scores.items() if identity in support),
        key=lambda row: (-row[1], row[0]),
    )[:2]
    zero_filled = sorted(
        ((identity, score if identity in support else 0.0) for identity, score in base_scores.items()),
        key=lambda row: (-row[1], row[0]),
    )[:2]

    return {
        "normalize_composed_query": {
            "detected": correct_ids != mutated_ids,
            "correct_cardinality": len(correct_ids),
            "mutated_cardinality": len(mutated_ids),
            "composed_query_norm": float(np.linalg.norm(raw_query)),
        },
        "drop_tie_break": {
            "detected": stable != input_order_mutation,
            "correct": stable,
            "mutated": input_order_mutation,
        },
        "zero_fill_mask": {
            "detected": restricted != zero_filled,
            "correct": [[identity, score] for identity, score in restricted],
            "mutated": [[identity, score] for identity, score in zero_filled],
        },
    }


def main():
    if not VASWANI.exists():
        raise FileNotFoundError(
            f"missing fixture: {VASWANI}; set VASWANI_DB to the Vaswani SQLite fixture"
        )
    db = sqlite3.connect(f"file:{VASWANI}?mode=ro", uri=True)
    mapping = {
        "R": {"kind": "relation", "relation": "documents", "key": "docno", "columns": ["text"]},
        "E": {"kind": "matrix", "relation": "embeddings", "key": "docno", "columns": ["vector"]},
        "q1": {"kind": "query", "relation": "query_vectors", "key": "qid", "value": "1", "columns": ["vector"], "expression": 'similar("q1")'},
        "q2": {"kind": "query", "relation": "embeddings", "key": "docno", "value": "2008", "columns": ["vector"], "expression": 'similar("q2")'},
        "m": {"kind": "mask", "relation": "pyterrier_bm25_results", "key": "docno", "selector": {"qid": "1"}, "expression": 'mask("m")'},
        "w": {"kind": "weight", "relation": "pyterrier_bm25_results", "key": "docno", "columns": ["score"], "selector": {"qid": "1"}, "normalize": "max", "expression": 'weight("w")'},
    }
    contract = Contract.from_mapping(mapping)
    ids, matrix, queries = _fixture(db)
    surface = _math_surface()

    faiss_index = faiss.IndexFlatIP(matrix.shape[1])
    faiss_index.add(matrix)
    faiss_q1 = _faiss_scores(faiss_index, queries["q1"], len(ids))
    faiss_q2 = _faiss_scores(faiss_index, queries["q2"], len(ids))
    faiss_linear = faiss_q1 - np.float32(0.5) * faiss_q2

    def embed_query(name):
        return queries[name]

    cases = {}

    direct_expression = "τ₁₀(E @ q1)"
    direct_ir = surface.parse_math(direct_expression, contract.bindings())
    direct_rows, _ = planner.run_records(direct_ir, embed_query, matrix, ids, db)
    direct_reference = _relation(ids, faiss_q1)
    cases["direct_scoring"] = _case(direct_expression, direct_rows, direct_reference)

    support = contract.mask_ids(db, "m")
    rerank_expression = "τ₁₀(m ▷ (E @ q1))"
    rerank_ir = surface.parse_math(rerank_expression, contract.bindings())
    rerank_rows, _ = planner.run_records(
        rerank_ir, embed_query, matrix, ids, db, contract=contract
    )
    rerank_reference = _relation(ids, faiss_q1, support=support)
    cases["first_pass_vector_rerank"] = _case(
        rerank_expression, rerank_rows, rerank_reference,
        candidate_members=len(support),
    )

    linear_expression = "τ₁₀((E @ q1) - 0.5·(E @ q2))"
    linear_ir = surface.parse_math(linear_expression, contract.bindings())
    before = rewrite.matmul_count(linear_ir.expr)
    after = rewrite.matmul_count(rewrite.collapse(linear_ir.expr))
    linear_rows, _ = planner.run_records(linear_ir, embed_query, matrix, ids, db)
    linear_reference = _relation(ids, faiss_linear)
    cases["linear_composition"] = _case(
        linear_expression, linear_rows, linear_reference,
        matmul_count={"before_collapse": before, "after_collapse": after},
        reference_computation="two independent FAISS IndexFlatIP searches, then q1_score - 0.5*q2_score",
    )

    cutoff_expression = "τ₅₀₀(threshold(0.20, E @ q1))"
    cutoff_ir = surface.parse_math(cutoff_expression, contract.bindings())
    cutoff_rows, _ = planner.run_records(cutoff_ir, embed_query, matrix, ids, db)
    cutoff_reference = _relation(ids, faiss_q1, cutoff=0.20, top=500)
    cases["score_cutoff"] = _case(
        cutoff_expression, cutoff_rows, cutoff_reference, threshold=0.20
    )

    linear_cutoff_expression = (
        "τ₅₀₀(threshold(0.20, (E @ q1) - 0.5·(E @ q2)))"
    )
    linear_cutoff_ir = surface.parse_math(
        linear_cutoff_expression, contract.bindings()
    )
    linear_cutoff_rows, _ = planner.run_records(
        linear_cutoff_ir, embed_query, matrix, ids, db
    )
    linear_cutoff_reference = _relation(
        ids, faiss_linear, cutoff=0.20, top=500
    )
    cases["linear_composition_with_cutoff"] = _case(
        linear_cutoff_expression,
        linear_cutoff_rows,
        linear_cutoff_reference,
        threshold=0.20,
        reference_computation="two independent FAISS searches combined before cutoff",
    )

    native = db.execute(
        "SELECT docno,score FROM pyterrier_bm25_results WHERE qid='1'"
    ).fetchall()
    maximum = max(float(row[1]) for row in native)
    weights = {str(row[0]): float(row[1]) / maximum for row in native}
    weighted_expression = "τ₁₀(m ▷ (w ⊙ (E @ q1)))"
    weighted_ir = surface.parse_math(weighted_expression, contract.bindings())
    weighted_rows, _ = planner.run_records(
        weighted_ir, embed_query, matrix, ids, db, contract=contract
    )
    weighted_reference = _relation(
        ids, faiss_q1, support=support, weights=weights
    )
    cases["mask_weight_composition"] = _case(
        weighted_expression, weighted_rows, weighted_reference,
        mask_members=len(support),
    )

    mutations = _mutation_checks(ids, faiss_index, queries, linear_cutoff_reference)
    bundle_sha256, bundle_files = _bundle_hash()
    passed = (
        all(not case["differences"] for case in cases.values())
        and all(case["max_score_delta"] <= SCORE_ATOL for case in cases.values())
        and (before, after) == (2, 1)
        and len(linear_cutoff_rows) == len(linear_cutoff_reference)
        and all(check["detected"] for check in mutations.values())
    )
    receipt = {
        "fixture": {
            "dataset": "ir_datasets:vaswani",
            "documents": len(ids),
            "dimensions": matrix.shape[1],
        },
        "score_device_contract": asdict(CONTRACT),
        "scorer_oracle": {
            "implementation": "faiss.IndexFlatIP",
            "faiss_version": faiss.__version__,
            "query_normalization": "none inside scorer",
            "pyterrier_role": "relation construction, cutoff, and top-k",
        },
        "tie_policy": "score DESC, id ASC",
        "cases": cases,
        "mutations": mutations,
        "bundle": {"sha256": bundle_sha256, "files": bundle_files},
        "passed": passed,
    }
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
