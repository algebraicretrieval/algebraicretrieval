#!/usr/bin/env python3
"""Run the paper's three examples and its bounded floating-point comparison."""

import json
import math
import sqlite3

import faiss
import numpy as np
import pandas as pd
import pyterrier as pt
import sqlite_vec

import run as proof
from score_semantics import normalize_primitive, score_inner_product


CASES = {
    "linear_composition": (
        "τ₁₀((E @ q1) - 0.5 * (E @ q2))",
        """SELECT docno AS id,
       (1 - vec_distance_cosine(vector, :q1))
       - 0.5 * (1 - vec_distance_cosine(vector, :q2)) AS score
FROM embeddings
ORDER BY score DESC, id ASC
LIMIT 10;""",
    ),
    "first_pass_vector_rerank": (
        "τ₁₀(m ▷ (E @ q1))",
        """SELECT e.docno AS id, 1 - vec_distance_cosine(e.vector, :q1) AS score
FROM embeddings AS e JOIN m ON m.id = e.docno
ORDER BY score DESC, id ASC
LIMIT 10;""",
    ),
    "mask_weight_composition": (
        "τ₁₀(w ⊙ (m ▷ (E @ q1)))",
        """SELECT e.docno AS id,
       w.weight * (1 - vec_distance_cosine(e.vector, :q1)) AS score
FROM embeddings AS e
JOIN m ON m.id = e.docno JOIN w ON w.id = e.docno
ORDER BY score DESC, id ASC
LIMIT 10;""",
    ),
}


def compare(reference, alternative):
    """Compare scores by identity even when the returned ranks differ."""
    expected, actual = dict(reference), dict(alternative)
    return {
        "same_document_set": expected.keys() == actual.keys(),
        "same_order": [i for i, _ in reference] == [i for i, _ in alternative],
        "max_common_score_delta": max(
            (abs(expected[i] - actual[i]) for i in expected.keys() & actual.keys()),
            default=0.0,
        ),
    }


def score_id_order(frame):
    out = frame.sort_values(
        ["score", "docno"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    out["rank"] = np.arange(len(out))
    return out


def cosine_steps(row, query):
    """Expose sqlite-vec v0.1.9's scalar float32 accumulation and rounding.

    Source: https://github.com/asg017/sqlite-vec/blob/v0.1.9/sqlite-vec.c#L436
    The returned distance is float32; SQLite then evaluates 1 - distance.
    """
    dot, aa, bb = np.float32(0), np.float32(0), np.float32(0)
    for x, y in zip(row, query):
        dot = np.float32(dot + np.float32(x * y))
        aa = np.float32(aa + np.float32(x * x))
        bb = np.float32(bb + np.float32(y * y))
    cosine = float(dot) / (math.sqrt(float(aa)) * math.sqrt(float(bb)))
    distance = float(np.float32(1 - cosine))
    return {
        "dot": float(dot), "row_norm_squared": float(aa),
        "query_norm_squared": float(bb), "cosine_before_rounding": cosine,
        "rounded_distance": distance, "sql_similarity": 1 - distance,
    }


def execute(db):
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript("""CREATE TEMP VIEW m AS
SELECT docno AS id FROM pyterrier_bm25_results WHERE qid='1';
CREATE TEMP VIEW w AS
SELECT docno AS id, score / max(score) OVER () AS weight
FROM pyterrier_bm25_results WHERE qid='1';""")
    contract = proof._contract()
    ids, E, queries = proof._fixture(db)
    surface = proof._math_surface()
    params = {
        "q1": db.execute("SELECT vector FROM query_vectors WHERE qid='1'").fetchone()[0],
        "q2": db.execute("SELECT vector FROM embeddings WHERE docno='2008'").fetchone()[0],
    }
    bm25 = pd.read_sql_query(
        "SELECT qid,docno,score,rank FROM pyterrier_bm25_results WHERE qid='1' ORDER BY docno", db
    )
    maximum = float(bm25.score.max())
    weight = dict(zip(bm25.docno, bm25.score / maximum))
    sql_weight = dict(db.execute("SELECT id,weight FROM w"))
    mask = contract.mask_ids(db, "m")
    runtime_weight = dict(zip(mask, contract.weight_values(db, "w", mask)))
    weights_match = weight == sql_weight == runtime_weight and set(mask) == set(weight)

    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    score1, score2 = (
        dict(zip(ids, proof._faiss_scores(index, queries[q], len(ids))))
        for q in ("q1", "q2")
    )
    D = pt.Transformer.from_df(pd.DataFrame({"qid": "1", "docno": ids}))
    M = pt.Transformer.from_df(bm25)
    A = pt.apply.doc_score(lambda r: float(score1[r["docno"]]))
    B = pt.apply.doc_score(lambda r: float(score2[r["docno"]]))
    W = pt.apply.doc_score(lambda r: r["score"] * weight[r["docno"]])
    T = pt.apply.generic(score_id_order)
    pipelines = [(D >> (A + (-0.5) * B) >> T) % 10,
                 (M >> A >> T) % 10, (M >> A >> W >> T) % 10]
    topics = pd.read_sql_query("SELECT qid,query FROM query_vectors WHERE qid='1'", db)
    report = {
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "pyterrier": pt.__version__, "faiss": faiss.__version__,
                     "sqlite": sqlite3.sqlite_version,
                     "sqlite_vec": db.execute("SELECT vec_version()").fetchone()[0]},
        "fixture": {"documents": len(ids), "dimensions": E.shape[1]},
        "weights": {"rows": len(weight), "qid": "1", "maximum": maximum,
                    "exactly_equal": weights_match},
        "absolute_score_tolerance": proof.SCORE_ATOL, "cases": {},
    }
    for (name, (expression, sql)), pipeline in zip(CASES.items(), pipelines):
        top = surface.parse_math(expression, contract.bindings())
        algebra, _ = proof.planner.run_records(
            top, queries.__getitem__, E, ids, db, contract=contract
        )
        sql_rows = db.execute(sql, params).fetchall()
        pt_rows = [(r.docno, float(r.score)) for r in pipeline.transform(topics).itertuples()]
        report["cases"][name] = {
            "expression": expression, "sql": sql,
            "algebra": proof._rows(algebra), "sqlite_vec": proof._rows(sql_rows),
            "pyterrier": proof._rows(pt_rows),
            "sql_comparison": compare(algebra, sql_rows),
            "pyterrier_comparison": compare(algebra, pt_rows),
        }

    q1, q2 = (normalize_primitive(queries[q]) for q in ("q1", "q2"))
    folded = score_inner_product(E, q1 - np.float32(0.5) * q2)
    s1, s2 = score_inner_product(E, q1), score_inner_product(E, q2)
    separate = s1 - np.float32(0.5) * s2
    pair, mechanism_matches = {}, True
    for identity in ("7429", "9217"):
        i = ids.index(identity)
        steps = {q: cosine_steps(E[i], np.frombuffer(params[q], dtype=np.float32))
                 for q in ("q1", "q2")}
        actual = db.execute("""SELECT 1-vec_distance_cosine(vector,:q1),
            1-vec_distance_cosine(vector,:q2) FROM embeddings WHERE docno=:id""",
            {**params, "id": identity}).fetchone()
        mechanism_matches &= actual == tuple(steps[q]["sql_similarity"] for q in ("q1", "q2"))
        pair[identity] = {"folded": float(folded[i]), "two_pass": float(separate[i]),
                          "row_norm_float64": float(np.linalg.norm(E[i].astype(np.float64))),
                          "sql_score": actual[0] - 0.5 * actual[1], "cosine_steps": steps}
    first, second = pair["7429"], pair["9217"]
    report["numerics"] = {
        "pair": pair, "scalar_reproduction_matches_sql": mechanism_matches,
        "reference_gap": first["folded"] - second["folded"],
        "two_pass_gap": first["two_pass"] - second["two_pass"],
        "sql_gap_9217_over_7429": second["sql_score"] - first["sql_score"],
    }
    report["passed"] = (
        weights_match and len(weight) == 52 and mechanism_matches
        and first["folded"] == second["folded"] == first["two_pass"] == second["two_pass"]
        and second["sql_score"] > first["sql_score"]
        and all(c[k]["same_document_set"] and c[k]["max_common_score_delta"] <= proof.SCORE_ATOL
                for c in report["cases"].values() for k in ("sql_comparison", "pyterrier_comparison"))
        and all(c["pyterrier_comparison"]["same_order"] for c in report["cases"].values())
    )
    return report


if __name__ == "__main__":
    db = sqlite3.connect(proof.VASWANI.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        report = execute(db)
    finally:
        db.close()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
