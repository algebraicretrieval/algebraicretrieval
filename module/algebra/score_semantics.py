"""Normative score-device contract for Algebraic Retrieval.

One Algebra execution is one query partition. A scorer consumes:

    ids: unique identity keys describing support A
    E:   float32 matrix with one L2-normalized row per id
    q:   float32 query vector in E's declared coordinate space

and returns one finite inner-product score per input identity, aligned to the
same support and order. Primitive query vectors are L2-normalized individually;
linear combinations retain their magnitude. Therefore direct primitive scoring
is cosine-equivalent while linear collapse is score-preserving:

    (E @ q1) - alpha * (E @ q2) == E @ (q1 - alpha * q2)

up to SCORE_ATOL from float32 summation order. Scorers do not rank, truncate,
change support, normalize composed queries, or apply thresholds. Selection alone
creates ranks under (score DESC, id ASC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCORE_ATOL = 1e-6
INNER_PRODUCT_F32 = "inner_product:f32"
TIE_ORDER = ("score:desc", "id:asc")


@dataclass(frozen=True)
class ScorerDeviceContract:
    score_unit: str = INNER_PRODUCT_F32
    dtype: str = "float32"
    row_normalization: str = "l2"
    primitive_query_normalization: str = "l2"
    composed_query_normalization: str = "none"
    partitioning: str = "one query per execution"
    support_policy: str = "preserve input identities and order"
    tie_order: tuple[str, str] = TIE_ORDER
    absolute_tolerance: float = SCORE_ATOL


CONTRACT = ScorerDeviceContract()


def normalize_primitive(vector: np.ndarray) -> np.ndarray:
    """Return one L2-normalized float32 primitive query vector."""
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("degenerate primitive query vector")
    return value / norm


def compose_query(terms) -> np.ndarray:
    """Compose normalized primitive vectors without normalizing the result.

    ``terms`` is an iterable of ``(coefficient, vector)`` pairs. Coefficients
    are applied after primitive normalization. The returned magnitude is part
    of the score semantics and is observable by threshold and later operators.
    """
    query = None
    for coefficient, vector in terms:
        value = normalize_primitive(vector) * np.float32(coefficient)
        query = value if query is None else query + value
    if query is None:
        raise ValueError("compose_query() needs at least one term")
    if float(np.linalg.norm(query)) < 1e-8:
        raise ValueError("degenerate composed query vector")
    return np.asarray(query, dtype=np.float32)


def stable_top_indices(ids, scores: np.ndarray, limit: int) -> np.ndarray:
    """Return exact top indices under ``score DESC, id ASC``.

    ``argpartition`` finds the cutoff cheaply; every identity tied at that
    cutoff is restored before deterministic ordering, so input order cannot
    choose a boundary winner.
    """
    values = np.asarray(scores)
    finite = np.flatnonzero(np.isfinite(values))
    if limit <= 0 or finite.size == 0:
        return np.array([], dtype=np.int64)
    if finite.size <= limit:
        candidates = finite
    else:
        local = np.argpartition(-values[finite], limit - 1)[:limit]
        cutoff = values[finite[local]].min()
        candidates = finite[values[finite] >= cutoff]
    ordered = sorted(
        (int(index) for index in candidates),
        key=lambda index: (-float(values[index]), str(ids[index])),
    )
    return np.asarray(ordered[:limit], dtype=np.int64)


def score_inner_product(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Score every row by raw float32 inner product; never normalize ``query``."""
    rows = np.asarray(matrix, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if rows.ndim != 2:
        raise ValueError(f"score matrix must be rank 2, got shape {rows.shape}")
    if vector.shape != (rows.shape[1],):
        raise ValueError(
            f"query vector dimension mismatch: q={vector.shape[0]}, E={rows.shape[1]}"
        )
    scores = rows @ vector
    if not np.all(np.isfinite(scores)):
        raise ValueError("scorer produced a non-finite score")
    return np.asarray(scores, dtype=np.float32)
