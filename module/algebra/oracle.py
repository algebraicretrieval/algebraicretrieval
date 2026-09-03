"""M0 — the referee.

The obviously-correct slow path: score the FULL matrix, THEN filter to the mask
and take top-n. No pushdown, no score_candidates, no shared optimization that
could hide a bug. If run_m0 (pushdown) agrees with this, the pushdown is sound.

This is the in-process equivalent of "batched flex calls" — same semantics,
used only by tests.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from materializers import _where     # noqa: E402
from planner import _compose_q       # noqa: E402


def run_oracle(top, embed_fn, E, ids, db):
    """Score all rows, filter by mask, take top-n. Returns [(id, score)]."""
    expr = top.expr
    q = _compose_q(expr, embed_fn)

    s = E @ q                                       # full matmul, every row

    keep = None
    if expr.mask:
        rows = db.execute("SELECT id FROM chunks WHERE " + _where(expr.mask.preds)).fetchall()
        keep = {r[0] for r in rows}

    order = np.argsort(-s)
    out = []
    for i in order:
        _id = ids[int(i)]
        if keep is None or _id in keep:
            out.append((_id, float(s[int(i)])))
            if len(out) >= top.n:
                break
    return out
