"""M0 — the Scope cable + per-query Ctx.

Scope is the algebra's closure type: every operator is Scope -> Scope, so
expressions compose arbitrarily. Its SQL projection is the temp table
(id, score, _cols) flex already builds, so Scope ≅ temp table — the same value
seen from numpy and from SQL.

Convention:
  ids is None  -> "no restriction yet" (full corpus; a root scorer scans all)
  ids == []    -> "explicitly empty" (a mask matched nothing)
  scores is None -> unscored pool (a selection/weight op fed this is an error)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class Scope:
    ids: Optional[List[str]] = None          # None = full corpus; [] = empty
    scores: Optional[np.ndarray] = None      # aligned with ids; None = unscored pool
    signal: Optional[np.ndarray] = None      # corpus-null z (σ above the scored pool's
                                             # null) aligned with ids; None = not computed
                                             # (kw/fuse/transform paths). distinctiveness,
                                             # NOT relevance.
    cols: dict = field(default_factory=dict) # _-prefixed enrichment by id (future phases)

    @property
    def scored(self) -> bool:
        return self.scores is not None

    @property
    def restricted(self) -> bool:
        return self.ids is not None


@dataclass
class Ctx:
    """Per-query shared context: the warm matrix, the embedder, the connection,
    and the counters tests assert against (matmul_rows proves pushdown)."""
    embed_fn: object
    E: np.ndarray
    ids: List[str]
    db: object
    contract: object = None
    id_to_idx: dict = None
    matmul_rows: int = 0          # rows that actually entered a matmul (pushdown evidence)
    materializations: int = 0     # temp tables / substrate crossings emitted
    signal_by_id: dict = None     # set by run_chain at its terminal return: id -> corpus-null
                                  # z of the FINAL result set. The surface reads it (run_records)
                                  # so run_chain's return stays a 2-tuple — consumers unaffected.
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.id_to_idx is None:
            self.id_to_idx = {x: i for i, x in enumerate(self.ids)}
