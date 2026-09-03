"""Bind native SQLite relations and columns to algebraic operand symbols.

The contract does not copy or reshape data.  It records only the semantic facts
SQLite cannot infer reliably (identity, vector, score, rank, and content roles)
and projects those bindings into the existing algebra orient row shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quoted_name(name: str) -> str:
    """Quote a table/view name, allowing an explicit SQLite schema prefix."""
    parts = name.split(".")
    if not parts or any(not _IDENT.fullmatch(part) for part in parts):
        raise ValueError(f"unsafe SQLite relation name: {name!r}")
    return ".".join(f'"{part}"' for part in parts)


def _columns(db, relation: str) -> tuple[str, ...]:
    """Return native columns through an authorizer-safe zero-row query."""
    try:
        cur = db.execute(f"SELECT * FROM {_quoted_name(relation)} LIMIT 0")
        return tuple(column[0] for column in cur.description or ())
    except Exception:
        return ()


def _selector_sql(selector: tuple[tuple[str, Any], ...]) -> tuple[str, tuple[Any, ...]]:
    if not selector:
        return "", ()
    clause = " AND ".join(f"{_quoted_name(column)}=?" for column, _ in selector)
    return f" WHERE {clause}", tuple(value for _, value in selector)


@dataclass(frozen=True)
class Operand:
    """One algebraic symbol bound to a native SQL relation."""

    symbol: str
    kind: str
    relation: str
    key: str | None = None
    columns: tuple[str, ...] = ()
    domain: str = ""
    expression: str | None = None
    value: Any = None
    dtype: str = "float32"
    selector: tuple[tuple[str, Any], ...] = ()
    normalize: str = "none"

    @classmethod
    def from_mapping(cls, symbol: str, value: Mapping[str, Any]) -> "Operand":
        columns = value.get("columns", ())
        if isinstance(columns, str):
            columns = (columns,)
        selector = value.get("selector", {})
        if not isinstance(selector, Mapping):
            raise ValueError(f"operand {symbol!r} selector must be a mapping")
        return cls(
            symbol=symbol,
            kind=str(value["kind"]),
            relation=str(value["relation"]),
            key=(str(value["key"]) if value.get("key") is not None else None),
            columns=tuple(str(column) for column in columns),
            domain=str(value.get("domain", "")),
            expression=(
                str(value["expression"])
                if value.get("expression") is not None
                else None
            ),
            value=value.get("value"),
            dtype=str(value.get("dtype", "float32")),
            selector=tuple((str(column), selected) for column, selected in selector.items()),
            normalize=str(value.get("normalize", "none")),
        )

    def orient_row(self, db) -> tuple[str, str, str, str]:
        available = _columns(db, self.relation)
        required = tuple(
            column
            for column in ((self.key,) + self.columns + tuple(name for name, _ in self.selector))
            if column is not None
        )
        missing = [column for column in required if column not in available]

        if not available:
            status = f"inert (missing relation {self.relation})"
        elif missing:
            status = f"inert (missing columns: {', '.join(missing)})"
        elif self.kind == "query" and self.key is not None and self.value is not None:
            row = db.execute(
                f"SELECT 1 FROM {_quoted_name(self.relation)} "
                f"WHERE {_quoted_name(self.key)}=? LIMIT 1",
                (self.value,),
            ).fetchone()
            status = "ok" if row else f"inert (missing {self.key}={self.value})"
        elif self.selector:
            where, params = _selector_sql(self.selector)
            row = db.execute(
                f"SELECT 1 FROM {_quoted_name(self.relation)}{where} LIMIT 1", params
            ).fetchone()
            status = "ok" if row else "inert (selector matched no rows)"
        else:
            status = "ok"

        shape = self.domain or f"{self.relation}({', '.join(required or available)})"
        return self.kind, self.symbol, shape, status


@dataclass(frozen=True)
class Contract:
    """A query-local algebraic view of one native SQLite database."""

    operands: tuple[Operand, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Mapping[str, Any]]) -> "Contract":
        return cls(tuple(Operand.from_mapping(symbol, value) for symbol, value in values.items()))

    @classmethod
    def from_json(cls, value: str | bytes | Mapping[str, Any]) -> "Contract":
        if isinstance(value, (str, bytes)):
            value = json.loads(value)
        operands = value.get("operands", value)
        if not isinstance(operands, Mapping):
            raise ValueError("contract JSON must contain an operand mapping")
        return cls.from_mapping(operands)

    @classmethod
    def discover(cls, db) -> "Contract | None":
        """Load the registered contract or the database's optional sidecar."""
        configured = os.environ.get("FLEX_ALGEBRA_CONTRACT", "").strip()
        if configured:
            if configured.startswith("{"):
                return cls.from_json(configured)
            return cls.from_json(Path(configured).expanduser().read_text())

        main_path = next(
            (row[2] for row in db.execute("PRAGMA database_list") if row[1] == "main"),
            "",
        )
        if not main_path:
            return None
        sidecar = Path(main_path + ".algebra.json")
        return cls.from_json(sidecar.read_text()) if sidecar.is_file() else None

    def resolve(self, symbol: str) -> Operand:
        for operand in self.operands:
            if operand.symbol == symbol:
                return operand
        raise KeyError(symbol)

    def orient_rows(self, db) -> list[tuple[str, str, str, str]]:
        """Return the declared operands plus the contract runtime's core algebra."""
        rows = [operand.orient_row(db) for operand in self.operands]
        rows.extend([
            ("score", "E @ q", "row-normalized Matrix[K,d] × Query[d] → Scored[K,inner_product:f32]", "ok"),
            ("restrict", "m ▷ S", "Mask[K] × Scored[K] → support-restricted Scored[K]", "ok"),
            ("modulate", "w ⊙ S", "Weight[K] × Scored[K] → Scored[K]", "ok"),
            ("fuse", "S₁ ⊕ S₂", "Ranked[K] × Ranked[K] → RRF-scored relation", "ok"),
            ("select", "τₖ(S)", "Scored[K] → Ranked[K]", "ok"),
        ])
        return rows

    def bindings(self) -> dict[str, str]:
        """Return symbols that can already lower to the existing Algebra syntax."""
        return {
            operand.symbol: operand.expression
            for operand in self.operands
            if operand.expression is not None
        }

    def mask_ids(self, db, symbol: str) -> list[str]:
        operand = self.resolve(symbol)
        if operand.kind != "mask" or operand.key is None:
            raise ValueError(f"operand {symbol!r} is not a keyed mask")
        where, params = _selector_sql(operand.selector)
        rows = db.execute(
            f"SELECT {_quoted_name(operand.key)} FROM {_quoted_name(operand.relation)}{where}",
            params,
        ).fetchall()
        ids = [str(row[0]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"mask operand {symbol!r} contains duplicate ids")
        return ids

    def weight_values(self, db, symbol: str, ids: list[str]) -> np.ndarray:
        operand = self.resolve(symbol)
        if operand.kind != "weight" or operand.key is None or len(operand.columns) != 1:
            raise ValueError(f"operand {symbol!r} is not a keyed scalar weight")
        where, params = _selector_sql(operand.selector)
        rows = db.execute(
            f"SELECT {_quoted_name(operand.key)}, {_quoted_name(operand.columns[0])} "
            f"FROM {_quoted_name(operand.relation)}{where}",
            params,
        ).fetchall()
        values = {}
        for identity, raw in rows:
            identity = str(identity)
            if identity in values:
                raise ValueError(f"weight operand {symbol!r} contains duplicate ids")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"weight operand {symbol!r} contains a non-finite value")
            values[identity] = value
        if operand.normalize == "max":
            maximum = max(values.values(), default=0.0)
            if maximum <= 0.0:
                raise ValueError(f"weight operand {symbol!r} needs a positive maximum")
            values = {identity: value / maximum for identity, value in values.items()}
        elif operand.normalize != "none":
            raise ValueError(f"unknown weight normalization: {operand.normalize}")
        return np.array([values.get(str(identity), 0.0) for identity in ids], dtype=np.float64)

    def runtime(self, db):
        """Load the matrix and named query vectors declared by this contract."""
        matrices = [operand for operand in self.operands if operand.kind == "matrix"]
        if len(matrices) != 1:
            raise ValueError("contract runtime requires exactly one matrix operand")
        matrix_operand = matrices[0]
        if matrix_operand.key is None or len(matrix_operand.columns) != 1:
            raise ValueError("matrix operand requires one key and one vector column")

        rows = db.execute(
            f"SELECT {_quoted_name(matrix_operand.key)}, "
            f"{_quoted_name(matrix_operand.columns[0])} "
            f"FROM {_quoted_name(matrix_operand.relation)} "
            f"ORDER BY {_quoted_name(matrix_operand.key)}"
        ).fetchall()
        if not rows:
            raise ValueError("matrix operand is empty")
        ids = [str(row[0]) for row in rows]
        vectors = [np.frombuffer(row[1], dtype=np.dtype(matrix_operand.dtype)) for row in rows]
        dimensions = {vector.shape for vector in vectors}
        if len(dimensions) != 1 or next(iter(dimensions)) == (0,):
            raise ValueError("matrix operand has inconsistent vector dimensions")
        matrix = np.vstack(vectors).astype(np.float32, copy=False)

        queries = {}
        for operand in self.operands:
            if operand.kind != "query":
                continue
            if operand.key is None or operand.value is None or len(operand.columns) != 1:
                raise ValueError(f"query operand {operand.symbol!r} requires key, value, and vector column")
            row = db.execute(
                f"SELECT {_quoted_name(operand.columns[0])} "
                f"FROM {_quoted_name(operand.relation)} "
                f"WHERE {_quoted_name(operand.key)}=? LIMIT 1",
                (operand.value,),
            ).fetchone()
            if row is None:
                raise ValueError(f"missing query operand: {operand.symbol}")
            vector = np.frombuffer(row[0], dtype=np.dtype(operand.dtype)).astype(np.float32, copy=False)
            if vector.shape != (matrix.shape[1],):
                raise ValueError(f"query operand dimension mismatch: {operand.symbol}")
            norm = float(np.linalg.norm(vector))
            if norm < 1e-8:
                raise ValueError(f"degenerate query operand: {operand.symbol}")
            queries[operand.symbol] = vector / norm

        def embed_fn(symbol: str):
            try:
                return queries[symbol]
            except KeyError as error:
                raise ValueError(f"unknown contract query operand: {symbol}") from error

        return embed_fn, matrix, ids
