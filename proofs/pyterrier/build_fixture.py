#!/usr/bin/env python3
"""Build the deterministic Vaswani SQLite fixture used by the parity receipt."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
import sqlite3
import struct

import ir_datasets
import pyterrier as pt


DIMENSIONS = 128
QUERY = "What are chemical reactions?"
PYTERRIER_VERSION = "1.1.2"


def signed_hash_vector(text: str) -> bytes:
    """Return the fixture's L2-normalized signed-hash bag-of-words vector."""
    values = [0.0] * DIMENSIONS
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        index = number % DIMENSIONS
        sign = 1.0 if (number >> 63) == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return struct.pack(
        f"<{DIMENSIONS}f", *(value / norm for value in values)
    )


def build(path: Path) -> None:
    if pt.__version__ != PYTERRIER_VERSION:
        raise RuntimeError(
            f"fixture requires pyterrier=={PYTERRIER_VERSION}; found {pt.__version__}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
        CREATE TABLE documents(
          docno TEXT PRIMARY KEY,
          text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(
          docno UNINDEXED,
          text,
          content='documents',
          content_rowid='rowid',
          tokenize='porter unicode61'
        );
        CREATE TABLE pyterrier_reference(
          qid TEXT,
          query TEXT,
          docno TEXT,
          score REAL,
          rank INTEGER,
          PRIMARY KEY(qid, rank)
        );
        CREATE TABLE pyterrier_bm25_results(
          qid TEXT,
          docno TEXT,
          score REAL,
          rank INTEGER
        );
        CREATE TABLE pyterrier_tfidf_results(
          qid TEXT,
          docno TEXT,
          score REAL,
          rank INTEGER
        );
        CREATE TABLE embeddings(
          docno TEXT PRIMARY KEY,
          dim INTEGER NOT NULL,
          vector BLOB NOT NULL
        );
        CREATE TABLE query_vectors(
          qid TEXT PRIMARY KEY,
          query TEXT NOT NULL,
          dim INTEGER NOT NULL,
          vector BLOB NOT NULL
        );
        CREATE TABLE metadata(
          key TEXT PRIMARY KEY,
          value TEXT
        );
        """)

        dataset = ir_datasets.load("vaswani")
        connection.executemany(
            "INSERT INTO documents(docno,text) VALUES(?,?)",
            ((document.doc_id, document.text) for document in dataset.docs_iter()),
        )
        connection.execute(
            "INSERT INTO documents_fts(documents_fts) VALUES('rebuild')"
        )

        rows = connection.execute(
            "SELECT docno,text FROM documents"
        ).fetchall()
        connection.executemany(
            "INSERT INTO embeddings VALUES(?,?,?)",
            (
                (docno, DIMENSIONS, signed_hash_vector(text))
                for docno, text in rows
            ),
        )
        connection.execute(
            "INSERT INTO query_vectors VALUES(?,?,?,?)",
            ("1", QUERY, DIMENSIONS, signed_hash_vector(QUERY)),
        )

        index = pt.terrier.TerrierIndex.from_hf("pyterrier/vaswani.terrier")
        bm25 = (index.bm25() % 100).search(QUERY)
        tfidf = (index.retriever("TF_IDF") % 100).search(QUERY)

        connection.executemany(
            "INSERT INTO pyterrier_reference VALUES(?,?,?,?,?)",
            (
                (
                    str(row.qid),
                    QUERY,
                    str(row.docno),
                    float(row.score),
                    int(row.rank),
                )
                for row in bm25.itertuples()
            ),
        )
        for table, frame in (
            ("pyterrier_bm25_results", bm25),
            ("pyterrier_tfidf_results", tfidf),
        ):
            connection.executemany(
                f"INSERT INTO {table} VALUES(?,?,?,?)",
                (
                    (
                        str(row.qid),
                        str(row.docno),
                        float(row.score),
                        int(row.rank),
                    )
                    for row in frame.itertuples()
                ),
            )

        document_count = connection.execute(
            "SELECT count(*) FROM documents"
        ).fetchone()[0]
        if document_count != 11429:
            raise RuntimeError(
                f"unexpected Vaswani document count: {document_count}"
            )
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)",
            [
                ("dataset", "ir_datasets:vaswani"),
                ("query", QUERY),
                ("pyterrier_version", pt.__version__),
                ("documents", str(document_count)),
                ("vector_model", "signed-hash-bow-128-l2"),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "vaswani.db",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        if not arguments.force:
            raise SystemExit(f"refusing to overwrite {output}; pass --force")
        output.unlink()

    build(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    connection = sqlite3.connect(f"file:{output}?mode=ro", uri=True)
    try:
        counts = {
            name: connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in (
                "documents",
                "embeddings",
                "query_vectors",
                "pyterrier_bm25_results",
                "pyterrier_tfidf_results",
            )
        }
    finally:
        connection.close()
    print(f"fixture={output}")
    print(f"sha256={digest}")
    for name, count in counts.items():
        print(f"{name}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
