#!/usr/bin/env python3
"""Gate G-K0b: can this interpreter load sqlite-vec and run a correct cosine KNN?

Run it on every host that will serve the knowledge base (plan §21, first risk):

    uv run python scripts/kb_sqlite_vec_check.py                  # the Mac
    docker run --rm -v $PWD:/w -w /w python:3.12-slim sh -c \\
      "pip install -q sqlite-vec==0.1.9 numpy && python scripts/kb_sqlite_vec_check.py"   # the Pi

It needs only ``sqlite-vec`` and ``numpy`` (not the hal9000 package), so it runs
in a bare container. It checks, in order:

1. ``sqlite3`` allows loading extensions, and sqlite-vec loads;
2. FTS5 is compiled in (BM25 keyword search);
3. a ``vec0`` table with ``distance_metric=cosine`` and a metadata column
   accepts 1,000 random 1,536-dimension vectors;
4. KNN returns the same top-k order and distances as a numpy brute force,
   both unfiltered and filtered on the metadata column inside the KNN.

Exit code 0 means pass; anything else prints what failed.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import sys

N = 1000
DIM = 1536
K = 10
QUERIES = 5
DOC_TYPES = ("invoice", "estimate", "inspection_report", "pest_report")


def run_check(n: int = N, dim: int = DIM, k: int = K, queries: int = QUERIES, seed: int = 7):
    """Run every check; returns a result dict. Raises AssertionError on a wrong answer."""
    import numpy as np
    import sqlite_vec

    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    doc_types = [DOC_TYPES[i % len(DOC_TYPES)] for i in range(n)]

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    (vec_version,) = conn.execute("SELECT vec_version()").fetchone()

    conn.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text, tokenize='porter unicode61')")
    conn.execute("INSERT INTO fts_probe (text) VALUES ('water heater valve leaking')")
    hits = conn.execute("SELECT rowid FROM fts_probe WHERE fts_probe MATCH '\"leak\"'").fetchall()
    assert hits == [(1,)], f"FTS5 porter stemming failed: {hits}"

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_probe USING vec0("
        f"chunk_id integer primary key, embedding float[{dim}] distance_metric=cosine, "
        f"doc_type text)"
    )
    with conn:
        conn.executemany(
            "INSERT INTO vec_probe (chunk_id, embedding, doc_type) VALUES (?, ?, ?)",
            [(i + 1, vectors[i].tobytes(), doc_types[i]) for i in range(n)],
        )

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    max_error = 0.0
    for q in range(queries):
        query = rng.standard_normal(dim).astype(np.float32)
        cosine_distance = 1.0 - unit @ (query / np.linalg.norm(query))
        for doc_type in (None, DOC_TYPES[q % len(DOC_TYPES)]):
            if doc_type is None:
                rows = conn.execute(
                    "SELECT chunk_id, distance FROM vec_probe "
                    "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (query.tobytes(), k),
                ).fetchall()
                candidates = np.arange(n)
            else:
                rows = conn.execute(
                    "SELECT chunk_id, distance FROM vec_probe "
                    "WHERE embedding MATCH ? AND k = ? AND doc_type = ? ORDER BY distance",
                    (query.tobytes(), k, doc_type),
                ).fetchall()
                candidates = np.array([i for i in range(n) if doc_types[i] == doc_type])
            order = candidates[np.argsort(cosine_distance[candidates], kind="stable")][:k]
            expected_ids = [int(i) + 1 for i in order]
            got_ids = [int(r[0]) for r in rows]
            assert got_ids == expected_ids, (
                f"KNN order differs from brute force (filter={doc_type}): "
                f"{got_ids} vs {expected_ids}"
            )
            errors = [abs(r[1] - float(cosine_distance[i])) for r, i in zip(rows, order)]
            max_error = max(max_error, *errors)
            assert max_error < 1e-4, f"cosine distances differ by {max_error}"

    return {
        "ok": True,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "sqlite_vec": vec_version,
        "machine": platform.machine(),
        "system": platform.system(),
        "vectors": n,
        "dimensions": dim,
        "k": k,
        "queries": queries,
        "max_distance_error": max_error,
    }


def main() -> int:
    try:
        result = run_check()
    except AttributeError as error:
        print(f"FAIL: this Python's sqlite3 cannot load extensions ({error}).", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print("PASS: sqlite-vec loads and cosine KNN matches brute force.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
