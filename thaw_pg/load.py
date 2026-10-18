"""Create the step-1 schema, load the old index, build the indexes.

Tables (the design's, minus anything step 1 does not exercise):

  docs          id, doc_id, text, emb_old vector(d) + HNSW
  docs_new      id -> docs, emb vector(768), origin 'repair'|'write'
                + partial HNSW over origin = 'write' only
  repair_claims id, owner, claimed_at: the claim protocol under test
  claim_done    stand-in for docs_new during the concurrency check

docs_new is filled with random unit vectors: step 1 tests whether the planner
uses the partial index, not what the vectors mean. Step 2 replaces them with
real BGE vectors.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

SCHEMA = """
DROP TABLE IF EXISTS claim_done, repair_claims, docs_new, docs CASCADE;
CREATE TABLE docs (
    id         bigint PRIMARY KEY,
    doc_id     text NOT NULL,
    text       text,
    emb_old    vector({d}),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE docs_new (
    id          bigint PRIMARY KEY REFERENCES docs (id) ON DELETE CASCADE,
    emb         vector({d_new}) NOT NULL,
    origin      text NOT NULL CHECK (origin IN ('repair', 'write')),
    repaired_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE repair_claims (
    id         bigint PRIMARY KEY,
    owner      text NOT NULL,
    claimed_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE claim_done (id bigint PRIMARY KEY, owner text NOT NULL);
"""


def _texts(path: Path, n: int):
    if not path.exists():
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
    if len(out) != n:
        raise SystemExit(f"{path} has {len(out)} rows, vectors have {n}")
    return out


def copy_docs(conn, V: np.ndarray, doc_ids, texts) -> float:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        with cur.copy("COPY docs (id, doc_id, text, emb_old) "
                      "FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int8", "text", "text", "vector"])
            for i in range(len(V)):
                cp.write_row([i, str(doc_ids[i]),
                              None if texts is None else texts[i], V[i]])
                if i and i % 100_000 == 0:
                    print(f"  copied {i:,} rows", flush=True)
    return time.perf_counter() - t0


def copy_docs_new(conn, ids: np.ndarray, origin: str, d_new: int, seed: int):
    rng = np.random.default_rng(seed)
    with conn.cursor() as cur:
        with cur.copy("COPY docs_new (id, emb, origin) "
                      "FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int8", "vector", "text"])
            for lo in range(0, len(ids), 10_000):
                block = rng.normal(size=(min(10_000, len(ids) - lo), d_new))
                block = (block / np.linalg.norm(block, axis=1, keepdims=True)
                         ).astype(np.float32)
                for j, v in enumerate(block):
                    cp.write_row([int(ids[lo + j]), v, origin])


def run(conn, data: dict, out_dir: Path, m: int, ef_construction: int,
        workers: int, maint_mem: str, d_new: int = 768,
        repair_frac: float = 0.3, write_frac: float = 0.05) -> dict:
    V = data["V_old"]
    N, d = V.shape
    rep = dict(N=int(N), d=int(d), m=m, ef_construction=ef_construction)
    conn.execute(SCHEMA.format(d=d, d_new=d_new))

    texts = _texts(out_dir / "texts.jsonl", N)
    print(f"COPY {N:,} rows into docs ({'with' if texts else 'without'} text)")
    rep["copy_docs_s"] = round(copy_docs(conn, V, data["doc_ids"], texts), 1)

    conn.execute(f"SET maintenance_work_mem = '{maint_mem}'")
    conn.execute(f"SET max_parallel_maintenance_workers = {int(workers)}")
    print(f"building HNSW on docs.emb_old (m={m}, ef_construction="
          f"{ef_construction}, {workers} workers, maintenance_work_mem={maint_mem})")
    t0 = time.perf_counter()
    conn.execute(f"CREATE INDEX docs_emb_old_hnsw ON docs USING hnsw "
                 f"(emb_old vector_cosine_ops) WITH (m = {m}, "
                 f"ef_construction = {ef_construction})")
    rep["hnsw_build_s"] = round(time.perf_counter() - t0, 1)
    print(f"  built in {rep['hnsw_build_s']} s")

    # docs_new: a repaired share (fetched by id only) and a post-switch share
    # (the only rows the partial index covers). Disjoint id sets.
    rng = np.random.default_rng(1)
    perm = rng.permutation(N)
    n_rep, n_wr = int(repair_frac * N), int(write_frac * N)
    print(f"filling docs_new: {n_rep:,} 'repair' + {n_wr:,} 'write' rows "
          f"({d_new}-d random unit vectors)")
    copy_docs_new(conn, perm[:n_rep], "repair", d_new, seed=2)
    copy_docs_new(conn, perm[n_rep:n_rep + n_wr], "write", d_new, seed=3)
    t0 = time.perf_counter()
    conn.execute("CREATE INDEX docs_new_write_hnsw ON docs_new USING hnsw "
                 "(emb vector_cosine_ops) WHERE origin = 'write'")
    rep["partial_hnsw_build_s"] = round(time.perf_counter() - t0, 1)
    conn.execute("ANALYZE docs; ANALYZE docs_new")

    sizes = conn.execute(
        "SELECT pg_relation_size('docs'), pg_total_relation_size('docs'), "
        "pg_relation_size('docs_emb_old_hnsw'), pg_relation_size('docs_new_write_hnsw')"
    ).fetchone()
    rep.update(docs_heap_mb=round(sizes[0] / 2**20, 1),
               docs_total_mb=round(sizes[1] / 2**20, 1),
               hnsw_old_mb=round(sizes[2] / 2**20, 1),
               hnsw_partial_mb=round(sizes[3] / 2**20, 1),
               docs_new_repair=n_rep, docs_new_write=n_wr)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.results_io import environment, write_results
    rep["environment"] = environment()
    write_results(out_dir / "load_report.json", rep)
    print(json.dumps(rep, indent=2))
    return rep
