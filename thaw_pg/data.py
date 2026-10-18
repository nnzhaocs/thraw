"""Produce the vectors step 1 needs, as one .npz file.

Two sources:

  msmarco    the same 500k gold-aware MS MARCO subset and the same 5,000 dev
             queries the paper's 500k run used (semview's loader, seed 0), with
             all-MiniLM-L6-v2 vectors. Reuses semview's corpus cache, so if you
             have cache/ from the 500k run nothing is re-encoded.
  synthetic  clustered random vectors, no model, no download. For checking the
             database setup in minutes before the real run.

The file holds:
  V_old    (N, d) float32, L2-normalised: the old index
  Q_old    (D, d) float32, L2-normalised: old-space query vectors
  doc_ids  (N,)  original corpus ids, as strings (row i is database id i)
  source   which of the two produced it
Passage text for the msmarco source goes to texts.jsonl beside it; step 1 does
not need text, but loading it now means step 2 starts from a complete table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def synthetic(n: int, d: int, n_queries: int, clusters: int = 500,
              seed: int = 0, noise: float = 0.35) -> dict:
    """Clustered vectors, so an HNSW graph faces neighbourhoods rather than
    uniform noise, where every point is roughly equidistant."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, d)).astype(np.float32)
    assign = rng.integers(0, clusters, size=n)
    V = centres[assign] + noise * rng.normal(size=(n, d)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    pick = rng.choice(n, size=n_queries, replace=False)
    Q = V[pick] + 0.3 * rng.normal(size=(n_queries, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return dict(V_old=V.astype(np.float32), Q_old=Q.astype(np.float32),
                doc_ids=np.array([str(i) for i in range(n)]), source="synthetic")


def msmarco(beir_path: str, limit: int, max_queries: int, old_model: str,
            cache_dir: str | None, device: str | None, out_dir: Path,
            seed: int = 0) -> dict:
    """Mirror `semview.migration.main` exactly, so row i here is unit i there:
    same loader, same subset, same query sample, same prefixes, same cache key."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.migration import (corpus_key, encode_all, encode_with,
                                   load_beir, resolve_prefixes)

    units, queries = load_beir("msmarco", beir_path, limit, seed, "dev")
    if max_queries and len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        queries = [queries[i] for i in
                   rng.choice(len(queries), max_queries, replace=False)]
    utext = [t for _, t in units]
    qtext = [t for _, t in queries]
    oq, op = resolve_prefixes(old_model)
    print(f"MS MARCO subset N={len(units):,}, queries D={len(qtext):,}, "
          f"old model {old_model}")
    V, fitted = encode_all(utext, old_model, device=device, prefix=op,
                           cache_dir=cache_dir,
                           cache_key=corpus_key(old_model, op, (i for i, _ in units)))
    Q = encode_with(fitted, qtext, old_model, prefix=oq)
    with open(out_dir / "texts.jsonl", "w", encoding="utf-8") as f:
        for uid, t in units:
            f.write(json.dumps({"doc_id": uid, "text": t}) + "\n")
    return dict(V_old=np.ascontiguousarray(V, dtype=np.float32),
                Q_old=np.ascontiguousarray(Q, dtype=np.float32),
                doc_ids=np.array([u for u, _ in units]), source="msmarco")


def save(data: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / "vectors.npz"
    np.savez(f, **data)
    V, Q = data["V_old"], data["Q_old"]
    print(f"wrote {f}: V_old {V.shape}, Q_old {Q.shape}, source {data['source']}")
    return f


def load(out_dir: Path) -> dict:
    z = np.load(out_dir / "vectors.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}
