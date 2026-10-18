"""A serving prototype, for the one cost the measurement tools cannot report.

`migration` and `estimate` count forward passes. They say nothing about the
latency a user sees when their query lands on a window nobody has touched yet,
because that cost is not a total -- it is a distribution over queries, and it
decays as the migration proceeds. This module serves queries one at a time and
records where the time went.

It is a prototype, and the boundaries are worth stating plainly, because a
reader should not have to infer them:

  * Single process, single thread, synchronous. A query waits for its own
    repairs. That makes the reported latency an upper bound: a real deployment
    could return the un-repaired ranking immediately and repair behind it, or
    schedule repairs off the serving device.
  * Exact search (`IndexFlatIP` semantics via a dense matmul). An ANN index
    would add its own recall error and its own insert cost for repaired
    vectors; both belong in a system evaluation and neither is measured here.
  * Repairs and serving share one device, serialised. Nothing overlaps, so
    there is no scheduling policy to credit or blame.
  * Persistence is a flat file, written on demand. It exists to answer what a
    restart costs, not to be a storage design.

What it does measure, and what the paper needs from it: the per-query latency
distribution during a migration, split into retrieve, repair and rank; how that
distribution decays as the reached fraction grows; and how much of it is
decided by the repair chunk size, which turns out to be most of it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .migration import _safe_normalize, encode_with, topk_chunked


class ArrayIndex:
    """The old index as a dense matrix. Exact, and with no insert semantics.

    Adequate for measuring the query path, inadequate for anything about the
    index itself: a real deployment must write each repaired vector somewhere,
    and a flat array makes that free. `QdrantIndex` does not.
    """

    def __init__(self, V: np.ndarray):
        self.V = V

    def search(self, q: np.ndarray, k: int) -> list[int]:
        return [int(u) for u in topk_chunked(q[None, :], self.V, k)[0]]

    def upsert(self, ids: list[int], vecs: np.ndarray) -> None:
        return None                      # nothing to do; see the docstring


class QdrantIndex:  # pragma: no cover - runs only where the client is installed
    """The new-space index in a real vector database, in local mode.

    This exists for the one number an in-memory prototype cannot produce: what
    it costs to put a repaired vector into a live index. Local mode keeps the
    artifact reproducible --- no server to stand up --- while preserving real
    insert and search semantics, including the work an HNSW graph does on
    write, which is the cost a lazy migration pays once per repaired unit.
    """

    def __init__(self, dim: int, path: str | None = None,
                 collection: str = "thaw"):
        from qdrant_client import QdrantClient, models

        self.models, self.collection = models, collection
        self.client = QdrantClient(path=path) if path else QdrantClient(":memory:")
        # recreate_collection is deprecated in recent clients: delete + create.
        # `collection_exists` is likewise not in every version, so probe for it
        # the way search() is probed -- otherwise an older client passes the
        # search guard and then dies here instead, which is the same failure the
        # guard exists to prevent.
        if hasattr(self.client, "collection_exists"):
            exists = self.client.collection_exists(collection)
        else:  # pragma: no cover - older clients
            exists = collection in [
                c.name for c in self.client.get_collections().collections]
        if exists:
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE))

    def search(self, q: np.ndarray, k: int) -> list[int]:
        # qdrant-client removed `search()` in 1.13, replacing it with
        # `query_points()`. Both are tried so the artifact runs on either, rather
        # than dying with AttributeError on a current client -- which is exactly
        # how this line failed the first time the suite ran.
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                self.collection, query=q.tolist(), limit=k)
            pts = getattr(res, "points", res)
        else:  # pragma: no cover - older clients
            pts = self.client.search(
                self.collection, query_vector=q.tolist(), limit=k)
        return [int(h.id) for h in pts]

    def upsert(self, ids: list[int], vecs: np.ndarray) -> None:
        # A list of PointStruct rather than the deprecated models.Batch, which
        # recent clients warn about and older ones require.
        points = [self.models.PointStruct(id=int(i), vector=v.tolist())
                  for i, v in zip(ids, vecs)]
        self.client.upsert(self.collection, points=points)


@dataclass
class QueryTrace:
    """Where one query's time went."""
    retrieve_ms: float
    repair_ms: float
    insert_ms: float      # writing repaired vectors into the index
    rank_ms: float
    n_stale: int          # units this query had to repair
    reached: float        # corpus fraction repaired after this query

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.repair_ms + self.insert_ms + self.rank_ms


@dataclass
class ThawServer:
    """Serve queries from an old index while repairing what they reach.

    `V_old` is the existing index. `texts` are the unit texts, needed because
    repair recomputes from the original. `encode_new` maps a list of texts to
    new-space vectors; `encode_old_query` maps a query string to the old space.
    """
    V_old: np.ndarray
    texts: list[str]
    encode_new: callable
    encode_old_query: callable
    encode_new_query: callable
    kprime: int = 500
    k: int = 10
    chunk: int = 8                      # repair batch size; see the docstring
    new_dim: int | None = None     # width of the new space; avoids the probe below
    V_new: np.ndarray = field(default=None, repr=False)
    repaired: np.ndarray = field(default=None, repr=False)
    new_index: object = None      # where repaired vectors are written, if anywhere

    def __post_init__(self):
        n = len(self.V_old)
        if self.V_new is None:
            dim = self.new_dim
            if dim is None:
                # A probe encode, the only one this class makes outside a query.
                # It is not a repair, so it does not appear in any trace -- which
                # is why callers who know the width should pass `new_dim` and
                # skip it. It used to index `self.texts[0]` unconditionally, so an
                # empty corpus raised IndexError instead of saying what was wrong.
                if not self.texts:
                    raise SystemExit(
                        "cannot infer the new-space width: the corpus is empty, so "
                        "there is no text to probe. Pass new_dim (or V_new) when "
                        "the corpus has no units.")
                dim = int(self.encode_new([self.texts[0]]).shape[1])
            self.V_new = np.zeros((n, dim), dtype=np.float32)
        if self.repaired is None:
            self.repaired = np.zeros(n, dtype=bool)

    # ---------------------------------------------------------------- serving

    def query(self, text: str) -> tuple[list[int], QueryTrace]:
        """Serve one query: candidates from the old space, ranking in the new.

        The repair is issued in chunks rather than as one call. That is not an
        optimisation detail: on a memory-tight device the single-call version
        of a 500-unit repair is an order of magnitude slower per unit, so the
        chunk size is most of what a user feels.
        """
        t0 = time.perf_counter()
        q_old = self.encode_old_query(text)
        window = topk_chunked(q_old[None, :], self.V_old, self.kprime)[0]
        t1 = time.perf_counter()

        stale = [int(u) for u in window if not self.repaired[u]]
        insert_ms = 0.0
        for i in range(0, len(stale), self.chunk):
            batch = stale[i:i + self.chunk]
            vecs = self.encode_new([self.texts[u] for u in batch])
            self.V_new[batch] = vecs
            self.repaired[batch] = True
            if self.new_index is not None:
                ti = time.perf_counter()
                self.new_index.upsert(batch, vecs)
                insert_ms += (time.perf_counter() - ti) * 1e3
        t2 = time.perf_counter()

        q_new = self.encode_new_query(text)
        sub = self.V_new[window]
        order = np.argsort(-(sub @ q_new))[:self.k]
        results = [int(window[j]) for j in order]
        t3 = time.perf_counter()

        return results, QueryTrace(
            retrieve_ms=(t1 - t0) * 1e3,
            repair_ms=(t2 - t1) * 1e3 - insert_ms,      # encoding only
            insert_ms=insert_ms, rank_ms=(t3 - t2) * 1e3,
            n_stale=len(stale), reached=float(self.repaired.mean()))

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Persist what has been repaired, so a restart does not redo it."""
        np.savez(path, V_new=self.V_new, repaired=self.repaired)

    def load(self, path: str) -> None:
        z = np.load(path)
        self.V_new, self.repaired = z["V_new"], z["repaired"]


def replay(server: ThawServer, queries: list[str]) -> list[QueryTrace]:
    """Serve a query stream in arrival order and return one trace per query."""
    return [server.query(q)[1] for q in queries]


def summarise(traces: list[QueryTrace]) -> dict:
    """The distribution, not the mean: first-touch cost is a tail problem.

    A mean latency hides exactly what matters here, because the first queries
    of a migration pay for hundreds of repairs and later ones pay for none.
    """
    if not traces:
        # Every field below is a percentile or max over the trace array, and
        # `max()` on an empty array has no identity (ValueError). Refuse with a
        # reason instead of crashing inside `float(tot.max())`.
        raise SystemExit("summarise() was given an empty replay: no query was "
                         "served, so there is no latency distribution to report")
    tot = np.array([t.total_ms for t in traces])
    rep = np.array([t.repair_ms for t in traces])
    ins = np.array([t.insert_ms for t in traces])
    stale = np.array([t.n_stale for t in traces])
    pct = lambda a, p: float(np.percentile(a, p))
    n = len(traces)
    first, last = slice(0, max(1, n // 10)), slice(max(1, 9 * n // 10), n)
    return {
        "queries": n,
        "total_p50": pct(tot, 50), "total_p95": pct(tot, 95),
        "total_p99": pct(tot, 99), "total_max": float(tot.max()),
        "repair_share": float(rep.sum() / max(tot.sum(), 1e-9)),
        "insert_share": float(ins.sum() / max(tot.sum(), 1e-9)),
        "insert_ms_per_unit": float(
            ins.sum() / max(sum(t.n_stale for t in traces), 1)),
        "stale_per_query_mean": float(stale.mean()),
        "first_decile_p50": pct(tot[first], 50),
        "last_decile_p50": pct(tot[last], 50),
        "reached_final": traces[-1].reached if traces else 0.0,
    }


def report(s: dict) -> None:
    print(f"  queries               {s['queries']:,}")
    print(f"  latency p50 / p95 / p99  {s['total_p50']:.1f} / "
          f"{s['total_p95']:.1f} / {s['total_p99']:.1f} ms   (max {s['total_max']:.1f})")
    print(f"  repair share of time  {100 * s['repair_share']:.1f}%")
    if s.get("insert_share"):
        print(f"  index insert share    {100 * s['insert_share']:.1f}%"
              f"   ({s['insert_ms_per_unit']:.2f} ms per repaired unit)")
    print(f"  stale units / query   {s['stale_per_query_mean']:.1f}")
    print(f"  p50, first decile     {s['first_decile_p50']:.1f} ms")
    print(f"  p50, last decile      {s['last_decile_p50']:.1f} ms"
          f"   ({s['first_decile_p50'] / max(s['last_decile_p50'], 1e-9):.0f}x lower)")
    print(f"  reached at end        {s['reached_final']:.3f}")


def main() -> None:  # pragma: no cover
    import argparse

    from .migration import encode_all, load_beir, load_cipbench, load_jsonl

    ap = argparse.ArgumentParser(
        prog="serve", description="measure first-touch latency during a migration")
    ap.add_argument("--corpus", default="cipbench",
                    choices=["cipbench", "jsonl", "beir"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--beir-name", default=None)
    ap.add_argument("--qrels-split", default=None,
                    help="BEIR: restrict queries to this split and keep their "
                         "gold documents in any --limit subset (dev for MS MARCO)")
    ap.add_argument("--queries-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--old", default="lexical:word")
    ap.add_argument("--new", default="lexical:char")
    ap.add_argument("--kprime", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunks", default="8,512",
                    help="repair batch sizes to compare; the difference between "
                         "them is most of the latency a user feels")
    ap.add_argument("--max-queries", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="serve.json")
    a = ap.parse_args()

    if a.corpus == "cipbench":
        units, queries = load_cipbench(a.path, a.limit, 0)
    elif a.corpus == "jsonl":
        units, queries = load_jsonl(a.path, a.queries_file, a.limit, 0)
    else:
        units, queries = load_beir(a.beir_name, a.path, a.limit, 0, a.qrels_split)
    utext = [t for _, t in units]
    qtext = [t for _, t in queries][:a.max_queries]
    if not utext:
        raise SystemExit("the corpus is empty: nothing to serve or repair")
    if not qtext:
        raise SystemExit("the query set is empty: the latency sweep has no "
                         "stream to replay, so there is nothing to measure")
    print(f"corpus N={len(units):,}  queries={len(qtext):,}  K'={a.kprime}")

    V_old, fit_old = encode_all(utext, a.old, device=a.device)
    # The new encoder is only needed as a callable -- the whole point is that
    # the corpus is NOT embedded with it up front. A neural model just loads;
    # the lexical stand-in has to be fitted on the corpus first, so for it we
    # fit and discard the vectors, which a real run never does.
    if a.new.startswith("lexical:"):
        _, fit_new = encode_all(utext, a.new, device=a.device)
    else:                                              # pragma: no cover
        _, fit_new = encode_all([utext[0]], a.new, device=a.device)

    out = {}
    # Resolve the new-space width once for every chunk. Reading it off the model
    # where it is exposed costs no forward pass at all; the fallback encodes one
    # text. Doing it here rather than in the server matters: the server is built
    # once per chunk, so its own probe ran once per chunk.
    _dim = getattr(fit_new, "get_sentence_embedding_dimension", None)
    new_dim = int(_dim()) if callable(_dim) else None
    if new_dim is None:
        if not utext:
            raise SystemExit("cannot infer the new-space width from an empty "
                             "corpus: there is no text to probe")
        new_dim = int(encode_with(fit_new, [utext[0]], a.new).shape[1])
    for chunk in (int(c) for c in a.chunks.split(",")):
        server = ThawServer(
            V_old=V_old, texts=utext, new_dim=new_dim,
            encode_new=lambda ts: _safe_normalize(
                encode_with(fit_new, ts, a.new, batch=max(chunk, 1))),
            encode_old_query=lambda t: _safe_normalize(
                encode_with(fit_old, [t], a.old))[0],
            encode_new_query=lambda t: _safe_normalize(
                encode_with(fit_new, [t], a.new))[0],
            kprime=min(a.kprime, len(units)), k=a.k, chunk=chunk)
        s = summarise(replay(server, qtext))
        out[str(chunk)] = s
        print(f"\nrepair chunk {chunk}:")
        report(s)
    from .results_io import environment, write_results
    out["environment"] = environment(device=getattr(a, "device", None))
    write_results(a.out, out)


if __name__ == "__main__":
    main()
