"""The Thaw query path on PostgreSQL + pgvector: step 2's vertical slice.

    window   old-space top-K' from docs.emb_old  (HNSW, or exact for E1)
    present  which window members already have a row in docs_new
    repair   encode the rest with the new model, in chunks; write to docs_new
    rank     top-k of the window by the new vector, in SQL
    union    (optional) also rank the new-space top-K_n among already-repaired
             units, found through an HNSW index on docs_new

Deduplication is in memory (RepairCoordinator): one process, many threads,
each unit encoded at most once. With one coordinator nothing more is needed;
claims in the database are left to step 3, where several coordinators share
the work. Step 1 on the real 500k data measured a database claim at 1.8 ms p50
(4.2 ms p99) against a 50 ms window search, so moving claims there later costs
little.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    """Where one query's time went, in milliseconds."""
    q_old_ms: float = 0.0       # encode the query with the old model
    q_new_ms: float = 0.0       # encode the query with the new model
    window_ms: float = 0.0      # old-space top-K'
    present_ms: float = 0.0     # which window members are already repaired
    text_ms: float = 0.0        # fetch passage text for the stale ones
    encode_ms: float = 0.0      # new-model passes for the stale ones
    write_ms: float = 0.0       # COPY into docs_new
    wait_ms: float = 0.0        # waiting on units another thread is repairing
    rank_ms: float = 0.0        # top-k of the window in the new space
    window_size: int = 0
    n_stale: int = 0            # units this query encoded itself
    n_waited: int = 0           # units another query was already encoding
    repaired_total: int = 0     # units repaired so far, after this query
    results: list = field(default_factory=list)
    window: list = field(default_factory=list)
    union_ms: float = 0.0       # new-space search over the repaired set
    extra: list = field(default_factory=list)   # union candidates outside the window
    wall_ms: float = 0.0        # the query's own wall clock, start to finish

    @property
    def total_ms(self) -> float:
        return (self.q_old_ms + self.q_new_ms + self.window_ms + self.present_ms
                + self.text_ms + self.encode_ms + self.write_ms + self.wait_ms
                + self.union_ms + self.rank_ms)


class RepairCoordinator:
    """At most one encode per unit within this process.

    `repaired` is a bitmap of units this process has written; `inflight` maps a
    unit being encoded to a Future. Both are read and claimed under one lock,
    so a unit seen as stale is claimed by exactly one thread. Claims are taken
    in sorted order -- the order step 1 showed the database needs, kept here so
    step 3 can swap in database claims without changing callers.
    """

    def __init__(self, n_units: int, encoder, fetch_text, write):
        self.encoder, self.fetch_text, self.write = encoder, fetch_text, write
        self.repaired = np.zeros(n_units, dtype=bool)
        self.inflight: dict[int, Future] = {}
        self.lock = threading.Lock()
        self.encoded_units = 0

    def mark_present(self, ids) -> None:
        with self.lock:
            self.repaired[list(ids)] = True

    def repair(self, ids: list[int], trace: Trace) -> None:
        with self.lock:
            mine, waits = [], []
            for i in sorted(ids):
                if self.repaired[i]:
                    continue
                f = self.inflight.get(i)
                if f is None:
                    self.inflight[i] = Future()
                    mine.append(i)
                else:
                    waits.append(f)
        trace.n_stale, trace.n_waited = len(mine), len(waits)
        if mine:
            try:
                t0 = time.perf_counter()
                texts = self.fetch_text(mine)
                t1 = time.perf_counter()
                vecs = self.encoder.encode(texts)
                t2 = time.perf_counter()
                self.write(mine, vecs)
                t3 = time.perf_counter()
                trace.text_ms = (t1 - t0) * 1e3
                trace.encode_ms = (t2 - t1) * 1e3
                trace.write_ms = (t3 - t2) * 1e3
            except BaseException as e:
                with self.lock:
                    for i in mine:
                        self.inflight.pop(i).set_exception(e)
                raise
            with self.lock:
                self.repaired[mine] = True
                self.encoded_units += len(mine)
                for i in mine:
                    self.inflight.pop(i).set_result(True)
        if waits:
            t0 = time.perf_counter()
            for f in waits:
                f.result(timeout=300)
            trace.wait_ms = (time.perf_counter() - t0) * 1e3


class ThawRetriever:
    """One instance per thread (each holds its own connection); share the
    coordinator between them."""

    def __init__(self, conn, coordinator: RepairCoordinator, enc_old_query,
                 enc_new_query, kprime: int, k: int = 10, window: str = "hnsw",
                 iterative: str = "relaxed_order", max_scan_tuples: int = 100_000,
                 ef_max: int = 1000, union_k: int = 0):
        self.conn, self.coord = conn, coordinator
        self.enc_old_query, self.enc_new_query = enc_old_query, enc_new_query
        self.kprime, self.k, self.window_mode = kprime, k, window
        # Candidate union: also take the new-space top-union_k among units that
        # already have a new vector (an HNSW index over docs_new). They cost no
        # forward pass, so savings are unchanged; ranking is still exact in the
        # new space over all candidates, so recall can only rise.
        self.union_k = int(union_k)
        self._configure(window, iterative, max_scan_tuples, ef_max)

    def _configure(self, window, iterative, max_scan, ef_max):
        """Set this connection's mode, once, in __init__.

        The `RESET`s are session-level, and so are the `SET`s: `hnsw.ef_search`
        and the iterative-scan settings persist for the whole session, not for a
        transaction. That is only sound because **one connection serves exactly
        one window mode for its lifetime** -- each thread builds its own
        ThawRetriever over its own connection and never changes `window_mode`.

        If a connection pool is ever put in front of this, `_configure` must be
        called on every checkout (or moved to `SET LOCAL` inside the transaction
        a query runs in), because a pooled connection handed a different mode
        would silently keep the previous caller's `ef_search` and scan settings.
        """
        c = self.conn
        if window == "exact":
            # A sequential scan and a sort give the exact top-K', for E1. The
            # index switch is scoped to the vector searches alone (_exact_scope):
            # set for the whole session it also disabled the primary-key
            # lookups on docs_new, which made exact-mode latency meaningless.
            c.execute("RESET enable_indexscan")
            c.execute("RESET enable_bitmapscan")
        elif window == "hnsw":
            c.execute("RESET enable_indexscan")
            c.execute("RESET enable_bitmapscan")
            c.execute(f"SET hnsw.ef_search = {min(self.kprime, ef_max)}")
            c.execute(f"SET hnsw.iterative_scan = {iterative}")
            c.execute(f"SET hnsw.max_scan_tuples = {int(max_scan)}")
        else:
            raise ValueError(f"window must be exact or hnsw, got {window!r}")

    def _exact_scope(self):
        """In exact mode, a transaction in which vector searches cannot use an
        index; otherwise a no-op."""
        import contextlib
        if self.window_mode != "exact":
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def scope():
            with self.conn.transaction():
                self.conn.execute("SET LOCAL enable_indexscan = off")
                self.conn.execute("SET LOCAL enable_bitmapscan = off")
                yield
        return scope()

    def plan_indexes(self, q_old: np.ndarray) -> list[str]:
        from .checks import _index_names
        import json as _json
        with self._exact_scope():
            row = self.conn.execute(
                "EXPLAIN (FORMAT JSON) SELECT id FROM docs ORDER BY emb_old <=> %s "
                "LIMIT %s", (q_old, self.kprime)).fetchone()
        plan = row[0] if not isinstance(row[0], str) else _json.loads(row[0])
        return _index_names(plan)

    def query(self, text: str | None = None, q_old: np.ndarray | None = None,
              q_new: np.ndarray | None = None) -> Trace:
        """Serve one query. Pass precomputed q_old / q_new to take the query
        encoders out of the measurement (E1); otherwise `text` is encoded."""
        tr = Trace()
        t_query = time.perf_counter()
        c = self.conn
        if q_old is None:
            t0 = time.perf_counter()
            q_old = self.enc_old_query.encode([text])[0]
            tr.q_old_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        with self._exact_scope():
            window = [r[0] for r in c.execute(
                "SELECT id FROM docs ORDER BY emb_old <=> %s LIMIT %s",
                (q_old, self.kprime)).fetchall()]
        t1 = time.perf_counter()
        present = [r[0] for r in c.execute(
            "SELECT id FROM docs_new WHERE id = ANY(%s)", (window,)).fetchall()]
        t2 = time.perf_counter()
        tr.window_ms, tr.present_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        tr.window_size = len(window)
        tr.window = window
        if q_new is None:
            t0 = time.perf_counter()
            q_new = self.enc_new_query.encode([text])[0]
            tr.q_new_ms = (time.perf_counter() - t0) * 1e3
        candidates = window
        if not self.union_k:
            # Rows written after the switch (inserts and updates) have no old
            # vector, so no old-space window can surface them. They are found
            # through the partial HNSW index over origin='write' rows, and their
            # top-k joins the candidates. With no writes the index is empty and
            # this costs well under a millisecond.
            t0 = time.perf_counter()
            written = [r[0] for r in c.execute(
                "SELECT id FROM docs_new WHERE origin = 'write' "
                "ORDER BY emb <=> %s LIMIT %s", (q_new, self.k)).fetchall()]
            tr.union_ms = (time.perf_counter() - t0) * 1e3
            in_window = set(window)
            tr.extra = [i for i in written if i not in in_window]
            candidates = window + tr.extra
        else:
            # The full docs_new index covers written rows as well as repaired
            # ones, so this search finds both. It runs before this query's own
            # repairs: it sees what earlier traffic (and writes) left.
            t0 = time.perf_counter()
            with self._exact_scope():          # exact mode: exact union too
                found = [r[0] for r in c.execute(
                    "SELECT id FROM docs_new ORDER BY emb <=> %s LIMIT %s",
                    (q_new, self.union_k)).fetchall()]
            tr.union_ms = (time.perf_counter() - t0) * 1e3
            in_window = set(window)
            tr.extra = [i for i in found if i not in in_window]
            candidates = window + tr.extra

        self.coord.mark_present(present)
        self.coord.repair(window, tr)

        t0 = time.perf_counter()
        # Ranking must be exact over the candidates. Written plainly, the planner
        # may answer "WHERE id = ANY(...) ORDER BY emb <=> q LIMIT k" with the
        # HNSW index on docs_new and filter afterwards, which returns too few or
        # misordered rows (found by the union self-test). A materialized CTE
        # cannot use the vector index, so the rank is an exact sort.
        tr.results = [r[0] for r in c.execute(
            "WITH cand AS MATERIALIZED (SELECT id, emb FROM docs_new "
            "WHERE id = ANY(%s)) SELECT id FROM cand ORDER BY emb <=> %s LIMIT %s",
            (candidates, q_new, self.k)).fetchall()]
        tr.rank_ms = (time.perf_counter() - t0) * 1e3
        tr.repaired_total = self.coord.encoded_units
        # Latency is this wall clock, not the sum of the parts: the parts leave
        # out lock waits in the coordinator, mark_present and Python overhead,
        # which grow with threads.
        tr.wall_ms = (time.perf_counter() - t_query) * 1e3
        return tr


def db_io(dsn: str, d_new: int):
    """fetch_text and write callables, each on its own connection and lock, so
    repair I/O from many threads does not interleave on one socket."""
    from .db import connect

    conn = connect(dsn)
    lock = threading.Lock()

    def fetch_text(ids):
        with lock:
            rows = dict(conn.execute("SELECT id, text FROM docs WHERE id = ANY(%s)",
                                     (ids,)).fetchall())
        return [rows[i] for i in ids]

    def write(ids, vecs):
        with lock, conn.cursor() as cur:
            with cur.copy("COPY docs_new (id, emb, origin) "
                          "FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int8", "vector", "text"])
                for i, v in zip(ids, vecs):
                    cp.write_row([int(i), v, "repair"])

    return fetch_text, write, conn
