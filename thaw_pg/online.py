"""The online policy against PostgreSQL: the encoder off the query path.

`retriever.ThawRetriever` is the synchronous realization. It repairs every
window member before ranking, so each answer is exact over its window and the
user waits for up to K' forward passes -- 1,362 ms of a 1,400 ms median on the
measured path, and a 99th percentile of 16 s once eight threads queue behind one
GPU.

This module serves the same schema under the policy in `semview.online`: at most
`budget_m` inline repairs per query, chosen by old-space rank, the rest to a
background repairer that drains in device-sized batches, and the inline budget
suppressed when the queue is deep. The queue and the claim set are imported
rather than reimplemented, so the policy measured offline by
`semview.baselines` and the policy running here are the same policy.

Three things differ from `ThawRetriever` and all three are deliberate:

  * **Presence and the claim both live in the database, not in the process.**
    `RepairCoordinator` keeps a `repaired` array and a `threading.Lock`, which
    are correct for one process and mean nothing to a second one. Here the
    window's presence comes from `docs_new` on every query and the claim is
    the `repair_claims` row -- `INSERT ... ON CONFLICT (id) DO NOTHING
    RETURNING id`, taken in key order, which is atomic across processes and is
    the protocol step 1's check 5 gated with eight concurrent workers and zero
    double claims. An earlier version of this file read presence from the
    database but still claimed with an in-process `ClaimSet` while this
    docstring promised multi-process safety: two services would each have seen
    a unit absent, each have claimed it locally, and both have encoded it --
    the exact cost guarantee the ledger rests on, broken by the one deployment
    shape the file was written for. Writes are idempotent as well, so a unit
    encoded twice after a reclaimed lease is wasted work and not a primary-key
    error.

  * **A query never waits for another query's in-flight repair.** The
    coordinator blocks on a Future; this takes the claim or moves on. A unit is
    still encoded at most once.

  * **Ranking covers what is present, not what the window contains.** The SQL
    is the same materialized-CTE exact sort; it is simply given fewer
    candidates, which is the quality this policy trades for its latency.

NOT RUN. This file was written against the schema and the helpers in
`retriever.py` and `db_io`, and its policy is covered by the tests in
`semview/tests.py`, but it has not been executed against a live PostgreSQL with
a GPU encoder. Treat it as a patch to review and run, not as a measured result.
"""

from __future__ import annotations

import contextlib
import sys as _sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from semview.online import RepairQueue


class PgClaimSet:
    """The claim, in the database, so it holds across service processes.

    `INSERT INTO repair_claims ... ON CONFLICT (id) DO NOTHING RETURNING id`
    is atomic: of any number of processes offering the same unit, exactly one
    gets the row back and the rest get nothing. Rows are offered in key order
    for the same reason `checks.py` does it -- two overlapping windows claiming
    in opposite orders can otherwise wait on each other.

    Release is a `DELETE`. A holder that dies releases nothing, so its row
    stays and the unit would be unclaimable forever; `reclaim_expired` is the
    sweep for that, deleting claims older than the lease whose unit still has
    no vector. `docs_new` is the done marker: a unit with a vector needs no
    claim, expired or not.

    This is step 1's check 5 (`claims`, 8 workers, 0 rows claimed twice,
    `lease_reclaimed` 100) used in anger rather than measured and set aside.
    """

    def __init__(self, owner: str, lease_s: float = 300.0):
        """`lease_s` must exceed the slowest batch this deployment can
        produce. It is the time after which a claim is treated as abandoned,
        so a lease shorter than an encode makes a slow worker look dead, and
        the unit is then encoded twice. 300 s against a batch of 16 on the
        measured hardware is two orders of margin; a deployment that raises
        the batch size or shares the device raises this with it."""
        self.owner, self.lease_s = str(owner), float(lease_s)

    def claim(self, conn, ids) -> list[int]:
        ids = sorted(int(i) for i in ids)
        if not ids:
            return []
        return [r[0] for r in conn.execute(
            "INSERT INTO repair_claims (id, owner) "
            "SELECT x, %s FROM unnest(%s::bigint[]) AS x ORDER BY x "
            "ON CONFLICT (id) DO NOTHING RETURNING id",
            (self.owner, ids)).fetchall()]

    def release(self, conn, ids) -> None:
        ids = [int(i) for i in ids]
        if ids:
            conn.execute("DELETE FROM repair_claims WHERE id = ANY(%s) "
                         "AND owner = %s", (ids, self.owner))

    def renew(self, conn, ids) -> int:
        """Push these claims' `claimed_at` forward: this owner is still alive.

        Without this the lease is a race against the encoder. A worker whose
        batch takes longer than `lease_s` has its OWN live claim reclaimed by
        somebody else's sweep, that somebody re-claims the unit and encodes it
        too, and the unit is encoded twice -- the cost guarantee broken not by
        a crash but by a slow device, which is the ordinary case on a loaded
        GPU. The write is idempotent so nothing corrupts; the ledger is simply
        wrong.

        A holder renews while it works, so a claim that stops being renewed is
        genuinely abandoned and the sweep means what it says.
        """
        ids = [int(i) for i in ids]
        if not ids:
            return 0
        # RETURNING, not len(ids): the update matches only rows this owner
        # still holds, so a claim already reclaimed by somebody else renews
        # NOTHING -- and reporting the number asked for made the heartbeat
        # counter say the opposite of what had happened. A caller that sees
        # fewer rows than it asked for has lost those claims and should not
        # go on believing it holds them.
        got = conn.execute(
            "UPDATE repair_claims SET claimed_at = now() "
            "WHERE id = ANY(%s) AND owner = %s RETURNING id",
            (ids, self.owner)).fetchall()
        return len(got)

    def release_completed(self, conn) -> int:
        """Delete claims whose unit already has a vector, whoever holds them.

        `reclaim_expired` only takes claims whose unit is still MISSING, which
        leaves a hole: a worker that dies between the write and the release
        leaves a claim on a unit that is finished. It is past the lease and it
        is never reclaimed, because the sweep reads the vector's presence as
        proof that somebody is on it -- so the row sits there forever, every
        other process fails to claim that id, re-queues it, and fails again.
        The recover scenario counted exactly those ids as "still blocked".

        Finishing and abandoning are different events and need different
        sweeps: a claim on a repaired unit is completed work to tidy up, a
        claim on an unrepaired unit past its lease is abandoned work to
        reclaim. This is the first.
        """
        rows = conn.execute(
            "DELETE FROM repair_claims c "
            "WHERE EXISTS (SELECT 1 FROM docs_new d WHERE d.id = c.id) "
            "RETURNING id").fetchall()
        return len(rows)

    def reclaim_expired(self, conn) -> int:
        """Drop claims past the lease whose unit still has no vector.

        Returns how many were reclaimed. Worth logging: a number that keeps
        growing means owners are dying mid-encode, which no latency percentile
        would show.
        """
        rows = conn.execute(
            "DELETE FROM repair_claims c "
            "WHERE c.claimed_at < now() - make_interval(secs => %s) "
            "AND NOT EXISTS (SELECT 1 FROM docs_new d WHERE d.id = c.id) "
            "RETURNING id", (self.lease_s,)).fetchall()
        return len(rows)


def online_io(dsn: str):
    """`fetch_text` and an **idempotent** `write`, each on its own connection.

    `retriever.db_io`'s writer is a bare binary `COPY` into `docs_new`, which
    raises on a duplicate primary key. That is the right strictness for the
    synchronous path, where one process owns every repair. Across processes a
    duplicate is reachable without anything being wrong -- an owner whose
    lease was reclaimed mid-encode finishes and writes after the new owner did
    -- and the second writer should lose quietly rather than raise inside a
    serving request. So this one stages the batch in a TEMP table and inserts
    `ON CONFLICT (id) DO NOTHING`.
    """
    from .db import connect

    conn = connect(dsn)
    lock = threading.Lock()
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS stage_new "
                 "(id bigint PRIMARY KEY, emb vector, origin text)")

    def fetch_text(ids):
        with lock:
            rows = dict(conn.execute(
                "SELECT id, text FROM docs WHERE id = ANY(%s)", (ids,)).fetchall())
        return [rows[i] for i in ids]

    def write(ids, vecs):
        with lock:
            conn.execute("TRUNCATE stage_new")
            with conn.cursor() as cur:
                with cur.copy("COPY stage_new (id, emb, origin) "
                              "FROM STDIN WITH (FORMAT BINARY)") as cp:
                    cp.set_types(["int8", "vector", "text"])
                    for i, v in zip(ids, vecs):
                        cp.write_row([int(i), v, "repair"])
            conn.execute("INSERT INTO docs_new (id, emb, origin) "
                         "SELECT id, emb, origin FROM stage_new "
                         "ON CONFLICT (id) DO NOTHING")

    return fetch_text, write, conn


@dataclass
class OnlineTrace:
    """Where one query's time went. The same split as `retriever.Trace`, plus
    what the policy gave up: how much of the window was actually ranked."""
    q_old_ms: float = 0.0
    q_new_ms: float = 0.0
    q_new_wait_ms: float = 0.0   # what the request waited, after overlapping
    written_ms: float = 0.0      # searching the partial index over post-switch writes
    n_written: int = 0           # candidates no old-space window could reach
    window_ms: float = 0.0
    present_ms: float = 0.0
    inline_repair_ms: float = 0.0
    rank_ms: float = 0.0
    wall_ms: float = 0.0
    window_size: int = 0
    n_present: int = 0
    n_inline: int = 0
    n_queued: int = 0
    n_contended: int = 0
    shed: bool = False
    queue_depth: int = 0
    repair_error: str | None = None
    results: list = field(default_factory=list)

    @property
    def window_coverage(self) -> float:
        return self.n_present / self.window_size if self.window_size else 1.0


class OnlinePgRetriever:
    """One instance per thread (each holds its own connection); share the
    queue and the encoder between them. The background repairer is a thread
    too, so it gets a connection of its own as well -- see
    `BackgroundRepairer`.

    The claim is NOT shared in the process: it is a `repair_claims` row, so it
    is shared with every other service process against the same database. The
    queue stays per-process, which is correct -- it is a work list, not a
    guarantee, and two processes wasting a queue slot on the same unit costs a
    presence check, not an encode.

    `encode_new(texts) -> (n, d)` and `fetch_text(ids) -> [str]` and
    `write(ids, vecs)` are callables in the shape `db_io` returns, but use
    `online_io`, whose writer is idempotent: see its docstring for why a bare
    `COPY` is wrong once more than one process can repair.
    """

    def __init__(self, conn, encode_new, fetch_text, write,
                 enc_old_query, enc_new_query, kprime: int, k: int = 10,
                 budget_m: int = 16, shed_at: int = 10_000,
                 drain_batch: int = 16, overlap_query_encode: bool = True,
                 queue: RepairQueue | None = None,
                 claims: PgClaimSet | None = None,
                 owner: str | None = None, lease_s: float = 300.0,
                 iterative: str = "relaxed_order", ef_max: int = 1000):
        self.conn = conn
        self.encode_new, self.fetch_text, self.write = encode_new, fetch_text, write
        self.enc_old_query, self.enc_new_query = enc_old_query, enc_new_query
        self.kprime, self.k = int(kprime), int(k)
        self.budget_m, self.shed_at = int(budget_m), int(shed_at)
        self.drain_batch = int(drain_batch)
        self.overlap_query_encode = bool(overlap_query_encode)
        self.encode_join_timeout = 30.0
        self.abandoned_encodes = 0
        # One queue per service process, as one coordinator was shared before.
        self.queue = queue if queue is not None else RepairQueue()
        # The claim crosses processes, so the owner has to identify this
        # process and not just this object: a reclaimed lease is read back by
        # owner, and two services calling themselves the same thing would
        # release each other's claims.
        if claims is None:
            import os
            claims = PgClaimSet(
                owner or f"{os.uname().nodename}:{os.getpid()}", lease_s=lease_s)
        self.claims = claims
        self.encoded_units = 0
        self.put_failures = 0
        self._enc_lock = threading.Lock()
        # pgvector caps hnsw.ef_search at 1000, so a window deeper than that
        # needs iterative scans to come back full -- the same configuration the
        # synchronous path established, kept here so the two are comparable.
        conn.execute(f"SET hnsw.ef_search = {min(self.kprime, ef_max)}")
        conn.execute(f"SET hnsw.iterative_scan = {iterative}")

    # ------------------------------------------------------------- repairing
    def _encode_and_store(self, ids: list[int], conn=None) -> int:
        """Encode and persist these units; return how many were encoded.

        The presence re-check that `semview.online` does against its bitmap is
        done here by the database, and it is the second line of defence rather
        than the first: the `repair_claims` row is what makes an encode
        exclusive, and this filters the units a reclaimed lease or a
        just-finished owner has already written, so they are not encoded again
        for nothing.

        `conn` is which connection that re-check runs on, and it is a
        parameter rather than `self.conn` because this method has two callers
        on two different threads. A query thread passes nothing and uses its
        own connection, which it owns for its lifetime. The background
        repairer passes its own. A psycopg connection is one socket carrying
        one transaction, and `db_io` already gives the repair I/O its own, so
        letting the repairer thread issue this SELECT on a query thread's
        connection would interleave two conversations on one socket -- the
        failure shows up as a mispaired result or a broken connection under
        exactly the concurrency this policy exists to serve.
        """
        if not ids:
            return 0
        c = self.conn if conn is None else conn
        ids = sorted(int(i) for i in ids)
        have = {r[0] for r in c.execute(
            "SELECT id FROM docs_new WHERE id = ANY(%s)", (ids,)).fetchall()}
        ids = [i for i in ids if i not in have]
        if not ids:
            self.queue.forget(have)
            return 0
        # One exception-safe region, from reading the texts to persisting
        # them. This guarded only `write()` while the numpy realization of the
        # same choke point guarded all three, so the two disagreed about what
        # a failed fetch or a failed encode means -- and the encoder is the
        # likely failure, not the insert. An inline repair's units come
        # straight off a window and were never queued, so losing them here
        # loses them entirely.
        try:
            vecs = self.encode_new(self.fetch_text(ids))
            self.write(ids, vecs)
        except Exception:                                          # noqa: BLE001
            # The unit is not repaired until it is persisted. Counting the
            # encode and forgetting the unit here would spend the forward pass
            # and leave nothing behind: the next query would find it absent
            # again, so the work is not lost, only repeated -- but a run that
            # did not surface the failure would report a saving it never made.
            # Put it back in the queue and count it where a summary will show
            # it.
            why = f"{type(_sys.exc_info()[1]).__name__}: {_sys.exc_info()[1]}"
            with self._enc_lock:
                self.put_failures += len(ids)
            # same poison rule as the numpy path: a batch failure is evidence
            # about the batch, not about each unit, so a suspect is retried
            # alone and only a unit that fails by itself is quarantined
            dead = self.queue.record_failure(ids, why.splitlines()[0],
                                             isolated=len(ids) == 1)
            self.queue.offer([u for u in ids if u not in set(dead)])
            raise
        with self._enc_lock:
            self.encoded_units += len(ids)
        self.queue.forget(ids)
        # and clear the poison counter, which the numpy path does and this one
        # did not: a unit that failed twice early on, succeeded, then met a
        # transient fault months later would carry the old strikes and be
        # quarantined far too soon. Two implementations of one policy, drifting
        # again.
        self.queue.clear_failures(ids)
        return len(ids)

    # ---------------------------------------------------------------- serving
    def query(self, text: str | None = None, q_old: np.ndarray | None = None,
              q_new: np.ndarray | None = None) -> OnlineTrace:
        tr = OnlineTrace()
        t_query = time.perf_counter()
        c = self.conn
        if q_old is None:
            t0 = time.perf_counter()
            q_old = self.enc_old_query.encode([text])[0]
            tr.q_old_ms = (time.perf_counter() - t0) * 1e3

        # The new-space query vector depends on nothing this request is about
        # to do: not the window, not the presence read, not the repairs. It
        # used to be encoded AFTER the inline repair, which put a GPU call at
        # the end of a request that had just spent 400-500 ms in the database
        # with the GPU idle. Starting it here lets the two resources run at
        # once; the request waits only for whatever is left when the window
        # and the repairs are done.
        #
        # This is unmeasured. It cannot make the critical path longer -- the
        # work is the same and the wait is bounded by the encode -- but how
        # much it saves depends on whether this encoder contends with the
        # repair encoder for one device, which is exactly what the 1-thread to
        # 8-thread measurement says dominates. `q_new_ms` is what the encode
        # took and `q_new_wait_ms` is what the request actually waited, so the
        # difference is the saving and a run can report it instead of assuming
        # it.
        pending = None
        if q_new is None and text is not None and self.overlap_query_encode:
            box: dict = {}

            def _encode_new_query():
                s = time.perf_counter()
                try:
                    box["v"] = self.enc_new_query.encode([text])[0]
                except Exception as e:                             # noqa: BLE001
                    box["err"] = e
                box["ms"] = (time.perf_counter() - s) * 1e3

            pending = threading.Thread(target=_encode_new_query,
                                       name="thaw-qnew", daemon=True)
            pending.start()

        # Everything from here is inside a region that joins the encode
        # thread. It is joined on the normal path too, a few lines down; this
        # exists for the paths that do NOT get there -- a window or presence
        # read that raises used to leave a daemon thread running a forward
        # pass for an answer nobody would read, and under a database fault
        # every failing request would start another one, so the device would
        # fill with work for discarded queries exactly when it was least able
        # to spare it.
        try:

            # The window comes back in old-space distance order, which is the order
            # the inline budget spends itself along: nearest first.
            t0 = time.perf_counter()
            window = [r[0] for r in c.execute(
                "SELECT id FROM docs ORDER BY emb_old <=> %s LIMIT %s",
                (q_old, self.kprime)).fetchall()]
            t1 = time.perf_counter()
            present = {r[0] for r in c.execute(
                "SELECT id FROM docs_new WHERE id = ANY(%s)", (window,)).fetchall()}
            t2 = time.perf_counter()
            tr.window_ms, tr.present_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3
            tr.window_size = len(window)

            miss = [u for u in window if u not in present]
            tr.queue_depth = self.queue.depth
            tr.shed = tr.queue_depth > self.shed_at
            m = 0 if tr.shed else max(self.budget_m, 0)

            t0 = time.perf_counter()
            if m and miss:
                head = miss[:m]
                mine = self.claims.claim(c, head)
                # Contended now means another PROCESS or thread holds the claim,
                # which is the number worth watching: it is how much of the inline
                # budget a second service is already spending.
                tr.n_contended = len(head) - len(mine)
                try:
                    tr.n_inline = self._encode_and_store(mine)
                    present.update(mine)
                except Exception as e:                                 # noqa: BLE001
                    # A repair that could not be persisted must not fail the
                    # request: this policy's whole premise is that an answer over
                    # part of the window is better than no answer. The units go
                    # back on the queue in `_encode_and_store`, the coverage this
                    # query reports is the lower one it actually had, and the
                    # reason is in the trace rather than in a log nobody reads.
                    tr.repair_error = f"{type(e).__name__}: {e}".splitlines()[0]
                finally:
                    self.claims.release(c, mine)
                rest = miss[len(head):]
            else:
                rest = miss
            if rest:
                self.queue.offer(rest)
            tr.n_queued = len(rest)
            tr.inline_repair_ms = (time.perf_counter() - t0) * 1e3

            if q_new is None:
                t0 = time.perf_counter()
                if pending is not None:
                    pending.join()
                    tr.q_new_wait_ms = (time.perf_counter() - t0) * 1e3
                    tr.q_new_ms = float(box.get("ms", 0.0))
                    if "err" in box:
                        raise box["err"]
                    q_new = box["v"]
                else:
                    q_new = self.enc_new_query.encode([text])[0]
                    tr.q_new_ms = tr.q_new_wait_ms = (time.perf_counter() - t0) * 1e3

            # Rows written after the switch -- inserts and updates -- have no
            # old vector, so NO old-space window can ever surface them. The
            # synchronous path has always searched the partial HNSW index over
            # origin='write' rows and joined that top-k to its candidates;
            # this path did not, which meant a document created after the
            # migration started was invisible to it for as long as the
            # migration ran. That is a correctness gap rather than a tuning
            # choice: the two realizations answered different questions. With
            # no writes the partial index is empty and this costs well under a
            # millisecond.
            t0 = time.perf_counter()
            written = [r[0] for r in c.execute(
                "SELECT id FROM docs_new WHERE origin = 'write' "
                "ORDER BY emb <=> %s LIMIT %s", (q_new, self.k)).fetchall()]
            tr.written_ms = (time.perf_counter() - t0) * 1e3
            in_window = set(window)
            tr.n_written = len([i for i in written if i not in in_window])

            # Rank what is present. The materialized CTE is load-bearing: without
            # it the planner may answer with the HNSW index on docs_new and filter
            # afterwards, returning too few or misordered rows (the union self-test
            # found this on the synchronous path and it applies unchanged here).
            candidates = [u for u in window if u in present]
            tr.n_present = len(candidates)
            candidates = candidates + [i for i in written if i not in in_window]
            t0 = time.perf_counter()
            tr.results = [r[0] for r in c.execute(
                "WITH cand AS MATERIALIZED (SELECT id, emb FROM docs_new "
                "WHERE id = ANY(%s)) SELECT id FROM cand ORDER BY emb <=> %s LIMIT %s",
                (candidates, q_new, self.k)).fetchall()] if candidates else []
            tr.rank_ms = (time.perf_counter() - t0) * 1e3
            tr.wall_ms = (time.perf_counter() - t_query) * 1e3
            return tr
        finally:
            if pending is not None and pending.is_alive():
                pending.join(self.encode_join_timeout)
                if pending.is_alive():
                    # cannot be cancelled; count it so a run can see the
                    # device being held by abandoned work
                    with self._enc_lock:
                        self.abandoned_encodes += 1


class BackgroundRepairer:
    """Drains the queue on its own connection, in device-sized batches.

    Batching across queries rather than within one is the point: a query's stale
    set averages 15.7 units, while the repairer can always issue the device's
    measured best batch. That removes the small-batch penalty the cost model has
    to argue about on the synchronous path.

    "Its own connection" is literal, and it is the caller's job to supply one:
    pass `conn`, or pass `dsn` and one is opened here on first use. Both the
    repairer thread and `drain_now` then read presence on that connection
    instead of on a query thread's, which is the only thing in this file that
    two threads would otherwise share. Handing it a retriever's connection is
    refused rather than tolerated.
    """

    def __init__(self, retriever: OnlinePgRetriever, conn=None,
                 dsn: str | None = None, idle_sleep: float = 0.005,
                 sweep_every_s: float = 60.0):
        if conn is not None and conn is retriever.conn:
            raise SystemExit(
                "the repairer needs its own connection: it runs on its own "
                "thread, and sharing a query thread's connection interleaves "
                "two transactions on one socket. Pass dsn=... instead, or "
                "thaw_pg.db.connect() a second connection for it.")
        self.r, self.idle_sleep = retriever, idle_sleep
        self.dsn, self._conn, self._owns_conn = dsn, conn, False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.encoded = 0
        self.errors = 0
        self.reclaimed = 0
        self.released = 0
        self.renewals = 0
        self.renewals_lost = 0
        self._backoff = 0.0
        self.max_backoff_s = 30.0
        # a third of the lease: three missed beats before anyone calls this
        # worker dead, which tolerates a stalled connection without tolerating
        # a stalled process
        self.heartbeat_s = float(retriever.claims.lease_s) / 3.0
        self.sweep_every_s = float(sweep_every_s)
        self.last_error: str | None = None

    @property
    def conn(self):
        """The repairer's connection, opened on first use if it was given a DSN.

        Opened lazily and on the thread that first needs it, so a repairer
        constructed in a service's setup does not hold an idle backend until
        it is actually started.
        """
        if self._conn is None:
            if self.dsn is None:
                raise SystemExit(
                    "BackgroundRepairer has no connection of its own: "
                    "construct it with conn=... or dsn=...")
            from .db import connect
            self._conn, self._owns_conn = connect(self.dsn), True
        return self._conn

    def _repair(self, batch: list[int]) -> tuple[int, list[int], int]:
        """Claim what is free in `batch`, encode it.

        Returns (encoded, unclaimed, claimed): `encoded` can be 0 while
        `claimed` is not, when the units turned out to be present already --
        that is progress, and `drain_now` must not confuse it with being
        blocked.
        """
        mine, c = [], None
        try:
            # `self.conn` can raise -- it opens the connection on first use --
            # and so can the claim, both of them OUTSIDE the old try. The
            # caller had already taken this batch off the queue, so a failure
            # there dropped it silently: in no queue, claimed by nobody, and
            # never asked for again. Whoever takes from the queue returns it.
            c = self.conn
            mine = self.r.claims.claim(c, batch)
            unclaimed = [u for u in batch if u not in set(mine)]
            # Renew for the whole encode, not once before it. Renewing once
            # only moves the deadline: an encode longer than `lease_s` still
            # has its own claim swept by another process, which re-claims and
            # re-encodes the unit -- the encode-once guarantee lost to a slow
            # device rather than a crash. The heartbeat holds the claim for as
            # long as this worker is really working, which is what makes a
            # claim that STOPS being renewed mean something.
            self.r.claims.renew(c, mine)
            with self._heartbeat(mine):
                n = self.r._encode_and_store(mine, conn=c)
        except Exception:                                          # noqa: BLE001
            self.r.queue.offer(batch)
            raise
        finally:
            if c is not None and mine:
                try:
                    self.r.claims.release(c, mine)
                except Exception:                                  # noqa: BLE001
                    # the claim row outlives us; the lease sweep reclaims it
                    pass
        return n, unclaimed, len(mine)

    def sweep_leases(self) -> int:
        """Reclaim claims whose owner died, and say how many.

        `PgClaimSet.reclaim_expired` existed and nothing called it, which made
        the lease a comment rather than a mechanism: a process that died
        holding claims left rows that no other worker could take, so those
        units were unrepairable for the life of the database and the migration
        quietly stopped converging. A claim is only reclaimed when its unit
        still has no vector, so a slow owner that finishes after the sweep
        loses nothing but the claim it no longer needs.
        """
        # tidy finished claims first, then reclaim abandoned ones. A claim on
        # a repaired unit is never "expired" to the second sweep, so without
        # the first it would sit forever and block that id for everyone.
        done = self.r.claims.release_completed(self.conn)
        if done:
            self.released += done
        n = self.r.claims.reclaim_expired(self.conn)
        if n:
            self.reclaimed += n
            # Reclaiming does not re-queue: nothing here knows which units
            # those were. They return to the queue the next time a query's
            # window finds them missing, which is the same path every other
            # stale unit takes.
        return n

    @contextlib.contextmanager
    def _heartbeat(self, ids):
        """Keep renewing `ids` until the body finishes.

        On its own connection, because the body is using the repairer's and
        this module allows one connection one thread. With no DSN to open one
        from -- a test passing `conn=` directly -- it renews on exit instead,
        which still shortens the window and cannot interleave.
        """
        if not ids or self.heartbeat_s <= 0:
            yield
            return
        stop = threading.Event()
        own = None
        if self.dsn is not None:
            from .db import connect
            own = connect(self.dsn)

        def beat():
            while not stop.wait(self.heartbeat_s):
                try:
                    got = self.r.claims.renew(own, ids)
                    self.renewals += 1
                    if got < len(ids):
                        # somebody reclaimed part of this batch while it was
                        # being encoded: the encode-once guarantee is already
                        # gone for those ids, and a run has to be able to see
                        # that rather than read a heartbeat count as proof
                        self.renewals_lost += len(ids) - got
                except Exception:                                  # noqa: BLE001
                    return          # the sweep will decide; do not crash here

        t = None
        if own is not None:
            t = threading.Thread(target=beat, name="thaw-pg-heartbeat",
                                 daemon=True)
            t.start()
        try:
            yield
        finally:
            stop.set()
            if t is not None:
                t.join(5.0)
            if own is not None:
                own.close()
            elif self._conn is not None:
                # no second connection: renew once more now that the encode is
                # done and the connection is free again
                try:
                    self.r.claims.renew(self._conn, ids)
                    self.renewals += 1
                except Exception:                                  # noqa: BLE001
                    pass

    def _loop(self) -> None:
        # once at startup, so a restart after a crash clears the dead process's
        # claims rather than inheriting them
        last_sweep = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_sweep >= self.sweep_every_s:
                last_sweep = now
                try:
                    self.sweep_leases()
                except Exception as e:                             # noqa: BLE001
                    self.errors += 1
                    self.last_error = f"lease sweep: {type(e).__name__}: {e}"
            batch = self.r.queue.take(self.r.drain_batch)
            if not batch:
                self._stop.wait(self.idle_sleep)
                continue
            # see semview.online: a unit taken from the queue and claimed by
            # someone else is in no queue, so a holder that dies without
            # releasing would strand it permanently
            try:
                done, unclaimed, _claimed = self._repair(batch)
            except Exception as e:                                 # noqa: BLE001
                # An exception here used to kill the thread, and `stop()` then
                # reported success because a dead thread joins instantly: the
                # queue would grow for the rest of the run with nothing
                # draining it and nothing saying so. Count it, record the last
                # one, and keep draining; `_repair` has returned the batch to
                # the queue on its way out.
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}".splitlines()[0]
                # Back off, growing: a database that went away or an inference
                # server returning 503 fails the next batch too, and retrying
                # at `idle_sleep` turns an outage into a tight loop against a
                # service that is already struggling.
                self._backoff = min(max(self._backoff * 2, self.idle_sleep * 4),
                                    self.max_backoff_s)
                self._stop.wait(self._backoff)
                continue
            self._backoff = 0.0
            if done == 0 and unclaimed:
                # nothing was ours this round: another process holds these.
                # Re-queueing and immediately re-claiming is how two services
                # on the same hot ids turn into a pair of database hammers.
                self._stop.wait(min(self.idle_sleep * 8, self.max_backoff_s))
            self.encoded += done
            self.r.queue.offer(unclaimed)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="thaw-pg-repairer",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> bool:
        """Ask the repairer to stop and wait for it. True if it actually did.

        The reference is dropped only when the thread has exited. Clearing it
        after a timed-out join -- which is what this did -- loses the handle to
        a thread that is still mid-encode: `start()` would then launch a second
        repairer beside it, and `close()` would close the connection the live
        one is using, which surfaces as a corrupt result or an
        InterfaceError from inside a serving request.
        """
        self._stop.set()
        w = self._thread
        if w is None:
            return True
        w.join(timeout)
        if w.is_alive():
            return False
        self._thread = None
        return True

    def close(self, timeout: float = 10.0) -> None:
        """Stop, and give back the connection if this object opened it."""
        if not self.stop(timeout):
            raise SystemExit(
                f"the repairer thread is still running after {timeout}s, so "
                f"its connection is still in use and closing it would break "
                f"whatever statement is in flight. It is most likely inside a "
                f"forward pass; wait and call close() again.")
        if self._owns_conn and self._conn is not None:
            self._conn.close()
            self._conn, self._owns_conn = None, False

    def drain_now(self) -> int:
        """Finish the backlog on this thread. For ending a measured run.

        Refuses while the background thread is alive. It runs on the caller's
        thread but uses `self.conn`, which the repairer thread is also using,
        and this module's whole connection discipline is that one connection
        serves one thread. `integration.py` called `start()` and then
        `drain_now()` without stopping in between -- the convention broken by
        the file written to demonstrate it. Making it refuse is better than
        remembering: stop the worker, then drain.
        """
        if self._thread is not None and self._thread.is_alive():
            raise SystemExit(
                "drain_now() while the repairer thread is running: it would "
                "issue statements on the connection that thread is using. "
                "Call stop() first, then drain_now().")
        done = 0
        while True:
            batch = self.r.queue.take(self.r.drain_batch)
            if not batch:
                return done
            n, unclaimed, claimed = self._repair(batch)
            done += n
            self.r.queue.offer(unclaimed)
            if unclaimed and not claimed:
                # everything left is held by someone else; spinning on it would
                # busy-wait on another thread's encode
                return done


def summarise(traces: list[OnlineTrace], retriever: OnlinePgRetriever | None = None,
              repairer: "BackgroundRepairer | None" = None) -> dict:
    """Latency beside coverage, because coverage is what the latency cost.

    Pass the retriever and the repairer and the failures come too. A repair
    that could not be persisted makes a run look faster, so a summary that
    reported only the percentiles would price a migration that did not happen.
    """
    if not traces:
        raise SystemExit("no query was served: there is no distribution to report")
    wall = np.array([t.wall_ms for t in traces])
    cov = np.array([t.window_coverage for t in traces])
    pct = lambda a, p: float(np.percentile(a, p))
    n = len(traces)
    first, last = slice(0, max(1, n // 10)), slice(max(1, 9 * n // 10), n)
    comp = {f"{k}_mean": float(np.mean([getattr(t, k) for t in traces]))
            for k in ("window_ms", "present_ms", "inline_repair_ms", "rank_ms",
                      "q_old_ms", "q_new_ms", "q_new_wait_ms")}
    return {
        "queries": n,
        "wall_p50": pct(wall, 50), "wall_p95": pct(wall, 95),
        "wall_p99": pct(wall, 99), "wall_max": float(wall.max()),
        "window_coverage_mean": float(cov.mean()),
        "window_coverage_p05": pct(cov, 5),
        "inline_per_query_mean": float(np.mean([t.n_inline for t in traces])),
        "queued_total": int(sum(t.n_queued for t in traces)),
        "contended_total": int(sum(t.n_contended for t in traces)),
        "shed_queries": int(sum(1 for t in traces if t.shed)),
        "queue_depth_p95": pct(np.array([t.queue_depth for t in traces]), 95),
        "first_decile_p50": pct(wall[first], 50),
        "last_decile_p50": pct(wall[last], 50),
        "queries_with_a_repair_error": int(sum(1 for t in traces if t.repair_error)),
        # what overlapping the new-query encode with the database work saved:
        # the encode's own time minus what the request actually waited for it
        "q_new_overlap_saved_ms_mean": float(np.mean(
            [max(0.0, t.q_new_ms - t.q_new_wait_ms) for t in traces])),
        "put_failures": int(getattr(retriever, "put_failures", 0)),
        "repairer_errors": int(getattr(repairer, "errors", 0)),
        # a number that keeps growing means owners are dying mid-encode, which
        # no latency percentile would show
        "claims_reclaimed": int(getattr(repairer, "reclaimed", 0)),
        "claim_renewals": int(getattr(repairer, "renewals", 0)),
        # non-zero means a claim was taken from a live worker mid-encode
        "claim_renewals_lost": int(getattr(repairer, "renewals_lost", 0)),
        "claims_released_after_completion": int(getattr(repairer, "released", 0)),
        "quarantined_units": len(retriever.queue.quarantined) if retriever else 0,
        "queue_dropped": int(retriever.queue.dropped) if retriever else 0,
        "queue_depth_at_end": int(retriever.queue.depth) if retriever else 0,
        "abandoned_encodes": int(getattr(retriever, "abandoned_encodes", 0)),
        "last_repairer_error": getattr(repairer, "last_error", None),
        **comp,
    }
