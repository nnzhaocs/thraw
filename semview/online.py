"""An online serving path: the encoder off the critical path, under a budget.

`serve.py` is a synchronous prototype and says so: a query waits for its own
repairs, which makes its latency an upper bound. That bound is not small. On the
realization's measured query path, 1,362 ms of a 1,400 ms median is the encoder,
and with eight workers against one GPU the 99th percentile reaches 16 s, because
nothing stops queries queueing behind each other's forward passes. No retrieval
service with a sub-second objective can be deployed in that shape.

This module is the deployable shape. It changes three things and nothing else:

  * **The encoder leaves the critical path.** A query ranks the window members
    that already have a new vector and returns. The rest are handed to a
    background repairer. Quality then depends on how much traffic came before,
    rather than being fixed by the window depth.

  * **A bounded inline budget buys quality back.** Returning immediately gives
    the earliest queries the worst answers, so a query may repair at most `m` of
    its missing members before answering. Which `m` is not arbitrary: the only
    evidence available before a unit has a new vector is its OLD score, so the
    window is ordered by that and the budget is spent on its head. Latency is
    then bounded by `window + m * unit_ms + rank`, which is a number an operator
    sets against an objective rather than discovers.

  * **Admission control bounds the tail.** When the repair queue is deeper than
    `shed_at`, the inline budget drops to zero and queries serve from what is
    already repaired. This is what makes the 99th percentile a property of
    policy instead of a property of queue depth.

Two further choices are worth stating because they differ from the synchronous
coordinator in `§4.2`, and the difference is the point:

  * **A query never waits for another query's in-flight repair.** The
    synchronous coordinator blocks, which guarantees a unit is encoded once and
    makes every answer exact over its window. Here a unit already claimed by
    someone else is treated as missing for this query and ranked without. A unit
    is still encoded at most once; what is given up is the guarantee that an
    answer sees every member of its window.

  * **The queue drops rather than blocks when full.** A dropped unit is not
    lost: the next query that misses it raises its miss count and re-queues it,
    so the backlog self-corrects toward whatever traffic actually wants. A
    bounded queue that blocks would put the encoder back on the critical path
    through the back door.

What this module does NOT do, so that a reader does not have to infer it: it
does not schedule across devices, it does not persist the queue (a restart
re-learns it from traffic, which costs repeats but no correctness), and it does
not implement the write path for corpus updates. Search is whatever the injected
index does, so an ANN index contributes its own recall error here exactly as it
does in the realization.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import numpy as np


# -------------------------------------------------------------------- tracing ---
class Tracer:
    """A JSONL event log for the concurrent path.

    `log.txt` records what a run printed; it cannot show the interleaving of
    two queries and a background repairer over one device, which is where this
    path's only correctness risk lives. Each line carries a monotonic offset, an
    event name, the thread, and the units involved, so the sequence that caused
    a problem can be read off rather than inferred.

    Specifically: the double-encode this module once had -- a claim won after
    its previous owner published and released, on a stale absence -- appears as
    `claim` then `skip_already_present` for the same unit. That bug was found by
    a test, and with this it would also have been visible without one.

    Writing is under a lock and flushed per line, because a trace that is
    buffered when the process dies is a trace of the wrong thing.
    """

    def __init__(self, path=None):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8") if path else None
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.n = 0

    def __call__(self, event: str, **fields) -> None:
        if self._fh is None:
            return
        rec = {"t": round(time.perf_counter() - self._t0, 6), "event": event,
               "thread": threading.current_thread().name, **fields}
        line = json.dumps(rec, separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            self.n += 1

    def close(self) -> None:
        if self._fh is not None:
            with self._lock:
                self._fh.close()
                self._fh = None


#: a tracer that costs nothing when no path is given, so the serving path has
#: one call site rather than a branch at every event
NO_TRACE = Tracer(None)


# --------------------------------------------------------------------- queue ---
class RepairQueue:
    """A bounded, de-duplicating queue ordered by how often a unit was missed.

    Arrival order is the obvious discipline and it is not the best one. A unit
    that several queries have already wanted is more likely to be wanted again
    than one seen once, so draining by miss count converges on the hot set
    faster than FIFO does. This is the demand-side counterpart to the
    corpus-side ordering prior work proposes, and it costs one counter per unit.

    `capacity` bounds the backlog. At capacity a new unit is dropped rather than
    queued, because the alternatives are worse: blocking puts the encoder back
    on the query path, and an unbounded queue turns a traffic spike into
    unbounded memory. A dropped unit returns on its next miss with a higher
    count, so the queue converges on what traffic repeatedly wants.
    """

    def __init__(self, capacity: int = 100_000, max_attempts: int = 5):
        self.capacity = int(capacity)
        self._misses: dict[int, int] = {}       # unit -> miss count
        self._buckets: dict[int, set[int]] = {}  # miss count -> units
        self._top = 0                            # highest non-empty bucket
        self._lock = threading.Lock()
        self.dropped = 0
        self.offered = 0
        # Poison handling. Returning a failed batch to the queue is right --
        # it is how work survives a transient fault -- but on its own it turns
        # a PERMANENT fault into a loop: one undecodable document fails its
        # batch, the batch comes back, and the repairer spends the device on
        # it forever while every healthy unit waits behind it. That failure
        # mode was introduced by the re-queue fix and has to be closed with
        # it. A unit that has failed `max_attempts` times is quarantined:
        # taken out of rotation, counted, and listed, so it shows up as a
        # named gap rather than as a migration that silently stops converging.
        self.max_attempts = int(max_attempts)
        self.split_after = 2
        self._attempts: dict[int, int] = {}
        self._solo: set[int] = set()
        self.quarantined: dict[int, str] = {}
        self.transient_failures = 0
        self.last_transient: str | None = None

    #: Failures that say nothing about the document. A database that went
    #: away, an inference server that returned 503, a socket that timed out --
    #: these fail every unit in the batch equally and will fail the next batch
    #: too, so counting them against the DOCUMENT quarantines healthy work for
    #: an outage. The queue counts them separately and backs off instead.
    TRANSIENT = ("connection", "timeout", "timed out", "temporarily",
                 "503", "502", "504", "econnreset", "broken pipe",
                 "server closed", "could not connect", "operationalerror",
                 "interfaceerror", "cannot allocate", "out of memory",
                 "cuda error", "device-side", "too many clients")

    @classmethod
    def looks_transient(cls, reason: str) -> bool:
        r = (reason or "").lower()
        return any(t in r for t in cls.TRANSIENT)

    def record_failure(self, ids, reason: str = "",
                       isolated: bool = False) -> list[int]:
        """Count a failed attempt on these units; return those now quarantined.

        The caller re-queues what this does NOT return.

        A batch fails as a unit, so a batch failure says *one of these* is bad
        and not *all of these are*. Counting it against everyone and
        quarantining on the count alone takes the innocent down with the
        guilty -- a first version of this quarantined twelve units when one
        was undecodable. So a failure in a batch only earns a unit
        `split_after` strikes, after which it is retried ALONE; and only a
        unit that fails on its own, `max_attempts` times, is quarantined. By
        then the evidence is specific: this unit, by itself, cannot be
        encoded.
        """
        if self.looks_transient(reason):
            # The service failed, not the document. Count it where an operator
            # can see it, leave every unit's strikes alone, and let the caller
            # back off: an outage would otherwise spend one strike per unit
            # per retry and quarantine a healthy corpus for being unlucky.
            with self._lock:
                self.transient_failures += 1
                self.last_transient = (reason or "")[:200]
            return []
        out = []
        with self._lock:
            for u in ids:
                u = int(u)
                self._attempts[u] = self._attempts.get(u, 0) + 1
                if self._attempts[u] >= self.split_after:
                    self._solo.add(u)
                if isolated and self._attempts[u] >= self.max_attempts:
                    self.quarantined[u] = reason[:200]
                    self._drop(u)
                    self._solo.discard(u)
                    out.append(u)
        return out

    def clear_failures(self, ids) -> None:
        """Forget the attempt history of units that succeeded."""
        with self._lock:
            for u in ids:
                self._attempts.pop(int(u), None)
                self._solo.discard(int(u))

    def _bump(self, u: int, to: int) -> None:
        """Move a unit into the bucket for miss count `to`. Caller holds the lock."""
        old = self._misses.get(u)
        if old is not None:
            b = self._buckets.get(old)
            if b is not None:
                b.discard(u)
                if not b:
                    del self._buckets[old]
        self._misses[u] = to
        self._buckets.setdefault(to, set()).add(u)
        if to > self._top:
            self._top = to

    def offer(self, ids) -> int:
        """Queue these units, or raise their miss count. Returns how many landed."""
        landed = 0
        with self._lock:
            for u in ids:
                u = int(u)
                if u in self.quarantined:
                    continue              # out of rotation; see record_failure
                self.offered += 1
                if u in self._misses:
                    self._bump(u, self._misses[u] + 1)
                    landed += 1
                elif len(self._misses) < self.capacity:
                    self._bump(u, 1)
                    landed += 1
                else:
                    self.dropped += 1
        return landed

    def take(self, n: int) -> list[int]:
        """Pop up to `n` of the most-missed units, highest count first.

        A unit that has already failed in a batch comes back ALONE, so the
        batch that proves it bad contains nothing else. Isolation beats
        throughput here: a suspect unit is at most one small batch of lost
        device time, while a wrong quarantine is a document that silently
        never migrates.
        """
        with self._lock:
            if not self._misses:
                return []
            suspect = next((u for u in self._solo if u in self._misses), None)
            if suspect is not None:
                self._drop(suspect)
                return [suspect]
            out = []
            # Walk the buckets from the highest miss count down. This used to
            # sort the whole queue on every take, under the lock that `offer`
            # also needs -- 15 ms at the default capacity of 100,000, which a
            # draining repairer pays continuously while every query trying to
            # enqueue its misses waits behind it. Same order out, no sort.
            while self._top > 0 and len(out) < n:
                b = self._buckets.get(self._top)
                if not b:
                    self._top -= 1
                    continue
                while b and len(out) < n:
                    u = b.pop()
                    del self._misses[u]
                    out.append(u)
                if not b:
                    self._buckets.pop(self._top, None)
            return out

    def snapshot(self) -> list[int]:
        """What is queued, without taking it.

        `take` is a work-handing-out call and it deliberately hands out a
        suspect unit ALONE, so it is the wrong way to ask what the queue
        holds: a caller draining with `take(10_000)` would get one unit and
        conclude the rest had been lost.
        """
        with self._lock:
            return sorted(self._misses)

    def _drop(self, u: int) -> None:
        """Remove a unit from the queue entirely. Caller holds the lock."""
        c = self._misses.pop(u, None)
        if c is None:
            return
        b = self._buckets.get(c)
        if b is not None:
            b.discard(u)
            if not b:
                del self._buckets[c]

    def forget(self, ids) -> None:
        """Drop these from the queue: someone else has repaired them."""
        with self._lock:
            for u in ids:
                self._drop(int(u))

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._misses)


# ------------------------------------------------------------------- claims ---
class ClaimSet:
    """Guarantees a unit is encoded at most once, without anyone waiting.

    The synchronous coordinator takes a lock and makes the loser wait for the
    winner's result. Here the loser is told no and serves without that unit,
    because waiting is the cost this module exists to remove. The guarantee
    kept is the one that matters for cost -- no unit is encoded twice -- and the
    one given up is that an answer sees its whole window.
    """

    def __init__(self):
        self._held: set[int] = set()
        self._lock = threading.Lock()

    def claim(self, ids) -> list[int]:
        """Return the subset this caller now owns; the rest are someone else's."""
        got = []
        with self._lock:
            for u in ids:
                u = int(u)
                if u not in self._held:
                    self._held.add(u)
                    got.append(u)
        return got

    def release(self, ids) -> None:
        with self._lock:
            for u in ids:
                self._held.discard(int(u))


# -------------------------------------------------------------------- traces ---
@dataclass
class OnlineTrace:
    """Where one query's time went, and what it gave up to spend so little."""
    window_ms: float
    inline_repair_ms: float
    rank_ms: float
    n_window: int
    n_visible: int          # window members that had a new vector when ranked
    n_inline: int           # repaired on the critical path
    n_queued: int           # handed to the repairer
    n_contended: int        # missing, and claimed by another query in flight
    shed: bool              # the inline budget was suppressed by load
    queue_depth: int
    repair_error: str | None = None   # an inline repair failed; answered anyway

    @property
    def total_ms(self) -> float:
        return self.window_ms + self.inline_repair_ms + self.rank_ms

    @property
    def window_coverage(self) -> float:
        """Share of the window that was ranked. 1.0 is the synchronous answer."""
        return self.n_visible / self.n_window if self.n_window else 1.0


# -------------------------------------------------------------------- server ---
@dataclass
class OnlineServer:
    """Serve from an old index, repair under a budget, queue the remainder.

    The injected pieces are deliberately narrow so this runs against the numpy
    stand-ins in `serve.py`, against a real vector store, or against PostgreSQL
    through `thaw_pg`:

      `search(q_old, kprime)`    -> (ids, scores), the window, best score first
      `encode_new(texts)`        -> (n, d) new-space vectors
      `encode_old_query(text)`   -> old-space query vector
      `encode_new_query(text)`   -> new-space query vector
      `texts[u]`                 -> the unit's original text, for repair
      `put(ids, vecs)`           -> persist repaired vectors (optional)

    `budget_m` is the inline repair budget per query, `shed_at` the queue depth
    above which it drops to zero, and `drain_batch` the size the background
    repairer uses -- which should be the device's measured best batch, since
    batching across queries is what lets it be.
    """
    search: callable
    encode_new: callable
    encode_old_query: callable
    encode_new_query: callable
    texts: list[str]
    new_dim: int
    kprime: int = 500
    k: int = 10
    budget_m: int = 16
    shed_at: int = 10_000
    drain_batch: int = 16
    queue_capacity: int = 100_000
    put: callable = None
    trace: object = None           # a Tracer, or None for no event log

    V_new: np.ndarray = field(default=None, repr=False)
    repaired: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        n = len(self.texts)
        if self.V_new is None:
            self.V_new = np.zeros((n, self.new_dim), dtype=np.float32)
        if self.repaired is None:
            self.repaired = np.zeros(n, dtype=bool)
        self.queue = RepairQueue(self.queue_capacity)
        self.claims = ClaimSet()
        if self.trace is None:
            self.trace = NO_TRACE
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.encoded_units = 0
        self.put_failures = 0
        self.repairer_errors = 0
        self.last_repairer_error = None
        self._enc_lock = threading.Lock()

    # ------------------------------------------------------------- repairing
    def _encode_and_store(self, ids: list[int]) -> None:
        """Encode these units, publish them, and return how many were encoded.

        This is the only place that encodes.

        A claim alone does not prevent a double encode, which is the race the
        test suite found. A claim is exclusive only while it is held: once an
        owner publishes and releases, a second caller that read `repaired`
        BEFORE that publish still holds a stale False, and its claim then
        succeeds. Re-reading `repaired` here closes the window, because an
        owner always publishes before releasing -- so anyone who wins the claim
        afterwards sees the flag set. This is the only choke point for
        encoding, which is why the check belongs here rather than at each call
        site.
        """
        asked = list(ids)
        ids = [u for u in ids if not self.repaired[u]]
        skipped = [u for u in asked if u not in set(ids)]
        if skipped:
            # the stale-absence race, made visible: this caller won a claim on
            # a unit another owner had already published and released
            self.trace("skip_already_present", units=skipped)
        if not ids:
            return 0
        self.trace("encode_start", units=ids, n=len(ids))
        t0 = time.perf_counter()
        # Everything from here to the publish is one exception-safe region.
        # It used to guard only `put`, which left two ways to lose work: the
        # ENCODER raising (the likely failure -- a device fault, an OOM, a
        # dead inference server) and reading the texts raising. The background
        # loop re-offers what it took, but the inline caller's units came
        # straight off a window and were in no queue at all, so a failed
        # inline repair dropped them with nothing to notice. The choke point
        # for encoding is the right place for the choke point for failure.
        try:
            vecs = self.encode_new([self.texts[u] for u in ids])
            self.trace("encode_done", units=ids,
                       ms=round((time.perf_counter() - t0) * 1e3, 3))
            self.V_new[ids] = vecs
            # Persist BEFORE publishing. `repaired` used to be set first, so a
            # `put` that raised left the unit marked repaired in this process
            # with nothing stored anywhere: every later query would consider
            # it done, no query would try again, and the run would report a
            # migration that had not happened. Ordering it this way costs a
            # reader one extra query's worth of absence, which is exactly what
            # this policy is for.
            if self.put is not None:
                self.put(ids, vecs)
        except Exception as e:                                     # noqa: BLE001
            why = f"{type(e).__name__}: {e}".splitlines()[0]
            self.trace("repair_failed", units=ids, error=why)
            with self._enc_lock:
                self.put_failures += len(ids)
            dead = self.queue.record_failure(ids, why, isolated=len(ids) == 1)
            self.queue.offer([u for u in ids if u not in set(dead)])
            if dead:
                self.trace("quarantined", units=dead, error=why)
            raise
        # `repaired` is published last of all, so no reader can see the flag
        # set while the vector is still zeros. Writing the other way round
        # would let a concurrent query rank against an all-zero vector, which
        # looks like a quality bug and is a visibility one.
        self.repaired[ids] = True
        self.queue.clear_failures(ids)     # a success resets the poison count
        self.trace("publish", units=ids, repaired_total=int(self.repaired.sum()))
        with self._enc_lock:
            self.encoded_units += len(ids)
        self.queue.forget(ids)
        return len(ids)

    # ---------------------------------------------------------------- serving
    def query(self, text: str) -> tuple[list[int], OnlineTrace]:
        t0 = time.perf_counter()
        q_old = self.encode_old_query(text)
        ids, scores = self.search(q_old, self.kprime)
        win = np.asarray(ids, dtype=np.int64)
        if scores is not None and win.size:
            # The budget is spent on the window's head, so the ordering is part
            # of the policy rather than a display detail. Sorting here rather
            # than trusting the index to return sorted keeps that explicit.
            win = win[np.argsort(-np.asarray(scores, dtype=np.float64))]
        t1 = time.perf_counter()

        miss = win[~self.repaired[win]] if win.size else win
        depth = self.queue.depth
        shed = depth > self.shed_at
        m = 0 if shed else max(int(self.budget_m), 0)

        self.trace("window", n_window=int(win.size), n_missing=int(miss.size),
                   queue_depth=depth, shed=shed, budget=m)
        n_inline = n_contended = 0
        if m and miss.size:
            head = [int(u) for u in miss[:m]]
            mine = self.claims.claim(head)
            self.trace("claim", units=mine,
                       contended=[u for u in head if u not in set(mine)])
            n_contended = len(head) - len(mine)
            repair_error = None
            try:
                n_inline = self._encode_and_store(mine)
            except Exception as e:                                 # noqa: BLE001
                # Answer anyway, over whatever is already present. This policy
                # exists on the premise that part of a window beats no answer,
                # and an inline repair that failed used to propagate out of
                # `query` and fail the request instead -- the PostgreSQL path
                # already degraded here and the two disagreed. The units are
                # back on the queue (see `_encode_and_store`), so the
                # background repairer will try them again.
                repair_error = f"{type(e).__name__}: {e}".splitlines()[0]
            finally:
                self.claims.release(mine)
                self.trace("claim_released", units=mine)
            rest = miss[len(head):]
        else:
            repair_error = None
            rest = miss
        if rest.size:
            landed = self.queue.offer(rest)
            self.trace("queued", n=int(rest.size), landed=landed,
                       dropped=self.queue.dropped)
        t2 = time.perf_counter()

        q_new = self.encode_new_query(text)
        vis = win[self.repaired[win]] if win.size else win
        if vis.size:
            order = np.argsort(-(self.V_new[vis] @ q_new))[:self.k]
            results = [int(vis[j]) for j in order]
        else:
            results = []
        t3 = time.perf_counter()

        return results, OnlineTrace(
            window_ms=(t1 - t0) * 1e3,
            inline_repair_ms=(t2 - t1) * 1e3,
            rank_ms=(t3 - t2) * 1e3,
            n_window=int(win.size), n_visible=int(vis.size),
            n_inline=n_inline,
            n_queued=int(rest.size), n_contended=n_contended,
            shed=shed, queue_depth=depth, repair_error=repair_error)

    # ----------------------------------------------------------- the repairer
    def _drain_loop(self, idle_sleep: float) -> None:
        while not self._stop.is_set():
            batch = self.queue.take(self.drain_batch)
            if not batch:
                self._stop.wait(idle_sleep)
                continue
            # This loop OWNS the batch from `take()` until it is either encoded
            # or put back. Nothing else can return it: `take` removed it from
            # the queue, so a unit dropped here is in no queue and claimed by
            # nobody, and no later query will ever ask for it again unless it
            # happens to fall in another window. An earlier version caught the
            # failure, recorded it, released the claim and returned nothing,
            # on the strength of a comment saying the batch was "already back
            # on the queue" -- which was true only when the *persist* step
            # raised, and false for the much more likely case of the encoder
            # raising. The ledger then counted a repair that never happened.
            mine = []
            try:
                mine = self.claims.claim(batch)
                # Units claimed by someone else have left the queue while
                # nobody here is encoding them. Normally their holder
                # publishes; a holder that dies without releasing would strand
                # them, so they go back now.
                self.queue.offer([u for u in batch if u not in set(mine)])
                self._encode_and_store(mine)
            except Exception as e:                                 # noqa: BLE001
                # Letting this out kills the thread, and a dead thread joins
                # instantly -- so `stop_repairer` would report success while
                # the queue grew untouched for the rest of the run.
                self.repairer_errors += 1
                self.last_repairer_error = f"{type(e).__name__}: {e}".splitlines()[0]
                # The whole batch, not just what was claimed: `claim` itself
                # can raise, and re-offering a unit that did get repaired is
                # harmless -- the queue de-duplicates and the encode choke
                # point filters anything already present.
                self.queue.offer(batch)
                self._stop.wait(idle_sleep)
            finally:
                self.claims.release(mine)

    def start_repairer(self, idle_sleep: float = 0.005) -> None:
        """Run the background repairer. Idempotent."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._drain_loop, args=(idle_sleep,),
            name="thaw-repairer", daemon=True)
        self._worker.start()

    def stop_repairer(self, timeout: float = 5.0) -> bool:
        """Ask the repairer to stop and wait. True if it actually stopped.

        The handle is dropped only once the thread has exited. Dropping it
        after a timed-out join -- the old behaviour -- loses the reference to a
        thread still inside a forward pass, and `start_repairer` would then
        start a second one beside it, double-encoding units the first is
        holding claims on.
        """
        self._stop.set()
        w = self._worker
        if w is None:
            return True
        w.join(timeout)
        if w.is_alive():
            return False
        self._worker = None
        return True

    def drain_now(self, max_batches: int | None = None) -> int:
        """Drain the queue on this thread. For measurement, not for serving.

        A replay that ends with a full queue has not finished the migration, so
        a run that wants the end state calls this rather than sleeping and
        hoping. Returns the number of units encoded.
        """
        done, rounds = 0, 0
        while True:
            if max_batches is not None and rounds >= max_batches:
                return done
            batch = self.queue.take(self.drain_batch)
            if not batch:
                return done
            mine = []
            try:
                mine = self.claims.claim(batch)
                unclaimed = [u for u in batch if u not in set(mine)]
                done += self._encode_and_store(mine)
            except Exception:                                      # noqa: BLE001
                # same ownership rule as the loop: what this took and did not
                # encode goes back before the exception leaves
                self.queue.offer(batch)
                raise
            finally:
                self.claims.release(mine)
            if unclaimed and not mine:
                # nothing was ours and nothing was done: another worker holds
                # them all, so returning them and spinning would not progress
                self.queue.offer(unclaimed)
                return done
            self.queue.offer(unclaimed)
            rounds += 1

    # ------------------------------------------------------------ bookkeeping
    @property
    def reached(self) -> float:
        return float(self.repaired.mean())


# ------------------------------------------------------------------ reporting ---
def replay(server: OnlineServer, queries: list[str],
           background: bool = True) -> list[OnlineTrace]:
    """Serve a stream in arrival order. With `background`, the repairer runs."""
    if background:
        server.start_repairer()
    try:
        return [server.query(q)[1] for q in queries]
    finally:
        if background:
            server.stop_repairer()


def summarise(traces: list[OnlineTrace], server: "OnlineServer | None" = None) -> dict:
    """The latency distribution and what it cost in window coverage.

    Coverage is reported beside latency on purpose. The whole trade this module
    makes is coverage for latency, so a summary that gave only the latency would
    make the policy look free.

    Pass `server` and the run's failures come with it: units whose `put` raised
    and batches the repairer could not finish. Those are invisible in a
    latency distribution -- a repair that failed makes the run look *faster* --
    so a summary without them can report a migration that did not happen.
    """
    if not traces:
        raise SystemExit("summarise() was given an empty replay: no query was "
                         "served, so there is no distribution to report")
    tot = np.array([t.total_ms for t in traces])
    cov = np.array([t.window_coverage for t in traces])
    inl = np.array([t.n_inline for t in traces])
    pct = lambda a, p: float(np.percentile(a, p))
    n = len(traces)
    first, last = slice(0, max(1, n // 10)), slice(max(1, 9 * n // 10), n)
    return {
        "queries": n,
        "total_p50": pct(tot, 50), "total_p95": pct(tot, 95),
        "total_p99": pct(tot, 99), "total_max": float(tot.max()),
        "window_coverage_mean": float(cov.mean()),
        "window_coverage_p05": pct(cov, 5),
        "inline_per_query_mean": float(inl.mean()),
        "inline_per_query_p95": pct(inl, 95),
        "queued_total": int(sum(t.n_queued for t in traces)),
        "contended_total": int(sum(t.n_contended for t in traces)),
        "shed_queries": int(sum(1 for t in traces if t.shed)),
        "queue_depth_p95": pct(np.array([t.queue_depth for t in traces]), 95),
        "first_decile_p50": pct(tot[first], 50),
        "last_decile_p50": pct(tot[last], 50),
        # a run with either of these non-zero did not migrate what it says
        "queries_with_a_repair_error": int(sum(1 for t in traces if t.repair_error)),
        "put_failures": int(getattr(server, "put_failures", 0)),
        "quarantined_units": len(getattr(server.queue, "quarantined", {}))
                             if server is not None else 0,
        "queue_dropped": int(server.queue.dropped) if server is not None else 0,
        "repairer_errors": int(getattr(server, "repairer_errors", 0)),
        "last_repairer_error": getattr(server, "last_repairer_error", None),
        "queue_depth_at_end": int(server.queue.depth) if server is not None else None,
    }


def report(s: dict) -> None:
    print(f"  queries                  {s['queries']:,}")
    print(f"  latency p50 / p95 / p99  {s['total_p50']:.1f} / "
          f"{s['total_p95']:.1f} / {s['total_p99']:.1f} ms"
          f"   (max {s['total_max']:.1f})")
    print(f"  window coverage          {s['window_coverage_mean']:.3f} mean, "
          f"{s['window_coverage_p05']:.3f} at p05")
    print(f"  inline repairs / query   {s['inline_per_query_mean']:.1f} mean, "
          f"{s['inline_per_query_p95']:.0f} at p95")
    print(f"  queued / contended       {s['queued_total']:,} / "
          f"{s['contended_total']:,}")
    print(f"  shed queries             {s['shed_queries']:,}"
          f"   (queue depth p95 {s['queue_depth_p95']:.0f})")
    print(f"  p50 first / last decile  {s['first_decile_p50']:.1f} / "
          f"{s['last_decile_p50']:.1f} ms")
    if s.get("put_failures") or s.get("repairer_errors"):
        print(f"  FAILURES                 {s['put_failures']:,} units not "
              f"persisted, {s['repairer_errors']:,} repairer batches failed"
              f"   ({s.get('last_repairer_error')})")
        print(f"  this run did not migrate what the coverage above implies")
