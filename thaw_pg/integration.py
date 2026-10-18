"""D-07: the online path against a live PostgreSQL and a real encoder.

Four scenarios, each answering a question the unit suite cannot. The suite runs
`OnlinePgRetriever` against a fake database -- one that claims atomically,
enforces one thread per connection, carries `origin='write'` rows and expires
leases -- and a good fake still cannot show what a real server does under real
concurrency, nor what one GPU does when two things want it.

    python -m thaw_pg.integration --dsn "$THAW_PG_DSN" \\
        --encoder local:BAAI/bge-base-en-v1.5 \\
        --old-query local:sentence-transformers/all-MiniLM-L6-v2 \\
        --scenario all --out results/d07.json

Run it through `semview.runlog` so it lands in a run directory with its command
and environment:

    python -m semview.runlog --name d07_online_pg --tag d07 -- \\
        python -m thaw_pg.integration --dsn ... --out d07.json

**Prepare the database first.** This expects the step-2 schema and a loaded
corpus: `python -m thaw_pg prepare` then `load`, as `README_STEP2.md` describes.
It will refuse to run against a `docs` table that holds real text unless
`--force` is given, for the reason `__main__.py`'s guard exists.

### The scenarios

1. `exclusive` -- two **OS processes**, not two threads, serving the same query
   stream against one database. Asserts every unit was encoded at most once, by
   comparing the encoder's own call count across both processes against the
   rows in `docs_new`. This is the guarantee every cost number in the paper
   rests on, and the one the fake can only assert.

2. `recover` -- kill a worker outright mid-encode, restart it, and check
   that the lease sweep reclaims the dead owner's claims and the units are
   repaired afterwards. A `kill -9` leaves committed claim rows behind; nothing
   but the sweep brings them back, and until this runs, nothing has ever
   watched it happen.

3. `writes` -- insert, update and delete documents **while the replay runs**,
   and check that an inserted or updated document reaches answers through the
   `origin='write'` partial index and that a deleted one disappears from them.
   Post-switch writes have no old vector, so no window can reach them at any
   depth.

4. `overlap` -- the same replay with `overlap_query_encode` true and false,
   reporting latency percentiles and `q_new_overlap_saved_ms_mean` for both.
   Overlapping the new-query encode with the window read cannot lengthen the
   critical path, but whether it *shortens* it depends on contention for one
   device, which is the thing the shipped measurements say dominates. This
   scenario exists to stop that change being reported as an improvement on
   anything but evidence.

### What it still does not cover

A single machine. Replication, a connection pooler, a network partition
between the service and the database, and several GPUs are all outside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ helpers --
def _connect(dsn: str):
    from .db import connect
    return connect(dsn)


def _guard(conn, force: bool) -> None:
    """Refuse a destructive run against a corpus that looks real."""
    if force:
        return
    row = conn.execute(
        "SELECT count(*) FROM docs WHERE text IS NOT NULL "
        "AND text NOT LIKE 'k:%'").fetchone()
    if row and row[0] > 0:
        raise SystemExit(
            f"refusing: `docs` holds {row[0]:,} rows of real text and these "
            f"scenarios write, delete and truncate. Point --dsn at a scratch "
            f"database, or pass --force if this IS the scratch database.")


def _retriever(dsn, enc_new, enc_old_q, enc_new_q, kprime, k, budget_m,
               overlap=True, owner=None):
    from . import online as pg_online
    fetch_text, write, _io = pg_online.online_io(dsn)
    return pg_online.OnlinePgRetriever(
        _connect(dsn), encode_new=enc_new, fetch_text=fetch_text, write=write,
        enc_old_query=enc_old_q, enc_new_query=enc_new_q,
        kprime=kprime, k=k, budget_m=budget_m,
        overlap_query_encode=overlap, owner=owner)


def _encoders(spec_new: str, spec_old_q: str, batch: int = 32,
              device: str | None = None):
    """Build both encoders, with the prefix each model wants.

    `encoders.make(spec, prefix, batch, device=None)` -- the prefix and the
    batch are required, and calling it with the spec alone raised TypeError on
    the first line of every scenario. The prefix is per model and not
    cosmetic: `semview.migration.resolve_prefixes` knows which models want
    "passage: " / "query: " and getting it wrong changes what is retrieved.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.migration import resolve_prefixes

    from . import encoders
    name_new = spec_new.partition(":")[2]
    name_old = spec_old_q.partition(":")[2]
    p_pass, _ = resolve_prefixes(name_new, name_new)
    _, q_old = resolve_prefixes(name_old, name_old)
    enc_new = encoders.make(spec_new, p_pass, batch, device)
    enc_old_q = encoders.make(spec_old_q, q_old, batch, device)
    counted = {"n": 0}

    def encode_new(texts):
        counted["n"] += len(texts)
        return enc_new.encode(texts)

    return encode_new, enc_old_q, enc_new, counted


def _queries(conn, n: int) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT text FROM docs WHERE text IS NOT NULL ORDER BY id LIMIT %s",
        (n,)).fetchall()]


def _reset_new(conn) -> None:
    conn.execute("TRUNCATE docs_new, repair_claims, claim_done")


# ---------------------------------------------------------------- scenarios --
def scenario_exclusive(a) -> dict:
    """Two OS processes, one database, one query stream.

    Each child reports how many units its encoder was asked for. The sum, set
    against the rows in `docs_new`, is the cost guarantee: a unit encoded by
    both processes shows up as a sum larger than the row count.
    """
    import subprocess

    conn = _connect(a.dsn)
    _guard(conn, a.force)
    _reset_new(conn)
    n_docs = conn.execute("SELECT count(*) FROM docs").fetchone()[0]

    env = dict(os.environ, THAW_PG_DSN=a.dsn)
    procs = [subprocess.Popen(
        [sys.executable, "-m", "thaw_pg.integration", "--dsn", a.dsn,
         "--encoder", a.encoder, "--old-query", a.old_query,
         "--scenario", "_child", "--queries", str(a.queries),
         "--kprime", str(a.kprime), "--budget-m", str(a.budget_m),
         "--child-id", str(i), "--force"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for i in range(2)]
    out = [p.communicate(timeout=a.timeout) for p in procs]

    reports = []
    for i, (so, se) in enumerate(out):
        line = [l for l in so.splitlines() if l.startswith("{")]
        if not line:
            raise SystemExit(f"child {i} produced no report:\n{se[-2000:]}")
        reports.append(json.loads(line[-1]))

    encoded = sum(r["encoded_units"] for r in reports)
    stored = conn.execute("SELECT count(*) FROM docs_new").fetchone()[0]
    left = conn.execute("SELECT count(*) FROM repair_claims").fetchone()[0]
    return {
        "scenario": "exclusive", "processes": len(reports), "corpus": n_docs,
        "per_process": reports,
        "encoded_units_total": encoded, "rows_in_docs_new": stored,
        "claims_left_behind": left,
        "pass": encoded == stored and left == 0,
        "says": ("every unit was encoded at most once across two OS processes"
                 if encoded == stored else
                 f"{encoded - stored} unit(s) were encoded more than once"),
    }


def scenario_child(a) -> dict:
    """One service process: serve the stream, drain, report. Not run directly."""
    encode_new, enc_old_q, enc_new, counted = _encoders(a.encoder, a.old_query)
    from . import online as pg_online

    r = _retriever(a.dsn, encode_new, enc_old_q, enc_new, a.kprime, a.k,
                   a.budget_m, owner=(a.owner or f"child-{a.child_id}"))
    rep = pg_online.BackgroundRepairer(r, dsn=a.dsn, idle_sleep=0.005)
    rep.start()
    traces = []
    try:
        for q in _queries(_connect(a.dsn), a.queries):
            traces.append(r.query(text=q))
        # stop the worker BEFORE draining on this thread: `drain_now` uses the
        # repairer's connection, and that thread is using it too
        rep.stop(timeout=300)
        rep.drain_now()
    finally:
        rep.close(timeout=120)
    s = pg_online.summarise(traces, retriever=r, repairer=rep)
    return {"child": a.child_id, "encoded_units": counted["n"],
            "retriever_encoded": r.encoded_units, "summary": s}


def scenario_recover(a) -> dict:
    """Kill a worker mid-encode; the lease must bring its claims back.

    An unannounced death leaves committed `repair_claims` rows with no
    owner alive. The
    sweep is the only thing that releases them, and `reclaim_expired` sat
    unwired for a round precisely because nothing ever watched this happen.
    """
    import subprocess

    conn = _connect(a.dsn)
    _guard(conn, a.force)
    _reset_new(conn)

    child = subprocess.Popen(
        [sys.executable, "-m", "thaw_pg.integration", "--dsn", a.dsn,
         "--encoder", a.encoder, "--old-query", a.old_query,
         "--scenario", "_child", "--queries", str(a.queries * 4),
         "--kprime", str(a.kprime), "--budget-m", str(a.budget_m),
         "--child-id", "9", "--owner", "doomed-child", "--force"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=dict(os.environ, THAW_PG_DSN=a.dsn))

    deadline = time.time() + a.timeout
    while time.time() < deadline:
        held = conn.execute("SELECT count(*) FROM repair_claims").fetchone()[0]
        if held:
            break
        time.sleep(0.05)
    else:
        child.kill()
        raise SystemExit("no claim was ever taken: nothing to kill mid-encode")

    # which units the dead process was holding, by id. `stored > 0` was the
    # old pass condition and it proves nothing: the restarted worker repairs
    # whatever traffic reaches, so a run where NONE of the killed process's
    # units came back still ended with rows in docs_new and called itself a
    # pass. The contract is about these ids.
    held_ids = [r[0] for r in conn.execute(
        "SELECT id FROM repair_claims ORDER BY id").fetchall()]
    # SIGKILL does not exist on Windows; `Popen.kill()` is the portable
    # spelling and maps to TerminateProcess there. The point is an
    # unannounced death with no chance to release claims, which both give.
    child.kill()
    child.wait(timeout=60)
    stranded = conn.execute(
        "SELECT count(*) FROM repair_claims").fetchone()[0]
    # age the rows so the lease applies without waiting it out
    conn.execute("UPDATE repair_claims SET claimed_at = now() - interval '1 hour'")

    from . import online as pg_online
    encode_new, enc_old_q, enc_new, counted = _encoders(a.encoder, a.old_query)
    r = _retriever(a.dsn, encode_new, enc_old_q, enc_new, a.kprime, a.k,
                   a.budget_m, owner="after-restart")
    rep = pg_online.BackgroundRepairer(r, dsn=a.dsn, idle_sleep=0.005,
                                       sweep_every_s=1.0)
    rep.start()
    try:
        time.sleep(3.0)                       # let the sweep run
        reclaimed = rep.reclaimed
        for q in _queries(conn, a.queries):
            r.query(text=q)
        rep.stop(timeout=300)
        rep.drain_now()
    finally:
        rep.close(timeout=120)

    after = conn.execute("SELECT count(*) FROM repair_claims").fetchone()[0]
    stored = conn.execute("SELECT count(*) FROM docs_new").fetchone()[0]

    # The units the dead process held: are they claimable again, and did the
    # ones traffic asked for actually get repaired? Thaw repairs on demand, so
    # the contract is NOT that every stranded unit is now in docs_new -- it is
    # that none of them is stuck: each is either repaired or freely claimable.
    still_blocked = [r[0] for r in conn.execute(
        "SELECT id FROM repair_claims WHERE id = ANY(%s) AND owner = %s",
        (held_ids, "doomed-child")).fetchall()] if held_ids else []
    repaired_of_held = conn.execute(
        "SELECT count(*) FROM docs_new WHERE id = ANY(%s)",
        (held_ids,)).fetchone()[0] if held_ids else 0
    free = conn.execute(
        "SELECT count(*) FROM unnest(%s::bigint[]) AS u(id) "
        "WHERE NOT EXISTS (SELECT 1 FROM repair_claims c WHERE c.id = u.id) "
        "AND NOT EXISTS (SELECT 1 FROM docs_new d WHERE d.id = u.id)",
        (held_ids,)).fetchone()[0] if held_ids else 0
    accounted = repaired_of_held + free
    return {
        "scenario": "recover",
        "units_held_at_kill": len(held_ids),
        "claims_stranded_by_the_kill": stranded,
        "reclaimed_by_the_sweep": reclaimed,
        "claims_left_at_end": after, "rows_in_docs_new": stored,
        "of_the_held_units_repaired": int(repaired_of_held),
        "of_the_held_units_claimable_again": int(free),
        "still_blocked_by_the_dead_owner": len(still_blocked),
        "encoded_after_restart": counted["n"],
        "pass": (len(held_ids) > 0 and stranded > 0
                 and reclaimed >= stranded
                 and accounted == len(held_ids)
                 and not still_blocked),
        "says": (f"of {len(held_ids)} units the killed process held, "
                 f"{repaired_of_held} were repaired and {free} are claimable "
                 f"again -- none is stuck behind a dead owner's claim"
                 if accounted == len(held_ids) and held_ids else
                 f"{len(held_ids) - accounted} unit(s) the killed process "
                 f"held are neither repaired nor claimable: they are stranded"),
    }


def scenario_writes(a) -> dict:
    """Insert, update and delete while the replay runs.

    A document written after the switch has no old vector, so no window
    reaches it at any depth; only the partial index over `origin='write'`
    rows can. A deleted one must leave the answers entirely.
    """
    conn = _connect(a.dsn)
    _guard(conn, a.force)
    _reset_new(conn)

    from . import online as pg_online
    encode_new, enc_old_q, enc_new, _ = _encoders(a.encoder, a.old_query)
    r = _retriever(a.dsn, encode_new, enc_old_q, enc_new, a.kprime, a.k,
                   a.budget_m, owner="writes")
    rep = pg_online.BackgroundRepairer(r, dsn=a.dsn, idle_sleep=0.005)
    rep.start()
    try:
        qs = _queries(conn, a.queries)
        for q in qs[:len(qs) // 2]:
            r.query(text=q)

        # a document created mid-migration, written straight into docs_new
        probe = "a document inserted during the migration, mentioning quokkas"
        new_id = conn.execute("SELECT max(id) + 1 FROM docs").fetchone()[0]
        conn.execute("INSERT INTO docs (id, doc_id, text) VALUES (%s, %s, %s)",
                     (new_id, f"d{new_id}", probe))
        vec = enc_new.encode([probe])[0]
        conn.execute("INSERT INTO docs_new (id, emb, origin) VALUES (%s, %s, 'write')",
                     (new_id, vec))
        tr_ins = r.query(text=probe)
        found_insert = new_id in tr_ins.results

        # the same document, updated: its vector is replaced, still origin='write'
        probe2 = "the same document, rewritten, now about pangolins"
        conn.execute("UPDATE docs SET text = %s WHERE id = %s", (probe2, new_id))
        conn.execute("UPDATE docs_new SET emb = %s, origin = 'write' WHERE id = %s",
                     (enc_new.encode([probe2])[0], new_id))
        tr_upd = r.query(text=probe2)
        found_update = new_id in tr_upd.results

        # and deleted: gone from docs, and from docs_new by cascade
        conn.execute("DELETE FROM docs WHERE id = %s", (new_id,))
        tr_del = r.query(text=probe2)
        found_delete = new_id in tr_del.results
        still_there = conn.execute(
            "SELECT count(*) FROM docs_new WHERE id = %s", (new_id,)).fetchone()[0]
    finally:
        rep.close(timeout=120)

    return {
        "scenario": "writes", "probe_id": int(new_id),
        "insert_reached_answers": bool(found_insert),
        "update_reached_answers": bool(found_update),
        "delete_removed_from_answers": not found_delete,
        "rows_left_after_delete": int(still_there),
        "n_written_candidates": [tr_ins.n_written, tr_upd.n_written],
        "pass": bool(found_insert and found_update and not found_delete
                     and still_there == 0),
        "says": ("documents written during the migration are reachable and a "
                 "deleted one is not" if found_insert and found_update
                 and not found_delete else
                 "a post-switch write was not reachable, or a delete was"),
    }


def scenario_overlap(a) -> dict:
    """The encode overlap, A/B, on the device that actually has to run it.

    Three things this has to get right, and a first version got none of them.
    It ran `overlap=False` and then `overlap=True`, once each, and reported
    the difference in p50. On a GPU the second arm inherits a warm device, a
    warm model and a warm page cache, so the ordering alone can produce the
    result -- and the arm that looks good is the one that ran second, which
    was always the one being argued for.

    So: a warm-up pass that is measured and discarded, `rounds` alternating
    which arm goes first, and a spread rather than one number. If the two
    orders disagree, the warm-up was not enough and the result says so
    instead of averaging the confusion away.
    """
    from . import online as pg_online

    def one(flag: str) -> dict:
        conn = _connect(a.dsn)
        _guard(conn, a.force)
        _reset_new(conn)
        encode_new, enc_old_q, enc_new, _ = _encoders(a.encoder, a.old_query)
        r = _retriever(a.dsn, encode_new, enc_old_q, enc_new, a.kprime, a.k,
                       a.budget_m, overlap=(flag == "on"),
                       owner=f"overlap-{flag}")
        rep = pg_online.BackgroundRepairer(r, dsn=a.dsn, idle_sleep=0.005)
        rep.start()
        try:
            traces = [r.query(text=q) for q in _queries(conn, a.queries)]
            rep.stop(timeout=300)
            rep.drain_now()
        finally:
            rep.close(timeout=120)
        return pg_online.summarise(traces, retriever=r, repairer=rep)

    # a measured warm-up, thrown away: the first pass over a cold device and a
    # cold cache is not a measurement of anything this is comparing
    warm = one("on")

    arms = {"on": [], "off": []}
    order = []
    for i in range(a.rounds):
        first, second = ("off", "on") if i % 2 == 0 else ("on", "off")
        order.append(first)
        arms[first].append(one(first))
        arms[second].append(one(second))

    def p50s(rs):
        return sorted(round(x["wall_p50"], 2) for x in rs)

    on_p50, off_p50 = p50s(arms["on"]), p50s(arms["off"])
    med = lambda v: v[len(v) // 2] if v else float("nan")
    delta = med(off_p50) - med(on_p50)
    saved = [round(x.get("q_new_overlap_saved_ms_mean", 0.0), 2)
             for x in arms["on"]]
    # did the two orderings agree about the sign?
    by_order = {}
    for i, first in enumerate(order):
        d = arms["off"][i]["wall_p50"] - arms["on"][i]["wall_p50"]
        by_order.setdefault(first, []).append(round(d, 2))
    agree = (len(by_order) < 2
             or all(d > 0 for ds in by_order.values() for d in ds)
             or all(d <= 0 for ds in by_order.values() for d in ds))
    verdict = ("consistently across both orderings" if agree else
               "but the two orderings disagree on the sign, so this is "
               "warm-up or noise and not an effect")
    return {
        "scenario": "overlap", "rounds": a.rounds,
        "warmup_p50": round(warm["wall_p50"], 2),
        "p50_on": on_p50, "p50_off": off_p50,
        "median_delta_ms": round(delta, 2),
        "delta_by_which_arm_ran_first": by_order,
        "orders_agree_on_the_sign": bool(agree),
        "saved_ms_mean_per_round": saved,
        "p95_on": [round(x["wall_p95"], 2) for x in arms["on"]],
        "p95_off": [round(x["wall_p95"], 2) for x in arms["off"]],
        "pass": True,                 # a measurement, not a gate
        "says": (f"median p50 moved {delta:+.1f} ms with overlapping on, "
                 f"{verdict}. A negative delta means the two encoders contend "
                 f"and the overlap costs more than it saves -- which is a "
                 f"result, not a failure."),
    }


SCENARIOS = {"exclusive": scenario_exclusive, "recover": scenario_recover,
             "writes": scenario_writes, "overlap": scenario_overlap,
             "_child": scenario_child}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="thaw_pg.integration", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=os.environ.get("THAW_PG_DSN"))
    ap.add_argument("--encoder", required=True,
                    help="new-model encoder: local:<model> or tei:<url>")
    ap.add_argument("--old-query", required=True,
                    help="old-model QUERY encoder")
    ap.add_argument("--scenario", default="all",
                    choices=["all", *SCENARIOS])
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--kprime", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget-m", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--rounds", type=int, default=3,
                    help="overlap A/B: how many alternating rounds after the "
                         "warm-up. More rounds, more confidence in the sign.")
    ap.add_argument("--child-id", type=int, default=0)
    ap.add_argument("--owner", default=None,
                    help="the claim owner this process writes, so a scenario "
                         "can tell whose claims it is looking at")
    ap.add_argument("--force", action="store_true",
                    help="this IS a scratch database; skip the real-data guard")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not a.dsn:
        raise SystemExit("no DSN: pass --dsn or set THAW_PG_DSN")

    if a.scenario == "_child":                       # one line of JSON, no noise
        print(json.dumps(scenario_child(a)))
        return 0

    names = [n for n in ("exclusive", "recover", "writes", "overlap")
             if a.scenario in ("all", n)]
    results, bad = [], 0
    for n in names:
        print(f"\n=== {n} ===", flush=True)
        t0 = time.perf_counter()
        try:
            r = SCENARIOS[n](a)
        except Exception as e:                                     # noqa: BLE001
            r = {"scenario": n, "pass": False,
                 "error": f"{type(e).__name__}: {e}"[:600]}
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        bad += not r.get("pass")
        print(f"  {'ok  ' if r.get('pass') else 'FAIL'}  {r.get('says', r.get('error'))}")

    print(f"\n{len(results) - bad} of {len(results)} scenarios passed")
    if a.out:
        from semview.results_io import environment, write_results
        write_results(a.out, {"tool": "d07-integration", "scenarios": results,
                              "queries": a.queries, "kprime": a.kprime,
                              "budget_m": a.budget_m, "encoder": a.encoder,
                              "environment": environment()})
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
