"""The five gated checks of step 1, plus a preview of E2.

Each check answers one question the architecture depends on. A failure here
changes the design, so it is cheaper to find now than after step 2 is built.

  1 index_use   Does the planner use the old HNSW index, the partial HNSW
                index on post-switch rows, and the primary key for window
                fetches, and NOT the partial index where it cannot apply?
  2 depth       Can pgvector return a full window of K' rows, and at which
                ef_search / iterative-scan setting? Overlap with the exact
                window and latency are reported (they are E2), not gated. (K' = 2000 is reported, not
                gated: only the DPR pair needs it.)
  3 null_skip   After UPDATE ... SET emb_old = NULL, is the row never returned?
  4 dead_tuples Before VACUUM, do dead index entries shrink a window below K'?
  5 claims      Do concurrent queries with overlapping windows ever claim (and
                so encode) the same row twice, and does an expired claim from a
                crashed worker get reclaimed while live ones do not?
"""
from __future__ import annotations

import json
import random
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

OLD_INDEX = "docs_emb_old_hnsw"
PARTIAL_INDEX = "docs_new_write_hnsw"
WINDOW_SQL = "SELECT id FROM docs ORDER BY emb_old <=> %s LIMIT %s"


# ------------------------------------------------------------------- helpers
def exact_topk(Q: np.ndarray, V: np.ndarray, k: int, mask=None) -> np.ndarray:
    """Exact cosine top-k (vectors are unit length). `mask` marks rows to
    exclude, e.g. rows whose emb_old was set to NULL."""
    out = np.empty((len(Q), k), dtype=np.int64)
    for lo in range(0, len(Q), 32):
        S = Q[lo:lo + 32] @ V.T
        if mask is not None:
            S[:, mask] = -np.inf
        idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        out[lo:lo + 32] = idx
    return out


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=float), p)) if xs else None


def ef_search_max(conn) -> int:
    conn.execute("SELECT '[1]'::vector <=> '[1]'::vector")    # load the library
    row = conn.execute("SELECT max_val FROM pg_settings "
                       "WHERE name = 'hnsw.ef_search'").fetchone()
    return int(row[0]) if row and row[0] else 1000


def has_iterative(conn) -> bool:
    row = conn.execute("SELECT count(*) FROM pg_settings "
                       "WHERE name = 'hnsw.iterative_scan'").fetchone()
    return bool(row[0])


def configure(conn, mode: str, k: int, ef_max: int, max_scan: int):
    """mode: plain | relaxed | strict."""
    conn.execute(f"SET hnsw.ef_search = {min(k, ef_max)}")
    if mode == "plain":
        if has_iterative(conn):
            conn.execute("SET hnsw.iterative_scan = off")
    else:
        conn.execute(f"SET hnsw.iterative_scan = {mode}_order")
        conn.execute(f"SET hnsw.max_scan_tuples = {int(max_scan)}")


def window(conn, q: np.ndarray, k: int) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    ids = [r[0] for r in conn.execute(WINDOW_SQL, (q, k)).fetchall()]
    return ids, (time.perf_counter() - t0) * 1e3


def _index_names(plan) -> list[str]:
    names = []

    def walk(node):
        if isinstance(node, dict):
            if "Index Name" in node:
                names.append(node["Index Name"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(plan)
    return names


def explain(conn, sql: str, params) -> list[str]:
    row = conn.execute("EXPLAIN (FORMAT JSON) " + sql, params).fetchone()
    plan = row[0] if not isinstance(row[0], str) else json.loads(row[0])
    return _index_names(plan)


# -------------------------------------------------------------------- checks
def check_index_use(conn, Q: np.ndarray, d_new: int = 768) -> dict:
    conn.execute("SET hnsw.ef_search = 500")
    rng = np.random.default_rng(0)
    qn = rng.normal(size=d_new).astype(np.float32)
    qn /= np.linalg.norm(qn)
    some_ids = [r[0] for r in conn.execute(
        "SELECT id FROM docs_new ORDER BY id LIMIT 500").fetchall()]
    cases = {
        "old_window_uses_old_hnsw": (WINDOW_SQL, (Q[0], 500), OLD_INDEX, True),
        "post_switch_uses_partial_hnsw": (
            "SELECT id FROM docs_new WHERE origin = 'write' "
            "ORDER BY emb <=> %s LIMIT 10", (qn,), PARTIAL_INDEX, True),
        "unfiltered_does_not_use_partial": (
            "SELECT id FROM docs_new ORDER BY emb <=> %s LIMIT 10", (qn,),
            PARTIAL_INDEX, False),
        "window_fetch_uses_pkey": (
            "SELECT id, emb FROM docs_new WHERE id = ANY(%s)", (some_ids,),
            "docs_new_pkey", True),
    }
    out, ok = {}, True
    for name, (sql, params, index, want) in cases.items():
        used = explain(conn, sql, params)
        entry = dict(indexes_in_plan=used, expected=index, expect_used=want)
        if name == "old_window_uses_old_hnsw":
            # Must be CHOSEN at real scale: a silent seq scan would make every
            # window exact and every latency number wrong. On a toy corpus,
            # where K' is a large share of N, a seq scan is the right plan.
            conn.execute("SET enable_seqscan = off")
            forced = explain(conn, sql, params)
            conn.execute("RESET enable_seqscan")
            n = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
            entry.update(chosen_by_planner=index in used, capable=index in forced,
                         gate=("chosen" if n >= 100_000 else "capable (N < 100k)"))
            good = index in used if n >= 100_000 else index in forced
        elif name == "post_switch_uses_partial_hnsw":
            # Whether the planner CAN use the partial index is the gate; whether
            # it chooses to is a cost decision that flips with the row count
            # (a seq scan over a few thousand rows can be the cheaper plan).
            conn.execute("SET enable_seqscan = off")
            forced = explain(conn, sql, params)
            conn.execute("RESET enable_seqscan")
            entry.update(chosen_by_planner=index in used,
                         indexes_with_seqscan_off=forced,
                         capable=index in forced)
            good = index in forced
        else:
            good = (index in used) == want
        entry["ok"] = good
        ok &= good
        out[name] = entry
    t0 = time.perf_counter()
    for _ in range(20):
        conn.execute(cases["post_switch_uses_partial_hnsw"][0], (qn,)).fetchall()
    out["post_switch_query_ms"] = round((time.perf_counter() - t0) * 1e3 / 20, 2)
    out["pass"] = bool(ok)
    return out


def check_depth(conn, V, Q, kprimes, n_queries: int, max_scan: int,
                gate_kprimes=(200, 500, 1000), min_overlap: float = 0.90) -> dict:
    ef_max = ef_search_max(conn)
    modes = ["plain"] + (["relaxed", "strict"] if has_iterative(conn) else [])
    sel = np.random.default_rng(0).choice(len(Q), min(n_queries, len(Q)),
                                          replace=False)
    Qs = Q[sel]
    rows, best, overlap = [], {}, {}
    for k in kprimes:
        if k > len(V) // 2:
            continue
        truth = exact_topk(Qs, V, k)
        for mode in modes:
            configure(conn, mode, k, ef_max, max_scan)
            got, lat, ov = [], [], []
            for qi, q in enumerate(Qs):
                ids, ms = window(conn, q, k)
                got.append(len(ids))
                lat.append(ms)
                ov.append(len(set(ids) & set(truth[qi].tolist())) / k)
            r = dict(kprime=k, mode=mode, ef_search=min(k, ef_max),
                     rows_min=int(min(got)), rows_mean=round(float(np.mean(got)), 1),
                     full_window=bool(min(got) >= k),
                     overlap_with_exact=round(float(np.mean(ov)), 4),
                     p50_ms=round(pct(lat, 50), 2), p99_ms=round(pct(lat, 99), 2))
            rows.append(r)
            print(f"  K'={k:<5} {mode:<8} ef={r['ef_search']:<5} rows min/mean "
                  f"{r['rows_min']}/{r['rows_mean']:<7} overlap {r['overlap_with_exact']:.3f}"
                  f"  p50 {r['p50_ms']:.1f} ms  p99 {r['p99_ms']:.1f} ms", flush=True)
        usable = [r for r in rows if r["kprime"] == k and r["full_window"]]
        # The mode this gate blesses must itself clear the overlap bar. Taking
        # the fastest full-window mode and then asking whether ANY mode's
        # overlap was adequate blesses a configuration no measurement supports,
        # because the fastest and the most accurate mode need not be the same
        # one. The quality criterion was previously computed here, printed, and
        # discarded -- so the gate asked only whether K' rows came back, which
        # is a row count and not a retrieval result.
        good = [r for r in usable if r["overlap_with_exact"] >= min_overlap]
        chosen = min(good, key=lambda r: r["p50_ms"]) if good else None
        best[k] = chosen["mode"] if chosen else None
        overlap[k] = (round(max(r["overlap_with_exact"] for r in usable), 4)
                      if usable else None)
        if usable and not good:
            print(f"  ! K'={k}: full windows, but no mode reaches overlap "
                  f"{min_overlap} with the exact window (best {overlap[k]}); "
                  "raise m / ef_construction (load) before step 2 measures "
                  "cross-space recall on top of it")
    gated = [k for k in gate_kprimes if k in best]
    absent = [int(k) for k in gate_kprimes if k not in best]
    # Fail closed, twice over. `all([])` is True, so a corpus small enough that
    # every gate depth was skipped at the top of the loop used to pass this
    # gate having checked nothing. And a gate that silently narrows itself to
    # whichever depths happened to run is the same defect one level down: an
    # operator measuring a deliberately small corpus passes a narrower
    # `gate_kprimes` explicitly rather than having one inferred from what fit.
    passed = bool(gated) and not absent and all(best[k] is not None for k in gated)
    return dict(ef_search_max=ef_max, iterative_available=len(modes) > 1,
                queries=int(len(Qs)), rows=rows, usable_mode_by_kprime=best,
                min_overlap=min_overlap, overlap_by_kprime=overlap,
                gate_kprimes_absent=absent,
                pass_=passed, gated_kprimes=gated)


def _restore(conn, V, victims):
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE restore_old (id bigint, emb vector(%d))"
                    % V.shape[1])
        with cur.copy("COPY restore_old FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int8", "vector"])
            for i in victims:
                cp.write_row([int(i), V[i]])
        cur.execute("UPDATE docs d SET emb_old = r.emb FROM restore_old r "
                    "WHERE d.id = r.id")
        cur.execute("DROP TABLE restore_old")
    conn.execute("VACUUM ANALYZE docs")


def check_updates(conn, V, Q, k: int, n_queries: int, frac: float,
                  max_scan: int, modes) -> dict:
    """Checks 3 and 4 together: NULL the old vector of `frac` of the rows that
    the sampled queries' windows contain, then search before and after VACUUM."""
    ef_max = ef_search_max(conn)
    sel = np.random.default_rng(1).choice(len(Q), min(n_queries, len(Q)),
                                          replace=False)
    Qs = Q[sel]
    union = np.unique(exact_topk(Qs, V, k))
    rng = np.random.default_rng(2)
    n_victims = int(frac * len(union))
    # If this rounds to 0 the UPDATE below is a no-op, no row is NULLed, and
    # every check that follows passes vacuously -- null_skip_pass is all-zero
    # because nothing was nulled, and the dead-tuple check never sees a dead
    # tuple. A gate that can pass while testing nothing is worse than no gate,
    # so refuse rather than report a pass.
    if n_victims < 1:
        raise SystemExit(
            f"check_updates would NULL 0 rows (frac={frac} of {len(union)} "
            "window rows); raise --update-frac or the queries' k, or lower N. "
            "Refusing to report a vacuous pass.")
    victims = np.sort(rng.choice(union, n_victims, replace=False))
    vset = set(victims.tolist())
    mask = np.zeros(len(V), dtype=bool)
    mask[victims] = True
    truth = exact_topk(Qs, V, k, mask=mask)

    conn.execute("ALTER TABLE docs SET (autovacuum_enabled = false)")
    conn.execute("UPDATE docs SET emb_old = NULL, updated_at = now() "
                 "WHERE id = ANY(%s)", (victims.tolist(),))
    phases = {}
    try:
        for phase in ("before_vacuum", "after_vacuum"):
            if phase == "after_vacuum":
                conn.execute("VACUUM docs")
            res = {}
            for mode in modes:
                configure(conn, mode, k, ef_max, max_scan)
                got, leaked, ov = [], 0, []
                for qi, q in enumerate(Qs):
                    ids, _ = window(conn, q, k)
                    got.append(len(ids))
                    leaked += sum(1 for i in ids if i in vset)
                    ov.append(len(set(ids) & set(truth[qi].tolist())) / k)
                res[mode] = dict(rows_min=int(min(got)),
                                 rows_mean=round(float(np.mean(got)), 1),
                                 full_window=bool(min(got) >= k),
                                 nulled_rows_returned=int(leaked),
                                 overlap_with_exact=round(float(np.mean(ov)), 4))
                print(f"  {phase:<14} {mode:<8} rows min/mean {res[mode]['rows_min']}/"
                      f"{res[mode]['rows_mean']}  nulled returned "
                      f"{res[mode]['nulled_rows_returned']}  overlap "
                      f"{res[mode]['overlap_with_exact']:.3f}", flush=True)
            phases[phase] = res
    finally:
        _restore(conn, V, victims)
        conn.execute("ALTER TABLE docs RESET (autovacuum_enabled)")
    null_ok = all(r["nulled_rows_returned"] == 0
                  for p in phases.values() for r in p.values())
    dead_ok = any(r["full_window"] for r in phases["before_vacuum"].values())
    return dict(kprime=k, queries=int(len(Qs)), window_union=int(len(union)),
                nulled=int(len(victims)), phases=phases,
                null_skip_pass=bool(null_ok), dead_tuples_pass=bool(dead_ok),
                full_window_modes_before_vacuum=[
                    m for m, r in phases["before_vacuum"].items() if r["full_window"]])


def check_claims(dsn: str, V, Q, k: int, n_windows: int, workers: int,
                 encode_ms_per_unit: float, lease_s: int = 300) -> dict:
    from .db import connect

    windows = [w.tolist() for w in exact_topk(
        Q[np.random.default_rng(3).choice(len(Q), min(n_windows, len(Q)),
                                          replace=False)], V, k)]
    # every window twice, shuffled: the same popular query arriving at two
    # workers at once is exactly the case the claim protocol exists for
    tasks = windows + windows
    random.Random(4).shuffle(tasks)
    admin = connect(dsn)
    admin.execute("TRUNCATE repair_claims, claim_done")

    lock = threading.Lock()
    claimed_by: Counter = Counter()
    stats = dict(claim_ms=[], wait_ms=[], timeouts=0, insert_conflicts=0,
                 errors=[])
    queue = list(range(len(tasks)))

    def worker(name: str):
        conn = connect(dsn)
        while True:
            with lock:
                if not queue:
                    break
                t = queue.pop()
            win = tasks[t]
            done = {r[0] for r in conn.execute(
                "SELECT id FROM claim_done WHERE id = ANY(%s)", (win,)).fetchall()}
            # sorted: two multi-row inserts that take the same keys in different
            # orders deadlock each other (found by this check on its first run)
            stale = sorted(i for i in win if i not in done)
            t0 = time.perf_counter()
            try:
                # ORDER BY fixes the order rows are INSERTED, hence the order
                # their key locks are taken -- that is what prevents the
                # deadlock. RETURNING's order is not guaranteed and is not
                # relied on: only the set of ids won is used.
                mine = [r[0] for r in conn.execute(
                    "INSERT INTO repair_claims (id, owner) "
                    "SELECT x, %s FROM unnest(%s::bigint[]) AS x ORDER BY x "
                    "ON CONFLICT (id) DO NOTHING RETURNING id",
                    (name, stale)).fetchall()]
            except Exception as e:  # noqa: BLE001 -- counted, and fails the gate
                with lock:
                    stats["errors"].append(f"{type(e).__name__}: {e}".splitlines()[0])
                continue
            claim_ms = (time.perf_counter() - t0) * 1e3
            time.sleep(len(mine) * encode_ms_per_unit / 1e3)        # "encode"
            n_ins = len(conn.execute(
                "INSERT INTO claim_done (id, owner) SELECT unnest(%s::bigint[]), %s "
                "ON CONFLICT (id) DO NOTHING RETURNING id", (mine, name)).fetchall())
            others = list(set(stale) - set(mine))
            t1 = time.perf_counter()
            timed_out = False
            while others:
                have = conn.execute("SELECT count(*) FROM claim_done "
                                    "WHERE id = ANY(%s)", (others,)).fetchone()[0]
                if have == len(others):
                    break
                if time.perf_counter() - t1 > 30:
                    timed_out = True
                    break
                time.sleep(0.005)
            with lock:
                claimed_by.update(mine)
                stats["claim_ms"].append(claim_ms)
                stats["wait_ms"].append((time.perf_counter() - t1) * 1e3)
                stats["insert_conflicts"] += len(mine) - n_ins
                stats["timeouts"] += int(timed_out)
        conn.close()

    threads = [threading.Thread(target=worker, args=(f"w{i}",))
               for i in range(workers)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - t0

    union = {i for w in windows for i in w}
    n_done = admin.execute("SELECT count(*) FROM claim_done").fetchone()[0]
    double = sum(1 for c in claimed_by.values() if c > 1)

    # lease: a crashed worker's claims expire; live claims and finished rows stay
    pool = [i for i in range(len(V)) if i not in union][:200]
    dead, alive = pool[:100], pool[100:200]
    admin.execute("INSERT INTO repair_claims (id, owner, claimed_at) "
                  "SELECT unnest(%s::bigint[]), 'crashed', now() - interval '1 hour'",
                  (dead,))
    admin.execute("INSERT INTO repair_claims (id, owner) "
                  "SELECT unnest(%s::bigint[]), 'alive'", (alive,))
    reclaimed = {r[0] for r in admin.execute(
        "DELETE FROM repair_claims c WHERE c.claimed_at < now() - make_interval(secs => %s) "
        "AND NOT EXISTS (SELECT 1 FROM claim_done d WHERE d.id = c.id) RETURNING id",
        (lease_s,)).fetchall()}
    retaken = len(admin.execute(
        "INSERT INTO repair_claims (id, owner) SELECT unnest(%s::bigint[]), 'w-new' "
        "ON CONFLICT (id) DO NOTHING RETURNING id", (dead,)).fetchall())
    lease_ok = reclaimed == set(dead) and retaken == len(dead)
    admin.close()

    ok = (double == 0 and stats["insert_conflicts"] == 0 and stats["timeouts"] == 0
          and not stats["errors"] and n_done == len(union) and lease_ok)
    return dict(windows=len(windows), tasks=len(tasks), workers=workers,
                window_union=len(union), rows_done=int(n_done),
                rows_claimed_twice=int(double),
                insert_conflicts=int(stats["insert_conflicts"]),
                waiter_timeouts=int(stats["timeouts"]),
                errors=stats["errors"][:10], error_count=len(stats["errors"]),
                claim_ms_p50=round(pct(stats["claim_ms"], 50), 2),
                claim_ms_p99=round(pct(stats["claim_ms"], 99), 2),
                wait_ms_p99=round(pct(stats["wait_ms"], 99), 2),
                elapsed_s=round(elapsed, 2),
                lease_reclaimed=len(reclaimed), lease_expected=len(dead),
                lease_live_claims_untouched=bool(not (reclaimed & set(alive))),
                lease_pass=bool(lease_ok), pass_=bool(ok))


# ---------------------------------------------------------------------- run
def run(dsn: str, data: dict, out: Path, kprimes, n_queries: int, max_scan: int,
        update_frac: float, claim_windows: int, claim_workers: int,
        encode_ms: float) -> int:
    from .db import connect, server_info

    V, Q = data["V_old"], data["Q_old"]
    conn = connect(dsn)
    info = server_info(conn)
    print(f"server: Postgres {info['postgres']}, pgvector {info['pgvector']}")
    report = dict(server=info, source=str(data["source"]), N=int(len(V)),
                  queries_available=int(len(Q)))

    print("\n[1/5] index use")
    report["index_use"] = r1 = check_index_use(conn, Q)
    for k_, v in r1.items():
        if isinstance(v, dict):
            print(f"  {'ok  ' if v['ok'] else 'FAIL'} {k_}: {v['indexes_in_plan']}")

    print("\n[2/5] window depth (and a preview of E2)")
    report["depth"] = r2 = check_depth(conn, V, Q, kprimes, n_queries, max_scan)
    print(f"  ef_search max {r2['ef_search_max']}; usable mode by K': "
          f"{r2['usable_mode_by_kprime']}")

    modes = ["plain"] + (["relaxed"] if r2["iterative_available"] else [])
    k_upd = 500 if 500 <= len(V) // 2 else min(kprimes)
    print(f"\n[3-4/5] NULLed old vectors and dead index entries (K'={k_upd})")
    report["updates"] = r34 = check_updates(conn, V, Q, k_upd,
                                            min(n_queries, 100), update_frac,
                                            max_scan, modes)
    conn.close()

    print(f"\n[5/5] concurrent claims ({claim_workers} workers)")
    report["claims"] = r5 = check_claims(dsn, V, Q, min(500, len(V) // 2),
                                         claim_windows, claim_workers, encode_ms)
    print(f"  {r5['window_union']:,} distinct rows across {r5['tasks']} window "
          f"tasks; claimed twice: {r5['rows_claimed_twice']}; claim p50/p99 "
          f"{r5['claim_ms_p50']}/{r5['claim_ms_p99']} ms; lease reclaim "
          f"{r5['lease_reclaimed']}/{r5['lease_expected']}; errors {r5['error_count']}")

    gates = {
        "1 index_use": r1["pass"],
        ("2 depth (K' in %s)" % r2["gated_kprimes"] if r2["gated_kprimes"]
         else "2 depth (no gated K' requested)"): r2["pass_"],
        "3 null_skip": r34["null_skip_pass"],
        "4 dead_tuples": r34["dead_tuples_pass"],
        "5 claims": r5["pass_"],
    }
    report["gates"] = gates
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.results_io import environment, write_results
    report["environment"] = environment()
    write_results(out, report)
    print("\n" + "=" * 60)
    for g, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {g}")
    print("=" * 60)
    print(f"report: {out}")
    return 0 if all(gates.values()) else 1
