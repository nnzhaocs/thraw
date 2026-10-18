"""Step 2: the vertical slice, measured.

  prepare2         ground truth for the evaluation: the new model's exact
                   top-10 for every query, plus a sample of reference vectors
                   for the encoder check. This full-corpus encode is the
                   evaluation's cost, not the system's.
  verify-encoder   the system's encoder (TEI or local) must produce the same
                   vectors as the ones the ground truth was built from.
  step2            replay queries through the real query path, from an empty
                   docs_new, with the exact window (E1) and the HNSW window
                   (E2), at several K'.
  selftest2        the same code path on synthetic data with a table-lookup
                   encoder: checks the machinery with no model and no GPU.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- prepare2
def prepare2(out_dir: Path, beir_path: str, limit: int, max_queries: int,
             old_model: str, new_model: str, cache_dir: str | None,
             device: str | None, seed: int = 0, n_check: int = 256) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.migration import (corpus_key, encode_all, encode_with,
                                   load_beir, resolve_prefixes, topk_chunked)

    units, queries = load_beir("msmarco", beir_path, limit, seed, "dev")
    if max_queries and len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        queries = [queries[i] for i in
                   rng.choice(len(queries), max_queries, replace=False)]
    vec = np.load(out_dir / "vectors.npz")
    if vec["Q_old"].shape[0] != len(queries) or vec["V_old"].shape[0] != len(units):
        raise SystemExit("step1_data/vectors.npz was built from a different subset; "
                         "rerun `prepare --source msmarco` with the same --limit, "
                         "--max-queries and seed")
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid, t in queries:
            f.write(json.dumps({"qid": qid, "text": t}) + "\n")

    nq, np_ = resolve_prefixes(new_model)
    print(f"new model {new_model}: query prefix {nq!r}, passage prefix {np_!r}")
    print("encoding the corpus with the NEW model for ground truth only "
          "(reuses cache/ from the 500k run if present)")
    V, fitted = encode_all([t for _, t in units], new_model, device=device,
                           prefix=np_, cache_dir=cache_dir,
                           cache_key=corpus_key(new_model, np_, (i for i, _ in units)))
    Q = encode_with(fitted, [t for _, t in queries], new_model, prefix=nq)
    print("exact new-space top-10 for every query")
    top10 = topk_chunked(Q, V, 10)
    rng = np.random.default_rng(7)
    check = np.sort(rng.choice(len(units), n_check, replace=False))
    np.savez(out_dir / "truth.npz", Q_new=Q.astype(np.float32),
             top10=top10.astype(np.int64), check_ids=check,
             check_vecs=np.asarray(V[check], dtype=np.float32),
             new_model=new_model, query_prefix=nq, passage_prefix=np_)
    print(f"wrote {out_dir / 'truth.npz'} and queries.jsonl ({len(queries):,} queries)")


def _texts(out_dir: Path, ids) -> list[str]:
    want, out = set(int(i) for i in ids), {}
    with open(out_dir / "texts.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                out[i] = json.loads(line)["text"]
    return [out[int(i)] for i in ids]


def _queries(out_dir: Path) -> list[str]:
    with open(out_dir / "queries.jsonl", encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


# ----------------------------------------------------------- verify-encoder
def verify_encoder(out_dir: Path, enc_new: str, enc_old: str | None, chunk: int,
                   device: str | None, old_model: str,
                   threshold: float = 0.999) -> int:
    from . import encoders

    t = np.load(out_dir / "truth.npz")
    v = np.load(out_dir / "vectors.npz")
    rep = {}
    e = encoders.make(enc_new, str(t["passage_prefix"]), chunk, device)
    got = e.encode(_texts(out_dir, t["check_ids"]))
    cos_p = np.sum(got * t["check_vecs"], axis=1)
    qs = _queries(out_dir)[:256]
    eq = encoders.make(enc_new, str(t["query_prefix"]), 64, device)
    cos_q = np.sum(eq.encode(qs) * t["Q_new"][:256], axis=1)
    rep["new_passages_min_cos"] = float(cos_p.min())
    rep["new_queries_min_cos"] = float(cos_q.min())
    if enc_old:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from semview.migration import resolve_prefixes
        oq, _ = resolve_prefixes(old_model)
        eo = encoders.make(enc_old, oq, 64, device)
        cos_o = np.sum(eo.encode(qs) * v["Q_old"][:256], axis=1)
        rep["old_queries_min_cos"] = float(cos_o.min())
    ok = all(x >= threshold for x in rep.values())
    for k, x in rep.items():
        print(f"  {'ok  ' if x >= threshold else 'FAIL'} {k} = {x:.5f}")
    print(f"{'PASS' if ok else 'FAIL'}: every vector within cosine {threshold} "
          "of the ground truth's" if ok else
          "FAIL: the system's encoder does not reproduce the ground-truth vectors "
          "(pooling, truncation, prefix or model revision differs); fix before step2")
    return 0 if ok else 1


# --------------------------------------------------------------------- run
def _exact_window_recall(Q_old, V_old, top10, k):
    """Offline recall@K' with exact windows, and how many queries have a score
    tie at the window boundary (rank K' vs K'+1). Ties make the window -- and so
    recall -- depend on tie-breaking, which Postgres and numpy do differently;
    real embeddings almost never tie, hashed or lexical stand-ins often do.

    `k` is clamped to the corpus size, as `migration.frontier` does: at K' = N
    the window is the whole corpus and recall is 1 by construction, and there is
    no rank K'+1 to tie with. Passing k = N unclamped made np.partition raise
    `kth(=N) out of bounds`, so the offline reference crashed on exactly the
    K' that frontier documents itself as supporting.
    """
    n = V_old.shape[0]
    k = min(k, n)
    if k < 1:
        raise ValueError(f"k must be at least 1, got k={k} for N={n}")
    hit = np.empty(len(Q_old))
    ties = 0
    for lo in range(0, len(Q_old), 32):
        S = Q_old[lo:lo + 32] @ V_old.T
        if k < n:
            # The tie check needs rank K'+1; at k == n there is none.
            part = np.partition(-S, kth=[k - 1, k], axis=1)
            ties += int(np.sum(np.isclose(part[:, k - 1], part[:, k], atol=1e-6)))
        idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        for j, row in enumerate(idx):
            hit[lo + j] = len(set(row.tolist()) & set(top10[lo + j].tolist())) / top10.shape[1]
    return hit, ties


def run_one(dsn, V_old, Q_old, Q_new, top10, qtexts, kprime, window, enc_new,
            enc_old_q, enc_new_q, threads, iterative, max_scan, fresh,
            precomputed, union_k: int = 0) -> tuple[dict, list]:
    from .db import connect
    from .retriever import RepairCoordinator, ThawRetriever, db_io

    admin = connect(dsn)
    if fresh:
        admin.execute("TRUNCATE docs_new")
    # The union needs an HNSW index over every row of docs_new, maintained on
    # each repair. Base runs drop it, so they are not charged its insert cost.
    #
    # DO NOT "optimise" this by hoisting the CREATE out of the sweep loop. It
    # has been proposed as a bottleneck fix and it would corrupt the one
    # comparison this experiment exists to make: with the index present during
    # a base run, the base variant pays the union variant's index-maintenance
    # cost on every repair, and base-versus-union stops measuring the union.
    # The cost is real but bounded and deliberate -- the index is built over
    # the rows repaired so far, not over N, because `fresh or first` TRUNCATEs
    # away the random vectors step 1 left in docs_new. It is Postgres CPU, not
    # GPU, and it is in this experiment (E1/E2) only: `analysis/chain.py`,
    # which runs D-08, contains no database code at all.
    #
    # Reordering the sweep so the union variants run consecutively is not free
    # either: under --warm each variant sees whatever its predecessors
    # repaired, so changing the order changes the numbers.
    if union_k:
        admin.execute("CREATE INDEX IF NOT EXISTS docs_new_emb_hnsw ON docs_new "
                      "USING hnsw (emb vector_cosine_ops)")
    else:
        admin.execute("DROP INDEX IF EXISTS docs_new_emb_hnsw")
    fetch_text, write, io_conn = db_io(dsn, Q_new.shape[1])
    coord = RepairCoordinator(len(V_old), enc_new, fetch_text, write)
    n = len(Q_old)
    traces = [None] * n
    order = list(range(n))
    lock = threading.Lock()
    errors = []
    plan = {}

    def worker(tid):
        conn = connect(dsn)
        r = ThawRetriever(conn, coord, enc_old_q, enc_new_q, kprime,
                          window=window, iterative=iterative,
                          max_scan_tuples=max_scan, union_k=union_k)
        if tid == 0:
            plan["indexes"] = r.plan_indexes(Q_old[0])
        while True:
            with lock:
                if not order:
                    break
                i = order.pop(0)
            try:
                if precomputed:
                    traces[i] = r.query(q_old=Q_old[i], q_new=Q_new[i])
                else:
                    traces[i] = r.query(text=qtexts[i])
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"q{i}: {type(e).__name__}: {e}")
        conn.close()

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    rows = admin.execute("SELECT count(*), count(DISTINCT id) FROM docs_new "
                         "WHERE origin = 'repair'").fetchone()
    admin.close()
    io_conn.close()
    if errors:
        raise SystemExit(f"{len(errors)} queries failed, first: {errors[0]}")

    # recall divides by the width of the truth, not by a hard-coded 10: `top10`
    # is named for the default, but it is whatever the caller built
    e2e = np.array([len(set(t.results) & set(top10[i].tolist())) / top10.shape[1]
                    for i, t in enumerate(traces)])
    win = np.array([len(set(t.window) & set(top10[i].tolist())) / top10.shape[1]
                    for i, t in enumerate(traces)])
    cand = np.array([len((set(t.window) | set(t.extra)) & set(top10[i].tolist())) / top10.shape[1]
                     for i, t in enumerate(traces)])
    hits = np.array([[x in set(t.results) for x in top10[i].tolist()]
                     for i, t in enumerate(traces)])
    tot = np.array([t.wall_ms for t in traces])           # wall clock per query
    parts = np.array([t.total_ms for t in traces])        # sum of the timed parts
    dec = max(1, n // 10)
    comp = {f: round(float(np.mean([getattr(t, f) for t in traces])), 3)
            for f in ("q_old_ms", "q_new_ms", "window_ms", "present_ms", "text_ms",
                      "encode_ms", "write_ms", "wait_ms", "union_ms", "rank_ms")}
    stale = np.array([t.n_stale for t in traces])
    enc_units = int(stale.sum())
    rep = dict(
        window=window, kprime=kprime, union_k=union_k, queries=n, threads=threads,
        iterative=iterative if window == "hnsw" else None,
        plan_indexes=plan.get("indexes"),
        recall_e2e=round(float(e2e.mean()), 4),
        recall_window=round(float(win.mean()), 4),
        recall_candidates=round(float(cand.mean()), 4),
        union_gain=round(float(e2e.mean() - win.mean()), 4),
        recall_e2e_by_decile=[round(float(e2e[i * dec:(i + 1) * dec].mean()), 4)
                              for i in range(10) if i * dec < n],
        hit_rate_by_rank=[round(float(x), 4) for x in hits.mean(axis=0)],
        per_query_p10=round(float(np.percentile(e2e, 10)), 4),
        worst_decile_mean=round(float(np.sort(e2e)[:dec].mean()), 4),
        extra_per_query_mean=round(float(np.mean([len(t.extra) for t in traces])), 2),
        e2e_equals_window=round(float(np.mean(np.isclose(e2e, cand))), 4),
        window_min=int(min(t.window_size for t in traces)),
        latency_ms=dict(p50=round(float(np.percentile(tot, 50)), 2),
                        p95=round(float(np.percentile(tot, 95)), 2),
                        p99=round(float(np.percentile(tot, 99)), 2),
                        max=round(float(tot.max()), 2),
                        first_decile_p50=round(float(np.percentile(tot[:dec], 50)), 2),
                        last_decile_p50=round(float(np.percentile(tot[-dec:], 50)), 2)),
        mean_component_ms=comp,
        unaccounted_ms_mean=round(float((tot - parts).mean()), 3),
        stale_per_query_mean=round(float(stale.mean()), 2),
        encoded_units=enc_units,
        reached=round(enc_units / len(V_old), 5),
        encode_ms_per_unit=round(sum(t.encode_ms for t in traces)
                                 / max(enc_units, 1), 4),
        write_ms_per_unit=round(sum(t.write_ms for t in traces)
                                / max(enc_units, 1), 4),
        docs_new_rows=int(rows[0]), docs_new_distinct=int(rows[1]),
        encoder_calls=enc_new.calls, wall_s=round(wall, 1),
    )
    return rep, traces


def plan_gates(window: str, union_k: int, plan_indexes, n_units: int,
               recall_e2e: float, recall_offline_exact: float) -> dict:
    """The gates that judge how the window query was planned.

    Pure, and separate from `run`, so it can be checked without Postgres. This
    is where an exact window WITH the union was once required to use an HNSW
    index it deliberately avoids: the branch tested `exact and not union_k`, so
    exact+union fell through to the HNSW rule, every such configuration failed,
    and the whole run exited non-zero with its numbers sound.

    Which plan is correct is decided by `window` alone. Whether end-to-end
    recall should reproduce the offline number is decided by `union_k` as well:
    the base design's end-to-end recall IS its window recall, while the union
    raises end-to-end recall above the window's by construction.
    """
    g = {}
    if window == "exact":
        g["exact plan has no HNSW"] = not plan_indexes
        if not union_k:
            g["E1 reproduces offline recall"] = (
                abs(recall_e2e - recall_offline_exact) <= 0.005)
    elif n_units >= 100_000:
        # on a toy corpus, where K' is a large share of N, a seq scan is the
        # planner's right answer (step 1 gates the same way)
        g["HNSW plan"] = "docs_emb_old_hnsw" in (plan_indexes or [])
    return g


def run(dsn: str, out_dir: Path, kprimes, windows, n_queries: int, enc_new_spec,
        enc_old_spec, chunk: int, threads: int, iterative: str, max_scan: int,
        fresh: bool, precomputed: bool, device: str | None, out: Path,
        truth=None, encoders_override=None,
        old_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        union_ks=(0,)) -> int:
    from . import encoders

    v = np.load(out_dir / "vectors.npz")
    t = truth if truth is not None else np.load(out_dir / "truth.npz")
    n = min(n_queries, len(v["Q_old"]))
    V_old, Q_old = v["V_old"], v["Q_old"][:n]
    Q_new, top10 = t["Q_new"][:n], t["top10"][:n]
    qtexts = None if precomputed else _queries(out_dir)[:n]

    if encoders_override:
        enc_new, enc_old_q, enc_new_q = encoders_override
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from semview.migration import resolve_prefixes
        oq, _ = resolve_prefixes(old_model)
        enc_new = encoders.make(enc_new_spec, str(t["passage_prefix"]), chunk, device)
        enc_new_q = enc_old_q = None
        if not precomputed:
            enc_new_q = encoders.make(enc_new_spec, str(t["query_prefix"]), 1, device)
            enc_old_q = encoders.make(enc_old_spec, oq, 1, device)

    print("offline reference: exact old-space windows, numpy")
    exact_by_k, ties_by_k = {}, {}
    for k in kprimes:
        h, ties = _exact_window_recall(Q_old, V_old, top10, k)
        exact_by_k[k], ties_by_k[k] = float(h.mean()), ties
        if ties > 0.01 * n:
            print(f"  ! K'={k}: {ties} of {n} queries have a score tie at the window "
                  "boundary; E1 can then differ from the offline number by "
                  "tie-breaking alone. Duplicate passages tie (MS MARCO has "
                  "them), and stand-in encoders tie often")
    report = dict(queries=n, N=int(len(V_old)), chunk=chunk, precomputed=precomputed,
                  threads=threads, offline_exact_recall=exact_by_k,
                  boundary_ties=ties_by_k, runs=[])
    traces_out = {}
    for window, k, uk in [(w, k, u) for w in windows for k in kprimes
                          for u in union_ks]:
            print(f"\n== window={window}  K'={k}  union={uk or 'off'}  ({n} queries, "
                  f"{'fresh' if fresh else 'warm'} docs_new)")
            enc_new.calls = enc_new.items = 0
            # always start from an empty docs_new: step 1 left random vectors
            # there. --warm only keeps repairs between runs of one invocation.
            first = not report["runs"]
            r, trs = run_one(dsn, V_old, Q_old, Q_new, top10, qtexts, k, window,
                             enc_new, enc_old_q, enc_new_q, threads, iterative,
                             max_scan, fresh or first, precomputed, uk)
            r["recall_offline_exact"] = round(exact_by_k[k], 4)
            r["ann_loss"] = round(exact_by_k[k] - r["recall_window"], 4)
            report["runs"].append(r)
            traces_out[f"{window}_{k}_u{uk}"] = [
                dict(wall_ms=round(x.wall_ms, 3), parts_ms=round(x.total_ms, 3),
                     n_stale=x.n_stale,
                     n_waited=x.n_waited, encode_ms=round(x.encode_ms, 3),
                     window_ms=round(x.window_ms, 3), rank_ms=round(x.rank_ms, 3),
                     union_ms=round(x.union_ms, 3), n_extra=len(x.extra),
                     repaired_total=x.repaired_total) for x in trs]
            L = r["latency_ms"]
            print(f"  recall@10 e2e {r['recall_e2e']:.4f}  window {r['recall_window']:.4f}"
                  f"  offline-exact {r['recall_offline_exact']:.4f}  "
                  f"ANN loss {r['ann_loss']:+.4f}"
                  + (f"  union gain {r['union_gain']:+.4f} "
                     f"({r['extra_per_query_mean']} extra/query)" if uk else ""))
            print(f"  latency p50/p95/p99 {L['p50']}/{L['p95']}/{L['p99']} ms; "
                  f"first-decile p50 {L['first_decile_p50']} -> last {L['last_decile_p50']}")
            print(f"  stale/query {r['stale_per_query_mean']}  encoded {r['encoded_units']:,}"
                  f" (reached {r['reached']:.3f})  encode {r['encode_ms_per_unit']} ms/unit"
                  f"  write {r['write_ms_per_unit']} ms/unit")
            print(f"  plan: {r['plan_indexes'] or 'seq scan'}")

    gates = {}
    for r in report["runs"]:
        tag = f"{r['window']} K'={r['kprime']} u={r['union_k']}"
        gates[f"persisted once ({tag})"] = (
            r["docs_new_rows"] == r["docs_new_distinct"] == r["encoded_units"]
            if fresh else r["docs_new_rows"] == r["docs_new_distinct"])
        gates[f"rank is exact within the candidates ({tag})"] = r["e2e_equals_window"] >= 0.99
        gates[f"full window ({tag})"] = r["window_min"] >= r["kprime"]
        if r["union_k"]:
            base = [b for b in report["runs"] if b["window"] == r["window"]
                    and b["kprime"] == r["kprime"] and not b["union_k"]]
            # This one holds by construction when the rank is exact (the rank
            # gate is what catches a broken rank); it is kept as a cheap check.
            gates[f"union never below base ({tag})"] = r["recall_e2e"] >= r["recall_window"] - 1e-9
            # Unlike the others, this one fails if the union silently does
            # nothing -- an empty docs_new index, a wrong predicate, a search
            # that returns only window members.
            gates[f"union produced candidates outside the window ({tag})"] = (
                r["extra_per_query_mean"] > 0)
            if base and fresh:
                gates[f"union costs no extra passes ({tag})"] = (
                    r["encoded_units"] == base[0]["encoded_units"])
        for name, ok in plan_gates(r["window"], r["union_k"], r["plan_indexes"],
                                   len(V_old), r["recall_e2e"],
                                   r["recall_offline_exact"]).items():
            gates[f"{name} ({tag})"] = ok
    report["gates"] = gates
    # The comparison the union exists for: does union at K' reach what the base
    # design needs a deeper window for? Reported, not gated -- it is a result.
    matches = []
    for u in (r for r in report["runs"] if r["union_k"]):
        for b in (r for r in report["runs"] if not r["union_k"]
                  and r["window"] == u["window"] and r["kprime"] > u["kprime"]):
            matches.append(dict(window=u["window"], union_kprime=u["kprime"],
                                union_k=u["union_k"], union_recall=u["recall_e2e"],
                                union_encoded=u["encoded_units"],
                                base_kprime=b["kprime"], base_recall=b["recall_e2e"],
                                base_encoded=b["encoded_units"],
                                union_reaches_base=u["recall_e2e"] >= b["recall_e2e"]))
    report["union_vs_deeper_base"] = matches
    for m in matches:
        print(f"union K'={m['union_kprime']} ({m['union_recall']:.4f}, "
              f"{m['union_encoded']:,} encoded) vs base K'={m['base_kprime']} "
              f"({m['base_recall']:.4f}, {m['base_encoded']:,} encoded): "
              f"{'reaches it' if m['union_reaches_base'] else 'below'}")
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.results_io import environment, write_results
    report["environment"] = environment()
    write_results(out, report)
    # the per-query traces are the largest and costliest output: same checks
    write_results(out.with_suffix(".traces.json"),
                  dict(environment=report["environment"], traces=traces_out),
                  indent=None)
    print("\n" + "=" * 70)
    print(f"  {'window':<7}{'K':>6}{'union':>7}{'e2e':>8}{'exact':>8}{'ANN loss':>10}"
          f"{'p50':>9}{'p99':>9}{'reached':>9}")
    for r in report["runs"]:
        print(f"  {r['window']:<7}{r['kprime']:>6}{(r['union_k'] or '-'):>7}"
              f"{r['recall_e2e']:>8.4f}"
              f"{r['recall_offline_exact']:>8.4f}{r['ann_loss']:>+10.4f}"
              f"{r['latency_ms']['p50']:>9.1f}{r['latency_ms']['p99']:>9.1f}"
              f"{r['reached']:>9.4f}")
    print("-" * 70)
    for g, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {g}")
    print("=" * 70)
    print(f"report: {out}")
    return 0 if all(gates.values()) else 1


# ---------------------------------------------------------------- selftest
def selftest2(dsn: str, out_dir: Path, threads: int = 4) -> int:
    """Synthetic: a new space correlated with the old one, an encoder that looks
    vectors up (with a sleep standing in for GPU time), real Postgres."""
    from .db import connect
    from .encoders import TableEncoder

    v = np.load(out_dir / "vectors.npz")
    V_old, Q_old = v["V_old"], v["Q_old"]
    rng = np.random.default_rng(11)
    R = rng.normal(size=(V_old.shape[1], 768)).astype(np.float32) / np.sqrt(768)
    # a new space that disagrees with the old one: a nonlinear distortion,
    # so windows miss some new top-10 members and the union has work to do
    R2 = rng.normal(size=(V_old.shape[1], 768)).astype(np.float32)
    V_new = V_old @ R + 0.6 * np.tanh(V_old @ R2) / np.sqrt(768)
    V_new /= np.linalg.norm(V_new, axis=1, keepdims=True)
    Q_new = Q_old @ R + 0.6 * np.tanh(Q_old @ R2) / np.sqrt(768)
    Q_new /= np.linalg.norm(Q_new, axis=1, keepdims=True)
    top10 = np.empty((len(Q_new), 10), dtype=np.int64)
    for lo in range(0, len(Q_new), 64):
        S = Q_new[lo:lo + 64] @ V_new.T
        top10[lo:lo + 64] = np.argsort(-S, axis=1)[:, :10]
    conn = connect(dsn)
    conn.execute("UPDATE docs SET text = 'k:' || id WHERE text IS NULL "
                 "OR text NOT LIKE 'k:%'")
    conn.close()
    truth = dict(Q_new=Q_new, top10=top10, passage_prefix="", query_prefix="")
    enc = TableEncoder(V_new.astype(np.float32), batch=8, delay_ms=0.05)
    return run(dsn, out_dir, [200, 500], ["exact", "hnsw"], 200, None, None, 8,
               threads, "relaxed_order", 100_000, True, True, None,
               out_dir / "step2_selftest.json", truth=truth,
               encoders_override=(enc, None, None), union_ks=(0, 50))
