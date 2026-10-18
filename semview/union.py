"""Candidate union: let the repaired set nominate candidates too.

The base design takes every candidate from the old index, so quality is capped
at recall@K' of the old space. But every repair leaves a new-space vector
behind, and traffic concentrates repairs where queries go. Union adds a second
source of candidates:

    window(q)  = old-space top-K'(q)                       (as before)
    extra(q)   = new-space top-K_n(q) among REPAIRED units  (new)
    candidates = window(q) U extra(q);  rank all of them in the new space

Two properties make it cheap to reason about:

  * It costs no extra forward passes. Every unit in extra(q) already has a new
    vector; only window members are ever repaired, exactly as before. Savings
    at a given K' are therefore unchanged -- the simulation checks this rather
    than assuming it.
  * It can only help. Candidates are a superset of the window's, and ranking
    is exact in the new space, so per-query recall never falls. What it can
    buy is either higher quality at the same K', or the same quality at a
    smaller K', and a smaller K' does raise savings.

The simulation replays the query stream in order against a growing repaired
set. To keep it tractable the new-space search for a block of B queries uses
the repaired set as it stood at the start of the block; within-block repairs
are invisible to it, so the reported gain is a slight UNDERestimate.

    python -m semview.union --corpus beir --beir-name msmarco --path beir/ \\
        --qrels-split dev --limit 500000 \\
        --old sentence-transformers/all-MiniLM-L6-v2 \\
        --new BAAI/bge-base-en-v1.5 --cache-dir cache/ \\
        --kprimes 100,200,500,1000 --kn 10,100 --out results/union_msmarco_500k.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _topk_rows(Q: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k per query of Q against V (dense, row indices into V)."""
    k = min(k, V.shape[0])
    if k <= 0:
        return np.zeros((len(Q), 0), dtype=np.int64)
    from .migration import topk_chunked
    return topk_chunked(Q, V, k).astype(np.int64)


def replay(V_old, V_new, Q_old, Q_new, truth: np.ndarray, kprime: int,
           kns=(10,), k: int = 10, block: int = 50,
           old_windows: np.ndarray | None = None) -> dict:
    """Replay the queries in order; return per-query hits for the base design
    and for union at each K_n, plus the repaired count after every query.

    `truth` is the new model's exact top-k for every query (row indices).
    `old_windows` may be passed in (top-K'_max windows, prefix = top-K').
    """
    N, D = len(V_old), len(Q_old)
    W = (old_windows[:, :kprime] if old_windows is not None
         else _topk_rows(Q_old, V_old, kprime))
    repaired = np.zeros(N, dtype=bool)
    base_hits = np.zeros((D, k), dtype=bool)          # rank-resolved
    union_hits = {kn: np.zeros((D, k), dtype=bool) for kn in kns}
    union_extra = {kn: np.zeros(D, dtype=np.int32) for kn in kns}
    reached_after = np.zeros(D, dtype=np.int64)
    kmax = max(kns)
    for lo in range(0, D, block):
        hi = min(lo + block, D)
        ids = np.flatnonzero(repaired)                # state at block start
        extra = (ids[_topk_rows(Q_new[lo:hi], V_new[ids], kmax)]
                 if len(ids) else np.zeros((hi - lo, 0), dtype=np.int64))
        for j, q in enumerate(range(lo, hi)):
            win = W[q]
            t = truth[q, :k]
            in_win = np.isin(t, win)
            base_hits[q] = in_win
            for kn in kns:
                ex = extra[j, :kn] if extra.shape[1] else extra[j]
                union_hits[kn][q] = in_win | np.isin(t, ex)
                union_extra[kn][q] = int(np.setdiff1d(ex, win).size)
            repaired[win] = True                      # the only repairs made
            reached_after[q] = int(repaired.sum())
    return dict(base_hits=base_hits, union_hits=union_hits,
                union_extra=union_extra, reached_after=reached_after, N=N)


def _decile_bounds(n: int, parts: int = 10) -> list[tuple[int, int]]:
    """Contiguous buckets that together cover every index in range(n).

    Floor division alone leaves up to `parts - 1` queries uncovered -- at D = 11
    the 11th query belonged to no decile -- which made `recall_by_decile` a list
    that does not sum to the stream while `recall_last_decile` (built from
    `per_q[-dec:]`) did cover the tail: the two reported different last deciles.
    Every shipped union run used D = 5000, where 10 * 500 = 5000 exactly and
    nothing was lost, so no published number is affected.
    """
    if n <= 0:
        return []
    cut = max(1, n // parts)
    bounds = [(i * cut, min(n, (i + 1) * cut)) for i in range(parts)]
    bounds = [(a, b) for a, b in bounds if a < n]
    bounds[-1] = (bounds[-1][0], n)          # the tail absorbs the remainder
    return bounds


def summarise(res: dict, kprime: int, k: int = 10) -> dict:
    N = res["N"]
    D = len(res["base_hits"])
    if D == 0:
        # `reached_after[-1]` below is the whole savings figure, and it does not
        # exist without queries. Say so here rather than raising IndexError.
        raise SystemExit("summarise() was given an empty replay: no query was "
                         "served, so there is no reached set to report")
    dec = max(1, D // 10)

    def stats(h: np.ndarray) -> dict:
        per_q = h.mean(axis=1)
        bounds = _decile_bounds(D)
        # The first and last decile are the first and last BUCKET of the same
        # partition `recall_by_decile` reports, not a separate `D // 10` slice.
        # They used to be the latter, so for a D that is not a multiple of ten
        # the summary and the per-decile list disagreed about what the last
        # decile was -- at D=11 one covered 2 queries and the other 1. Every
        # published union run used D=5000, where the two coincide, so no
        # shipped number moves; and no claim reads this field.
        (f_lo, f_hi), (l_lo, l_hi) = bounds[0], bounds[-1]
        return dict(
            recall=round(float(per_q.mean()), 4),
            recall_first_decile=round(float(per_q[f_lo:f_hi].mean()), 4),
            recall_last_decile=round(float(per_q[l_lo:l_hi].mean()), 4),
            recall_by_decile=[round(float(per_q[a:b].mean()), 4)
                              for a, b in bounds],
            hit_rate_by_rank=[round(float(x), 4) for x in h.mean(axis=0)],
            top1_hit=round(float(h[:, 0].mean()), 4),
            per_query_p10=round(float(np.percentile(per_q, 10)), 4),
            worst_decile_mean=round(float(np.sort(per_q)[:dec].mean()), 4),
            queries_with_zero=int((per_q == 0).sum()),
        )
    out = dict(kprime=kprime, k=k, N=N, D=D,
               reached=round(float(res["reached_after"][-1] / N), 4),
               savings_pct=round(100 * (1 - res["reached_after"][-1] / N), 2),
               base=stats(res["base_hits"]), union={})
    for kn, h in res["union_hits"].items():
        s = stats(h)
        s["mean_extra_candidates"] = round(float(res["union_extra"][kn].mean()), 1)
        s["never_below_base"] = bool(np.all(h.mean(axis=1) >= res["base_hits"].mean(axis=1)))
        out["union"][str(kn)] = s
    return out


def run(V_old, V_new, Q_old, Q_new, kprimes, kns, k=10, block=50,
        verbose=True) -> list[dict]:
    """The full sweep. Truth and the deepest old windows are computed once."""
    if len(Q_old) == 0:
        # Every figure this tool prints is a function of the query stream, and
        # the savings come from the reached set that stream builds. An empty
        # stream is not a degenerate measurement, it is no measurement, so refuse
        # it here with a reason instead of failing later inside summarise().
        raise SystemExit(
            "no queries to replay: the query set is empty. Check the corpus and "
            "queries files, the qrels split, and any --max-queries that could "
            "have selected nothing.")
    truth = _topk_rows(Q_new, V_new, k)
    W = _topk_rows(Q_old, V_old, max(kprimes))
    rows = []
    for kp in sorted(kprimes):
        res = replay(V_old, V_new, Q_old, Q_new, truth, kp, kns, k, block,
                     old_windows=W)
        s = summarise(res, kp, k)
        rows.append(s)
        if verbose:
            u = "   ".join(f"union K_n={kn}: {v['recall']:.4f} "
                           f"(last decile {v['recall_last_decile']:.4f})"
                           for kn, v in s["union"].items())
            print(f"K'={kp:<5} savings {s['savings_pct']:5.1f}%   base "
                  f"{s['base']['recall']:.4f} (last decile "
                  f"{s['base']['recall_last_decile']:.4f})   {u}", flush=True)
    return rows


def main() -> None:  # pragma: no cover
    import argparse
    from .migration import (corpus_key, encode_all, encode_with, load_beir,
                            resolve_prefixes)

    ap = argparse.ArgumentParser(prog="union")
    ap.add_argument("--corpus", default="beir", choices=["beir"])
    ap.add_argument("--beir-name", default="msmarco")
    ap.add_argument("--path", default="beir/")
    ap.add_argument("--qrels-split", default="dev")
    ap.add_argument("--limit", type=int, default=500_000)
    ap.add_argument("--max-queries", type=int, default=5000)
    ap.add_argument("--old", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--new", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--cache-dir", default="cache/")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--kprimes", default="100,200,500,1000")
    ap.add_argument("--kn", default="10,100",
                    help="new-space candidates taken from the repaired set")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order-seed", type=int, default=None,
                    help="shuffle the query stream with this seed (default: "
                         "the same sample and order as semview.migration)")
    ap.add_argument("--out", default="union.json")
    a = ap.parse_args()

    units, queries = load_beir(a.beir_name, a.path, a.limit, a.seed, a.qrels_split)
    if a.max_queries and len(queries) > a.max_queries:
        rng = np.random.default_rng(a.seed)
        queries = [queries[i] for i in
                   rng.choice(len(queries), a.max_queries, replace=False)]
    if a.order_seed is not None:
        perm = np.random.default_rng(a.order_seed).permutation(len(queries))
        queries = [queries[i] for i in perm]
    if not queries:
        # Refuse before encoding anything: this is cheap to check here and
        # expensive to discover after two corpus passes.
        raise SystemExit(
            "the query set is empty, so the union sweep has nothing to replay. "
            "Check --qrels-split (a split with no matching queries empties it), "
            "the queries file, and --limit on the corpus.")
    if not units:
        raise SystemExit("the corpus is empty: no units to offer as candidates.")
    utext = [t for _, t in units]
    qtext = [t for _, t in queries]
    ids = [i for i, _ in units]
    oq, op = resolve_prefixes(a.old)
    nq, np_ = resolve_prefixes(a.new)
    V_old, f_old = encode_all(utext, a.old, device=a.device, prefix=op,
                              cache_dir=a.cache_dir, cache_key=corpus_key(a.old, op, ids))
    V_new, f_new = encode_all(utext, a.new, device=a.device, prefix=np_,
                              cache_dir=a.cache_dir, dtype=a.dtype,
                              cache_key=corpus_key(a.new, np_, ids))
    Q_old = encode_with(f_old, qtext, a.old, prefix=oq)
    Q_new = encode_with(f_new, qtext, a.new, prefix=nq)
    V_old = np.asarray(V_old, dtype=np.float32)
    V_new = np.asarray(V_new, dtype=np.float32)
    print(f"N={len(units):,} D={len(queries):,}  old={a.old}  new={a.new}")
    rows = run(V_old, V_new, Q_old, Q_new,
               [int(x) for x in a.kprimes.split(",")],
               [int(x) for x in a.kn.split(",")], a.k, a.block)
    from .migration import _fingerprint, _ids_hash
    from .results_io import environment, write_results
    write_results(a.out, dict(
        corpus=a.beir_name, N=len(units), D=len(queries), old=a.old, new=a.new,
        k=a.k, block=a.block, order_seed=a.order_seed,
        fingerprint=dict(old_head=_fingerprint(V_old), new_head=_fingerprint(V_new),
                         query_ids_sha256=_ids_hash(q[0] for q in queries),
                         unit_ids_sha256=_ids_hash(ids)),
        environment=environment(dtype=a.dtype, device=a.device), rows=rows))


if __name__ == "__main__":
    main()
