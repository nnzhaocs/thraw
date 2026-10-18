"""Why is recall@K' higher at 8.8M than at 500k? A test, not a guess.

Same 5,000 dev queries, same k, same prefixes in both runs (the query list
does not depend on --limit; both logs show the same prefixes). What differs
is the corpus: the 500k subset is every dev query's gold passages plus random
filler, the 8.8M corpus is everything.

Hypothesis H: in the 500k subset a query's new-model top-10 is its gold
passage(s) plus weak matches from the random filler, which sit close together
(near-ties) and which the two models order differently; at 8.8M the top-10 is
a tight cluster of related passages that both models find. If H holds:

  1. the top-10 is more tied at 500k (smaller gap between rank 10 and 11,
     more passages within a small margin of rank 10);
  2. within a bin of equal tie density, recall is about the same at both
     scales: the gain is a change in mix, not a different relationship;
  3. per query, the recall gain grows with the drop in tie density.

If (2) fails -- recall differs at equal tie density -- H does not explain it.

Two confounds are checked along the way:
  * storage precision: the 8.8M run stored new vectors as float16, the 500k
    run as float32. --fp16-check recomputes the 500k numbers with the new
    vectors rounded to float16.
  * query identity: the hash of the query id list is printed for both scales
    and must be equal.

    python analysis/margin_8m.py --path beir/ --cache-dir cache/ \\
        --scales 500000,full --dtype-full float16 --fp16-check \\
        --out results/margin_8m.json

Uses the cached corpus vectors from the 500k and 8.8M runs; with the caches
present only the 5,000 queries are encoded. RAM as for the 8.8M migration run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semview.migration import (_fingerprint, _ids_hash, corpus_key,  # noqa: E402
                               encode_all, encode_with, load_beir,
                               resolve_prefixes, topk_chunked)


def tie_bins(small_tie, large_tie, small_rec: dict, large_rec: dict, kps,
             quantiles=(0, .2, .4, .6, .8, 1.0)) -> list[dict]:
    """Bins of tie density shared by both scales (edges from the pooled
    quantiles), with each scale's share of queries and mean recall per bin.
    Bins are [lo, hi) except the last, which includes hi. Shares are exact,
    not rounded, so they can be used as weights."""
    small_tie, large_tie = np.asarray(small_tie), np.asarray(large_tie)
    edges = np.unique(np.quantile(np.concatenate([small_tie, large_tie]), quantiles))
    if len(edges) == 1:                       # every query equally tied
        edges = np.array([edges[0], edges[0]])
    bins = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = j == len(edges) - 2
        row = dict(tie_from=float(lo), tie_to=float(hi))
        for name, tie, rec in (("small", small_tie, small_rec),
                               ("large", large_tie, large_rec)):
            sel = (tie >= lo) & ((tie <= hi) if last else (tie < hi))
            row[f"{name}_share"] = float(sel.mean())
            for kp in kps:
                row[f"{name}_recall@{kp}"] = (float(np.asarray(rec[kp])[sel].mean())
                                              if sel.any() else None)
        bins.append(row)
    return bins


def reweight(bins: list[dict], kps) -> tuple[dict, float, dict]:
    """Small-scale recall averaged with the large scale's bin shares as weights.
    Bins with no small-scale queries cannot be used; the weights of the rest
    are renormalised to sum to one (skipping them without renormalising biases
    the result low).

    Returns the re-weighted recall, the WORST coverage across the depths, and
    the per-depth coverage. An earlier version assigned `covered = w` inside the
    loop and returned whatever the last depth happened to give. Today a bin is
    unusable exactly when it holds no small-scale queries, which does not depend
    on the depth, so every depth gives the same number and the bug was
    invisible -- but the caller prints that number as the share of queries the
    comparison covers, and a silent last-wins is how a caveat goes quietly
    wrong. The minimum is reported because the caveat should describe the
    weakest depth, not the luckiest, and the full map is written to the results
    so a disagreement is visible rather than averaged away.
    """
    out, per_kp = {}, {}
    for kp in kps:
        use = [b for b in bins if b[f"small_recall@{kp}"] is not None]
        w = sum(b["large_share"] for b in use)
        per_kp[kp] = round(w, 6)
        out[kp] = (round(sum(b["large_share"] * b[f"small_recall@{kp}"] for b in use)
                         / w, 4) if w > 0 else None)
    return out, (min(per_kp.values()) if per_kp else 0.0), per_kp


def sorted_topk(Q, V, m):
    """Top-m indices and similarities per query, best first.

    The similarities are recomputed here rather than read out of
    `topk_chunked`, because the tie analysis needs full float32 precision and
    the search may have run in float16. That costs one scattered gather of m
    rows per query from V, which on the 8.8M side is a 13.5 GB float16 memmap.

    A block-wise version (gather `block` index arrays at once, one einsum for
    the block) was written and measured, because the per-query gather is the
    slow part. It is not used: reassociating the float32 sum moves the
    similarities by up to 7e-8, which flips the ORDER of near-tied candidates
    for 9 of 2000 queries. The reported statistics move by at most 1e-7 and the
    top-10 sets are unchanged for every query, so the analysis is unaffected --
    but it would make the shipped `results/runs/.../margin_8m.json`
    irreproducible in its last digits, and that file is cited for the fp16
    control. Not worth it for a tool that already completes.
    """
    idx = np.asarray(topk_chunked(Q, V, m), dtype=np.int64)
    sims = np.empty(idx.shape, dtype=np.float32)
    for j in range(len(Q)):
        sims[j] = np.asarray(V[idx[j]], dtype=np.float32) @ Q[j]
    order = np.argsort(-sims, axis=1)
    return (np.take_along_axis(idx, order, axis=1),
            np.take_along_axis(sims, order, axis=1))


def _spearman(x, y) -> float | None:
    """Rank correlation with average ranks for ties; None if either is constant."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):              # average the ranks of tied values
            sel = v == val
            r[sel] = r[sel].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def margin_stats(sims: np.ndarray, k: int, delta: float) -> dict:
    s_k, s_k1 = sims[:, k - 1], sims[:, k]
    return dict(gap=s_k - s_k1,                                   # rank k vs k+1
                tie=(np.abs(sims - s_k[:, None]) <= delta).sum(1) - 1,
                s1=sims[:, 0], sk=s_k)


def recall_at(truth: np.ndarray, windows: np.ndarray, kprimes) -> dict:
    out = {}
    for kp in kprimes:
        w = windows[:, :kp]
        out[kp] = np.array([np.isin(t, wi).mean() for t, wi in zip(truth, w)])
    return out


def one_scale(a, limit, dtype_new):
    units, queries = load_beir("msmarco", a.path, limit, a.seed, "dev")
    if a.max_queries and len(queries) > a.max_queries:
        rng = np.random.default_rng(a.seed)
        queries = [queries[i] for i in
                   rng.choice(len(queries), a.max_queries, replace=False)]
    utext = [t for _, t in units]
    qtext = [t for _, t in queries]
    oq, op = resolve_prefixes(a.old)
    nq, np_ = resolve_prefixes(a.new)
    V_old, f_old = encode_all(utext, a.old, device=a.device, prefix=op,
                              cache_dir=a.cache_dir,
                              cache_key=corpus_key(a.old, op, (i for i, _ in units)))
    V_new, f_new = encode_all(utext, a.new, device=a.device, prefix=np_,
                              cache_dir=a.cache_dir, dtype=dtype_new,
                              cache_key=corpus_key(a.new, np_, (i for i, _ in units)))
    Q_old = encode_with(f_old, qtext, a.old, prefix=oq)
    Q_new = encode_with(f_new, qtext, a.new, prefix=nq)
    del utext
    return dict(units=units, queries=queries, V_old=V_old, V_new=V_new,
                Q_old=Q_old, Q_new=Q_new, prefixes=(oq, op, nq, np_))


def analyse(S, k, kprimes, m, delta, fp16=False):
    V_new = S["V_new"]
    if fp16:
        V_new = np.asarray(V_new, dtype=np.float16)
    idx, sims = sorted_topk(S["Q_new"], V_new, m)
    W = np.asarray(topk_chunked(S["Q_old"], S["V_old"], max(kprimes)), dtype=np.int64)
    # topk_chunked returns each query's top set; recall only needs membership
    return dict(truth=idx[:, :k], sims=sims, rec=recall_at(idx[:, :k], W, kprimes),
                **margin_stats(sims, k, delta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="beir/")
    ap.add_argument("--cache-dir", default="cache/")
    ap.add_argument("--scales", default="500000,full")
    ap.add_argument("--dtype-small", default="float32",
                    help="storage dtype the 500k run used for new vectors")
    ap.add_argument("--dtype-full", default="float16",
                    help="storage dtype the 8.8M run used for new vectors")
    ap.add_argument("--max-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--new", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kprimes", default="10,50,200,500")
    ap.add_argument("--m", type=int, default=100, help="new-space depth for margins")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="'tied with rank k' = cosine within this of rank k")
    ap.add_argument("--fp16-check", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="margin_8m.json")
    a = ap.parse_args()
    kps = [int(x) for x in a.kprimes.split(",")]

    res, raw = {}, {}
    for sc in a.scales.split(","):
        limit = None if sc == "full" else int(sc)
        S = one_scale(a, limit, a.dtype_full if limit is None else a.dtype_small)
        N = len(S["units"])
        print(f"\n== N={N:,}  D={len(S['queries']):,}  query ids "
              f"{_ids_hash(q for q, _ in S['queries'])}  prefixes {S['prefixes']}")
        r = analyse(S, a.k, kps, a.m, a.delta)
        raw[sc] = dict(r, ids=[u for u, _ in S["units"]], N=N,
                       qids=[q for q, _ in S["queries"]])
        entry = dict(N=N, query_ids_sha256=_ids_hash(q for q, _ in S["queries"]),
                     new_head=_fingerprint(S["V_new"]),
                     recall={kp: round(float(v.mean()), 4) for kp, v in r["rec"].items()},
                     gap_median=round(float(np.median(r["gap"])), 5),
                     tie_mean=round(float(r["tie"].mean()), 2),
                     sk_median=round(float(np.median(r["sk"])), 4))
        if a.fp16_check and limit is not None:
            r16 = analyse(S, a.k, kps, a.m, a.delta, fp16=True)
            entry["recall_new_vectors_fp16"] = {
                kp: round(float(v.mean()), 4) for kp, v in r16["rec"].items()}
            entry["truth_changed_by_fp16"] = round(float(np.mean(
                [len(set(x) ^ set(y)) > 0 for x, y in zip(r["truth"], r16["truth"])])), 4)
        res[sc] = entry
        print(f"  recall {entry['recall']}")
        print(f"  rank-{a.k} vs {a.k + 1} gap median {entry['gap_median']}, "
              f"passages tied with rank {a.k} (within {a.delta}): {entry['tie_mean']}")
        if "recall_new_vectors_fp16" in entry:
            print(f"  with new vectors in float16: {entry['recall_new_vectors_fp16']}"
                  f"  (top-{a.k} changed for {entry['truth_changed_by_fp16']:.1%} of queries)")
        del S

    scales = list(raw)
    if len(scales) == 2:
        sm, lg = raw[scales[0]], raw[scales[1]]
        if sm["qids"] != lg["qids"]:
            print("\nWARNING: the two scales used different queries; nothing below "
                  "compares like with like")
        # (1)-(2): recall within tie-density bins shared by both scales
        bins = tie_bins(sm["tie"], lg["tie"], sm["rec"], lg["rec"], kps)
        print(f"\nrecall@{kps[-1]} by tie-density bin (share of queries in brackets)")
        for b in bins:
            print(f"  tie {b['tie_from']:>5.0f}-{b['tie_to']:<5.0f}  "
                  f"{scales[0]} {b[f'small_recall@{kps[-1]}']} ({b['small_share']:.0%})   "
                  f"{scales[1]} {b[f'large_recall@{kps[-1]}']} ({b['large_share']:.0%})")
        adj, covered, cov_by_kp = reweight(bins, kps)
        if covered < 0.999:
            print(f"  note: {1 - covered:.1%} of the {scales[1]} queries fall in bins "
                  f"with no {scales[0]} queries; the re-weighting covers the rest")
        if len(set(cov_by_kp.values())) > 1:
            print(f"  note: coverage differs by depth {cov_by_kp}; the figure above "
                  f"is the worst of them")
        print(f"{scales[0]} recall re-weighted to {scales[1]}'s tie mix: {adj}")
        print(f"  measured {scales[0]} {res[scales[0]]['recall']}  "
              f"{scales[1]} {res[scales[1]]['recall']}")
        print(f"  H explains the gain to the extent the re-weighted numbers move "
              f"from {scales[0]} toward {scales[1]}.")
        # (3) per query
        d_tie = lg["tie"] - sm["tie"]
        corr = {kp: _spearman(d_tie, lg["rec"][kp] - sm["rec"][kp]) for kp in kps}
        print(f"per query: Spearman(change in ties, change in recall) {corr} "
              "(H predicts negative)")
        # composition: how much of the 8.8M top-k is even in the 500k subset
        small_ids = set(sm["ids"])
        present = np.array([np.mean([lg["ids"][i] in small_ids for i in t])
                            for t in lg["truth"]])
        print(f"share of the {scales[1]} top-{a.k} that exists in the {scales[0]} corpus: "
              f"{present.mean():.3f}")
        res["comparison"] = dict(bins=bins, small_reweighted_to_large_mix=adj,
                                 reweight_coverage=round(covered, 4),
                                 reweight_coverage_by_kprime=cov_by_kp,
                                 spearman_dtie_drecall=corr,
                                 large_topk_present_in_small=round(float(present.mean()), 4))

    from semview.results_io import environment, write_results
    write_results(a.out, dict(tool="margin_8m", k=a.k, kprimes=kps, m=a.m,
                              delta=a.delta, fingerprint={s: res[s].get("new_head")
                                                          for s in scales},
                              environment=environment(device=a.device),
                              scales=res))


if __name__ == "__main__":
    main()
