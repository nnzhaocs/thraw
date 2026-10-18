"""Why does cross-space recall RISE as the corpus grows? Two candidate answers, measured.

The `n_ladder` run records something the cost model does not predict. At a
fixed window of K'=500 and a fixed intent count D=5000, as MS MARCO grows from
500k to 8.84M rows:

    window as a fraction of the corpus   1/1,000  ->  1/17,700
    cross-space recall@500                0.9164  ->   0.9624

The window becomes a vanishing fraction of the corpus and quality *improves*.
The savings half of that observation is just `(1-K'/N)^D` restated -- D*K' is
fixed, so the reached set is capped and the saving fraction must rise. The
recall half is not predicted by anything, and it is what makes the savings free.

An unexplained monotone trend over four nested subsets of one corpus under one
model pair is exactly where a reviewer pushes, so this tool tests the two
explanations that matter, and they point in opposite directions.

**H1, the mechanism (hoped for).** At larger N the new model's top-k are
stronger absolute matches, so the two models agree on them more readily. The
disagreement lives in near-ties, whose density does not keep pace with N. If so,
the k-th score should pull away from the bulk as N grows, and the band of
documents that could displace the top-k should thin.

The band is the quantity from the research contract's section 10: documents
whose score sits within `band` of the k-th are the only ones a change of space
can promote into the top-k. A band that thins with N explains rising cross-space
recall directly, and ties the scaling result to the certificate.

**H2, the confound (feared).** MS MARCO contains duplicate passages -- the
step-2 report already counts boundary ties for this reason. If the top-k holds
more near-duplicates at larger N, then hitting *any* of them scores as a hit and
recall rises for a dull reason that generalises to no other corpus. The chance
correction in `n_ladder` does not cover this: duplicates are not chance, and
indeed the chance floor there is below 0.001 at every point while recall moves
by 5 points.

**Both can be true.** The tool reports both and does not decide; the numbers do.
If the band thins and duplicate density is flat, the scaling result has a
mechanism. If duplicate density rises with N, it is MS MARCO's composition and
should be reported as such before a spine is built on it.

Subsets come from `n_ladder.nested_indices` with the same seed, so they are
byte-identical to the ladder's -- comparing against differently drawn subsets
would answer a different question.

**`--queries` is a sample, and the recall column here is not the paper's.** The
ladder averages over all D = 5000 intents; this defaults to the first 1000, so
its recall carries sampling error and will not reproduce `n_ladder` to four
decimals. That is deliberate -- the trend being tested lives in the band and the
z-score, which are per-query quantities, and 1000 queries against 8.8M rows is
already the expensive half of the run. Quote recall from `n_ladder`.

`--q-new` is whatever `semview.migration --dump-queries DIR` wrote: that flag
creates DIR and saves `q_old.npy` and `q_new.npy` directly into it, with no
`arrays/` subdirectory -- an `arrays/` path appeared in this docstring and in
two runbook entries for a while and exists nowhere in the code. The real chain
legs used `dump/leg0` ... `dump/leg3`, so `dump/leg3/q_new.npy` is the newest
model's queries.

The two input paths below are **placeholders**. Both files are large (the 8.8M
corpus is about 13.5 GB as f16) and neither is in the evidence bundle, which
holds JSON only; take the real paths from the `n_ladder` run's `command.txt` on
the machine that built them. `RUNBOOK.md` section 3, B5, has the full invocation.

    python -m semview.runlog --name scaling_mechanism --tag d08 -- \\
      python analysis/scaling_mechanism.py \\
        --v-new cache/NEWMODELHASH.f16.npy \\
        --q-new results/runs/NLADDER_DUMPQ_RUN/dump/leg3/q_new.npy \\
        --sizes 500000,2000000,4000000,8841823 \\
        --k 10 --queries 1000 --out "{run}/scaling_mechanism.json"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview.results_io import environment, write_results  # noqa: E402

_HERE = Path(__file__).resolve().parent


def _nested_indices(n: int, sizes, seed: int = 0):
    """The ladder's own subsets, imported rather than reimplemented."""
    import importlib.util as _u
    spec = _u.spec_from_file_location("_n_ladder", _HERE / "n_ladder.py")
    mod = _u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.nested_indices(n, sizes, seed=seed)


def _blocks(idx: np.ndarray, rows: int):
    """Walk a sorted index array in contiguous chunks of `rows` entries."""
    for lo in range(0, len(idx), rows):
        yield idx[lo:lo + rows]


def measure_one(V, Q, idx: np.ndarray, k: int, band: float,
                dup_tau: float, rows: int = 200_000) -> dict:
    """Margin, band and duplicate statistics for one nested subset.

    Two passes over the subset, never materialising it. The subset of an 8.84M
    x 768 matrix does not fit in memory as a copy (13 GB at float16, more once
    cast), and `V[idx]` would make one, so scores are accumulated block by
    block against a memmap instead.

    Pass 1 keeps a running top-(k+1) per query, which gives the k-th and
    (k+1)-th scores. Pass 2 needs the k-th score to count the band, so it
    cannot be merged with the first.
    """
    nq = len(Q)
    Qf = np.ascontiguousarray(Q, dtype=np.float32)
    kk = k + 1

    # --- pass 1: running top-(k+1) --------------------------------------
    top_s = np.full((nq, kk), -np.inf, dtype=np.float32)
    top_i = np.zeros((nq, kk), dtype=np.int64)
    for blk in _blocks(idx, rows):
        B = np.ascontiguousarray(V[blk], dtype=np.float32)
        S = Qf @ B.T                                  # nq x len(blk)
        m = min(kk, S.shape[1])
        part = np.argpartition(-S, m - 1, axis=1)[:, :m]
        cand_s = np.take_along_axis(S, part, axis=1)
        cand_i = blk[part]
        all_cs = np.concatenate([top_s, cand_s], axis=1)
        all_i = np.concatenate([top_i, cand_i], axis=1)
        keep = np.argpartition(-all_cs, kk - 1, axis=1)[:, :kk]
        top_s = np.take_along_axis(all_cs, keep, axis=1)
        top_i = np.take_along_axis(all_i, keep, axis=1)
        del B, S
    order = np.argsort(-top_s, axis=1)
    top_s = np.take_along_axis(top_s, order, axis=1)
    top_i = np.take_along_axis(top_i, order, axis=1)
    s_k = top_s[:, k - 1]
    s_k1 = top_s[:, k]

    # --- pass 2: the band below the k-th score, and the score distribution
    band_n = np.zeros(nq, dtype=np.int64)
    tot = np.zeros(nq, dtype=np.float64)
    tot2 = np.zeros(nq, dtype=np.float64)
    lo_edge = (s_k - band).astype(np.float32)
    for blk in _blocks(idx, rows):
        B = np.ascontiguousarray(V[blk], dtype=np.float32)
        S = Qf @ B.T
        band_n += np.count_nonzero(
            (S >= lo_edge[:, None]) & (S <= s_k[:, None]), axis=1)
        # the k-th document itself always falls in [s_k - band, s_k]; it is not
        # a candidate to DISPLACE the top-k, so it is removed once, below
        tot += S.sum(axis=1, dtype=np.float64)
        tot2 += (S.astype(np.float64) ** 2).sum(axis=1)
        del B, S
    band_n = np.maximum(band_n - 1, 0)      # drop the k-th document itself
    N = len(idx)
    mean = tot / N
    var = np.maximum(tot2 / N - mean ** 2, 0.0)
    sd = np.sqrt(var)

    # --- duplicates inside the top-k ------------------------------------
    # only k rows per query, so these can be gathered
    dup_pairs = np.zeros(nq, dtype=np.int64)
    for i in range(nq):
        rows_i = np.ascontiguousarray(V[np.sort(top_i[i, :k])], dtype=np.float32)
        G = rows_i @ rows_i.T
        iu = np.triu_indices(len(rows_i), k=1)
        dup_pairs[i] = int(np.count_nonzero(G[iu] >= dup_tau))

    return {
        "N": int(N), "queries": int(nq), "k": int(k),
        "band": float(band), "dup_tau": float(dup_tau),
        # H1: does the k-th document pull away from the bulk?
        "s_k_mean": float(s_k.mean()),
        "s_k_minus_s_k1_mean": float((s_k - s_k1).mean()),
        "s_k_minus_s_k1_median": float(np.median(s_k - s_k1)),
        "z_of_s_k_mean": float(np.mean((s_k - mean) / np.maximum(sd, 1e-9))),
        "corpus_score_mean": float(mean.mean()),
        "corpus_score_sd": float(sd.mean()),
        # H1: does the band that could displace the top-k thin out?
        "band_count_mean": float(band_n.mean()),
        "band_frac_mean": float((band_n / N).mean()),
        # H2: is the top-k getting more duplicated?
        "dup_pairs_mean": float(dup_pairs.mean()),
        "queries_with_a_dup_frac": float(np.mean(dup_pairs > 0)),
    }


def _trend(vals: list[float]) -> tuple[bool, bool]:
    """(rises first-to-last, and is it monotone in that direction all the way).

    A flat series rises nowhere, so `rises` is False and `mono` describes
    flatness -- every step equal -- rather than being borrowed from the falling
    branch, where `b <= a` would call a flat series monotonically decreasing.
    The caller checks `rises` before it looks at `mono`, so this costs nothing
    today; it is written out because the two readings differ and the next reader
    should not have to work out which one was meant.
    """
    rises, falls = vals[-1] > vals[0], vals[-1] < vals[0]
    pairs = list(zip(vals, vals[1:]))
    if rises:
        mono = all(b >= a for a, b in pairs)
    elif falls:
        mono = all(b <= a for a, b in pairs)
    else:
        mono = all(a == b for a, b in pairs)
    return rises, mono


def verdict(rows: list[dict], dup_threshold: float = 1.1) -> dict:
    """State which hypothesis the numbers support, in every direction.

    **`band_frac_mean` is deliberately NOT a criterion.** It is
    `band_count / N`, so when N grows eighteenfold it falls eighteenfold even
    if the band holds exactly the same documents -- and an earlier version of
    this function used `band_frac_thins or band_count_thins`, which declared
    H1 on that arithmetic alone. Three independent reviews caught it, and it is
    the worst place in the project for such a bug: this tool exists precisely
    to be able to come out against the hypothesis the paper wants. The
    criterion is the absolute count; the fraction is reported as information
    only, under the name `band_frac_fell_by_division_not_evidence` -- a review
    pointed out that any boolean whose name contains `thins` will be read as a
    criterion by the next person or script, and `_INFORMATIONAL` is a weaker
    warning than putting the reason in the key itself.

    The decision is first-to-last, and monotonicity across every size is
    reported beside it, because a trend that reverses in the middle is not the
    clean scaling law the paper would be claiming.
    """
    if len(rows) < 2:
        # Same keys as every other branch, with None for what cannot be known
        # from one point. `main` merges this dict into the results file with
        # `**v`, so a short branch writes a JSON object with seven fields
        # missing, and anything reading the file -- a test, a later claim, the
        # audit -- gets a KeyError on a run that merely had one size.
        return {"verdict": "need at least two sizes to see a trend",
                "band_count_thins": None, "kth_z_rises": None,
                "dup_density_rises": None,
                "dup_threshold": float(dup_threshold),
                "band_count_monotone": None, "kth_z_monotone": None,
                "dup_monotone": None,
                "band_frac_fell_by_division_not_evidence": None}
    first, last = rows[0], rows[-1]

    band_thins, band_mono = _trend([-r["band_count_mean"] for r in rows])
    z_rises, z_mono = _trend([r["z_of_s_k_mean"] for r in rows])
    dup_rises = last["dup_pairs_mean"] > first["dup_pairs_mean"] * dup_threshold
    _, dup_mono = _trend([r["dup_pairs_mean"] for r in rows])
    frac_thins = last["band_frac_mean"] < first["band_frac_mean"]

    if dup_rises:
        v = ("H2: near-duplicate density in the top-k rises with N, so part of "
             "the recall gain is MS MARCO's composition. Report it as such "
             "before building on the scaling result. H2 wins even when H1's "
             "signals also look good: a real confound is not cancelled by a "
             "real mechanism.")
    elif z_rises and band_thins:
        v = ("H1: the k-th document pulls away from the bulk and the ABSOLUTE "
             "number of documents that could displace the top-k falls, with "
             "duplicate density flat. The scaling result has a mechanism.")
    else:
        v = ("neither: the k-th score does not pull away, or the absolute band "
             "does not thin, while duplicates stay flat -- so the recall rise "
             "is real and unexplained by either hypothesis. Report it as an "
             "observation and do not lead with it.")
    if (z_rises or band_thins) and not (z_mono and band_mono):
        v += (" NOTE: the trend is not monotone across all sizes, so this is a "
              "first-to-last comparison and not a scaling law.")
    return {
        "verdict": v,
        # criteria
        "band_count_thins": bool(band_thins),
        "kth_z_rises": bool(z_rises),
        "dup_density_rises": bool(dup_rises),
        "dup_threshold": float(dup_threshold),
        # monotonicity, reported not decided on
        "band_count_monotone": bool(band_mono),
        "kth_z_monotone": bool(z_mono),
        "dup_monotone": bool(dup_mono),
        # information only: falls mechanically with N, never a criterion
        "band_frac_fell_by_division_not_evidence": bool(frac_thins),
    }


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v-new", required=True,
                    help=".npy of the NEW model's corpus vectors, N x d. Loaded "
                         "as a memmap; never copied whole.")
    ap.add_argument("--q-new", required=True, help=".npy, D x d, the same "
                                                   "queries under the new model")
    ap.add_argument("--sizes", required=True,
                    help="comma-separated and increasing, the ladder's own "
                         "sizes so the subsets match it exactly")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=1000,
                    help="how many queries to measure; the first of them, so a "
                         "rerun with the same dump is identical")
    ap.add_argument("--band", type=float, default=0.01,
                    help="the score window below the k-th that a change of "
                         "space could promote from. 0.01 on normalised cosines "
                         "is about a hundredth of the usable range; sweep it if "
                         "the verdict sits on the boundary")
    ap.add_argument("--dup-tau", type=float, default=0.99,
                    help="cosine above which two top-k documents count as "
                         "near-duplicates of each other")
    ap.add_argument("--rows", type=int, default=200_000,
                    help="corpus rows per block; lower it if memory is tight")
    ap.add_argument("--dup-threshold", type=float, default=1.1,
                    help="how much duplicate density must rise, as a ratio, to "
                         "call H2. Recorded in the results file; 1.1 is a "
                         "judgement, so sweep it if the verdict sits on it")
    ap.add_argument("--seed", type=int, default=0,
                    help="must match the ladder's seed or the subsets differ")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    V = np.load(a.v_new, mmap_mode="r")
    Q = np.load(a.q_new)
    if Q.shape[1] != V.shape[1]:
        raise SystemExit(f"query width {Q.shape[1]} != corpus width {V.shape[1]}")
    if a.queries < 1 or a.queries > len(Q):
        raise SystemExit(f"--queries {a.queries} outside 1..{len(Q)}")
    Q = Q[:a.queries]
    sizes = [int(s) for s in a.sizes.split(",")]
    if min(sizes) <= a.k:
        raise SystemExit(
            f"the smallest subset ({min(sizes)}) must exceed k={a.k}: a subset "
            "with fewer than k+1 rows leaves the top-k register padded with "
            "index 0, and the duplicate count would then compare row 0 "
            "against itself")
    gb = (len(Q) * a.rows * 4 + a.rows * V.shape[1] * 4) / 1e9
    if gb > 2:
        print(f"  WARNING: {len(Q)} queries x {a.rows:,} rows per block needs "
              f"about {gb:.1f} GB at peak. Lower --rows (the corpus is read as "
              f"a memmap, so blocks are the only large allocation).")
    idxs = _nested_indices(len(V), sizes, seed=a.seed)

    rows = []
    for L, idx in zip(sizes, idxs):
        print(f"  N={L:,} ...", flush=True)
        r = measure_one(V, Q, idx, a.k, a.band, a.dup_tau, rows=a.rows)
        rows.append(r)
        print(f"    k-th score {r['s_k_mean']:.4f}  z {r['z_of_s_k_mean']:.2f}"
              f"   band {r['band_count_mean']:.1f} docs"
              f" ({r['band_frac_mean']:.2e} of N)"
              f"   dup pairs/query {r['dup_pairs_mean']:.2f}", flush=True)

    v = verdict(rows, dup_threshold=a.dup_threshold)
    print("\n  H1 wants: the k-th score's z rises AND the band thins.")
    print("  H2 warns:  duplicate pairs per query rise with N.")
    print(f"\n  {v['verdict']}")

    obj = {"tool": "scaling_mechanism",
           "inputs": {"v_new": a.v_new, "q_new": a.q_new, "sizes": sizes,
                      "seed": a.seed, "queries": int(len(Q)),
                      "band": a.band, "dup_tau": a.dup_tau,
                      "dup_threshold": a.dup_threshold},
           "environment": environment(dtype=str(V.dtype)),
           "rows": rows, **v}
    write_results(a.out, obj)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
