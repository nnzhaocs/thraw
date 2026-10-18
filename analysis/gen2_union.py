"""Does an unfinished upgrade help the next one? Second-generation candidate union.

The setting. A model is upgraded twice without either migration finishing:

    leg 1:  f_old  ->  f_mid     candidates from the OLD index
    leg 2:  f_mid  ->  f_new     candidates from ... ?

Read literally, Thaw answers "from the old index again": the intermediate index
is complete for nobody, so only the original index is a candidate source, the
ceiling is the already-measured old->new cross-space recall, and leg 1's repairs
are simply wasted. That is not a new measurement, it is the paper's existing
rho(D(t)) argument.

But the design's invariant permits something better, and this tool measures it.
Because a score from an old space only ever *nominates* -- all ranking happens in
the new space -- a partially migrated index from a previous generation can be a
**second candidate source** at no cost in forward passes: its vectors already
exist, and leg 2 needs one extra query encode per distinct query string. So

    W(q) = topK'(q; V_old)  u  { topK'_mid(q; V_mid) restricted to R1 }

where R1 is the unit set leg 1 reached. Ranking over W(q) is still exact in the
new space, so the argument of Proposition 1 is untouched, and the ceiling can
only rise. Whether it rises *materially* is an empirical question, and it is the question
this tool answers. The dominant factor is **how much of the corpus leg 1
reached**: the mid index can only ever nominate inside R1, so if the previous
migration saw little traffic it cannot help no matter how good its geometry is.
Set the leg-1 traffic to what that migration actually served, not to half of the
evaluation set; `--warm-frac` is the knob.

What it reports, at each K': recall of f_new's top-k under
    old-only      the design as written (the single-generation baseline)
    mid-only      the intermediate index alone, restricted to R1
    union         old K' plus mid K' inside R1 (the proposal; no extra passes)
    union-open    the same union with the mid list unrestricted (an upper bound
                  showing what restricting to R1 costs)
and the cost accounting: passes for leg 1, passes leg 2 adds, and the reached
fraction after both legs.

Inputs are `.npy` matrices, because corpus caches are keyed by a hash and query
vectors are never cached. Produce them with `--dump-queries` on `migration`
(see RUNTESTS.md section 8d): q_old and q_new from the f_old -> f_new run, and
q_mid from a second run whose `--old` is f_mid.

    python -m semview.runlog --name gen2_union_msmarco --tag final -- \
      python analysis/gen2_union.py \
        --v-old cache/OLD.npy --v-mid cache/MID.npy --v-new cache/NEW.f16.npy \
        --q-old dump/A/q_old.npy --q-mid dump/B/q_old.npy --q-new dump/A/q_new.npy \
        --k 10 --kprimes 200,500,1000,2000 --warm-frac 0.5 \
        --out "{run}/gen2_union.json"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview import baselines, migration  # noqa: E402
from semview.results_io import environment, write_results  # noqa: E402


def _topk_rows(Q, V, kprime: int) -> np.ndarray:
    """Exact top-K' of every query, as an index matrix."""
    return migration.topk_chunked(Q, V, int(min(kprime, len(V))))


def gen2_schemes(V_old, V_mid, V_new, Q_old_warm, Q_old_eval, Q_mid_eval,
                 Q_new_eval, k: int = 10, kprimes=(200, 500, 1000, 2000),
                 budget: int | None = None):
    """One row per (candidate rule, K'), all scored against f_new's own top-k.

    Two accounting facts decide the shape of this table.

    First, **a unit leg 1 repaired holds a MID vector, which is useless for
    ranking in the new space.** So at the start of leg 2 no unit has a usable new
    vector, every unit in leg 2's window has to be encoded with f_new, and this
    migration's cost is |W| -- the size of the window it builds. Leg 1's passes
    are through f_mid and are reported separately, because they were that
    migration's cost, already paid.

    Second, therefore, the four rules are NOT equal-cost: a union window is
    larger and costs more f_new passes. The control is in the same table -- the
    `old-only` row at a deeper K'. The mid index is worth using only if
    `union` at K' beats `old-only` at whatever K'' reaches the same |W|, i.e.
    only if a previous generation is a better candidate source per pass than the
    original index is.

    The warm-up/evaluation split is `baselines.compare_schemes`'s and for the
    same reason: leg 1 must be driven by traffic the quality measurement then
    does not see.
    """
    N = len(V_new)
    truth = migration.topk_chunked(Q_new_eval, V_new, min(k, N))

    # leg 1: candidates from the ORIGINAL index, repaired in the mid space. Its
    # reached set is what leg 2 inherits as the mid index's coverage.
    cap = N if budget is None else budget
    R1 = baselines._touched(Q_old_warm, V_old, max(kprimes), cap)
    passes_leg1 = int(R1.sum())

    rows = []
    for kp in kprimes:
        win_old = _topk_rows(Q_old_eval, V_old, kp)
        win_mid_all = _topk_rows(Q_mid_eval, V_mid, kp)
        win_mid = [[u for u in row.tolist() if R1[u]] for row in win_mid_all]

        def add(scheme, windows, note):
            windows = [np.asarray(w, dtype=np.int64) for w in windows]
            touched = np.zeros(N, dtype=bool)
            for w in windows:
                touched[w] = True
            rows.append(dict(
                scheme=scheme, kprime=int(kp), passes=int(touched.sum()),
                passes_leg1=passes_leg1, passes_train=0,
                passes_serve=int(touched.sum()),
                # candidate recall is the ceiling; served recall is what the user
                # gets after the window is repaired and reranked exactly
                recall_candidates=baselines._recall(truth, windows, k),
                recall_at_k=baselines._recall(
                    truth, baselines._rerank(Q_new_eval, V_new, windows, k), k),
                note=note))

        add("old-only", win_old,
            "candidates from the original index: the design read literally")
        add("mid-only", win_mid,
            "the intermediate index alone, restricted to what leg 1 reached")
        add("union", [np.array(sorted(set(a.tolist()) | set(b)), dtype=np.int64)
                      for a, b in zip(win_old, win_mid)],
            "old K' plus mid K' inside R1; the mid list costs no forward pass, "
            "only the extra units leg 2 must then encode")
        add("union-open", [np.array(sorted(set(a.tolist()) | set(b.tolist())),
                                    dtype=np.int64)
                           for a, b in zip(win_old, win_mid_all)],
            "the same union with the mid list unrestricted: what restricting it "
            "to R1 costs")

    # what each source contributes that the other does not, at the middle depth
    kp_mid = kprimes[len(kprimes) // 2]
    o = [set(r.tolist()) for r in _topk_rows(Q_old_eval, V_old, kp_mid)]
    m = [set(u for u in r.tolist() if R1[u])
         for r in _topk_rows(Q_mid_eval, V_mid, kp_mid)]
    ex = np.array([[len(set(t.tolist()) & (mi - oi)), len(set(t.tolist()) & (oi - mi)),
                    len(set(t.tolist()) & oi & mi), len(set(t.tolist()) - oi - mi)]
                   for t, oi, mi in zip(truth, o, m)]).mean(axis=0)
    rows.append(dict(scheme="diagnostics", kprime=int(kp_mid), passes=0,
                     passes_train=0, passes_serve=0,
                     recall_candidates=None, recall_at_k=None,
                     note=f"at K'={kp_mid}, per query: truth only in the mid list "
                          f"{ex[0]:.2f}, only in old {ex[1]:.2f}, in both {ex[2]:.2f}, "
                          f"in neither {ex[3]:.2f}"))
    rows.append(dict(scheme="coverage", kprime=int(kp_mid), passes=passes_leg1,
                     passes_train=0, passes_serve=passes_leg1,
                     recall_candidates=None, recall_at_k=None,
                     note=f"leg 1 reached {R1.sum():,} of {N:,} "
                          f"({100 * R1.sum() / N:.1f}%), which is all the mid index "
                          f"can ever nominate"))
    return rows, dict(R1=int(R1.sum()), N=int(N), passes_leg1=passes_leg1,
                      kprimes=[int(x) for x in kprimes])


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v-old", required=True, help="original index, N x d")
    ap.add_argument("--v-mid", required=True, help="intermediate index, N x d")
    ap.add_argument("--v-new", required=True, help="newest model, N x d (same rows)")
    ap.add_argument("--q-old", required=True, help="queries in the original space")
    ap.add_argument("--q-mid", required=True, help="the same queries in the mid space")
    ap.add_argument("--q-new", required=True, help="the same queries in the new space")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kprimes", default="200,500,1000,2000")
    ap.add_argument("--warm-frac", type=float, default=0.5,
                    help="fraction of queries that drive leg 1 (the rest measure)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    V_old, V_mid, V_new = np.load(a.v_old), np.load(a.v_mid), np.load(a.v_new)
    Q_old, Q_mid, Q_new = np.load(a.q_old), np.load(a.q_mid), np.load(a.q_new)
    if not (len(V_old) == len(V_mid) == len(V_new)):
        raise SystemExit("the three corpus matrices must share a row order and length")
    cut = int(round(a.warm_frac * len(Q_old)))
    if cut < 1 or cut >= len(Q_old):
        raise SystemExit(f"--warm-frac {a.warm_frac} leaves no evaluation queries")
    kprimes = tuple(int(x) for x in a.kprimes.split(","))
    rows, stats = gen2_schemes(V_old, V_mid, V_new, Q_old[:cut], Q_old[cut:],
                               Q_mid[cut:], Q_new[cut:], k=a.k, kprimes=kprimes)

    print(f"\n  N={stats['N']:,}  k={a.k}  leg-1 reached {stats['R1']:,} "
          f"({100 * stats['R1'] / stats['N']:.1f}%)  "
          f"warm-up {cut} / evaluation {len(Q_old) - cut} queries")
    print("\n  rule           K'     passes   % of N   cand.recall   served")
    print("  " + "-" * 62)
    for r in rows:
        if r["recall_at_k"] is None:
            continue
        print("  %-13s %-6d %7d %8.1f%% %12.4f %9.4f"
              % (r["scheme"], r["kprime"], r["passes"],
                 100 * r["passes"] / stats["N"], r["recall_candidates"], r["recall_at_k"]))
    print("\n  The control is in the table: compare `union` at K' with `old-only` "
          "at the\n  K'' whose passes match. The mid index is worth using only if "
          "the union wins.")
    for r in rows:
        if r["recall_at_k"] is None:
            print(f"\n  {r['scheme']}: {r['note']}")

    obj = {"tool": "gen2_union", "k": a.k, "kprimes": list(kprimes),
           "warm_frac": a.warm_frac,
           "inputs": {"v_old": a.v_old, "v_mid": a.v_mid, "v_new": a.v_new,
                      "q_old": a.q_old, "q_mid": a.q_mid, "q_new": a.q_new},
           "environment": environment(dtype=str(V_new.dtype)),
           "fingerprint": {"old_head": migration._fingerprint(V_old),
                           "mid_head": migration._fingerprint(V_mid),
                           "new_head": migration._fingerprint(V_new),
                           "old_strided": migration._fingerprint_strided(V_old),
                           "mid_strided": migration._fingerprint_strided(V_mid),
                           "new_strided": migration._fingerprint_strided(V_new),
                           "new_storage_dtype": str(V_new.dtype)},
           "coverage": stats, "schemes": rows}
    write_results(a.out, obj)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
