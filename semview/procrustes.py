"""The query-side adapter, measured the way its own paper measures it.

`baselines.py` has an `adapter` row and a `lazy+adapter` row, and both map the
**corpus** into the new space with a ridge least-squares fit. Drift-Adapter does
the opposite: it maps the **query** into the legacy space, q'_old = g(f_new(q)),
and searches the existing index of old vectors, so the corpus is never touched.
Its reported figure is an adaptation recall ratio against ground truth from an
exhaustive k-NN using the new model on both sides -- which is the same quantity
our cross-space recall measures, not an easier relevance metric.

So the paper's "adapters lose, and lose twice" currently rests on the weakest
adapter in the family applied in the other direction. This module measures the
comparison the way a reviewer who knows that work would run it:

  * **Orthogonal Procrustes**, closed form, no training. Given paired query
    embeddings it solves min over orthogonal W of ||Q_new W - Q_old||_F, which
    is W = U V^T from the SVD of Q_new^T Q_old. An orthogonal map preserves
    inner products exactly, which is why it is the right first thing to try and
    why it needs no regularisation.
  * **Low-rank affine**, also closed form: a ridge-regularised linear map with
    an intercept, truncated to `rank`. Strictly more expressive than Procrustes
    and strictly less than an MLP, so it brackets the family from below without
    a training loop.
  * The same **recall@k against the new model's own top-k** the rest of the
    paper reports, so the number drops straight into `tab:compare`.

Two honest asymmetries, stated because they decide how to read the result. The
adapter searches the old index at depth k and returns that as the answer, so its
error is permanent and present for every unit -- there is no exact rerank behind
it. And it costs one new-model pass per distinct query (to produce f_new(q)
before mapping it), which is the term our `lazy+adapter` row is also charged.

    python -m semview.procrustes --from-run results/runs/RUNDIR
    python -m semview.procrustes --q-old q_old.npy --q-new q_new.npy \\
        --v-old V_old.npy --v-new V_new.npy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .migration import _safe_normalize, topk_chunked


def fit_procrustes(Q_new: np.ndarray, Q_old: np.ndarray) -> np.ndarray:
    """The semi-orthogonal map taking new-space queries to the old space.

    The two spaces need not have the same width, and the common case is that
    they do not -- MiniLM is 384 and BGE-base 768. Nothing has to be padded or
    truncated first: M = Q_new^T Q_old is (d_new x d_old) whatever the widths,
    `svd(M, full_matrices=False)` gives U as (d_new x r) and Vt as (r x d_old)
    for r = min(d_new, d_old), and W = U Vt is the (d_new x d_old) solution of
    min ||Q_new W - Q_old||_F over the matrices with orthonormal rows or
    columns -- whichever side is narrower. Padding the new side with zeros
    would be harmless but pointless; **truncating** the wider side, which an
    earlier version of this module did, silently throws away half of MiniLM's
    or BGE's coordinates and reports the result as the adapter's accuracy.

    So the map is exact on the narrower space and loses only what no linear
    map between spaces of different widths can keep: when d_new > d_old it is
    a projection, and the rank deficit is the adapter's own, not an artefact
    of how it was fitted.
    """
    if Q_new.shape[0] != Q_old.shape[0]:
        raise SystemExit("fit_procrustes needs paired queries: "
                         f"{Q_new.shape[0]} new against {Q_old.shape[0]} old")
    M = Q_new.T @ Q_old
    U, _s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def fit_affine(Q_new: np.ndarray, Q_old: np.ndarray, ridge: float = 1e-3,
               rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A ridge linear map with an intercept, optionally truncated to `rank`.

    Rectangular like `fit_procrustes`: G is (d_new x d_new), so W comes out
    (d_new x d_old) and the intercept (d_old,) for any pair of widths. The
    `rank` truncation here is a deliberate capacity knob on the map, which is
    a different thing from truncating the *inputs*.
    """
    mu_n, mu_o = Q_new.mean(0), Q_old.mean(0)
    Xn, Xo = Q_new - mu_n, Q_old - mu_o
    G = Xn.T @ Xn + ridge * len(Xn) * np.eye(Xn.shape[1], dtype=Xn.dtype)
    W = np.linalg.solve(G, Xn.T @ Xo)
    if rank:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        r = min(int(rank), len(s))
        W = (U[:, :r] * s[:r]) @ Vt[:r]
    return W.astype(np.float32), (mu_o - mu_n @ W).astype(np.float32)


def evaluate(Q_old, Q_new, V_old, V_new, k: int = 10, ks=(10,),
             train_frac: float = 0.5, ridge: float = 1e-3,
             rank: int | None = 64, seed: int = 0) -> dict:
    """Fit on one half of the queries, score on the other, report both maps.

    The split matters: an adapter scored on the queries it was fitted on is
    measuring memorisation. Drift-Adapter fits on a calibration sample and
    reports on held-out queries, so this does the same.

    The two spaces may have different widths. Both maps are rectangular, so
    nothing is padded or truncated: the widths are checked against the two
    corpora, recorded in the result as `d_old` and `d_new`, and handed to the
    fits as they are.
    """
    rng = np.random.default_rng(seed)
    n = len(Q_old)
    if n < 8:
        raise SystemExit(f"need at least 8 queries to split; got {n}")
    if len(Q_new) != n:
        raise SystemExit("the two query matrices must be paired row by row: "
                         f"{n} old against {len(Q_new)} new")
    perm = rng.permutation(n)
    cut = max(2, int(n * train_frac))
    tr, ev = perm[:cut], perm[cut:]

    d_old, d_new = int(V_old.shape[1]), int(V_new.shape[1])
    # A query must live in the same space as the index it is about to search,
    # or the adapter would be measured against a mismatch rather than a drift.
    if Q_old.shape[1] != d_old:
        raise SystemExit(f"old queries are {Q_old.shape[1]}-d but the old "
                         f"corpus is {d_old}-d")
    if Q_new.shape[1] != d_new:
        raise SystemExit(f"new queries are {Q_new.shape[1]}-d but the new "
                         f"corpus is {d_new}-d")
    Qn_tr, Qn_ev, Qo_tr = Q_new[tr], Q_new[ev], Q_old[tr]

    truth = topk_chunked(Q_new[ev], V_new, k)
    recall = lambda served: float(np.mean(
        [len(set(t) & set(s)) / k for t, s in zip(truth, served)]))

    out = {"queries_fit": int(len(tr)), "queries_eval": int(len(ev)),
           "k": k, "N": int(len(V_old)), "d_old": d_old, "d_new": d_new,
           "schemes": []}

    # An adapter returns its top-k straight from the old index, so there is no
    # window and no depth to vary: these rows are one number each. Reporting
    # them per depth would imply a knob they do not have.
    served = [list(r[:k]) for r in topk_chunked(Q_old[ev], V_old, k)]
    out["schemes"].append(dict(
        scheme="old index, no adapter", kprime=None,
        recall_at_k=recall(served),
        note="the floor: what the stale index returns for a new-model query"))

    W = fit_procrustes(Qn_tr, Qo_tr)
    Wa, ba = fit_affine(Qn_tr, Qo_tr, ridge=ridge, rank=rank)
    for name, mapped in (("procrustes (orthogonal)",
                          _safe_normalize(Qn_ev @ W)),
                         (f"low-rank affine (r={rank}, ridge={ridge})",
                          _safe_normalize(Qn_ev @ Wa + ba))):
        served = [list(r[:k]) for r in topk_chunked(mapped, V_old, k)]
        out["schemes"].append(dict(
            scheme=name, kprime=None, recall_at_k=recall(served),
            note="query mapped into the old space, old index searched, answer "
                 "returned as-is -- no exact rerank behind it, so this error is "
                 "permanent for every unit"))

    # and the design, at the same depths, for the comparison that matters
    for kk in ks:
        win = topk_chunked(Q_old[ev], V_old, kk)
        served = []
        for q, w in zip(Q_new[ev], win):
            w = np.asarray(w, dtype=np.int64)
            served.append([int(w[j]) for j in np.argsort(-(V_new[w] @ q))[:k]])
        out["schemes"].append(dict(
            scheme="thaw: old window, exact rerank", kprime=int(kk),
            recall_at_k=recall(served),
            note="exact in the new space over the window; the units it reaches "
                 "are re-embedded"))
    return out


def report(res: dict) -> None:
    print(f"N={res['N']:,}  fit on {res['queries_fit']} queries, "
          f"scored on {res['queries_eval']}  k={res['k']}")
    d_old, d_new = res.get("d_old"), res.get("d_new")
    if d_old is not None:
        same = "same width" if d_old == d_new else "widths differ, map is rectangular"
        print(f"old space {d_old}-d, new space {d_new}-d ({same})")
    head_k = "K'"
    print(f"{'scheme':<44}{head_k:>7}{'recall@k':>10}")
    for r in res["schemes"]:
        depth = "-" if r["kprime"] is None else str(r["kprime"])
        print(f"{r['scheme']:<44}{depth:>7}{r['recall_at_k']:>10.4f}")


def main(argv=None) -> int:  # pragma: no cover - a thin CLI over evaluate()
    import argparse

    ap = argparse.ArgumentParser(
        prog="procrustes",
        description="the query-side adapter, measured against the old window")
    ap.add_argument("--from-run", default=None,
                    help="a run directory holding q_old.npy, q_new.npy "
                         "and the cached V_old / V_new")
    ap.add_argument("--q-old", default=None)
    ap.add_argument("--q-new", default=None)
    ap.add_argument("--v-old", default=None)
    ap.add_argument("--v-new", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--depths", default="10,200,500,2000")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    def pick(flag, *names):
        if flag:
            return Path(flag)
        if a.from_run:
            for n in names:
                hits = sorted(Path(a.from_run).rglob(n))
                if hits:
                    return hits[0]
        raise SystemExit(
            f"could not find {names[0]}: pass it explicitly, or point "
            f"--from-run at a directory that contains it")

    Q_old = np.load(pick(a.q_old, "q_old.npy"))
    Q_new = np.load(pick(a.q_new, "q_new.npy"))
    V_old = np.load(pick(a.v_old, "V_old.npy", "v_old.npy"))
    V_new = np.load(pick(a.v_new, "V_new.npy", "v_new.npy"))
    res = evaluate(Q_old, Q_new, V_old, V_new, k=a.k,
                   ks=tuple(int(x) for x in a.depths.split(",")),
                   train_frac=a.train_frac, ridge=a.ridge, rank=a.rank,
                   seed=a.seed)
    report(res)
    if a.out:
        from .results_io import environment, write_results
        res["environment"] = environment()
        write_results(a.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
