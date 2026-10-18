"""Which retrieval operators can a mixed index actually serve? Measured.

C1 asserts that an operator is safe over a heterogeneous index exactly when it
compares no scores across spaces, and the paper has been listing the
consequences as a table of yes/no. That table is a definition, so by the
project's own rule -- an experiment that could not have failed carries no
information -- it was evidence for nothing.

This makes it an experiment, and the experiment can come out either way.

**The construction.** A mixed index is precisely an operator that compares
numbers originating in two spaces. At the score level it is dimension-free and
can be built exactly, with no ANN library and no equal-dimension requirement:

    S_mixed[q,u] = S_old[q,u]   if u still holds an old vector
                   S_new[q,u]   if u has been repaired

Searching that matrix is what an ANN index built over mixed vectors computes,
up to the ANN approximation; thresholding it is what a score cutoff computes.
Both are compared against the new model's own top-k, at a sweep of repaired
fractions rho.

**Why it could fail.** Nothing forces the two spaces' score scales to disagree.
If they broadly agree, a mixed index works and C1's prohibition is empty. The
measured outcome is that the prohibition is *conditional*: it holds when the
score perturbation B (see `margin_cert.py`) is large and is false when B is
small. So the same scalar that derives K' also decides which operators the
index can serve, and the blanket yes/no table in the paper is wrong as written.

The control is in the same output: window-then-rerank returns the new model's
own ordering for any rho, because ranking happens entirely in the new space.
That is not measured because it is true by construction -- which is the honest
status of that row, and why it is labelled rather than scored.

    python analysis/operator_safety.py \
        --old cache/MINILM.npy --new cache/BGE.npy \
        --queries 500 --k 10 --out results/operator_safety_minilm_bge.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview.results_io import environment, write_results  # noqa: E402


def _unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def _recall(top: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / k
                          for a, b in zip(top, truth)]))


def measure(S_old: np.ndarray, S_new: np.ndarray, k: int = 10,
            rhos=(0.1, 0.25, 0.5, 0.75, 0.9), pct: float = 99.8,
            seed: int = 7) -> dict:
    """One row per (operator, rho). `pct` sets the threshold operator's cutoff
    as a percentile of the OLD score distribution, which is how a deployed
    threshold is calibrated: on the scores the operator could see when it was
    tuned."""
    nq, N = S_old.shape
    k = int(min(k, N))
    rng = np.random.default_rng(seed)
    B = float(2 * np.abs(S_new - S_old).max())
    truth = np.argpartition(-S_new, k - 1, axis=1)[:, :k]
    t = float(np.percentile(S_old, pct))
    sel_true = float((S_new >= t).sum(axis=1).mean())

    # One draw, thresholded at each rho, so the repaired sets are NESTED: the
    # sweep is one migration observed at several points rather than five
    # unrelated stores. Drawing inside the loop made rho=0.25's repaired set
    # not a superset of rho=0.1's, which no deployment does -- repair only ever
    # adds. (Measured separately: a traffic-driven repaired set, the union of
    # served queries' windows, tracks a uniform one within about 3 points of
    # recall here, so uniform is a fair stand-in for WHICH units are repaired;
    # it was never a fair stand-in for the sweep being a trajectory.)
    u = rng.random(N)
    rows = []
    for rho in rhos:
        repaired = u < rho
        S_mix = np.where(repaired[None, :], S_new, S_old)
        top = np.argpartition(-S_mix, k - 1, axis=1)[:, :k]
        sel_mix = float((S_mix >= t).sum(axis=1).mean())
        rows.append({
            "rho": float(rho),
            "repaired": int(repaired.sum()),
            "mixed_index_recall_at_k": _recall(top, truth, k),
            "threshold_selected_mixed": sel_mix,
            "threshold_selected_true": sel_true,
            "threshold_rel_error": abs(sel_mix - sel_true) / max(sel_true, 1e-9),
        })
    return {
        "queries": int(nq), "corpus": int(N), "k": k,
        "score_perturbation_band": B,
        "threshold_percentile": pct, "threshold_value": t,
        "rows": rows,
        "control": ("window-then-rerank returns the new model's own top-k at "
                    "any rho, by construction: every comparison happens in the "
                    "new space. Not scored, because a measurement of an "
                    "identity is not evidence."),
    }


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True,
                    help="may have a different dimension than --old")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    O = np.load(a.old).astype(np.float64)
    Nw = np.load(a.new).astype(np.float64)
    if len(O) != len(Nw):
        raise SystemExit("the two corpus matrices must share a row order and length")
    if a.queries < 1 or a.queries >= len(O):
        raise SystemExit(f"--queries {a.queries} leaves no corpus")
    O, Nw = _unit(O), _unit(Nw)
    nq = a.queries
    # The same pre-flight `margin_cert` carries, for the same reason: this tool
    # holds two (queries x corpus) score matrices and `measure` builds two more
    # temporaries per rho (`np.where` and the argpartition), so peak memory is
    # a few multiples of one matrix. It was added to margin_cert and missed
    # here, which is how one of a pair of twin tools keeps a fixed bug.
    gb = 6.0 * nq * (len(O) - nq) * 8 / 1e9
    if gb > 4:
        print(f"  WARNING: {nq} queries over a corpus of {len(O) - nq:,} needs "
              f"about {gb:.0f} GB of RAM (several multiples of one score "
              f"matrix, one per rho). Lower --queries, or sample the corpus: "
              f"what is being measured is how a mixed index behaves as a "
              f"function of the band, which a few thousand rows show as well "
              f"as a million.")
    rep = measure(O[:nq] @ O[nq:].T, Nw[:nq] @ Nw[nq:].T, k=a.k)
    rep["label"] = a.label or f"{Path(a.old).name} -> {Path(a.new).name}"
    rep["dims"] = [int(O.shape[1]), int(Nw.shape[1])]

    print(f"\n  {rep['label']}   dims {rep['dims'][0]} -> {rep['dims'][1]}"
          f"   N={rep['corpus']:,}  B={rep['score_perturbation_band']:.6f}")
    print("    rho    mixed-index recall    threshold selected (vs true)")
    for r in rep["rows"]:
        print(f"    {r['rho']:4.2f}   {r['mixed_index_recall_at_k']:.4f}"
              f"                {r['threshold_selected_mixed']:7.1f} vs "
              f"{r['threshold_selected_true']:5.1f}"
              f"  ({100 * r['threshold_rel_error']:.1f}%)")
    worst = min(r["mixed_index_recall_at_k"] for r in rep["rows"])
    if worst > 0.95:
        print("    -> the mixed index is SAFE for this pair: C1's prohibition "
              "does not hold\n       unconditionally, it holds for large B.")
    else:
        print("    -> the mixed index is destroyed for this pair, and the "
              "threshold operator\n       over-selects. C1's prohibition is "
              "measured here, not assumed.")
    print(f"    (control) {rep['control']}")

    obj = {"tool": "operator_safety", "inputs": {"old": a.old, "new": a.new},
           "environment": environment(dtype=str(np.load(a.new).dtype)),
           "report": rep}
    write_results(a.out, obj)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
