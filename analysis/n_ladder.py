"""Recall and savings on NESTED corpus subsets, to answer what recall does as N grows.

Why this tool exists. `semview.migration --limit L` draws an **independent**
random subset for each L (`_sample` uses `rng.choice(len(units), size=L)`), so
running the ladder that way would compare four different corpora and confound
corpus size with which units happened to be drawn. Recall rises with N on the
two scales we have (0.768 -> 0.824 at K' = 50, MS MARCO 500k -> 8.8M), and the
tie-density explanation of that rise was tested and **fails** (`margin_8m`: the
re-weighting moves the 500k numbers away from the full ones). Every quality
conclusion in the paper rests on `recall@K'`, so the shape of this curve is the
one thing a reviewer can reasonably demand before believing the metric.

This tool fixes ONE permutation of the units and takes prefixes of it, so the
subsets are genuinely nested: subset(L1) is a subset of subset(L2) for
L1 < L2, with the same queries at every scale. It reuses `migration.frontier`
unchanged, so recall, reached, savings and the closed-form predictions are
computed exactly as they are everywhere else.

Inputs are `.npy` embedding matrices, because the caches are keyed by a hash of
the corpus and query set rather than by path. Record the file names in the
run's `RUNS.md` line: they are what makes a ladder reproducible.

    python -m semview.runlog --name n_ladder_msmarco --tag final -- \\
      python analysis/n_ladder.py \\
        --v-old cache/OLDKEY.npy --v-new cache/NEWKEY.f16.npy \\
        --q-old cache/QOLDKEY.npy --q-new cache/QNEWKEY.npy \\
        --sizes 500000,2000000,4000000,8841823 \\
        --kprimes 10,50,200,500 --k 10 --d 5000 --seed 0 \\
        --out "{run}/n_ladder.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview import migration  # noqa: E402
from semview.results_io import environment, write_results  # noqa: E402


def nested_indices(n: int, sizes, seed: int = 0) -> list[np.ndarray]:
    """Sorted unit indices of each nested subset, smallest first.

    One permutation, then its prefixes. `np.sort` keeps each subset in corpus
    order, so a row's position does not encode which scale it came from and two
    runs with the same seed produce byte-identical subsets.
    """
    sizes = [int(s) for s in sizes]
    if sizes != sorted(sizes):
        raise ValueError(f"--sizes must be increasing, got {sizes}")
    if sizes and sizes[-1] > n:
        raise ValueError(f"--sizes {sizes[-1]} exceeds the corpus N={n}")
    if sizes and sizes[0] < 1:
        raise ValueError(f"--sizes must be positive, got {sizes}")
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[:s]) for s in sizes]


def ladder(V_old, V_new, Q_old, Q_new, sizes, k: int = 10,
           kprimes=(10, 50, 200, 500), d: int | None = None,
           seed: int = 0, repeats: int = 5) -> list[dict]:
    """`frontier` on each nested subset, nailing every row to its own N.

    The queries are the same at every scale, and so is D (`--d`), so the only
    thing that changes between rows is how many units the corpus has.
    """
    D = len(Q_old) if d is None else min(int(d), len(Q_old))
    frac = (D / len(Q_old),)
    rows: list[dict] = []
    for L, idx in zip(sizes, nested_indices(len(V_old), sizes, seed)):
        part = migration.frontier(V_old[idx], V_new[idx], Q_old, Q_new, k=k,
                                  kprimes=kprimes, d_fracs=frac, seed=seed,
                                  repeats=repeats)
        for r in part:
            r["N"] = int(L)
            r["subset"] = int(L)
            r["units_removed"] = int(len(V_old) - L)
        rows.extend(part)
    return rows


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v-old", required=True, help=".npy, N x d_old")
    ap.add_argument("--v-new", required=True, help=".npy, N x d_new (same row order)")
    ap.add_argument("--q-old", required=True, help=".npy, D x d_old")
    ap.add_argument("--q-new", required=True, help=".npy, D x d_new")
    ap.add_argument("--sizes", required=True, help="comma-separated, increasing")
    ap.add_argument("--kprimes", default="10,50,200,500")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--d", type=int, default=None,
                    help="distinct intents (queries) used at every scale; default all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    V_old = np.load(a.v_old)
    V_new = np.load(a.v_new)
    Q_old = np.load(a.q_old)
    Q_new = np.load(a.q_new)
    if V_old.shape[0] != V_new.shape[0]:
        raise SystemExit(f"row counts differ: {V_old.shape[0]} vs {V_new.shape[0]}")
    sizes = [int(s) for s in a.sizes.split(",")]
    kprimes = tuple(int(s) for s in a.kprimes.split(","))
    rows = ladder(V_old, V_new, Q_old, Q_new, sizes, k=a.k,
                  kprimes=kprimes, d=a.d, seed=a.seed, repeats=a.repeats)

    print("\n  N        K'    recall@K'   reached   savings%   chance-adj")
    print("  " + "-" * 62)
    for r in rows:
        print("  %-9d %-5d %-11.4f %-9.4f %-10.2f %.4f"
              % (r["N"], r["kprime"], r["recall_at_kprime"], r["reached"],
                 r["savings_pct"], r.get("recall_chance_corrected", float("nan"))))
    print("\n  recall@K' across N, one line per K':")
    for kp in sorted({r["kprime"] for r in rows}):
        pts = [(r["N"], r["recall_at_kprime"]) for r in rows if r["kprime"] == kp]
        pts.sort()
        print("    K'=%-5d " % kp + "  ".join("%d: %.4f" % p for p in pts))

    obj = {"tool": "n_ladder", "sizes": sizes, "kprimes": list(kprimes),
           "k": a.k, "d": a.d, "seed": a.seed, "repeats": a.repeats,
           "nesting": "one permutation, prefixes (see nested_indices)",
           "inputs": {"v_old": a.v_old, "v_new": a.v_new,
                      "q_old": a.q_old, "q_new": a.q_new},
           "N_full": int(V_old.shape[0]), "D_available": int(len(Q_old)),
           # provenance: `write_results` refuses a file without one, and the
           # cheap strided fingerprints are the same ones `migration` records,
           # so a ladder can be tied to the caches it ran on without hashing
           # 13 GB per matrix.
           "environment": environment(dtype=str(V_new.dtype)),
           "fingerprint": {
               "old_head": migration._fingerprint(V_old),
               "new_head": migration._fingerprint(V_new),
               "old_strided": migration._fingerprint_strided(V_old),
               "new_strided": migration._fingerprint_strided(V_new),
               "old_dim": int(V_old.shape[1]), "new_dim": int(V_new.shape[1]),
               "new_storage_dtype": str(V_new.dtype),
               "query_old_head": migration._fingerprint(Q_old),
               "query_new_head": migration._fingerprint(Q_new),
               "D_hashed": int(len(Q_old)),
           },
           "frontier": rows}
    write_results(a.out, obj)
    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":  # pragma: no cover
    main()
