"""How much of the chain result is the generator's assumption?

    python -m semview.runlog --name drift_sweep --tag final -- \
      python analysis/drift_sweep.py --n 50000 --legs 4 --kprime 500 \
        --eval-queries 60 --queries-per-leg 35 --out "{run}/drift_sweep.json"

`analysis/chain.py --synthetic` needs two things no single measurement gives
it. `alpha` is fitted so that ONE generation reproduces a measured pair, and
`--drift` says how the legs after the first are generated. The second is pure
assumption: a one-generation measurement constrains it not at all.

So this holds the measured quantity FIXED -- alpha is calibrated separately for
every variant so each one's first leg reproduces the same recall -- and varies
only the assumption. Whatever moves is the assumption's doing.

It is a falsifiable experiment about our own evidence: it could have shown the
chain gap is insensitive to the drift structure, which would have licensed
reading the simulated table as evidence. It did not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview.results_io import environment, write_results  # noqa: E402
from analysis import chain as ch  # noqa: E402


def sweep(N, d, T, kprime, k, eval_queries, queries_per_leg, target, seeds,
          drifts=ch.DRIFTS, rho=0.4):
    rows = []
    for drift in drifts:
        for seed in seeds:
            alpha, achieved = ch.calibrate_alpha(
                N, d, kprime, target, k, eval_queries, seed, drift)
            Vs, Qs = ch.synthetic_chain(
                N, d, T, alpha, n_queries=max(1200, 4 * eval_queries * T),
                seed=seed, drift=drift)
            crows, meta = ch.run_chain(Vs, Qs, k=k, kprimes=(kprime,), rho=rho,
                                       eval_queries=eval_queries, seed=seed,
                                       queries_per_leg=queries_per_leg)
            pick = lambda pol, f: next(
                r[f] for r in crows if r["policy"] == pol
                and r["leg"] == meta["T"] and r["kprime"] == kprime)
            lit = pick("original-only", "recall_at_k")
            rows.append(dict(
                drift=drift, seed=int(seed), alpha=round(float(alpha), 4),
                one_generation_recall=round(float(achieved), 4),
                literal_last=lit,
                nearest_last=pick("union-nearest", "recall_at_k"),
                all_last=pick("union-all", "recall_at_k"),
                gap_all_minus_literal=round(
                    pick("union-all", "recall_at_k") - lit, 4),
                total_passes_pct_literal=pick("original-only",
                                              "total_passes_pct"),
                total_passes_pct_all=pick("union-all", "total_passes_pct"),
                order_holds=bool(
                    pick("union-all", "recall_at_k")
                    >= pick("union-nearest", "recall_at_k") >= lit),
                reach_last=next(
                    L["reached_frac"] for L in meta["legs"]
                    if L["policy"] == "union-all" and L["leg"] == meta["T"]),
            ))
    gaps = [r["gap_all_minus_literal"] for r in rows]
    return rows, dict(
        N=int(N), dim=int(d), T=int(T), kprime=int(kprime), k=int(k),
        eval_queries=int(eval_queries), queries_per_leg=queries_per_leg,
        calibrate_to=float(target), seeds=[int(s) for s in seeds],
        drifts=list(drifts),
        gap_min=round(min(gaps), 4), gap_max=round(max(gaps), 4),
        gap_spread_ratio=(round(max(gaps) / min(gaps), 1)
                          if min(gaps) > 0 else None),
        order_holds_everywhere=all(r["order_holds"] for r in rows))


def report(rows, meta) -> None:
    print(f"\n  N={meta['N']:,}  {meta['T']} legs  K'={meta['kprime']}  "
          f"every variant calibrated so one generation = {meta['calibrate_to']}")
    print(f"\n  {'drift assumption':<20}{'seed':>5}{'alpha':>7}{'leg 1':>7}"
          f"{'literal':>9}{'+recent':>9}{'+all':>8}{'gap':>8}{'order':>7}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['drift']:<20}{r['seed']:>5}{r['alpha']:>7.3f}"
              f"{r['one_generation_recall']:>7.3f}{r['literal_last']:>9.3f}"
              f"{r['nearest_last']:>9.3f}{r['all_last']:>8.3f}"
              f"{r['gap_all_minus_literal']:>8.3f}"
              f"{'ok' if r['order_holds'] else 'BROKEN':>7}")
    print(f"\n  the gap the chain section argues from spans "
          f"{meta['gap_min']:.3f} to {meta['gap_max']:.3f}"
          + (f" ({meta['gap_spread_ratio']}x)" if meta["gap_spread_ratio"]
             else "")
          + f" with the measured\n  quantity held fixed, so the magnitude is "
            f"not identified by anything measured.")
    print(f"  the policy order {'holds in every variant' if meta['order_holds_everywhere'] else 'DOES NOT hold everywhere'}, "
          f"so the direction is the part that survives.")


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--legs", type=int, default=4)
    ap.add_argument("--kprime", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eval-queries", type=int, default=60)
    ap.add_argument("--queries-per-leg", type=int, default=35)
    ap.add_argument("--calibrate-to", type=float, default=0.929,
                    help="the measured one-generation recall@K' to fit alpha to")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows, meta = sweep(a.n, a.dim, a.legs, a.kprime, a.k, a.eval_queries,
                       a.queries_per_leg, a.calibrate_to, seeds)
    report(rows, meta)
    write_results(a.out, {"tool": "drift_sweep", "meta": meta, "rows": rows,
                          "environment": environment(dtype="float32")})
    print(f"\nwrote {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
