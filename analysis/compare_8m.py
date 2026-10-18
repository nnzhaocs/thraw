"""Read the 8.8M results beside the 500k ones and check the two acceptance
conditions in SIGMOD_PLAN.md: the recall curve keeps its shape, and measured
savings stay at or above (1 - K'/N)^D across the informative range.

    python analysis/compare_8m.py \
        --small results/migration_msmarco_500k.json \
        --large results/migration_msmarco_8m.json \
        --estimate results/estimate_msmarco_8m_dev.json \
                   results/estimate_msmarco_8m_all.json

Residuals come from two sources, reported separately and never pooled:
`migration` rows (dev queries, x up to ~0.28 at 8.8M) and `estimate` coverage
curves (any query population; each curve point is a (D, reached) pair).
"""
from __future__ import annotations

import argparse
import json

LO, HI = 0.05, 3.0          # the informative range of x = D K'/N


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_by_kprime(run):
    """One recall per K'. Recall does not move with D, so rows repeat it.

    Checked rather than assumed: `setdefault` silently kept the first value, so
    a run whose frontier disagreed with itself at a given K' would have had the
    disagreement dropped without a word, and the acceptance table is the thing
    that is supposed to catch exactly that kind of inconsistency.
    """
    out = {}
    for r in run["frontier"]:
        kp, rec = r["kprime"], r["recall_at_kprime"]
        if kp in out and abs(out[kp] - rec) > 1e-9:
            raise SystemExit(
                f"frontier is inconsistent at K'={kp}: recall {out[kp]} and "
                f"{rec}. Recall should not depend on D; a run that reports two "
                "values for one K' cannot be read by this table.")
        out[kp] = rec
    return out


def migration_residuals(run):
    rows = []
    for r in run["frontier"]:
        x = r["D"] * r["kprime"] / r["N"]
        if LO <= x <= HI:
            pred = 100 * (1 - r["kprime"] / r["N"]) ** r["D"]
            rows.append((r["kprime"], r["D"], x, r["savings_pct"],
                         pred, r["savings_pct"] - pred))
    return rows


def estimate_residuals(est):
    rows = []
    for e in est["estimates"]:
        kp, N = e["kprime"], e["N"]
        for d, reached in zip(e["curve_d"], e["curve_reached"]):
            x = d * kp / N
            if LO <= x <= HI:
                meas = 100 * (1 - reached)
                pred = 100 * (1 - kp / N) ** d
                rows.append((kp, d, x, meas, pred, meas - pred))
    return rows


def acceptance(s, l, ests=(), tol: float = 0.0) -> list[str]:
    """The acceptance conditions as checks, so a failure exits non-zero instead
    of waiting for someone to read a table. Empty list = accepted.

    1. like for like: same models, prefixes, k, and (when recorded) queries;
    2. the recall curve keeps its shape: non-decreasing in K' at both scales;
    3. measured savings never fall below (1 - K'/N)^D by more than `tol`
       points, on migration rows and estimate curves alike.
    """
    fails = []
    for key in ("old", "new", "prefixes", "k"):
        if s.get(key) != l.get(key):
            fails.append(f"runs differ in {key}: {s.get(key)!r} vs {l.get(key)!r}")
    qs = (s.get("fingerprint") or {}).get("query_ids_sha256")
    ql = (l.get("fingerprint") or {}).get("query_ids_sha256")
    if qs and ql and qs != ql:
        fails.append(f"runs used different queries ({qs} vs {ql})")
    for run, name in ((s, "small"), (l, "large")):
        r = recall_by_kprime(run)
        ks = sorted(r)
        bad = [(a, b) for a, b in zip(ks, ks[1:]) if r[b] < r[a] - 1e-9]
        if bad:
            fails.append(f"{name}: recall falls with K' at {bad}")
    rows = migration_residuals(s) + migration_residuals(l)
    for e in ests:
        rows += estimate_residuals(e)
    below = [r for r in rows if r[-1] < -tol]
    if below:
        fails.append(f"{len(below)} point(s) below (1-K'/N)^D by more than {tol}: "
                     + ", ".join(f"K'={r[0]} D={r[1]} {r[-1]:+.2f}" for r in below[:5]))
    return fails


def show(title, rows):
    print(f"\n{title}")
    if not rows:
        print("  (no point in the informative range)")
        return
    print(f"  {'K':>5} {'D':>7} {'x':>6} {'meas%':>7} {'pred%':>7} {'resid':>7}")
    for kp, d, x, m, p, res in rows:
        flag = "   <-- BELOW THE FORM" if res < 0 else ""
        print(f"  {kp:>5} {d:>7} {x:>6.3f} {m:>7.2f} {p:>7.2f} {res:>+7.2f}{flag}")
    res = [r[-1] for r in rows]
    print(f"  points {len(res)}   mean {sum(res)/len(res):+.2f}   min {min(res):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True, help="500k migration JSON")
    ap.add_argument("--large", required=True, help="8.8M migration JSON")
    ap.add_argument("--estimate", nargs="*", default=[],
                    help="8.8M estimate JSONs (dev queries, all queries)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="points below the closed form tolerated before failing")
    a = ap.parse_args()

    s, l = load(a.small), load(a.large)
    for run, name in ((s, "small"), (l, "large")):
        print(f"{name}: N={run['N']:,} D={run['D']:,} old={run['old']} "
              f"new={run['new']} new-query-prefix={run['prefixes']['new_query']!r}")
    if s["prefixes"] != l["prefixes"] or s["old"] != l["old"] or s["new"] != l["new"]:
        print("WARNING: the two runs differ in models or prefixes; "
              "the comparison is not like for like")

    rs, rl = recall_by_kprime(s), recall_by_kprime(l)
    print(f"\n  {'K':>5} {'recall 500k':>12} {'recall 8.8M':>12} {'delta':>7}")
    for kp in sorted(set(rs) | set(rl)):
        a_, b_ = rs.get(kp), rl.get(kp)
        d_ = "" if a_ is None or b_ is None else f"{b_ - a_:+.3f}"
        print(f"  {kp:>5} {a_ if a_ is not None else '-':>12} "
              f"{b_ if b_ is not None else '-':>12} {d_:>7}")

    show("closed-form residual, migration rows (500k)", migration_residuals(s))
    show("closed-form residual, migration rows (8.8M)", migration_residuals(l))
    ests = []
    for path in a.estimate:
        e = load(path)
        ests.append(e)
        show(f"closed-form residual, estimate curve {path} "
             f"(N={e['N']:,}, D={e['D']:,})", estimate_residuals(e))
    fails = acceptance(s, l, ests, a.tol)
    print("\n" + ("ACCEPTED" if not fails else "NOT ACCEPTED"))
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
