"""Regenerate the paper's conservatism table from result files, one row per
corpus, using the same rule as `compare_8m.py` and the same rule the caption
states: measured minus predicted savings, in points, over the informative range
x = D*K'/N in [0.05, 3].

Why this exists: the table in `body.tex` (`tab:form`) had no generator in the
bundle, so two of its five rows could not be reproduced from any archived file.
Applying this rule to `results/migration_cipbench.json` gives 6 points with mean
+27.6 and min +7.8, not the 11 / +9.75 / +0.00 the table printed. A number in a
table is only as good as the command that prints it, so here is the command.

    python analysis/tab_form.py \
        "MS~MARCO=results/migration_msmarco_500k.json" \
        "MS~MARCO (8.8M)=results/migration_msmarco_8m.json" \
        "DPR=results/migration_dpr_490k.json" \
        "FiQA=results/migration_fiqa.json" \
        "CIPBench=results/migration_cipbench.json" \
        "ORCAS (real log)=results/estimate_orcas_100k.json"

Each argument is `<label>=<path>`. A `migration` file contributes its `frontier`
rows; an `estimate` file contributes every point of every coverage curve. Both
give (measured savings, predicted savings) pairs, and they are never pooled
across rows -- a row is one corpus.

`--points` prints the per-point detail for one row, which is what to look at when
a row disagrees with the paper.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LO, HI = 0.05, 3.0          # the informative range, as the caption states


def residuals(doc: dict) -> list[tuple[float, float, float, float, float, float]]:
    """(K', D, x, measured%, predicted%, residual) for one result file.

    Dispatches on the file's own shape rather than on its name, so a file that
    carries both blocks (an estimate attached to a migration) is fully used.
    """
    rows = []
    for r in doc.get("frontier", []):
        N, kp, d = r["N"], r["kprime"], r["D"]
        x = d * kp / N
        if LO <= x <= HI:
            pred = 100 * (1 - kp / N) ** d
            rows.append((kp, d, x, r["savings_pct"], pred, r["savings_pct"] - pred))
    for e in doc.get("estimates", []):
        N, kp = e["N"], e["kprime"]
        for d, reached in zip(e["curve_d"], e["curve_reached"]):
            x = d * kp / N
            if LO <= x <= HI:
                meas = 100 * (1 - reached)
                pred = 100 * (1 - kp / N) ** d
                rows.append((kp, d, x, meas, pred, meas - pred))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="+", metavar="LABEL=PATH")
    ap.add_argument("--points", default=None,
                    help="also print every point of the row with this label")
    a = ap.parse_args()

    out, bad = [], 0
    for spec in a.spec:
        if "=" not in spec:
            print(f"expected LABEL=PATH, got {spec!r}", file=sys.stderr)
            return 2
        label, path = spec.split("=", 1)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = residuals(doc)
        if not rows:
            print(f"% {label}: NO POINT in x in [{LO}, {HI}] -- {path}")
            print(f"%   the row cannot be sourced from this file; a denser depth "
                  f"grid or a longer intent sweep is needed", file=sys.stderr)
            bad = 1
            continue
        res = [r[-1] for r in rows]
        n = doc.get("N") or (doc.get("estimates") or [{}])[0].get("N")
        N = f"{n:,}" if isinstance(n, int) else "--"
        mean, mn = sum(res) / len(res), min(res)
        neg = [r for r in rows if r[-1] < 0]
        mark = f"  <-- {len(neg)} point(s) BELOW the form" if neg else ""
        print(f"% {label}: {path}")
        print(f"{label:<22} & {N:>9} & {len(res):>2} & ${mean:+.2f}$ & "
              f"${mn:+.2f}$ \\\\{mark}")
        if a.points and a.points == label:
            print(f"%   {'K':>5} {'D':>7} {'x':>6} {'meas%':>7} {'pred%':>7} {'resid':>7}")
            for kp, d, x, m, p, r in sorted(rows, key=lambda t: t[2]):
                print(f"%   {kp:>5} {d:>7} {x:>6.3f} {m:>7.2f} {p:>7.2f} {r:>+7.2f}")
        out.append((label, res))

    if out:
        pooled = [r for _, rs in out for r in rs]
        print(f"% rows {len(out)}   pooled points {len(pooled)}   "
              f"min residual {min(pooled):+.2f}   "
              f"any below the form: "
              f"{'YES' if any(r < 0 for r in pooled) else 'no'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
