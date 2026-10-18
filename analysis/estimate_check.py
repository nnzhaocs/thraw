"""D-09: does `estimate` predict the reached fraction it claims to?

The pre-migration decision (C2) rests on being able to say, from a query log and
the *existing* index, how much of the corpus a migration will reach -- and
therefore how much it will save. The paper's closed form is the predictor:

    reached ~= 1 - (1 - K'/N)^D        (exact)      x = D*K'/N
             ~= 1 - exp(-D*K'/N)       (exponential, the one to avoid)

Every migration result file already records, per (K', D) row, the reached
fraction it *measured* and the savings the form *predicted*
(`pred_savings_indep` = 100*(1 - (1-K'/N)^D)). So the validation costs no
measurement at all: read both numbers out of the runs that are on disk.

Two things this tool does that reading the file alone does not:

  * it **recomputes** the prediction from the row's own N, K' and D, and reports
    any disagreement with the recorded `pred_savings_indep` -- a file whose
    recorded prediction does not match its own parameters is not evidence;
  * it reports the error **per corpus**, signed, in points, because the paper's
    decision is taken per corpus: a predictor that is right on average and wrong
    on the corpus in front of the operator does not support the decision.

Sign convention, matching `analysis/tab_form.py`: the error is in points of
**savings**, measured minus predicted. A positive error means the form
under-predicted the saving, i.e. the prediction of the reached fraction was
conservative (too high).

    python analysis/estimate_check.py \
        "MS MARCO=results/runs/RUN/migration.json" \
        "CIPBench=results/runs/migration_cipbench_/migration_cipbench.json" \
        --kprime 500 --out "{run}/estimate_check.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview.results_io import environment, write_results  # noqa: E402

# The informative range the paper reads the form over; outside it either the
# corpus is small enough that the form is trivially right (x < 0.05) or the
# whole corpus is touched and there is nothing to predict (x > 3).
LO, HI = 0.05, 3.0


def rows_of(doc: dict) -> list[dict]:
    """Every (K', D) point of a result file, with both numbers and the check."""
    out = []
    for r in doc.get("frontier", []):
        N, kp, d = r.get("N"), r.get("kprime"), r.get("D")
        if not all(isinstance(v, int) for v in (N, kp, d)):
            continue
        x = d * kp / N
        if not (LO <= x <= HI):
            continue
        measured_savings = r.get("savings_pct")
        recorded = r.get("pred_savings_indep")
        if measured_savings is None:
            continue
        # the paper's form, exactly: savings = (1 - K'/N)^D
        predicted_savings = 100 * (1 - kp / N) ** d
        out.append(dict(
            kprime=int(kp), D=int(d), x=round(x, 4),
            measured_reached=round(1 - measured_savings / 100, 6),
            predicted_reached=round(1 - predicted_savings / 100, 6),
            measured_savings_pct=round(measured_savings, 4),
            predicted_savings_pct=round(predicted_savings, 4),
            recorded_pred_savings_pct=recorded,
            error_points=round(measured_savings - predicted_savings, 4),
            recorded_vs_recomputed=(
                None if recorded is None
                else round(recorded - predicted_savings, 4)),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", nargs="+", metavar="LABEL=PATH")
    ap.add_argument("--kprime", type=int, default=None,
                    help="restrict to one K' (default: every K' whose x falls "
                         "in the informative range, as analysis/tab_form.py does)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    per_corpus, errors, mismatches = [], [], []
    for spec in a.spec:
        if "=" not in spec:
            print(f"expected LABEL=PATH, got {spec!r}", file=sys.stderr)
            return 2
        label, path = spec.split("=", 1)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [r for r in rows_of(doc)
                if a.kprime is None or r["kprime"] == a.kprime]
        if not rows:
            print(f"% {label}: no point at K'={a.kprime} in [{LO}, {HI}] -- {path}",
                  file=sys.stderr)
            return 1
        # The validation point per corpus: the deepest intent count inside the
        # informative range, because that is where the form is most tested.
        row = max(rows, key=lambda r: r["x"])
        errs = [r["error_points"] for r in rows]
        rec = [abs(r["recorded_vs_recomputed"]) for r in rows
               if r["recorded_vs_recomputed"] is not None]
        # the recorded field is rounded to two decimals, so the check is
        # against that granularity, not against float noise
        if rec and max(rec) > 0.005:
            mismatches.append((label, max(rec)))
        per_corpus.append(dict(label=label, path=path, N=doc.get("N"),
                               points=len(rows), **row,
                               min_error_points=min(errs),
                               max_error_points=max(errs)))
        errors.extend(errs)
        print(f"{label:<14} N={doc.get('N'):>9,}  x={row['x']:.2f}  "
              f"D={row['D']:<6} measured={row['measured_reached']:.4f} "
              f"predicted={row['predicted_reached']:.4f} "
              f"error={row['error_points']:+.2f} pts  "
              f"(span {min(errs):+.2f}..{max(errs):+.2f} over {len(rows)} pts)")

    if mismatches:
        print(f"\n  ! recorded pred_savings_indep disagrees with the recomputed "
              f"form: {mismatches}", file=sys.stderr)
        return 1

    err_max = max(errors, key=abs)
    worst = max(per_corpus, key=lambda r: abs(r["error_points"]))
    print(f"\n  {len(per_corpus)} corpora, {len(errors)} points: "
          f"the form never under-predicts savings (min error {min(errors):+.2f} pts); "
          f"max |error| {err_max:+.2f} pts on {worst['label']}")

    if a.out:
        write_results(a.out, {
            "tool": "estimate_check",
            "convention": "error_points = measured_savings - predicted_savings "
                          "(positive: the form was conservative about the saving)",
            "kprime": a.kprime,
            "x_range": [LO, HI],
            "corpora": per_corpus,
            "min_error_points": round(min(errors), 4),
            "max_error_points_abs": round(err_max, 4),
            "max_error_corpus": worst["label"],
            "environment": environment(),
        })
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
