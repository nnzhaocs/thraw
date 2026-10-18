"""Build the paper's evidence bundle: every result file a number in body.tex
rests on, copied under one directory with a manifest and a mapping README.

Why a script and not a copy: the mapping is the traceability claim. If a number
moves, this file is where the source is re-pointed, and re-running it rebuilds
the bundle and its hashes in one step.

    python analysis/collect_evidence.py --runs-root RUNSDIR [--runs-root MOREDIR ...] \
        --out RUNDIR/evidence

Each row below is (label, file pattern, field path, the value body.tex states).
`field` uses dotted paths into the JSON; a trailing `[k=v]` selects the row of a
list whose key equals v, and a second `[k=v]` selects within that row's list.
The script verifies each value against the JSON rather than trusting the table,
so a re-pointed source that does not carry the number fails loudly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# label, file pattern, field path, stated value, tolerance
CLAIMS = [
    # ---- R4: chained unfinished upgrades (sec:eval-chain) ------------------
    # Both runs write a file called chain.json, so these patterns carry the run
    # directory: a bare name would match whichever the walk reached first.
    ("R4 chain, literal design leg 1", "20261015-223042_chain_synthetic_traffic/chain.json",
     "sweeps[0].rows[policy=original-only][leg=1][kprime=500].recall_at_k",
     0.8167, 5e-5),
    ("R4 chain, literal design leg 4", "20261015-223042_chain_synthetic_traffic/chain.json",
     "sweeps[0].rows[policy=original-only][leg=4][kprime=500].recall_at_k",
     0.4493, 5e-5),
    ("R4 chain, all partial indexes leg 4", "20261015-223042_chain_synthetic_traffic/chain.json",
     "sweeps[0].rows[policy=union-all][leg=4][kprime=500].recall_at_k",
     0.7947, 5e-5),
    ("R4 chain, most recent only leg 4", "20261015-223042_chain_synthetic_traffic/chain.json",
     "sweeps[0].rows[policy=previous-only][leg=4][kprime=500].recall_at_k",
     0.5233, 5e-5),
    ("R4 chain, most recent leg 4 cost", "20261015-223042_chain_synthetic_traffic/chain.json",
     "sweeps[0].rows[policy=previous-only][leg=4][kprime=500].passes_pct",
     25.8, 5e-5),
    ("R4 chain, fixed rho, all leg 4", "20261015-223059_chain_synthetic_rho/chain.json",
     "sweeps[0].rows[policy=union-all][leg=4][kprime=500].recall_at_k",
     0.7747, 5e-5),
    # ---- R2: the nested-subset ladder (new in this round) ------------------
    ("R2 ladder, 8.8M K'=500 recall", "n_ladder.json",
     "frontier[N=8841823][kprime=500].recall_at_kprime", 0.9624, 5e-5),
    ("R2 ladder, 500k K'=500 recall", "n_ladder.json",
     "frontier[N=500000][kprime=500].recall_at_kprime", 0.9164, 5e-5),
    ("R2 ladder, 500k K'=10 recall", "n_ladder.json",
     "frontier[N=500000][kprime=10].recall_at_kprime", 0.4572, 5e-5),
    # ---- R3: the cross-generation union (new in this round) ----------------
    ("R3 union, old-only K'=500", "gen2_union.json",
     "schemes[scheme=old-only][kprime=500].recall_at_k", 0.9212, 5e-5),
    ("R3 union, mid-only K'=500", "gen2_union.json",
     "schemes[scheme=mid-only][kprime=500].recall_at_k", 0.9610, 5e-5),
    ("R3 union, union K'=200", "gen2_union.json",
     "schemes[scheme=union][kprime=200].recall_at_k", 0.9532, 5e-5),
    ("R3 union, union K'=200 passes", "gen2_union.json",
     "schemes[scheme=union][kprime=200].passes", 125780, 0.5),
    ("R3 union, old-only K'=500 passes", "gen2_union.json",
     "schemes[scheme=old-only][kprime=500].passes", 180647, 0.5),
    # ---- 3b: the baselines table, incl. the oracle and the last PENDING ----
    ("3b lazy recall (D=5000 column)", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=lazy].recall_at_k", 0.9310, 5e-5),
    ("3b stale recall", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=stale].recall_at_k", 0.5100, 5e-5),
    ("3b backfill-traffic recall", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=backfill-traffic].recall_at_k", 0.9617, 5e-5),
    ("3b backfill-oracle recall (new row)", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=backfill-oracle].recall_at_k", 0.9980, 5e-5),
    ("3b merge-calibrated recall", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=merge-calibrated].recall_at_k", 0.8956, 5e-5),
    ("3b merge-rrf recall", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=merge-rrf].recall_at_k", 0.6594, 5e-5),
    ("3b lazy+adapter recall (last PENDING)", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=lazy+adapter].recall_at_k", 0.5448, 5e-5),
    ("3b lazy+adapter query-path passes", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=lazy+adapter].passes_serve", 298535, 0.5),
    ("3b adapter-only recall", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "baselines[scheme=adapter].recall_at_k", 0.1958, 5e-5),
    ("3b hybrid recall at K'=500", "20261013-165911_migration_msmarco_500k_bsl/migration.json",
     "adapter.hybrid[kprime=500].hybrid_recall_at_kprime", 0.6973, 5e-5),
    # ---- D: CIPBench with a real model pair --------------------------------
    ("D cipbench N", "20261013-202002_migration_cipbench_real/migration.json",
     "N", 2366, 0.5),
    ("D cipbench K'=10 savings", "20261013-202002_migration_cipbench_real/migration.json",
     "frontier[kprime=10].savings_pct", 6.55, 0.02),
    # ---- S1: the step-1 checks ---------------------------------------------
    ("S1 claim median ms", "20261014-040335_step1_check/step1_report.json",
     "claims.claim_ms_p50", 2.35, 0.005),
    ("S1 gate count", "20261014-040335_step1_check/step1_report.json",
     "gates", None, None),          # checked for all-True instead of a value
    ("S1 overlap at K'=200", "20261014-040335_step1_check/step1_report.json",
     "depth.rows[mode=strict][kprime=200].overlap_with_exact", 0.9295, 5e-5),
    ("S1 overlap at K'=500", "20261014-040335_step1_check/step1_report.json",
     "depth.rows[mode=strict][kprime=500].overlap_with_exact", 0.9407, 5e-5),
    ("S1 overlap at K'=1000", "20261014-040335_step1_check/step1_report.json",
     "depth.rows[mode=strict][kprime=1000].overlap_with_exact", 0.9465, 5e-5),
    # ---- 7b: concurrency ---------------------------------------------------
    ("7b threads=1 wall seconds", "20261014-104501_step2_load_t1/step2.json",
     "runs[0].wall_s", 909.7, 0.05),
    ("7b threads=4 wall seconds", "20261014-110054_step2_load_t4/step2.json",
     "runs[0].wall_s", 656.5, 0.05),
    ("7b threads=8 wall seconds", "20261014-111235_step2_load_t8/step2.json",
     "runs[0].wall_s", 648.1, 0.05),
    ("7b recall identical across threads", "20261014-104501_step2_load_t1/step2.json",
     "runs[0].recall_e2e", 0.9364, 5e-5),
    ("7b units encoded, threads=1", "20261014-104501_step2_load_t1/step2.json",
     "runs[0].encoded_units", 183193, 0.5),
    # ---- 2b: contention at the ledger's own reference batch ----------------
    ("2b penalty with Postgres load (b16)", "20261013-191607_backfill_rate_b16_pgload/backfill_rate.json",
     "result.penalty", 1.0007, 5e-5),
    ("2b drift B over A", "20261013-191607_backfill_rate_b16_pgload/backfill_rate.json",
     "result.backfill_drift_B_over_A", 1.0016, 5e-5),
    ("2b pg-load window searches (b64)", "20261012-132839_backfill_rate_b64_pgload/backfill_rate.json",
     "pg_load.window_searches", 114190, 0.5),
    ("2b contention shift at b64", "20261012-132839_backfill_rate_b64_pgload/backfill_rate.json",
     "result.penalty", 0.9172, 5e-5),
]


def _match(rv, want) -> bool:
    if rv is None:
        return False
    try:
        return abs(float(rv) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return str(rv) == want


def _resolve(obj, path: str):
    """Walk a dotted path, honouring [key=value] selectors on lists.

    Consecutive selectors are predicates on the SAME list, applied together --
    `frontier[N=500000][kprime=500]` picks one row, it does not select a row and
    then try to iterate it.
    """
    cur = obj
    for part in path.split("."):
        if "[" not in part:
            cur = cur[part]
            continue
        name, rest = part.split("[", 1)
        bits = rest.rstrip("]").split("][")
        keys = [b.split("=", 1) for b in bits if "=" in b]     # [k=v] predicates
        idxs = [int(b) for b in bits if "=" not in b]          # [0] positions
        rows = cur[name] if name else cur
        if keys:
            hit = None
            for row in rows:
                if all(_match(row.get(k), v) for k, v in keys):
                    hit = row
                    break
            if hit is None:
                raise KeyError(f"no row matching {rest}")
        else:
            hit = rows
        for i in idxs:
            hit = hit[i]
        cur = hit
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", required=True,
                    help="directory to search for run dirs/files; repeatable")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    roots = [Path(r) for r in a.runs_root]
    out = Path(a.out)
    # Flat layout on purpose: one file per cited run, prefixed with the run
    # directory's name. Nested dirs would be tidier but the sandboxed collector
    # can create files in an existing directory and not new directories, and a
    # flat bundle with self-describing names is easier to read anyway.
    out.mkdir(exist_ok=True)
    # The stale files are removed AFTER collection, not before. Deleting first
    # is idempotent only while the sources are present: run this against a tree
    # that has the bundle but not the run directories it was built from -- a
    # distributed tarball, which is the normal case for a reader -- and the
    # unlink loop empties the bundle before discovering it cannot refill it.
    # That happened, and it destroyed every evidence file in the package. So:
    # collect, then prune what collection did not produce, and refuse to prune
    # at all if nothing resolved.
    had = {p.name for p in out.glob("*__*") if p.is_file()}
    written: set[str] = set()

    rows, missing, failed = [], [], []
    for label, pat, field, want, tol in CLAIMS:
        src = None
        want_path = pat.replace("\\", "/")
        for root in roots:
            cand = root / Path(pat)
            if cand.is_file():
                src = cand
                break
            hits = [p for p in root.rglob(Path(pat).name)
                    if str(p).replace("\\", "/").endswith(want_path)]
            if hits:
                # shortest path wins: the same run name can appear under several
                # archive roots, and the shallowest is the canonical copy
                src = sorted(hits, key=lambda p: len(str(p)))[0]
                break
        if src is None:
            missing.append((label, pat))
            continue
        obj = json.loads(src.read_text(encoding="utf-8"))
        try:
            got = _resolve(obj, field)
        except Exception as e:                                    # noqa: BLE001
            failed.append((label, field, f"{type(e).__name__}: {e}", src))
            continue
        if want is None:
            ok = isinstance(got, dict) and all(bool(v) for v in got.values())
            shown = f"all {len(got)} gates True" if ok else "SOME GATE FALSE"
        else:
            ok = abs(float(got) - float(want)) <= tol
            shown = f"{float(got):g}"
        if not ok:
            failed.append((label, field, f"got {shown}, paper says {want}", src))
        # copy the source, prefixed with its run directory's name
        stem = f"{src.parent.name}__{src.name}"
        dest = out / stem
        shutil.copy2(src, dest)
        written.add(dest.name)
        if (src.parent / (src.name + ".sha256")).is_file():
            shutil.copy2(src.parent / (src.name + ".sha256"), out / (stem + ".sha256"))
            written.add(stem + ".sha256")
        if (src.parent / "MANIFEST.sha256").is_file():
            shutil.copy2(src.parent / "MANIFEST.sha256", out / (src.parent.name + "__MANIFEST.sha256"))
            written.add(src.parent.name + "__MANIFEST.sha256")
        rows.append((label, field, want, shown, str(src), dest.name))
        print(f"{'ok  ' if ok else 'FAIL'} {label:<44} {field:<48} {shown}")

    if not written:
        print(f"\nrefusing to touch {out}: not one of the {len(CLAIMS)} cited "
              f"files resolved under {', '.join(str(r) for r in roots)}, so "
              f"there is nothing to collect. The {len(had)} file(s) already in "
              f"the bundle are left alone -- this is what a tree that has the "
              f"bundle but not the run directories looks like, and emptying it "
              f"would destroy the only copy.", file=sys.stderr)
        return 1
    # Pruning is only safe when the sources are somewhere other than the
    # bundle. Given `--runs-root evidence` the patterns resolve to the bundle's
    # own flattened files, whose parent directory is `evidence`, so collection
    # rewrites them under a different name and every original would look stale.
    inside = [r for r in roots if r.resolve() == out.resolve()
              or out.resolve() in r.resolve().parents
              or r.resolve() in out.resolve().parents]
    if inside:
        print(f"  not pruning: {', '.join(str(r) for r in inside)} is the "
              f"bundle itself, so a file already here is a source, not a "
              f"leftover")
    else:
        for name in sorted(had - written):
            (out / name).unlink()
            print(f"  pruned stale {name}")

    # -------- README + MANIFEST ------------------------------------------
    lines = ["# Evidence bundle", "",
             "Every result file a stated number in `body.tex` rests on, copied here by",
             "`analysis/collect_evidence.py`. Regenerate with:", "",
             "```bash",
             "python analysis/collect_evidence.py \\",
             "  --runs-root /path/to/result(1) --runs-root results \\",
             "  --out evidence",
             "```", "",
             "Columns: what the paper says, where it says it (JSON field path), the",
             "value read back from the file, and the copied file. `[k=v]` selects a row",
             "of a list by its key.", "",
             "| paper number | field path | read back | source run | bundle path |",
             "|---|---|---|---|---|"]
    for label, field, want, shown, src, dst in rows:
        lines.append(f"| {label} | `{field}` | {shown} | `{src}` | `{dst}` |")
    if missing:
        lines += ["", "## Not found in the searched roots", ""]
        lines += [f"- {lbl} (`{pat}`)" for lbl, pat in missing]
    if failed:
        lines += ["", "## MISMATCHES — a number in the paper disagrees with its file", ""]
        lines += [f"- **{lbl}**: {why} (`{src}`)" for lbl, _f, why, src in failed]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    man = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            man.append(f"{h}  {p.stat().st_size}  {p.relative_to(out).as_posix()}")
    (out / "MANIFEST.sha256").write_text("\n".join(man) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} claims resolved, {len(missing)} not found, "
          f"{len(failed)} mismatched -> {out}")
    for lbl, _f, why, src in failed:
        print(f"  MISMATCH {lbl}: {why}  ({src})")
    for lbl, pat in missing:
        print(f"  MISSING  {lbl}: no file matching {pat}")
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
