"""A7 batch: evidence entries, claims, pins, declarations, baseline.

Order matters: the claims' `source` files must be in evidence/ before verify
can read them, so the bundle is extended first, then the claims are added
through the CLI (one process per add, the sanctioned path), verified, and
frozen. Pins and declarations then go into audit/paper_numbers.json, and the
ratchet baseline is lowered to the new count.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EV = ROOT / "evidence"
MEASURE_ENV = {**os.environ, "THAW_MODE": "measure"}

# ---------------------------------------------------------------- evidence --
NEW_EVIDENCE = [
    "20261017-184214_chain_real_deep/chain.json",
    "20261017-225704_scaling_mechanism/scaling_mechanism.json",
    "20261017-225502_dpr_provenance_diag/dpr_diag.json",
    "20261017-150338_estimate_cipbench_dense/estimate.json",
    "20261017-150605_estimate_cipbench_dense/estimate.json",
    "20261017-150642_estimate_fiqa_dense/estimate.json",
]
for rel in NEW_EVIDENCE:
    src = ROOT / "results" / "runs" / rel
    run = src.parent.name
    dest = EV / f"{run}__{src.name}"
    if not dest.exists():
        shutil.copy2(src, dest)
        print(f"evidence + {dest.name}")
    for side in (f"{src.name}.sha256", "MANIFEST.sha256"):
        s = src.parent / side
        if s.is_file():
            name = f"{run}__{side}" if side.startswith("MANIFEST") else f"{run}__{side}"
            shutil.copy2(s, EV / name)

# ------------------------------------------------------------------ claims --
RUNS = {"CHAIN-DEEP": "20261017-184214_chain_real_deep",
         "SCALING": "20261017-225704_scaling_mechanism",
         "DPR-PROV": "20261017-225502_dpr_provenance_diag"}
DEEP = "20261017-184214_chain_real_deep__chain.json"
SCAL = "20261017-225704_scaling_mechanism__scaling_mechanism.json"
DIAG = "20261017-225502_dpr_provenance_diag__dpr_diag.json"
S1 = "sweeps[1].rows"

CLAIMS = [
    # ---- the deep chain table (tab:chain, K'=1000 unless said) -------------
    ("CHAIN-DEEP-ORIG-L1", "literal design (original index only) served recall@10 at leg 1, K'=1000, on the real four-generation chain", 0.9773, 5e-5, DEEP, f"{S1}[policy=original-only][leg=1][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ORIG-L2", "literal design served recall@10 at leg 2, K'=1000", 0.9613, 5e-5, DEEP, f"{S1}[policy=original-only][leg=2][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ORIG-L3", "literal design served recall@10 at leg 3, K'=1000", 0.8947, 5e-5, DEEP, f"{S1}[policy=original-only][leg=3][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ORIG-L4", "literal design served recall@10 at leg 4, K'=1000", 0.848, 5e-5, DEEP, f"{S1}[policy=original-only][leg=4][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ORIG-L4-PASSES", "the literal design's total passes at leg 4, K'=1000, as a percentage of N", 173.1, 0.05, DEEP, f"{S1}[policy=original-only][leg=4][kprime=1000].total_passes_pct", "tab:chain"),
    ("CHAIN-DEEP-PREV-L2", "most recent partial index only served recall@10 at leg 2, K'=1000", 0.4787, 5e-5, DEEP, f"{S1}[policy=previous-only][leg=2][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-PREV-L3", "most recent partial index only served recall@10 at leg 3, K'=1000", 0.416, 5e-5, DEEP, f"{S1}[policy=previous-only][leg=3][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-PREV-L4", "most recent partial index only served recall@10 at leg 4, K'=1000", 0.3967, 5e-5, DEEP, f"{S1}[policy=previous-only][leg=4][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-PREV-L4-PASSES", "the previous-only design's total passes at leg 4, K'=1000, as a percentage of N", 166.1, 0.05, DEEP, f"{S1}[policy=previous-only][leg=4][kprime=1000].total_passes_pct", "tab:chain"),
    ("CHAIN-DEEP-UNION-L2", "union with the most recent partial served recall@10 at leg 2, K'=1000", 0.9727, 5e-5, DEEP, f"{S1}[policy=union-nearest][leg=2][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-UNION-L3", "union with the most recent partial served recall@10 at leg 3, K'=1000", 0.918, 5e-5, DEEP, f"{S1}[policy=union-nearest][leg=3][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-UNION-L4", "union with the most recent partial served recall@10 at leg 4, K'=1000", 0.882, 5e-5, DEEP, f"{S1}[policy=union-nearest][leg=4][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-UNION-L4-PASSES", "the union design's total passes at leg 4, K'=1000, as a percentage of N", 176.0, 0.05, DEEP, f"{S1}[policy=union-nearest][leg=4][kprime=1000].total_passes_pct", "tab:chain"),
    ("CHAIN-DEEP-ALL-L3", "union with all partial indexes served recall@10 at leg 3, K'=1000", 0.9313, 5e-5, DEEP, f"{S1}[policy=union-all][leg=3][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ALL-L4", "union with all partial indexes served recall@10 at leg 4, K'=1000", 0.896, 5e-5, DEEP, f"{S1}[policy=union-all][leg=4][kprime=1000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ALL-L4-PASSES", "the union-all design's total passes at leg 4, K'=1000, as a percentage of N", 180.8, 0.05, DEEP, f"{S1}[policy=union-all][leg=4][kprime=1000].total_passes_pct", "tab:chain"),
    ("CHAIN-DEEP-ALL-L4-K4000", "union with all partial indexes served recall@10 at leg 4, K'=4000 -- the matched-budget row", 0.9427, 5e-5, DEEP, f"{S1}[policy=union-all][leg=4][kprime=4000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-ALL-L4-K4000-PASSES", "that row's total passes, as a percentage of N", 203.9, 0.05, DEEP, f"{S1}[policy=union-all][leg=4][kprime=4000].total_passes_pct", "tab:chain"),
    ("CHAIN-DEEP-LIT-K8000", "literal design served recall@10 at leg 4, K'=8000 -- the strongest control spending at least as much", 0.9307, 5e-5, DEEP, f"{S1}[policy=original-only][leg=4][kprime=8000].recall_at_k", "tab:chain"),
    ("CHAIN-DEEP-LIT-K8000-PASSES", "that control's total passes, as a percentage of N", 210.0, 0.05, DEEP, f"{S1}[policy=original-only][leg=4][kprime=8000].total_passes_pct", "tab:chain"),
    # ---- A4: the scaling mechanism ----------------------------------------
    ("SCALING-DUP-500K", "near-duplicate pairs per query in the top-10 at N=500,000", 0.031, 0.005, SCAL, "rows[0].dup_pairs_mean", "sec:eval-chain"),
    ("SCALING-DUP-8M", "near-duplicate pairs per query in the top-10 at N=8,841,823", 0.762, 0.005, SCAL, "rows[3].dup_pairs_mean", "sec:eval-chain"),
    ("SCALING-BAND-500K", "documents inside the 0.01 band below the k-th score at N=500,000", 4.976, 0.05, SCAL, "rows[0].band_count_mean", "sec:eval-chain"),
    ("SCALING-BAND-8M", "documents inside the 0.01 band below the k-th score at N=8,841,823", 5.549, 0.05, SCAL, "rows[3].band_count_mean", "sec:eval-chain"),
    # ---- A5: the DPR provenance diagnosis ----------------------------------
    ("DPR-PROV-JOINED", "mean cosine of re-encoded wiki_dpr passages with title and text encoded as one string", 0.9851, 5e-5, DIAG, "rows[case=title+text joined].mean_cos", "tab:ladder"),
    ("DPR-PROV-PAIR", "the same passages encoded as a (title, text) pair -- the form verify_dpr_provenance used", 0.9499, 5e-5, DIAG, "rows[case=max_length=256].mean_cos", "tab:ladder"),
    ("DPR-PROV-TRUNC-256", "of the 256 sampled passages, how many max_length=256 actually truncates", 0, 0, DIAG, "rows[case=max_length=256].tokens_at_or_over_limit", "tab:ladder"),
]

# every field path must resolve against its source before anything is added
sys.path.insert(0, str(ROOT))
from thaw_audit.paths import resolve  # noqa: E402

for cid, says, val, tol, src, field, paper in CLAIMS:
    obj = json.loads((EV / src).read_text(encoding="utf-8"))
    try:
        got = resolve(obj, field)
    except Exception as e:
        sys.exit(f"FIELD RESOLUTION FAILED for {cid}: {field!r}: {e}")
    if abs(float(got) - float(val)) > tol:
        sys.exit(f"VALUE MISMATCH for {cid}: field says {got}, claim says {val}")
    print(f"resolved {cid}: {field} = {got}")
print(f"all {len(CLAIMS)} field paths resolve and match\n")

added = []
for cid, says, val, tol, src, field, paper in CLAIMS:
    run_id = next(v for k, v in RUNS.items() if cid.startswith(k))
    cmd = [PY, "-m", "thaw_audit", "add", "--id", cid, "--says", says,
           "--value", json.dumps(val), "--tol", json.dumps(tol),
           "--source", src, "--field", field, "--paper", paper,
           "--run-id", run_id, "--replace"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=MEASURE_ENV)
    if r.returncode != 0:
        sys.exit(f"add failed for {cid}: {r.stdout}{r.stderr}")
    added.append(cid)
    print(f"added {cid}")
print(f"\n{len(added)} claims added")

if added:
    r = subprocess.run([PY, "-m", "thaw_audit", "--runs-root", "evidence", "verify"],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:])
    r = subprocess.run([PY, "-m", "thaw_audit", "freeze", *added],
                       cwd=ROOT, capture_output=True, text=True,
                       env=MEASURE_ENV)
    print(r.stdout[-800:])
    if r.returncode != 0:
        sys.exit(f"freeze failed: {r.stdout}{r.stderr}")
print("claims frozen")
