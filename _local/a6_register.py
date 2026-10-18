"""Register A6 (the rho=0.3 chain sweep) into the ledger and the bundle."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EV = ROOT / "evidence"
ENV = {**os.environ, "THAW_MODE": "measure"}
sys.path.insert(0, str(ROOT))
from thaw_audit.paths import resolve  # noqa: E402

RUN = "20261017-233537_chain_real_rho03"
SRC = ROOT / "results" / "runs" / RUN / "chain.json"
for side in ("chain.json.sha256", "MANIFEST.sha256"):
    s = SRC.parent / side
    if s.is_file():
        shutil.copy2(s, EV / f"{RUN}__{side}")
shutil.copy2(SRC, EV / f"{RUN}__chain.json")
print("evidence +", f"{RUN}__chain.json")

NAME = f"{RUN}__chain.json"
S0 = "sweeps[0].rows"
CLAIMS = [
    ("CHAIN-RHO03-ALL-L4-K4000",
     "union with all partial indexes served recall@10 at leg 4, K'=4000, rho=0.3 -- "
     "the pre-registered branch threshold, measured for the first time",
     0.9367, 5e-5, NAME, f"{S0}[policy=union-all][leg=4][kprime=4000].recall_at_k",
     "sec:eval-chain"),
    ("CHAIN-RHO03-ALL-L4-K4000-PASSES",
     "that row's total passes as a percentage of N",
     170.6, 0.05, NAME, f"{S0}[policy=union-all][leg=4][kprime=4000].total_passes_pct",
     "sec:eval-chain"),
    ("CHAIN-RHO03-LIT-K8000",
     "the strongest original-only row spending at least as much -- leg 4, K'=8000, "
     "rho=0.3 -- the control the branch criterion is decided against",
     0.9307, 5e-5, NAME, f"{S0}[policy=original-only][leg=4][kprime=8000].recall_at_k",
     "sec:eval-chain"),
    ("CHAIN-RHO03-LIT-K8000-PASSES",
     "that control's total passes as a percentage of N",
     179.0, 0.05, NAME, f"{S0}[policy=original-only][leg=4][kprime=8000].total_passes_pct",
     "sec:eval-chain"),
]
for cid, says, val, tol, src, field, paper in CLAIMS:
    got = resolve(json.loads((EV / src).read_text(encoding="utf-8")), field)
    assert abs(float(got) - float(val)) <= tol, (cid, got, val)
    r = subprocess.run(
        [PY, "-m", "thaw_audit", "add", "--id", cid, "--says", says,
         "--value", json.dumps(val), "--tol", json.dumps(tol),
         "--source", src, "--field", field, "--paper", paper,
         "--run-id", RUN, "--replace"],
        cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, (cid, r.stdout, r.stderr)
    print(f"added {cid} = {got}")

ids = [c[0] for c in CLAIMS]
r = subprocess.run([PY, "-m", "thaw_audit", "freeze", *ids],
                   cwd=ROOT, capture_output=True, text=True, env=ENV)
assert r.returncode == 0, r.stdout
print(f"froze {len(ids)}: {', '.join(ids)}")

# regenerate the bundle manifest and read everything back
files = [str(p) for p in sorted(EV.glob("*")) if p.is_file()]
r = subprocess.run([PY, "-m", "semview.results_io", "manifest", *files],
                   cwd=ROOT, capture_output=True, text=True)
print("manifest:", (r.stdout or r.stderr).strip()[:120])
r = subprocess.run([PY, "-m", "semview.results_io", "verify",
                    str(EV / "MANIFEST.sha256")],
                   cwd=ROOT, capture_output=True, text=True)
print("verify:", (r.stdout or r.stderr).strip().splitlines()[-1])
