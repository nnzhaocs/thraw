"""A6: the chain sweep at rho = 0.3, on the merged tree's NEW chain.py.

The five spaces and five query files are the ones 20261017-184214_chain_real_deep
used, so the rho=0.3 row is comparable to the rows either side of it. The
spaces arrive from the box (bge-m3 and Qwen3 500k matrices exist only there);
the queries are the per-leg dumps. CPU only -- no encoder runs.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

SPACES = [
    "cache/2aa13438bd13a33c3e6705a785ecfcc2.npy",
    "cache/8256b4ed43559037dee7d45f28746869.npy",
    "cache/5d103cb11869f81b2b5023901a9fc385.npy",
    "cache/c60e09474748d76ad5f3eafb61126bce.npy",
    "cache/ad3f5957fc3400a29de3e268ada99b08.npy",
]
QUERIES = [
    "dump/leg0/q_old.npy", "dump/leg1/q_old.npy", "dump/leg2/q_old.npy",
    "dump/leg3/q_old.npy", "dump/leg3/q_new.npy",
]

for p in SPACES + QUERIES:
    f = ROOT / p
    if not f.is_file():
        sys.exit(f"missing input: {p}")

env = {**os.environ, "THAW_MODE": "measure", "USE_TF": "0",
       "TRANSFORMERS_NO_TF": "1", "HF_HUB_OFFLINE": "1"}
cmd = [PY, "-m", "semview.runlog", "--name", "chain_real_rho03",
       "--tag", "d08", "--",
       PY, "analysis/chain.py",
       "--spaces", ",".join(SPACES),
       "--queries", ",".join(QUERIES),
       "--kprimes", "500,1000,2000,4000,8000",
       "--rho", "0.3", "--eval-queries", "150",
       "--out", "{run}/chain.json"]
print(" ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=ROOT, env=env)
sys.exit(r.returncode)
