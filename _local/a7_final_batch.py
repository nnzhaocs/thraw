"""A7 final batch: the claims, pins and declarations that survived review,
then the ratchet baseline lowered to the resulting count.

Every claim below was confirmed by reading the paper sentence beside the
evidence field (see _local/a7_exhaustive.txt). Numbers whose only candidates
were coincidences -- a percent that matches a latency, a count that matches a
thread count -- are deliberately left unregistered: the runbook's A7 is the
known submission blocker, and a claim frozen against a coincidence verifies
forever while meaning nothing.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EV = ROOT / "evidence"
ENV = {**os.environ, "THAW_MODE": "measure"}
sys.path.insert(0, str(ROOT))
from thaw_audit.paths import resolve  # noqa: E402

DEEP = "20261017-184214_chain_real_deep__chain.json"
RHO3 = "20261017-000011_chain_real_rho3__chain.json"
SEED1 = "20261016-023904_chain_synthetic_traffic_seed1__chain.json"
LAD = "20261013-160555_n_ladder_msmarco__n_ladder.json"
B16P = "20261013-191607_backfill_rate_b16_pgload__backfill_rate.json"
B64P = "20261012-132839_backfill_rate_b64_pgload__backfill_rate.json"
B16T = "20261017-010914_backfill_rate_b16_tokensort_pgload__backfill_rate.json"
TAB = "20261017-003632_tab_form_rows__tab_form.json"

CLAIMS = [
    ("CHAIN-DEEP-MARGINAL-K8000",
     "marginal evaluation passes of the previous-only design at K'=8000 as a percentage of N",
     13.92, 0.05, DEEP,
     "sweeps[1].rows[policy=previous-only][leg=4][kprime=8000].eval_passes_marginal_pct",
     "sec:eval-chain", "20261017-184214_chain_real_deep"),
    ("RHO3-MARGINAL-500",
     "marginal evaluation passes of the union design at K'=500 as a percentage of N -- the upper bound on savings",
     12.6, 0.05, RHO3,
     "sweeps[0].rows[policy=union-nearest][leg=2][kprime=500].eval_passes_marginal_pct",
     "sec:eval-chain", "20261017-000011_chain_real_rho3"),
    ("SEED1-REACH-447",
     "union-nearest reached fraction on the seed-1 synthetic traffic",
     0.4471, 5e-5, SEED1,
     "sweeps[0].meta.legs[policy=union-nearest][leg=2].reached_frac",
     "sec:eval-chain", "20261016-023904_chain_synthetic_traffic_seed1"),
    ("LADDER-PREDSAV-LINEAR-500",
     "the closed form's linear prediction of savings at K'=500 on the 4M corpus at D=5000",
     37.5, 0.06, LAD, "frontier[N=4000000][kprime=500].pred_savings_linear",
     "tab:ladder", "20261013-160555_n_ladder_msmarco"),
    ("COST-B16-PREFIX-BACKFILL",
     "pre-token-sort backfill rate at batch 16, ms per unit (the D-02 pre-fix number)",
     3.7801, 5e-5, B16P, "phases[0].ms_per_unit_series[27][1]",
     "6.4", "20261013-191607_backfill_rate_b16_pgload"),
    ("COST-B64-FWD",
     "the b64 run's forward-pass rate, ms per unit",
     3.7854, 5e-5, B64P, "phases[2].fwd_ms_per_unit",
     "6.4", "20261012-132839_backfill_rate_b64_pgload"),
    ("PAD-B16-TOKENSORT-FACTOR",
     "which-units tokens per unit at batch 16 after the token-sort fix",
     1.0158, 5e-5, B16T, "result.factors.which_units_tokens_per_unit",
     "6.4", "20261017-010914_backfill_rate_b16_tokensort_pgload"),
    ("DEVICE-FACTOR-B64",
     "device ms per padded token at batch 64",
     1.0206, 5e-5, B64P, "result.factors.device_ms_per_padded_token",
     "6.4", "20261012-132839_backfill_rate_b64_pgload"),
]

for cid, says, val, tol, src, field, paper, run in CLAIMS:
    obj = json.loads((EV / src).read_text(encoding="utf-8"))
    got = resolve(obj, field)
    assert abs(float(got) - float(val)) <= tol, (cid, got, val)
    r = subprocess.run(
        [PY, "-m", "thaw_audit", "add", "--id", cid, "--says", says,
         "--value", json.dumps(val), "--tol", json.dumps(tol),
         "--source", src, "--field", field, "--paper", paper,
         "--run-id", run, "--replace"],
        cwd=ROOT, capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, (cid, r.stdout, r.stderr)
    print(f"added {cid} = {got}")

ids = [c[0] for c in CLAIMS]
r = subprocess.run([PY, "-m", "thaw_audit", "freeze", *ids],
                   cwd=ROOT, capture_output=True, text=True, env=ENV)
assert r.returncode == 0, r.stdout
print(f"froze {len(ids)}")

PINS = {
    "1018:0.762": "LADDER-DPR-200", "1059:0.762": "LADDER-DPR-200",
    "1243:0.896": "CHAIN-DEEP-ALL-L4", "1291:0.896": "CHAIN-DEEP-ALL-L4",
    "1240:0.961": "CHAIN-DEEP-ORIG-L2",
    "1242:0.882": "CHAIN-DEEP-UNION-L4", "1290:0.882": "CHAIN-DEEP-UNION-L4",
}
DECLS = {
    "265:99": {"reason": "cited", "anchor": "residual MLP recovers",
               "why": "the 98--99% range another architectural family reports; "
                      "a citation, not a measurement of ours"},
    "1279:0.012": {"reason": "derived", "anchor": "+0.012",
                   "why": "0.9427 - 0.9307, the matched-cost margin between "
                          "CHAIN-DEEP-ALL-L4-K4000 and CHAIN-DEEP-LIT-K8000"},
    "1279:0.024": {"reason": "derived", "anchor": "margin reaches +0.024",
                   "why": "the same margin at the 60% reach setpoint of the "
                          "synthetic sweep"},
    "1281:0.004": {"reason": "derived", "anchor": "trails by 0.004",
                   "why": "the literal design's margin over the best union row "
                          "at that depth"},
    "1291:2.7": {"reason": "derived", "anchor": "2.7\\% more passes",
                 "why": "180.8 / 176.0 - 1, union-all's pass premium over "
                        "union-nearest at leg 4"},
    "1254:4.4": {"reason": "derived", "anchor": "4.4\\% more",
                 "why": "the deeper index's pass premium at the same depth"},
    "1406:0.003": {"reason": "derived", "anchor": "shift is $0.003",
                   "why": "the reached shift once the ledger depth is the one "
                          "actually loaded"},
}

PN = ROOT / "audit" / "paper_numbers.json"
d = json.loads(PN.read_text(encoding="utf-8"))
claims = {c["id"] for c in json.loads(
    (ROOT / "audit" / "claims.json").read_text(encoding="utf-8"))["claims"]}
for k, cid in PINS.items():
    assert cid in claims, k
    d[k] = {"claim": cid}
for k, v in DECLS.items():
    d[k] = v
PN.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len(PINS)} pins and {len(DECLS)} declarations written")
