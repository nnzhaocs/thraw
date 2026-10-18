"""A7 part 2: pin the ambiguous numbers, drop the stale declarations, declare
the design points, and lower the ratchet baseline to the new count.

A pin is {"claim": "ID"} keyed at "line:value". It resolves an ambiguous number
to the one claim that backs it, and makes deleting that claim a failure. A
declaration is {"reason": ..., "anchor": ..., "why": ...} keyed at "line:value"
(or by value alone with an anchor) for numbers that are not measurements.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PN = ROOT / "audit" / "paper_numbers.json"
d = json.loads(PN.read_text(encoding="utf-8"))

PINS = {
    # the MiniLM->BGE cross-space recall, printed at four precisions
    "109:0.929": "LADDER-MINILM-500", "789:0.929": "LADDER-MINILM-500",
    "1017:0.929": "LADDER-MINILM-500", "1031:0.929": "LADDER-MINILM-500",
    "1045:0.929": "LADDER-MINILM-500", "1058:0.929": "LADDER-MINILM-500",
    "1109:0.929": "LADDER-MINILM-500", "1604:0.929": "LADDER-MINILM-500",
    "1068:0.9295": "LADDER-MINILM-500", "1139:0.9295": "LADDER-MINILM-500",
    "1163:0.9295": "LADDER-MINILM-500", "1572:0.9295": "LADDER-MINILM-500",
    # tab:distance cells
    "1017:0.953": "DIST-MINILM-1000", "1018:0.930": "DIST-DPR-2000",
    "1040:0.930": "DIST-DPR-2000", "1019:0.942": "DIST-FIQA-500",
    "1032:0.942": "DIST-FIQA-500",
    # tab:ladder cells
    "1095:0.895": "LADDER-2M-200", "1095:0.931": "LADDER-8M-200",
    "1096:0.936": "LADDER-2M-500", "1096:0.962": "LADDER-8M-R",
    "1184:0.962": "STEP2-DECILE-MAX",
    # the PG offline exact recall, printed in body and appendix
    "1175:0.9364": "CONC-RECALL", "1181:0.9364": "CONC-RECALL",
    "86:0.9364": "CONC-RECALL",
    # tab:chain cells that several claims share a value with
    "1240:0.895": "CHAIN-DEEP-ORIG-L3", "1243:0.931": "CHAIN-DEEP-ALL-L3",
    "1278:0.931": "CHAIN-DEEP-LIT-K8000", "1280:0.931": "CHAIN-DEEP-LIT-K8000",
    # tab:compare cells
    "1538:0.936": "CMP-BF-RANDOM", "1597:0.936": "CMP-BF-RANDOM",
    "1542:0.931": "CMP-LAZY", "1565:0.931": "CMP-LAZY", "1606:0.931": "CMP-LAZY",
}

DECLARATIONS = {
    # reach setpoints of the synthetic sweep, and the adapter's fit fraction:
    # design choices, not measurements
    "1279:60": {"reason": "design", "anchor": "At 60\\% reach",
                "why": "the sweep's reach setpoint; the margin at each setpoint "
                       "is the measured quantity, the setpoint is not"},
    "1494:60": {"reason": "derived", "anchor": "saves about 60",
                "why": "closed form at the x = 0.5 stated in the same sentence; "
                       "illustrative, not measured"},
    "1602:4": {"reason": "design", "anchor": "Fitted on 4",
               "why": "the query-side adapter's fit fraction, a design choice"},
}

# stale: the paper moved these numbers; the pins above replace them
for k in ("1132:0.9364", "135:0.929"):
    if k in d:
        del d[k]
        print(f"dropped stale declaration {k}")

for k, cid in PINS.items():
    d[k] = {"claim": cid}
print(f"{len(PINS)} pins written")
for k, v in DECLARATIONS.items():
    d[k] = v
print(f"{len(DECLARATIONS)} declarations written")

# every pin must name a claim that exists and still matches, checked by the
# coverage tool itself below -- this only guards the obvious typo
claims = {c["id"] for c in json.loads(
    (ROOT / "audit" / "claims.json").read_text(encoding="utf-8"))["claims"]}
for k, cid in PINS.items():
    if cid not in claims:
        sys.exit(f"pin {k} names unknown claim {cid}")

PN.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("paper_numbers.json rewritten")
