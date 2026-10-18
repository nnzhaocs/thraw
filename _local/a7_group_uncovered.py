"""Group the paper's uncovered numbers into distinct registerable quantities.

For every uncovered (line, value) the coverage report's `suggest` names up to
four candidate evidence (source, field) pairs whose stored value rounds to the
printed number. Each candidate is resolved against its evidence file and the
grouped by (source, field, resolved value) -- one claim per group covers every
printing of that quantity in the paper.

The suggest docstring is explicit that this is a WORKLIST: a printed value can
collide with an unrelated field, so the printed line's context matters before a
candidate is registered. This script prints that context beside every group so
the confirmation is a reading, not a guess.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaw_audit.coverage import report as cov_report, suggest  # noqa: E402
from thaw_audit.paths import resolve  # noqa: E402

body = ROOT / "paper" / "body.tex"
sources = [body] + sorted(f for f in (ROOT / "paper").glob("*.tex")
                          if f.name not in {"body.tex", "main.tex"})
r = cov_report(sources, ROOT / "audit" / "claims.json",
               ROOT / "audit" / "paper_numbers.json")
unc = r["uncovered"]
print(f"uncovered: {len(unc)}")

sug = suggest(unc, ROOT / "evidence")

groups = {}
no_home = []
for row in sug:
    kept = None
    for c in row["candidates"]:
        src = ROOT / "evidence" / c["source"]
        try:
            v = resolve(json.loads(src.read_text(encoding="utf-8")), c["field"])
        except Exception:
            continue
        if isinstance(v, (int, float)) and \
                abs(float(v) - float(row["value"])) <= \
                max(abs(float(row["value"])) * 0.005, 0.005):
            kept = (c["source"], c["field"], float(v), c["value"])
            break
    if kept is None:
        no_home.append(row)
        continue
    key = (kept[0], kept[1], kept[2])
    groups.setdefault(key, {"lines": [], "context": row.get("context", "")})
    groups[key]["lines"].append(f"{Path(row['file']).name}:{row['line']}")

print(f"distinct quantities with an evidence home: {len(groups)}")
print(f"uncovered with NO usable home: {len(no_home)}")
print()
for (src, field, val), g in sorted(groups.items(), key=lambda kv: kv[0][0]):
    print(f"{val!r:<14} {src[:50]:<50} {field[:58]}")
    print(f"{'':<14} printed at {', '.join(g['lines'][:8])}"
          + (" ..." if len(g["lines"]) > 8 else ""))
    ctx = g["context"][:110]
    print(f"{'':<14} context: {ctx}")
print()
print("=== no usable home (need a declaration or a new measurement) ===")
for row in no_home:
    ctx = row.get("context", "")[:80]
    print(f"  {Path(row['file']).name}:{row['line']}  {row['value']}  {ctx}")
