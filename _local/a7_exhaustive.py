"""Exhaustive search: for each still-uncovered number, walk EVERY evidence file
and list every field whose value rounds to it at the paper's precision. The
coverage tool's `suggest` caps at four candidates; this does not, so a number
whose home is the fifth match is still found. Output is a review list, not a
registration -- a value can collide with an unrelated field.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from thaw_audit.coverage import report as cov_report  # noqa: E402
from thaw_audit.coverage import _candidates, _matches, _walk  # noqa: E402

body = ROOT / "paper" / "body.tex"
sources = [body] + sorted(f for f in (ROOT / "paper").glob("*.tex")
                          if f.name not in {"body.tex", "main.tex"})
r = cov_report(sources, ROOT / "audit" / "claims.json",
               ROOT / "audit" / "paper_numbers.json")
unc = [x for x in r["uncovered"]]

flat = []
for f in sorted((ROOT / "evidence").glob("*.json")):
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    flat.extend((f.name, p, v) for p, v in _walk(obj))
print(f"uncovered: {len(unc)}; evidence numeric leaves: {len(flat)}")

found, none = [], []
for row in unc:
    hits = []
    for probe, pdec, pf in _candidates(row["value"], row.get("is_percent", False)):
        for fn, path, ev in flat:
            if _matches(probe, pdec, ev, pct_form=pf):
                hits.append((fn, path, ev))
        if hits:
            break
    if hits:
        found.append((row, hits))
    else:
        none.append(row)

print(f"with at least one exact home: {len(found)}")
print(f"with NO home anywhere in the bundle: {len(none)}")
print()
for row, hits in found:
    print(f"{row['file']}:{row['line']}  {row['value']}")
    print(f"    {row['context'][:100]}")
    for fn, path, ev in hits[:4]:
        print(f"    -> {fn[:58]}  {path[:52]} = {ev!r}")
print()
print("=== NO home anywhere ===")
for row in none:
    print(f"  {row['file']}:{row['line']}  {row['value']}  {row['context'][:85]}")
