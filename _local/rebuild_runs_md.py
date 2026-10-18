"""Rebuild missing RUNS.md rows from each run directory's own env.json/log.txt.

A row is (started, name, tag, status, exit, dir, code_hash, paper). Everything
except `paper` is recorded per-run in env.json (started/run_name/tag/code_hash)
and log.txt (the `[exit N after M s]` line), so a row can be reconstructed
without re-running anything. Existing rows are preserved verbatim (their `paper`
column included); only runs with no row are appended, then the table is sorted
by start time.
"""
import json
import re
import sys
from pathlib import Path

results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
runs_dir = results / "runs"
reg = results / "RUNS.md"

header = ("| started | name | tag | status | exit | dir | code | paper |\n"
          "|---|---|---|---|---|---|---|---|\n")

rows = {}  # dir -> full row text (without trailing newline)
if reg.exists():
    for line in reg.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "started |" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7:
            rows[cells[5]] = line  # keyed on the dir column

added, skipped = [], []
for d in sorted(runs_dir.iterdir()):
    if not d.is_dir():
        continue
    rel = d.relative_to(results).as_posix()
    if rel in rows:
        continue
    env_p = d / "env.json"
    log_p = d / "log.txt"
    if not env_p.exists():
        skipped.append(f"{rel}: no env.json")
        continue
    env = json.loads(env_p.read_text(encoding="utf-8"))["environment"]
    started = env.get("started", "")
    name = env.get("run_name", "")
    tag = env.get("tag", "")
    code_hash = env.get("code_hash", "")
    code, has_manifest = 1, (d / "MANIFEST.sha256").exists()
    if log_p.exists():
        m = re.search(r"\[exit (\d+) after", log_p.read_text(encoding="utf-8", errors="replace"))
        if m:
            code = int(m.group(1))
    status = "ok" if code == 0 and has_manifest else "FAIL"
    row = (f"| {started} | {name} | {tag} | {status} | {code} | {rel} "
           f"| {code_hash} |  |")
    rows[rel] = row
    added.append(row)

def sort_key(r):
    cells = [c.strip() for c in r.strip().strip("|").split("|")]
    return cells[0]

body = "".join(r + "\n" for r in sorted(rows.values(), key=sort_key))
reg.write_text(header + body, encoding="utf-8")
print(f"rows now: {len(rows)} (added {len(added)}, skipped {len(skipped)})")
for r in added:
    print("  +", r)
for s in skipped:
    print("  ?", s)
