"""CIPBench loader.

Reads the `bench_all_clean` layout. Per repository:

    units.jsonl        unit_id, text, repo, lang, base_commit, n_versions
    updates.jsonl      unit_id, old_text, new_text, update_type, commit_idx, ...
    query_index.jsonl  query_id, query, issue_number, link_type, source_type
    gt_by_qid.json     {query_id: [unit_id, ...]}

In the migration work CIPBench has a narrow role: it is the negative result.
Its corpora run 23 to 678 units with roughly as many distinct queries as
units (pooled: N = 2,366, D = 2,378), so D*K'/N is well above 1 at any useful
K' and deferred re-embedding saves nothing -- which `analysis/coverage.py`
and the pooled `migration` run both demonstrate. It is also the realism check
on update streams for the incremental setting, which lives in the archive.

The stream-building half of the old adapter (interleaving updates and queries
into a replay workload) belonged to the read-repair engine and moved to the
archive with it.

    python -m semview.cipbench check bench_all_clean
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPOS = ("aiohttp", "click", "cobra", "echo", "express", "flask", "gin",
         "lodash", "pylint", "requests")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def load_repo(root: str | Path) -> dict:
    """Read one repository directory into plain records."""
    d = Path(root)
    return dict(
        name=d.name,
        units=_jsonl(d / "units.jsonl"),
        updates=_jsonl(d / "updates.jsonl"),
        queries=_jsonl(d / "query_index.jsonl"),
        gt=json.loads((d / "gt_by_qid.json").read_text(encoding="utf-8")),
    )


def ordered_updates(updates: list[dict]) -> list[dict]:
    """Updates in stream order. (commit_idx, to_date) is the order under which
    each unit's first update has old_text equal to its units.jsonl text."""
    return sorted(updates,
                  key=lambda x: (x.get("commit_idx", 0), x.get("to_date", "")))


def check(root: str | Path) -> dict:
    """Verify the properties the loaders depend on, for one repository."""
    r = load_repo(root)
    U = {u["unit_id"]: u["text"] for u in r["units"]}
    ups = ordered_updates(r["updates"])
    first: dict[str, dict] = {}
    for u in ups:
        first.setdefault(u["unit_id"], u)
    out = dict(
        repo=r["name"],
        units=len(U),
        updates=len(ups),
        queries=len(r["queries"]),
        orphan_updates=sum(1 for u in ups if u["unit_id"] not in U),
        first_old_text_mismatch=sum(
            1 for k, u in first.items() if k in U and u["old_text"] != U[k]),
        queries_without_gt=sum(1 for q in r["queries"]
                               if q["query_id"] not in r["gt"]),
        link_type=dict(Counter(q.get("link_type", "?") for q in r["queries"])),
    )
    out["ok"] = (out["orphan_updates"] == 0 and out["first_old_text_mismatch"] == 0
                 and out["queries_without_gt"] == 0)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="semview.cipbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("check", help="validate a repo, or every repo under a root")
    p0.add_argument("root")
    a = ap.parse_args()

    root = Path(a.root)
    dirs = ([root] if (root / "units.jsonl").exists()
            else [d for d in sorted(root.iterdir())
                  if d.is_dir() and (d / "units.jsonl").exists()])
    print(f"{'repo':<10}{'units':>7}{'upds':>7}{'qrys':>6}{'orph':>6}"
          f"{'mism':>6}{'nogt':>6}  link_type")
    bad = 0
    for d in dirs:
        c = check(d)
        bad += int(not c["ok"])
        print(f"{c['repo']:<10}{c['units']:>7}{c['updates']:>7}{c['queries']:>6}"
              f"{c['orphan_updates']:>6}{c['first_old_text_mismatch']:>6}"
              f"{c['queries_without_gt']:>6}  {c['link_type']}"
              f"  {'OK' if c['ok'] else 'PROBLEM'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
