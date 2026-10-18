"""Read an existing markdown claim table into the registry.

`ENGINEERING_MANUAL.md` already carries a claim-per-row table with ids like
`C028`, a status letter and a source file. This reads it rather than asking
anyone to retype it, which matters because retyping a provenance table is how a
provenance table stops matching its files.

The parser is deliberately conservative. It accepts a header row that mentions a
claim column and a status column, locates the other columns by name where it can
find them, and **skips any row it cannot read, reporting it**. Guessing at a row
would put a number into the ledger that nobody wrote, which is worse than
leaving it out: the ledger's whole value is that every row in it came from
somewhere.
"""
from __future__ import annotations

import re
from pathlib import Path

from .claims import Claim, STATUSES

#: column header -> claim field. Several spellings per field, because the
#: manual was written for people and its headers vary between tables.
HEADERS = {
    "claim": "id", "claim id": "id", "id": "id",
    "says": "says", "what it says": "says", "statement": "says",
    "number": "value", "value": "value", "数字": "value",
    "source": "source", "source file": "source", "file": "source", "源文件": "source",
    "field": "field", "path": "field",
    "run": "run_id", "run dir": "run_id", "run directory": "run_id", "run 目录": "run_id",
    "status": "status", "state": "status", "状态": "status",
    "paper": "paper", "where": "paper", "location": "paper",
    "note": "note", "notes": "note",
}
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _clean(s: str) -> str:
    """Strip the markdown a human table carries: code spans, bold, links."""
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    return s.replace("`", "").replace("**", "").replace("*", "").strip()


def parse_markdown(path: Path) -> tuple[list[dict], list[str]]:
    """Return (rows, skipped-reasons). Every table in the file is considered."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows, skipped, mapping = [], [], None
    for n, line in enumerate(lines, 1):
        if not line.strip().startswith("|"):
            mapping = None
            continue
        cells = _cells(line)
        if set("".join(cells)) <= set("-: "):          # the separator row
            continue
        low = [_clean(c).lower() for c in cells]
        if mapping is None:
            cand = {i: HEADERS[h] for i, h in enumerate(low) if h in HEADERS}
            if "id" in cand.values() and "status" in cand.values():
                mapping = cand
            continue
        row = {}
        for i, cell in enumerate(cells):
            key = mapping.get(i)
            if key:
                row[key] = _clean(cell)
        cid = row.get("id", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]{1,31}", cid):
            skipped.append(f"line {n}: {cid!r} is not a usable claim id")
            continue
        st = (row.get("status") or "").strip()[:1].upper()
        if st not in STATUSES:
            skipped.append(f"line {n}: claim {cid} has status {row.get('status')!r}, "
                           f"not one of {''.join(sorted(STATUSES))}")
            continue
        row["status"] = st
        rows.append(row)
    return rows, skipped


def import_markdown(path: Path, registry, replace: bool = False
                    ) -> tuple[int, list[str]]:
    rows, skipped = parse_markdown(path)
    added = 0
    for row in rows:
        value = None
        raw = row.get("value", "")
        hit = _NUM.search(raw) if raw else None
        if hit:
            value = float(hit.group()) if ("." in hit.group()
                                           or "e" in hit.group().lower()) \
                else int(hit.group())
        claim = Claim(
            id=row["id"], says=row.get("says", "") or raw,
            paper=row.get("paper", ""), value=value,
            source=row.get("source", ""), field=row.get("field", ""),
            run_id=row.get("run_id", ""), status=row["status"],
            note=(row.get("note", "") + (" | " if row.get("note") else "")
                  + f"imported from {Path(path).name}").strip(" |"))
        try:
            registry.add(claim, replace=replace)
            added += 1
        except SystemExit as e:
            skipped.append(f"{claim.id}: {e}")
    return added, skipped
