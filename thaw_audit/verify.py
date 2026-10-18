"""The audit pass. Reads files, compares fields, executes nothing.

The order of the checks is the whole design, because it is what bounds the work:

  1. **Frozen and the hash matches** -> settled. Return immediately. No field is
     read, no number recomputed, no command run. This is the stop condition, and
     it is the first branch so that a settled claim costs one cached hash.
  2. Frozen and the hash does NOT match -> `S`, loudly. A file changed under a
     number the paper states; a person decides whether the paper or the file is
     wrong. The tool never resolves this by re-measuring.
  3. Not frozen, source missing -> `X`.
  4. Not frozen, no run directory -> `P`, and the field is still read, because
     knowing the value is right is useful even when its provenance is not.
  5. Not frozen, field missing -> `S` with "field not found", which is a
     different repair from a value that moved: the claim points at the wrong
     place.
  6. Otherwise compare within the claim's tolerance -> `V` or `S`.

Nothing here writes to the registry. `verify_all` returns findings; the caller
decides whether to record them, and in AUDIT mode it may only write the report.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path

from .claims import Claim
from .hashes import sha256_cached
from .modes import Mode, require
from .paths import FieldMissing, resolve


@dataclass
class Finding:
    claim_id: str
    status: str                 # the status the evidence supports
    settled: bool               # frozen and intact: nothing was recomputed
    message: str
    read_value: object = None
    source_path: str = ""
    sha256: str = ""

    @property
    def changed(self) -> bool:
        return not self.settled


def _find_source(source: str, roots: list[Path]) -> Path | None:
    """Locate a source file by its ledger path, then by basename.

    The same run name can appear under several archive roots; the shallowest
    copy wins, which is the convention `collect_evidence.py` already uses.
    """
    want = source.replace("\\", "/")
    for root in roots:
        cand = root / Path(source)
        if cand.is_file():
            return cand
        hits = [p for p in root.rglob(Path(source).name)
                if str(p).replace("\\", "/").endswith(want)]
        if hits:
            return sorted(hits, key=lambda p: len(str(p)))[0]
    return None


def verify_claim(claim: Claim, roots: list[Path], use_cache: bool = True,
                 mode=None) -> Finding:
    # Bind the resolved mode. `require` RETURNS it, and discarding the return
    # value left `mode` as None on every call that did not pass one -- which is
    # every call from the CLI. `None is not Mode.AUDIT` is True, so
    # `write_cache` came out True and an audit pass wrote its hash cache into
    # the runs root, against this package's one standing promise.
    #
    # It was masked, not absent: `use_cache and not claim.frozen` is False
    # while every claim is frozen, so nothing reached the cache. One `thaw`
    # would have uncovered it. The existing test passed `mode=Mode.AUDIT`
    # explicitly and so never exercised the path the CLI actually takes.
    mode = require("read", mode)
    if claim.status == "A":
        return Finding(claim.id, "A", True,
                       "an argument; there is no file to check")

    src = _find_source(claim.source, roots) if claim.source else None
    if src is None:
        if claim.frozen:
            return Finding(
                claim.id, "S", False,
                f"FROZEN but its source is gone: {claim.source!r}. A settled "
                f"number must keep the file it was settled against.")
        return Finding(claim.id, "X", False,
                       f"source not found in any runs root: {claim.source!r}")

    # A frozen claim's hash IS the check, so it is computed from the bytes
    # whatever the caller asked for: a (size, mtime) cache entry can be
    # stale after an in-place edit of the same length, and then a settled
    # number verifies against a file that changed under it. `write_cache`
    # is off in a read-only mode so an audit pass leaves the tree alone.
    digest = sha256_cached(src, root=roots[0],
                           use_cache=use_cache and not claim.frozen,
                           write_cache=(mode is not Mode.AUDIT))

    # (1) and (2): the only two branches a frozen claim can take.
    if claim.frozen:
        if claim.sha256 and digest == claim.sha256:
            return Finding(claim.id, claim.status, True,
                           "frozen and intact", source_path=str(src),
                           sha256=digest)
        if not claim.sha256:
            return Finding(
                claim.id, "P", False,
                "frozen with no recorded hash, so there is nothing to compare; "
                "re-freeze it to record one", source_path=str(src),
                sha256=digest)
        return Finding(
            claim.id, "S", False,
            f"FROZEN and its source CHANGED: recorded {claim.sha256[:12]}, "
            f"found {digest[:12]}. Decide whether the paper or the file is "
            f"wrong -- this tool will not re-measure to settle it.",
            source_path=str(src), sha256=digest)

    try:
        obj = json.loads(src.read_text(encoding="utf-8"))
    except ValueError as e:
        return Finding(claim.id, "X", False, f"source is not valid JSON: {e}",
                       source_path=str(src), sha256=digest)
    try:
        got = resolve(obj, claim.field)
    except FieldMissing as e:
        return Finding(claim.id, "S", False,
                       f"field not found, so the claim points at the wrong "
                       f"place: {e}", source_path=str(src), sha256=digest)

    if claim.value is None:
        return Finding(claim.id, "M", False,
                       f"no value recorded in the ledger; the file says {got!r}",
                       read_value=got, source_path=str(src), sha256=digest)

    try:
        ok = abs(float(got) - float(claim.value)) <= float(claim.tol)
        detail = f"{got} vs {claim.value} (tol {claim.tol})"
    except (TypeError, ValueError):
        ok = str(got) == str(claim.value)
        detail = f"{got!r} vs {claim.value!r}"

    if not ok:
        return Finding(claim.id, "S", False, f"source disagrees: {detail}",
                       read_value=got, source_path=str(src), sha256=digest)
    status = "V" if claim.run_id else "P"
    msg = "matches" if claim.run_id else "matches, but has no run directory"
    return Finding(claim.id, status, False, msg, read_value=got,
                   source_path=str(src), sha256=digest)


def verify_all(registry, roots: list[Path], use_cache: bool = True,
               only: list[str] | None = None, mode=None) -> list[Finding]:
    mode = require("read", mode)   # bind it, then hand it down (see verify_claim)
    claims = registry.claims
    if only:
        wanted = set(only)
        claims = [c for c in claims if c.id in wanted]
    return [verify_claim(c, roots, use_cache=use_cache, mode=mode)
            for c in sorted(claims, key=lambda c: c.id)]


def report_markdown(registry, findings: list[Finding]) -> str:
    """The audit report, as the one table a reader should look at."""
    by_id = {f.claim_id: f for f in findings}
    settled = sum(1 for f in findings if f.settled)
    blocking = [f for f in findings if f.status in {"X", "P", "S"}]
    lines = [
        "# Audit report",
        "",
        f"As of {_dt.date.today().isoformat()}. "
        f"{len(findings)} claims checked, **{settled} settled without "
        f"recomputation**, {len(blocking)} blocking submission.",
        "",
        "`V` verified · `M` mapped · `A` argument · "
        "`X` no source · `P` no provenance · `S` stale. "
        "Only V, M and A may appear in a submitted version.",
        "",
        "| claim | paper | status | settled | what the check found |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(registry.claims, key=lambda c: c.id):
        f = by_id.get(c.id)
        if f is None:
            continue
        lines.append(
            f"| `{c.id}` | {c.paper or '—'} | {f.status} | "
            f"{'yes' if f.settled else 'no'} | {f.message} |")
    if blocking:
        lines += ["", "## Blocking", ""]
        for f in sorted(blocking, key=lambda f: f.claim_id):
            c = registry.get(f.claim_id)
            lines.append(f"- **`{f.claim_id}`** ({f.status}) — {c.says}. "
                         f"{f.message}")
    return "\n".join(lines) + "\n"
