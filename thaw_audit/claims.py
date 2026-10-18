"""What the paper says, where it comes from, and whether that is settled.

One claim is one number (or one string) the paper states, together with the
file and field it was read from. The registry is JSON rather than code so that
it diffs, reviews and merges like the results it describes.

The status vocabulary is the one the engineering manual already uses, and the
submission rule is unchanged: **only `V`, `M` and `A` may appear in a submitted
version.** `X`, `P` and `S` are the work list.

    V  verified: the field was re-read from the source and matches
    M  mapped: a source file and a run directory exist; the field was not re-read
    A  argument: a statement that rests on reasoning, not on a measurement
    X  no source: the file the number came from is not in the tree
    P  no provenance: a source file, but no run directory and no ledger line
    S  stale: the source says something different from the paper

`lesson` is the one field nothing can generate: what the run that produced this
number taught us about the system, as against which table it landed in. It is
worth more than the `paper` column six months later, because it says why a run
was worth doing rather than where its output went.

`frozen` is orthogonal to status and is the thing that stops an audit loop. A
frozen claim records its source's sha256, and verifying it is a hash
comparison. Freezing a claim that is not `V` is refused: there is no sense in
settling a number that has never been checked against its file.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field as _field
from pathlib import Path

STATUSES = {
    "V": "verified field-by-field against the source",
    "M": "source file and run directory exist; field not re-read",
    "A": "an argument, not a measurement",
    "X": "the source file is missing from the tree",
    "P": "a source file, but no run directory or ledger line",
    "S": "the source disagrees with the paper",
}
SUBMITTABLE = {"V", "M", "A"}


@dataclass
class Claim:
    id: str
    says: str                      # what the paper asserts, in words
    paper: str = ""                # where it appears: "tab:distance", "6.4"
    value: object = None           # the number the paper states
    tol: float = 5e-5
    source: str = ""               # path, relative to a runs root
    field: str = ""                # dotted path into that JSON
    run_id: str = ""               # the run directory that produced it
    sha256: str = ""               # the source's hash when frozen
    status: str = "X"
    frozen: bool = False
    note: str = ""
    lesson: str = ""               # what the run that produced it taught us
    checked: str = ""              # ISO date of the last verification

    def submittable(self) -> bool:
        return self.status in SUBMITTABLE

    def needs_work(self) -> bool:
        return not self.submittable()


class Registry:
    """The claims file. Loading never fails on a missing file: an empty
    registry is a legitimate starting state, and refusing to start is how a
    tool ends up bypassed."""

    SCHEMA = 1

    def __init__(self, path: Path, claims: list[Claim] | None = None,
                 updated: str = ""):
        self.path = Path(path)
        self.claims = claims or []
        self.updated = updated

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path) -> "Registry":
        path = Path(path)
        if not path.is_file():
            return cls(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != cls.SCHEMA:
            raise SystemExit(
                f"{path}: claims schema is {raw.get('schema')!r}, this tool "
                f"writes {cls.SCHEMA}. Refusing to read it rather than guess at "
                f"the field names.")
        known = {f for f in Claim.__dataclass_fields__}
        out = []
        for c in raw.get("claims", []):
            unknown = set(c) - known
            if unknown:
                raise SystemExit(
                    f"{path}: claim {c.get('id')!r} has unknown field(s) "
                    f"{sorted(unknown)}. A typo in a claims file would silently "
                    f"drop the field, so it is refused instead.")
            out.append(Claim(**c))
        ids = [c.id for c in out]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise SystemExit(f"{path}: duplicate claim id(s) {sorted(dupes)}")
        return cls(path, out, raw.get("updated", ""))

    def save(self) -> None:
        self.updated = _dt.date.today().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = dict(schema=self.SCHEMA, updated=self.updated,
                    claims=[asdict(c) for c in
                            sorted(self.claims, key=lambda c: c.id)])
        self.path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")

    # --------------------------------------------------------------- access
    def get(self, claim_id: str) -> Claim:
        for c in self.claims:
            if c.id == claim_id:
                return c
        raise KeyError(claim_id)

    def add(self, claim: Claim, replace: bool = False) -> None:
        try:
            existing = self.get(claim.id)
        except KeyError:
            self.claims.append(claim)
            return
        if not replace:
            raise SystemExit(
                f"claim {claim.id!r} already exists. Pass --replace to overwrite "
                f"it, so an accidental re-add cannot quietly change a recorded "
                f"number.")
        # Refuse ANY replacement of a frozen claim, not just one that would
        # unfreeze it. Replacing frozen with frozen used to be silent, which
        # meant a re-add could change a settled number's value or its recorded
        # hash without anyone saying so -- exactly the thing freezing exists to
        # prevent. Thawing is the documented path, and it demands a reason.
        if existing.frozen:
            raise SystemExit(
                f"claim {claim.id!r} is frozen and will not be replaced. Thaw "
                f"it first and say why:\n"
                f"    THAW_MODE=measure python -m thaw_audit thaw {claim.id} "
                f"--why '<what changed>'\n"
                f"A frozen number is one the paper already states.")
        self.claims[self.claims.index(existing)] = claim

    def counts(self) -> dict:
        out = {k: 0 for k in STATUSES}
        for c in self.claims:
            out[c.status] = out.get(c.status, 0) + 1
        out["frozen"] = sum(1 for c in self.claims if c.frozen)
        return out

    def blocking(self) -> list[Claim]:
        """Claims that may not appear in a submitted version."""
        return [c for c in sorted(self.claims, key=lambda c: c.id)
                if c.needs_work()]
