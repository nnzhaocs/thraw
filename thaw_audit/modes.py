"""Three modes, so that an audit cannot become a measurement by accident.

The failure this prevents is specific. An auditor checking a number finds no
run directory for it, decides the responsible thing is to re-measure, gets a
value that differs from both the paper and the previous re-measurement, and now
has three numbers where it had one. Nothing in that sequence is unreasonable;
the problem is that the task changed shape halfway through without anyone
deciding it should.

So the mode is declared before the work starts and enforced:

    AUDIT       read files, compare fields, write the audit report. Executes
                nothing and never re-measures. Cannot change a claim's value.
    MEASURE     may run a command, but through `semview.runlog`, so it gets a
                run directory before it produces a number.
    IMPLEMENT   may change code and tests. Cannot write to the ledger at all,
                because a code change is not evidence.

`require()` is called by the operations themselves rather than trusted to the
caller, so a mode cannot be bypassed by reaching past the CLI.
"""
from __future__ import annotations

import enum
import os
from pathlib import Path


class Mode(enum.Enum):
    AUDIT = "audit"
    MEASURE = "measure"
    IMPLEMENT = "implement"

    @classmethod
    def current(cls) -> "Mode":
        """The mode from THAW_MODE, defaulting to the most restrictive one.

        Defaulting to AUDIT is deliberate: a tool invoked with no mode set
        should be the one that cannot change anything.
        """
        raw = os.environ.get("THAW_MODE", "audit").strip().lower()
        try:
            return cls(raw)
        except ValueError:
            raise SystemExit(
                f"THAW_MODE={raw!r} is not one of "
                f"{', '.join(m.value for m in cls)}") from None


class ModeError(SystemExit):
    """Raised instead of doing the forbidden thing, with the fix in the text."""


#: what each capability needs. `freeze` is under MEASURE rather than AUDIT on
#: purpose: settling a number is a statement about evidence, and the person
#: making it should be the one who has the evidence in hand.
CAPABILITIES = {
    "read": {Mode.AUDIT, Mode.MEASURE, Mode.IMPLEMENT},
    "write_report": {Mode.AUDIT, Mode.MEASURE},
    "run_command": {Mode.MEASURE},
    "write_claim": {Mode.MEASURE},
    "freeze": {Mode.MEASURE},
    "write_code": {Mode.IMPLEMENT},
}


def require(capability: str, mode: Mode | None = None) -> Mode:
    mode = mode or Mode.current()
    allowed = CAPABILITIES.get(capability)
    if allowed is None:
        raise ModeError(f"unknown capability {capability!r}")
    if mode not in allowed:
        want = " or ".join(sorted(m.value for m in allowed))
        raise ModeError(
            f"{capability!r} needs THAW_MODE={want}; it is {mode.value}.\n"
            f"If that is really what you want, set it explicitly:\n"
            f"    THAW_MODE={want} python -m thaw_audit ...\n"
            f"The mode is not inferred, so that an audit cannot turn into a "
            f"measurement without someone saying so.")
    return mode


# ------------------------------------------------------- definition of done ---
def _find(runs_root, rel: str):
    """The first root that holds `rel`, or None.

    `runs_root` used to be one directory -- `_roots_for(...)[0]`, which is
    `evidence/`. A definition of done naming a file that landed in `results/`
    then read as "missing file", which is the same wrong answer the runs-root
    guard exists to prevent, one level down. Every root is searched now.
    """
    roots = runs_root if isinstance(runs_root, (list, tuple)) else [runs_root]
    for r in roots:
        p = Path(r) / rel
        if p.is_file():
            return p
    return None


def check_dod(item: dict, runs_root, registry) -> list[str]:
    """Evaluate one open item's definition of done. Returns the unmet parts.

    An item with no `dod` is reported as unmet with a reason, rather than
    silently passing: an open item nobody can finish is worse than one nobody
    has started, because it looks finishable.
    """

    import json as _json

    unmet = []
    checks = item.get("dod") or []
    if not checks:
        return ["no dod: this item cannot be finished because nothing says when "
                "it is done"]
    for chk in checks:
        kind = chk.get("kind")
        if kind == "file_exists":
            if _find(runs_root, chk["path"]) is None:
                unmet.append(f"missing file: {chk['path']}")
        elif kind == "json_has_field":
            p = _find(runs_root, chk["path"])
            if p is None:
                unmet.append(f"missing file: {chk['path']}")
                continue
            from .paths import FieldMissing, resolve
            try:
                resolve(_json.loads(p.read_text(encoding="utf-8")), chk["field"])
            except (FieldMissing, ValueError):
                unmet.append(f"{chk['path']} has no field {chk['field']}")
        elif kind == "claim_status":
            try:
                c = registry.get(chk["id"])
            except KeyError:
                unmet.append(f"no such claim: {chk['id']}")
                continue
            # An unrecognised key is refused rather than ignored. Eleven dod
            # checks were written with `"status": "V"` instead of
            # `"in": ["V"]`; the default made them accidentally equivalent, so
            # they looked wired and were not -- change `status` to `["V","M"]`
            # and nothing would happen. That is this project's recurring defect
            # class, and the cure is to make a silent default impossible.
            unknown = set(chk) - {"kind", "id", "in", "why"}
            if unknown:
                unmet.append(
                    f"dod check for {chk['id']} has unrecognised key(s) "
                    f"{sorted(unknown)}; claim_status reads `in`, so this "
                    "criterion is not doing what it appears to")
                continue
            want = chk.get("in", ["V"])
            if c.status not in want:
                unmet.append(f"claim {c.id} is {c.status}, wanted one of {want}")
        elif kind == "claim_frozen":
            try:
                c = registry.get(chk["id"])
            except KeyError:
                unmet.append(f"no such claim: {chk['id']}")
                continue
            if not c.frozen:
                unmet.append(f"claim {c.id} is not frozen")
        elif kind == "monotone":
            p = _find(runs_root, chk["path"])
            if p is None:
                unmet.append(f"missing file: {chk['path']}")
                continue
            from .paths import FieldMissing, resolve
            try:
                obj = resolve(_json.loads(p.read_text(encoding="utf-8")),
                              chk["field"])
            except (FieldMissing, ValueError):
                unmet.append(f"{chk['path']} has no field {chk['field']}")
                continue
            key = chk.get("key")
            # a curve stored as {"8": {...}, "16": {...}} keyed by batch size
            try:
                xs = sorted((float(k), v) for k, v in obj.items())
                ys = [float(v[key]) if key else float(v) for _, v in xs]
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                unmet.append(f"{chk['path']}:{chk['field']} is not a curve ({e})")
                continue
            direction = chk.get("direction", "any")
            tol = float(chk.get("tol", 0.0))
            # A shape check on one point is not a shape check. Every direction
            # here is vacuously true for a curve of length 1 -- there is no
            # pair to compare and no triple to turn -- so D-01, whose whole
            # purpose is to replace a five-point curve with a denser one, could
            # be closed by a file containing a single batch size. `min_points`
            # is what the item asks for, and `keys` pins WHICH points, since
            # ten samples clustered at the top would also pass a count.
            need = int(chk.get("min_points", 3))
            if len(ys) < need:
                unmet.append(
                    f"{chk['field']} has {len(ys)} point(s), and a "
                    f"{direction!r} shape needs at least {need}: "
                    f"{sorted(k for k, _ in xs)}")
                continue
            want_keys = chk.get("keys")
            if want_keys:
                have = {float(k) for k, _ in xs}
                missing = [k for k in want_keys if float(k) not in have]
                if missing:
                    unmet.append(f"{chk['field']} is missing the required "
                                 f"samples {missing}")
                    continue
            lo = all(b >= a for a, b in zip(ys, ys[1:]))
            hi = all(b <= a for a, b in zip(ys, ys[1:]))
            # A per-item throughput curve is NOT monotone and should not be
            # asked to be: it falls while per-call overhead amortises and rises
            # once padding waste takes over, so a correct curve has exactly one
            # interior minimum. Requiring monotonicity made the D-01 criterion
            # unsatisfiable -- it would have rejected the very curve item T is
            # meant to produce. `unimodal` allows at most one change of
            # direction; `tol` ignores wiggles smaller than that fraction of
            # the local magnitude, so measurement noise is not a turning point.
            turns = 0
            for x, y, z in zip(ys, ys[1:], ys[2:]):
                d1, d2 = y - x, z - y
                scale = max(abs(x), abs(y), abs(z), 1e-12)
                if abs(d1) <= tol * scale or abs(d2) <= tol * scale:
                    continue
                if (d1 > 0) != (d2 > 0):
                    turns += 1
            ok = {"up": lo, "down": hi, "any": lo or hi,
                  "unimodal": turns <= 1}[direction]
            if not ok:
                detail = (f"has {turns} turning points" if direction == "unimodal"
                          else f"is not monotone ({direction})")
                unmet.append(f"{chk['field']} {detail}: {ys}")
        else:
            unmet.append(f"unknown dod check kind {kind!r}")
    # several checks can fail for one reason -- three fields of one missing file
    # is one problem, and listing it three times reads as three
    seen, out = set(), []
    for u in unmet:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out
