"""Resolve a dotted field path into a results JSON.

The syntax is the one `analysis/collect_evidence.py` already uses, deliberately:
a claim's `field` is copied between the two without translation, and the test
suite asserts the two resolvers agree on the paths the evidence bundle uses.

    frontier[N=8841823][kprime=500].recall_at_kprime
    baselines[scheme=lazy].recall_at_k
    result.penalty
    runs[0].wall_s

Consecutive `[k=v]` selectors are predicates on the SAME list -- they pick one
row, they do not select a row and then index into it.
"""
from __future__ import annotations


class FieldMissing(KeyError):
    """The path does not exist in this file. Distinguished from a wrong value:
    a missing field means the claim points at the wrong place, which is a
    different repair from a number that moved."""


def _match(rv, want) -> bool:
    if rv is None:
        return False
    try:
        return abs(float(rv) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return str(rv) == str(want)


def _split_parts(path: str) -> list[str]:
    """Split on dots that are OUTSIDE brackets.

    A plain `path.split(".")` breaks any selector whose value has a decimal
    point -- `ridge_sweep[ridge=0.0001]` becomes `ridge_sweep[ridge=0` and
    `0001]`, and the lookup fails with a confusing message about a row that
    does not exist. The paths the evidence bundle happens to use all select on
    integers, so this never showed; it is still wrong, and a ledger that cannot
    address a float-keyed row would push claims towards positional indices,
    which break whenever a list is reordered.
    """
    parts, buf, depth = [], [], 0
    for ch in path:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return [p for p in parts if p != ""]


def resolve(obj, path: str):
    cur = obj
    for part in _split_parts(path):
        if "[" not in part:
            if not isinstance(cur, dict) or part not in cur:
                raise FieldMissing(f"no key {part!r} at {path!r}")
            cur = cur[part]
            continue
        name, rest = part.split("[", 1)
        bits = rest.rstrip("]").split("][")
        keys = [b.split("=", 1) for b in bits if "=" in b]
        idxs = [int(b) for b in bits if "=" not in b]
        if name:
            if not isinstance(cur, dict) or name not in cur:
                raise FieldMissing(f"no key {name!r} at {path!r}")
            rows = cur[name]
        else:
            rows = cur
        if keys:
            hit = None
            for row in rows:
                if isinstance(row, dict) and all(
                        _match(row.get(k), v) for k, v in keys):
                    hit = row
                    break
            if hit is None:
                raise FieldMissing(f"no row matching [{rest} in {path!r}")
        else:
            hit = rows
        for i in idxs:
            try:
                hit = hit[i]
            except (IndexError, TypeError) as e:
                raise FieldMissing(f"index {i} out of range at {path!r}") from e
        cur = hit
    return cur
