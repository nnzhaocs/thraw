"""Hash a file once, and only again when the file changes.

Recomputing a manifest over a results tree on every audit is the other half of
the loop this package exists to break: it is slow, so it gets skipped, and then
nobody knows whether the hashes are current. Caching on (size, mtime_ns) makes
it cheap enough to run every time, which is what keeps it honest.

**The cache is not trusted where the hash is the decision.** A frozen claim is
settled by comparing its source's hash against the recorded one, and that is
the only check standing between a settled number and a changed file -- so
`verify` reads the bytes for frozen claims and `freeze` reads them before
recording, both with `use_cache=False`. An earlier version of this docstring
argued that a stale entry needed size AND mtime to match and so meant a
"deliberate overwrite rather than an accident"; that is true and it is beside
the point. An editor that rewrites a file in place, a `git checkout` of a
same-length line, or `touch -r` after an edit all reproduce it, and a reviewer
did reproduce it. Where the hash decides, it is computed.

So the cache is for the bulk paths that re-hash a whole results tree to see
what moved, and `use_cache=False` is the default for anything a claim rests on.

It also does not write where the caller is only reading: `write_cache=False`
keeps an audit-mode pass from leaving a file in the evidence bundle, which is
the one directory that has to stay as it was shipped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_NAME = ".audit_hash_cache.json"


def _read_cache(root: Path) -> dict:
    p = root / CACHE_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}                      # a corrupt cache is not an error; rebuild


def _write_cache(root: Path, cache: dict) -> None:
    try:
        (root / CACHE_NAME).write_text(
            json.dumps(cache, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass                           # a read-only tree still audits fine


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_cached(path: Path, root: Path | None = None,
                  use_cache: bool = True, write_cache: bool = True) -> str:
    """The file's sha256, from the cache when `use_cache` and it is current.

    `use_cache=False` reads the bytes: correct, and what every caller whose
    decision rests on the hash passes. `write_cache=False` reads the cache but
    does not create or update it, for a pass that must not touch the tree.
    """
    path = Path(path)
    st = path.stat()
    key = str(path.resolve())
    root = Path(root) if root is not None else path.parent
    if not use_cache:
        return sha256_file(path)
    cache = _read_cache(root)
    hit = cache.get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime_ns:
        return hit["sha256"]
    digest = sha256_file(path)
    if write_cache:
        cache[key] = dict(size=st.st_size, mtime=st.st_mtime_ns, sha256=digest)
        _write_cache(root, cache)
    return digest
