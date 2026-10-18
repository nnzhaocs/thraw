"""Write results files so that a lost or empty file is caught by the tool, not
by a person weeks later.

    write_results(path, obj)   atomic write, read back, validate, sha256 sidecar;
                               exits non-zero if anything is wrong
    environment(dtype=...)     what produced the file: versions, GPU, precision,
                               host, command line

and a checker for copies:

    python -m semview.results_io check results/*.json
    python -m semview.results_io manifest results/*.json > MANIFEST.sha256
    python -m semview.results_io verify MANIFEST.sha256      # after copying

`check` fails on an empty file, invalid JSON, a missing provenance block, or a
sidecar checksum that doesn't match. Run `manifest` on the machine that made
the files and `verify` on the machine they were copied to, before releasing
the first one. The 8.8M JSON was lost because the copy was 0 bytes and nobody
looked.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

# A results file must carry at least one of these; which one depends on the tool.
PROVENANCE_KEYS = ("fingerprint", "environment", "provenance")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment(dtype: str | None = None, device: str | None = None) -> dict:
    """Everything about the run that a reader would need to tell two results
    files apart. Never raises: a missing package is recorded as None."""
    env = dict(python=platform.python_version(), platform=platform.platform(),
               host=socket.gethostname(), argv=list(sys.argv), dtype=dtype,
               device=device)
    for mod in ("numpy", "torch", "sentence_transformers", "transformers",
                "faiss", "psycopg"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            env[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return env


def validate(obj, required=PROVENANCE_KEYS) -> list[str]:
    """Problems with a parsed results object; empty if it is fine."""
    problems = []
    if not isinstance(obj, dict) or not obj:
        return ["not a non-empty JSON object"]
    if required and not any(k in obj and obj[k] for k in required):
        problems.append(f"no provenance block (one of {', '.join(required)})")
    return problems


def write_results(path, obj, required=PROVENANCE_KEYS, indent: int = 2) -> str:
    """Write atomically, read back, validate, write `<path>.sha256`. Returns the
    checksum. Exits with status 3 if the file on disk is not what was meant."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    problems = validate(obj, required)
    if problems:
        raise SystemExit(f"refusing to write {path}: " + "; ".join(problems))
    text = json.dumps(obj, indent=indent)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        back = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {path} does not read back as JSON: {e}", file=sys.stderr)
        raise SystemExit(3)
    if path.stat().st_size == 0 or back != json.loads(text):
        print(f"ERROR: {path} on disk differs from what was written", file=sys.stderr)
        raise SystemExit(3)
    digest = sha256_file(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes, sha256 {digest[:12]})")
    return digest


def check_file(path, required=PROVENANCE_KEYS) -> list[str]:
    path = Path(path)
    if not path.exists():
        return ["missing"]
    if path.stat().st_size == 0:
        return ["empty (0 bytes)"]
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"not valid JSON: {e}"]
    problems = validate(obj, required)
    side = Path(str(path) + ".sha256")
    if side.exists():
        # A sidecar that exists but holds nothing is the signature of a write
        # that was interrupted (disk full, machine released mid-copy). It must
        # be reported as the problem it is: indexing [0] here raised IndexError
        # and made runlog mark the whole run FAIL with a traceback that looked
        # like a bug in the checker rather than a damaged file.
        parts = side.read_text(encoding="utf-8").split()
        if not parts:
            problems.append("sha256 sidecar is empty")
        elif sha256_file(path) != parts[0]:
            problems.append("sha256 does not match its .sha256 sidecar")
    return problems


def main(argv=None) -> int:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(prog="results_io")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("files", nargs="+")
    c.add_argument("--no-provenance", action="store_true",
                   help="only check non-empty, valid JSON, checksum")
    m = sub.add_parser("manifest")
    m.add_argument("files", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    v.add_argument("--strict", action="store_true",
                   help="a file named by the manifest but absent is a failure "
                        "even in a partial export (see below)")
    a = ap.parse_args(argv)

    if a.cmd == "check":
        bad = 0
        for f in a.files:
            p = check_file(f, () if a.no_provenance else PROVENANCE_KEYS)
            print(f"  {'FAIL' if p else 'ok  '}  {f}" + (f": {'; '.join(p)}" if p else ""))
            bad += bool(p)
        return 1 if bad else 0
    if a.cmd == "manifest":
        for f in a.files:
            if Path(f).stat().st_size == 0:
                print(f"ERROR: {f} is empty", file=sys.stderr)
                return 1
            print(f"{sha256_file(f)}  {Path(f).stat().st_size}  {f}")
        return 0
    bad = absent = 0
    base = Path(a.manifest).parent
    # A per-run manifest inside a FLATTENED evidence export. The export copies
    # `<run>/n_ladder.json` to `<run>__n_ladder.json`, so the manifest written
    # inside the run directory still names `n_ladder.json` and matches nothing
    # -- every entry read as a missing file, which is how a perfectly good
    # bundle reports itself as destroyed. The prefix is recoverable from the
    # manifest's own name, so recover it.
    #
    # The export is also results-only: `command.txt`, `env.json` and `log.txt`
    # stay in the run directory and are not copied. Absence is therefore
    # expected here and is reported as absence, not as damage -- what would be
    # damage is a file that IS present and does not match. `--strict` is for
    # checking a complete run directory, where anything missing is wrong.
    name = Path(a.manifest).name
    prefix = name[:-len("MANIFEST.sha256")] if name.endswith("__MANIFEST.sha256") \
        else ""
    lines = Path(a.manifest).read_text(encoding="utf-8").splitlines()
    if not lines or not any(l.strip() for l in lines):
        print("FAIL  the manifest is empty -- nothing to verify")
        return 1
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split(None, 2)
        if len(fields) != 3:
            # The purpose of verify is to catch a truncated copy, so a
            # truncated *line* has to be a FAIL, not a traceback: unpacking
            # three names from fewer fields raised ValueError and told the
            # reader nothing about which file was damaged.
            print(f"  FAIL  line {n}: malformed manifest entry ({len(fields)} "
                  f"fields, expected 3): {line.strip()[:60]!r}")
            bad += 1
            continue
        digest, size, name = fields
        # Next to the manifest FIRST. A manifest's entries are relative to the
        # manifest, and preferring the working directory meant that verifying
        # `evidence/MANIFEST.sha256` from the project root checked the
        # project's own `README.md` against the bundle's -- reporting the
        # evidence bundle as corrupt because two files share a common name.
        # The cwd remains a fallback for a manifest written elsewhere.
        for cand in ([base / (prefix + name)] if prefix else []) + \
                    [base / name, Path(name)]:
            if cand.exists():
                p = cand
                break
        else:
            p = base / name
        if not p.exists():
            if prefix and not a.strict:
                print(f"  absent  {name}: not in this export")
                absent += 1
            else:
                print(f"  FAIL  {name}: missing")
                bad += 1
        elif p.stat().st_size != int(size):
            print(f"  FAIL  {name}: {p.stat().st_size} bytes, expected {size}"); bad += 1
        elif sha256_file(p) != digest:
            print(f"  FAIL  {name}: checksum differs"); bad += 1
        else:
            print(f"  ok    {name}")
    if absent:
        print(f"\n{absent} file(s) named by this manifest are not in this "
              f"export. A flattened evidence bundle carries the results files "
              f"and their checksums; command.txt, env.json and log.txt stay in "
              f"the run directory. Pass --strict to treat absence as failure, "
              f"which is what you want against a full run directory.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
