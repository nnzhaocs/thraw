"""Run any experiment so that what it produced can't be lost or confused.

    python -m semview.runlog --name 8m_migration -- \\
        python -m semview.migration ... --out {run}/migration.json

What it does:
  1. makes a new directory results/runs/<YYYYmmdd-HHMMSS>_<name>/ (never reuses
     one, so nothing is overwritten) and substitutes it for {run} in the command;
  2. writes command.txt (the exact argv), env.json (versions, GPU, host, and a
     hash of the code that ran), and log.txt (stdout and stderr, as they come);
  3. after the command exits, checks every .json in the directory with
     semview.results_io (non-empty, valid, provenance, checksum), writes
     MANIFEST.sha256 over everything in it, and appends one line to
     results/RUNS.md: when, name, tag, status, exit code, directory, code hash.

Exits non-zero if the command failed, produced no results file, or a file
failed its check. A failed run keeps its directory and is marked FAIL in
RUNS.md; nothing is deleted.

Copy a run off a remote machine and check it arrived whole:

    python -m semview.results_io verify results/runs/RUNDIR/MANIFEST.sha256
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("semview", "thaw_pg", "analysis")


def code_hash(root: Path = ROOT) -> str:
    """One hash over every .py file the bundle runs, in a fixed order: two runs
    with the same value ran the same code."""
    h = hashlib.sha256()
    for d in CODE_DIRS:
        for p in sorted((root / d).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            h.update(str(p.relative_to(root)).replace("\\", "/").encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def new_run_dir(results: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise SystemExit(f"--name {name!r}: letters, digits, _ . - only")
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = results / "runs" / f"{stamp}_{name}"
    n = 1
    while d.exists():                        # two runs in the same second
        n += 1
        d = results / "runs" / f"{stamp}_{name}_{n}"
    d.mkdir(parents=True)
    return d


def run(cmd: list[str], name: str, results: Path, tag: str = "",
        allow_no_json: bool = False) -> int:
    from .results_io import check_file, environment, sha256_file

    if not cmd:
        raise SystemExit("give the command after --")
    rd = new_run_dir(results, name)
    argv = [a.replace("{run}", str(rd)) for a in cmd]
    (rd / "command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
    env = environment()
    env.update(code_hash=code_hash(), run_name=name, tag=tag,
               started=_dt.datetime.now().isoformat(timespec="seconds"))
    (rd / "env.json").write_text(json.dumps(dict(environment=env), indent=2),
                                 encoding="utf-8")
    print(f"run directory: {rd}")
    t0 = time.time()
    with open(rd / "log.txt", "w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {' '.join(argv)}\n")
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = p.wait()
        log.write(f"\n[exit {code} after {time.time() - t0:.0f} s]\n")

    problems = []
    # anywhere under the run directory, so --out "{run}/sub/x.json" is checked too
    jsons = [f for f in sorted(rd.rglob("*.json")) if f != rd / "env.json"]
    if not jsons and not allow_no_json:
        problems.append("no results .json in the run directory -- did --out "
                        "point at {run}/...?")
    for f in jsons:
        req = () if f.name.endswith(".traces.json") else None
        p_ = check_file(f) if req is None else check_file(f, ())
        if p_:
            problems.append(f"{f.relative_to(rd).as_posix()}: {'; '.join(p_)}")
    with open(rd / "MANIFEST.sha256", "w", encoding="utf-8") as m:
        for f in sorted(rd.rglob("*")):
            if f.is_file() and f != rd / "MANIFEST.sha256":
                m.write(f"{sha256_file(f)}  {f.stat().st_size}  "
                        f"{f.relative_to(rd).as_posix()}\n")
    status = "ok" if code == 0 and not problems else "FAIL"
    reg = results / "RUNS.md"
    if not reg.exists():
        reg.write_text("| started | name | tag | status | exit | dir | code | paper |\n"
                       "|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(reg, "a", encoding="utf-8") as f:
        f.write(f"| {env['started']} | {name} | {tag} | {status} | {code} | "
                f"{rd.relative_to(results).as_posix()} | {env['code_hash']} |  |\n")
    print(f"\n{status}: {rd}")
    for pr in problems:
        print(f"  - {pr}")
    return 0 if status == "ok" else (code or 1)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="runlog")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="", help="e.g. smoke, final, paper")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--allow-no-json", action="store_true",
                    help="for commands that legitimately write no results file")
    return ap


def main(argv=None) -> int:  # pragma: no cover
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m semview.runlog --name NAME [--tag T] "
                         "-- <command ... {run}/out.json>")
    i = argv.index("--")
    a = parser().parse_args(argv[:i])
    return run(argv[i + 1:], a.name, Path(a.results), a.tag, a.allow_no_json)


if __name__ == "__main__":
    sys.exit(main())
