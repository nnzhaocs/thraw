"""The run index, and the bridge to `results/RUNS.md`.

`semview.runlog` already appends a line to `RUNS.md` for every run, with a
`paper` column meant to say which paper numbers it supports. That column has
been empty, which is the gap this closes: the claims registry knows which run
each number came from, so the column can be generated rather than remembered.

Generating it also makes one kind of error impossible. A run whose numbers the
paper uses but whose directory nobody recorded shows up here as a claim with no
`run_id`, and a run directory no claim points at shows up as unused. Both are
visible in one listing instead of being discovered by reading two files side by
side.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .modes import require


@dataclass
class Run:
    run_id: str
    path: Path
    command: str = ""
    code_hash: str = ""
    status: str = ""
    has_manifest: bool = False

    @property
    def json_files(self) -> list[Path]:
        return sorted(p for p in self.path.glob("*.json")
                      if not p.name.endswith(".sha256"))


class Ledger:
    """Run directories on disk, plus whatever `RUNS.md` says about them."""

    def __init__(self, runs_root: Path, runs_md: Path | None = None):
        self.runs_root = Path(runs_root)
        self.runs_md = Path(runs_md) if runs_md else self.runs_root / "RUNS.md"

    # ----------------------------------------------------------------- runs
    def discover(self) -> list[Run]:
        """Every run directory under the root, whether `RUNS.md` knows it or not.

        A directory is a run if it carries a `command.txt` or a
        `MANIFEST.sha256`; a bare folder of JSON is not, because the point of a
        run directory is that it records how it was made.
        """
        out = []
        roots = [self.runs_root]
        sub = self.runs_root / "runs"
        if sub.is_dir():
            roots.append(sub)
        seen = set()
        for root in roots:
            if not root.is_dir():
                continue
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                if d.resolve() in seen:
                    continue
                cmd, man = d / "command.txt", d / "MANIFEST.sha256"
                if not (cmd.is_file() or man.is_file()):
                    continue
                seen.add(d.resolve())
                code = ""
                env = d / "env.json"
                if env.is_file():
                    try:
                        code = json.loads(
                            env.read_text(encoding="utf-8")).get("code_hash", "")
                    except (ValueError, OSError):
                        code = ""
                out.append(Run(
                    run_id=d.name, path=d,
                    command=(cmd.read_text(encoding="utf-8").strip()
                             if cmd.is_file() else ""),
                    code_hash=code, has_manifest=man.is_file()))
        return out

    def run_ids(self) -> set[str]:
        return {r.run_id for r in self.discover()}

    def flattened_run_ids(self, roots: list[Path]) -> set[str]:
        """Runs present as a FLATTENED evidence export rather than a directory.

        `collect_evidence.py` ships one file per cited run named
        `<run id>__<file>`, deliberately flat. Without this, `reconcile` reports
        every run in a shipped bundle as "not on disk" -- which reads as a
        broken archive and sends the next reader hunting for directories that
        were never meant to be there. A flattened run is real evidence; what it
        lacks is the run directory's `command.txt` and `env.json`, and that is a
        different statement from being absent.
        """
        out = set()
        for root in roots:
            if not Path(root).is_dir():
                continue
            for p in Path(root).rglob("*__*"):
                if p.is_file() and "__" in p.name:
                    out.add(p.name.split("__", 1)[0])
        return out

    # -------------------------------------------------------------- RUNS.md
    def runs_md_rows(self) -> dict[str, str]:
        """The `paper` cell of each row in RUNS.md, keyed by run directory.

        Parsed loosely on purpose: RUNS.md is written for people, and a strict
        parser that refuses an unexpected row would make the tool the reason to
        stop maintaining the file.
        """
        if not self.runs_md.is_file():
            return {}
        out = {}
        for line in self.runs_md.read_text(encoding="utf-8").splitlines():
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # The run id is searched for inside the cell, not matched against
            # the whole of it: `semview.runlog` writes the directory cell as
            # `runs/<id>` and a flattened row as `evidence/<id>__*`, so an
            # anchored match recognised neither and `reconcile` could never
            # report clean for a run the project's own tooling had written.
            hit = None
            for c in cells:
                m = re.search(r"\d{8}-\d{6}_[A-Za-z0-9_.-]+", c)
                if m:
                    hit = m.group(0).rstrip("_").removesuffix("__*")
                    break
            if hit:
                out[hit] = cells[-1]
        return out

    def paper_column(self, registry) -> dict[str, str]:
        """What each run's `paper` cell should say, from the claims registry."""
        out: dict[str, list[str]] = {}
        for c in sorted(registry.claims, key=lambda c: c.id):
            if not c.run_id:
                continue
            label = c.paper or c.id
            out.setdefault(c.run_id, [])
            if label not in out[c.run_id]:
                out[c.run_id].append(label)
        return {k: ", ".join(v) for k, v in out.items()}

    def lesson_column(self, registry) -> dict[str, str]:
        """What each run taught us, from the claims that came out of it.

        Unlike the `paper` column this cannot be derived from anything -- a
        lesson is a judgement -- so it is carried on the claim and collected
        here. A run with no lesson on any of its claims is reported by
        `reconcile` so it does not quietly stay blank the way the `paper`
        column did.
        """
        out: dict[str, list[str]] = {}
        for c in sorted(registry.claims, key=lambda c: c.id):
            if c.run_id and c.lesson:
                out.setdefault(c.run_id, [])
                if c.lesson not in out[c.run_id]:
                    out[c.run_id].append(c.lesson)
        return {k: "; ".join(v) for k, v in out.items()}

    def reconcile(self, registry, roots: list[Path] | None = None) -> dict:
        """Where the registry and the run directories disagree.

        Four ways, each a different repair: a claim naming a run that is
        nowhere (the archive is incomplete), one present only as a flattened
        evidence export (fine for the numbers, but its command and environment
        are not shipped), a run on disk no claim uses (dead weight, or a number
        nobody wired up), and a run whose `paper` cell is blank or stale.
        """
        on_disk = self.run_ids()
        flat = self.flattened_run_ids(roots or []) - on_disk
        wanted = self.paper_column(registry)
        current = self.runs_md_rows()
        named = {c.run_id for c in registry.claims if c.run_id}
        missing_runs = sorted(named - on_disk - flat)
        flattened_only = sorted(named & flat)
        unused_runs = sorted(on_disk - set(wanted))
        stale_paper = sorted(
            rid for rid, should in wanted.items()
            if rid in on_disk and current.get(rid, "") != should)
        lessons = self.lesson_column(registry)
        no_lesson = sorted(rid for rid in wanted if rid not in lessons)
        return dict(missing_runs=missing_runs, unused_runs=unused_runs,
                    flattened_only=flattened_only,
                    stale_paper_cells=stale_paper, paper_column=wanted,
                    lesson_column=lessons, runs_without_a_lesson=no_lesson)

    # ------------------------------------------------------------- freezing
    def freeze(self, registry, claim_ids: list[str], roots: list[Path],
               mode=None) -> list[str]:
        """Settle these claims: record the source hash so a check is a compare.

        Refuses anything that is not `V`. Freezing an unverified number would
        make the stop condition a lie -- the audit would report "settled" for a
        figure nobody ever matched against its file.
        """
        require("freeze", mode)
        from .hashes import sha256_cached
        from .verify import _find_source, verify_claim

        frozen = []
        for cid in claim_ids:
            c = registry.get(cid)
            f = verify_claim(c, roots, mode=mode)
            if f.status != "V":
                raise SystemExit(
                    f"refusing to freeze {cid}: it is {f.status}, not V "
                    f"({f.message}). Freeze records that a number is settled, "
                    f"so it has to be verified first.")
            src = _find_source(c.source, roots)
            # from the bytes, never the cache: this hash is what every
            # later audit compares against, so a stale entry here would
            # settle the claim against content that was already gone
            c.sha256 = sha256_cached(src, root=roots[0], use_cache=False)
            c.frozen, c.status = True, "V"
            frozen.append(cid)
        return frozen

    def thaw(self, registry, claim_ids: list[str], why: str,
             mode=None) -> list[str]:
        """Unsettle claims, with a reason recorded in the note.

        Thawing without a reason is the start of the loop this package exists to
        stop, so the reason is required rather than encouraged.
        """
        require("write_claim", mode)
        if not why.strip():
            raise SystemExit(
                "thaw needs --why: a number that was settled and is not any "
                "more has to say what changed, or the next audit has no way to "
                "know whether to trust it.")
        out = []
        for cid in claim_ids:
            c = registry.get(cid)
            c.frozen = False
            c.note = (c.note + " | " if c.note else "") + f"thawed: {why.strip()}"
            out.append(cid)
        return out
