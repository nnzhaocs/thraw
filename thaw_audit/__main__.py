"""The audit CLI.

    # what is settled and what is blocking, without touching anything
    python -m thaw_audit status

    # the audit pass: read fields, compare, write the report
    python -m thaw_audit verify --runs-root evidence --report AUDIT_REPORT.md

    # settle numbers that verified (needs MEASURE, and refuses anything not V)
    THAW_MODE=measure python -m thaw_audit freeze T-PENALTY-B16 T-LADDER-8M-500

    # can an open item be closed?
    python -m thaw_audit dod D-01

    # what each RUNS.md `paper` cell should say
    python -m thaw_audit reconcile

Default mode is `audit`, which cannot change a claim or run a command. That is
the point: the harmless invocation is the one you get by forgetting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .claims import Claim, Registry, STATUSES
from .ledger import Ledger
from .modes import Mode, check_dod, require
from .verify import report_markdown, verify_all

HERE = Path(__file__).resolve().parent.parent
DEFAULT_CLAIMS = HERE / "audit" / "claims.json"
DEFAULT_ITEMS = HERE / "audit" / "open_items.json"


def _roots(args) -> list[Path]:
    """The directories a claim's source file is looked for in.

    An explicit `--runs-root` replaces the defaults rather than adding to
    them, which is the point of the flag: verifying against the evidence
    bundle alone is what shows the settled numbers do not depend on anything
    in the working tree.

    That makes an empty root dangerous in a specific way, and `evidence/`
    can be empty -- in some hand-offs the bundle is a separate archive to
    unpack into it. Run before that and every source file is missing, so every frozen claim reads
    as a settled number whose evidence has disappeared: an archive-is-broken
    report from the most ordinary omission there is. So a root set holding no
    results file at all is an error here, before any claim is read.
    """
    roots = [Path(r) for r in (args.runs_root or [])]
    if not roots:
        for cand in (HERE / "evidence", HERE / "results", HERE):
            if cand.is_dir():
                roots.append(cand)
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        raise SystemExit(f"no such runs root: {', '.join(map(str, missing))}")
    return roots


def _require_sources(roots: list[Path], registry) -> None:
    """Refuse to audit when not one claim's source is findable.

    The first version of this guard asked whether any `*.json` existed under
    the roots, which the default root set satisfies with `audit/claims.json`
    -- the ledger's own file. So the case it was written for, an empty
    `evidence/` on a clean unpack, still printed all 64 frozen claims as
    settled numbers whose evidence had vanished. The question is not whether
    there is a JSON file somewhere; it is whether any file a claim rests on is
    present.

    One source found is enough to proceed: after that, a missing file is a
    real finding about that claim and belongs in the report.
    """
    sources = {c.source for c in registry.claims if c.source}
    if not sources:
        return
    from .verify import _find_source
    if any(_find_source(s, roots) is not None for s in sources):
        return
    where = ", ".join(map(str, roots))
    raise SystemExit(
        f"not one of the {len(sources)} files these claims rest on is under "
        f"{where}. That is a wrong or empty root, not a broken archive -- the "
        f"report would otherwise call every settled number's evidence lost. "
        f"If 'evidence/' is empty, unpack the evidence bundle into it first. "
        f"Omitting --runs-root searches 'evidence/', then 'results/', then "
        f"the project tree.")


def _roots_for(args, registry) -> list[Path]:
    """The roots, checked against what the claims actually need."""
    roots = _roots(args)
    _require_sources(roots, registry)
    return roots


def _items(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def cmd_status(args) -> int:
    reg = Registry.load(Path(args.claims))
    c = reg.counts()
    print(f"claims {len(reg.claims)}   frozen {c['frozen']}")
    for k, desc in STATUSES.items():
        print(f"  {k}  {c.get(k, 0):>4}   {desc}")
    blocking = reg.blocking()
    if blocking:
        print(f"\n{len(blocking)} claim(s) may not go in a submitted version:")
        for cl in blocking:
            print(f"  {cl.status}  {cl.id:<28} {cl.says[:62]}")
    items = _items(Path(args.items))
    if items:
        print(f"\nopen items: {len(items)}")
        for it in items:
            print(f"  {it['id']:<7} [{it.get('mode','?'):<9}] {it['title'][:60]}")
    return 1 if blocking else 0


def cmd_verify(args) -> int:
    reg = Registry.load(Path(args.claims))
    roots = _roots_for(args, reg)
    findings = verify_all(reg, roots, use_cache=not args.no_cache,
                          only=args.only or None)
    settled = sum(1 for f in findings if f.settled)
    for f in findings:
        mark = "settled" if f.settled else f.status
        print(f"  {mark:<8} {f.claim_id:<28} {f.message}")
    bad = [f for f in findings if f.status in {"X", "P", "S"}]
    print(f"\n{len(findings)} checked, {settled} settled without recomputation, "
          f"{len(bad)} blocking")
    if args.report:
        require("write_report")
        Path(args.report).write_text(report_markdown(reg, findings),
                                     encoding="utf-8")
        print(f"wrote {args.report}")
    if args.record:
        # recording the finding back into the registry changes it, so it needs
        # a mode that may: an audit reports, it does not rewrite the ledger
        require("write_claim")
        import datetime as _dt
        today = _dt.date.today().isoformat()
        for f in findings:
            cl = reg.get(f.claim_id)
            if cl.frozen:
                continue
            cl.status, cl.checked = f.status, today
            if cl.value is None and f.read_value is not None:
                cl.value = f.read_value
            if f.sha256:
                cl.sha256 = cl.sha256 or ""
        reg.save()
        print(f"recorded {len(findings)} findings into {reg.path}")
    return 1 if bad else 0


def cmd_freeze(args) -> int:
    reg = Registry.load(Path(args.claims))
    led = Ledger(Path(args.runs_dir))
    done = led.freeze(reg, args.ids, _roots_for(args, reg))
    reg.save()
    print(f"froze {len(done)}: {', '.join(done)}")
    print("a frozen claim is checked by hash from now on, and never re-measured")
    return 0


def cmd_thaw(args) -> int:
    reg = Registry.load(Path(args.claims))
    led = Ledger(Path(args.runs_dir))
    done = led.thaw(reg, args.ids, args.why)
    reg.save()
    print(f"thawed {len(done)}: {', '.join(done)}")
    return 0


def cmd_add(args) -> int:
    require("write_claim")
    reg = Registry.load(Path(args.claims))
    reg.add(Claim(id=args.id, says=args.says, paper=args.paper,
                  value=(json.loads(args.value) if args.value else None),
                  tol=args.tol, source=args.source, field=args.field,
                  run_id=args.run_id, status=args.status, note=args.note),
            replace=args.replace)
    reg.save()
    print(f"added {args.id} to {reg.path}")
    return 0


def cmd_dod(args) -> int:
    reg = Registry.load(Path(args.claims))
    items = {it["id"]: it for it in _items(Path(args.items))}
    want = args.ids or sorted(items)
    rc = 0
    for iid in want:
        it = items.get(iid)
        if it is None:
            print(f"  {iid}: no such open item")
            rc = 1
            continue
        unmet = check_dod(it, _roots_for(args, reg), reg)
        if unmet:
            rc = 1
            print(f"  OPEN  {iid}  {it['title']}")
            for u in unmet:
                print(f"          - {u}")
        else:
            print(f"  DONE  {iid}  {it['title']}")
    return rc


def _findings(args, reg):
    return verify_all(reg, _roots_for(args, reg), use_cache=True)


def cmd_evidence(args) -> int:
    from .docs import evidence_md
    require("write_report")
    reg = Registry.load(Path(args.claims))
    md = evidence_md(reg, _findings(args, reg), _items(Path(args.items)),
                     bundle=args.bundle, covers=args.covers,
                     verified_by=args.verified_by)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def cmd_project_status(args) -> int:
    from .docs import project_status_md
    require("write_report")
    reg = Registry.load(Path(args.claims))
    led = Ledger(Path(args.runs_dir))
    md = project_status_md(reg, _findings(args, reg), _items(Path(args.items)),
                           _roots_for(args, reg)[0], ledger=led)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def cmd_trail(args) -> int:
    from .docs import trail_md
    require("write_report")
    reg = Registry.load(Path(args.claims))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(trail_md(reg, _findings(args, reg)),
                              encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def cmd_coverage(args) -> int:
    """The direction `reconcile` structurally cannot see: paper -> claim.

    Exits non-zero while any result number in the paper is neither registered
    nor declared, so it gates the same way every other check here does.
    """
    from thaw_audit.coverage import main as coverage_main
    return coverage_main(["--suggest"] if args.suggest else [])


def cmd_reconcile(args) -> int:
    reg = Registry.load(Path(args.claims))
    led = Ledger(Path(args.runs_dir))
    r = led.reconcile(reg, _roots_for(args, reg))
    for key, label in (("missing_runs", "claims naming a run that is nowhere"),
                       ("flattened_only",
                        "runs present only as a flattened evidence export"),
                       ("unused_runs", "run directories no claim uses"),
                       ("runs_without_a_lesson", "runs with no recorded lesson"),
                       ("stale_paper_cells", "RUNS.md paper cells to update")):
        vals = r[key]
        print(f"{label}: {len(vals)}")
        for v in vals:
            if key == "stale_paper_cells":
                extra = (f"  -> paper: {r['paper_column'].get(v, '')}"
                         f"  |  lesson: {r['lesson_column'].get(v, '(none)')}")
            else:
                extra = ""
            print(f"  {v}{extra}")
    return 1 if r["missing_runs"] or r["stale_paper_cells"] else 0


def cmd_import_manual(args) -> int:
    """Fold an existing markdown claim table into the registry.

    The engineering manual's first table is already claim-per-row; this reads it
    rather than asking anyone to retype it. Rows it cannot parse are reported,
    never guessed at.
    """
    require("write_claim")
    from .manual import import_markdown
    reg = Registry.load(Path(args.claims))
    added, skipped = import_markdown(Path(args.path), reg, replace=args.replace)
    reg.save()
    print(f"imported {added} claim(s) into {reg.path}")
    for s in skipped:
        print(f"  skipped: {s}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="thaw_audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default=str(DEFAULT_CLAIMS))
    ap.add_argument("--items", default=str(DEFAULT_ITEMS))
    ap.add_argument("--runs-root", action="append",
                    help="directory to search for source files; repeatable")
    ap.add_argument("--runs-dir", default=str(HERE / "results"),
                    help="where run directories and RUNS.md live")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="what is settled and what blocks submission"
                   ).set_defaults(fn=cmd_status)

    v = sub.add_parser("verify", help="read fields and compare; executes nothing")
    v.add_argument("--report", default=None, help="write a markdown report here")
    v.add_argument("--only", action="append", help="claim id; repeatable")
    v.add_argument("--no-cache", action="store_true",
                   help="re-hash every source instead of trusting size+mtime")
    v.add_argument("--record", action="store_true",
                   help="write the findings back as statuses (needs write_claim)")
    v.set_defaults(fn=cmd_verify)

    f = sub.add_parser("freeze", help="settle verified claims (needs MEASURE)")
    f.add_argument("ids", nargs="+")
    f.set_defaults(fn=cmd_freeze)

    t = sub.add_parser("thaw", help="unsettle claims, with a reason")
    t.add_argument("ids", nargs="+")
    t.add_argument("--why", required=True)
    t.set_defaults(fn=cmd_thaw)

    a = sub.add_parser("add", help="register a claim")
    a.add_argument("--id", required=True)
    a.add_argument("--says", required=True)
    a.add_argument("--paper", default="")
    a.add_argument("--value", default=None, help="JSON scalar")
    a.add_argument("--tol", type=float, default=5e-5)
    a.add_argument("--source", default="")
    a.add_argument("--field", default="")
    a.add_argument("--run-id", default="")
    a.add_argument("--status", default="X", choices=sorted(STATUSES))
    a.add_argument("--note", default="")
    a.add_argument("--replace", action="store_true")
    a.set_defaults(fn=cmd_add)

    d = sub.add_parser("dod", help="is an open item finished?")
    d.add_argument("ids", nargs="*")
    d.set_defaults(fn=cmd_dod)

    sub.add_parser("reconcile", help="registry against the run directories"
                   ).set_defaults(fn=cmd_reconcile)

    # `reconcile` checks claim -> run and run -> claim. Both directions start
    # from the ledger, so a number typed into the paper that no claim covers is
    # invisible to it. This is the third direction.
    cv = sub.add_parser("coverage", help="every number in the paper against "
                                         "the registry: the direction "
                                         "reconcile cannot see")
    cv.add_argument("--suggest", action="store_true",
                    help="for each uncovered number, where in the evidence "
                         "bundle it might live. A worklist to confirm, not a "
                         "registration: a value can collide with an unrelated "
                         "field, and a claim frozen against a coincidence "
                         "verifies forever while meaning nothing")
    cv.set_defaults(fn=cmd_coverage)

    e = sub.add_parser("evidence", help="generate the bundle's EVIDENCE.md")
    e.add_argument("--out", default="evidence/EVIDENCE.md")
    e.add_argument("--bundle", default="evidence")
    e.add_argument("--covers", default="")
    e.add_argument("--verified-by", default="")
    e.set_defaults(fn=cmd_evidence)

    ps = sub.add_parser("project-status", help="generate PROJECT_STATUS.md")
    ps.add_argument("--out", default="PROJECT_STATUS.md")
    ps.set_defaults(fn=cmd_project_status)

    tr = sub.add_parser("trail", help="generate the per-number audit trail")
    tr.add_argument("--out", default="docs/AUDIT_TRAIL.md")
    tr.set_defaults(fn=cmd_trail)

    m = sub.add_parser("import-manual", help="read a markdown claim table")
    m.add_argument("path")
    m.add_argument("--replace", action="store_true")
    m.set_defaults(fn=cmd_import_manual)

    args = ap.parse_args(argv)
    print(f"[THAW_MODE={Mode.current().value}]")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
