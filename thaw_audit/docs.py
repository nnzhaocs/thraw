"""Generate the hand-off documents from the ledger, so they cannot drift.

Every review of this project asks for the same set of markdown files:
`EVIDENCE.md` with a frozen table and a do-not-re-run rule, a `PROJECT_STATUS.md`
saying where the work is, an audit trail per number. They are the right
documents. The reason to generate them rather than write them is in this
project's own history: `docs/history/PENDING_RUNS.md` went stale, the `paper` column of
`RUNS.md` stayed empty for thirty rows, and the engineering manual ended up
older than the evidence it described. A hand-kept provenance document drifts,
and a drifted one is worse than none, because it is read as current.

So the rule here is: anything derivable is derived. What a number is, where it
came from, whether it is settled, and what is still owed all live in
`audit/claims.json` and `audit/open_items.json`; these functions render them.
The only field nothing can generate is `lesson` -- what a run taught us -- which
is a judgement and is therefore carried on the claim.

    python -m thaw_audit evidence --out evidence/EVIDENCE.md
    python -m thaw_audit project-status --out PROJECT_STATUS.md
    python -m thaw_audit trail --out docs/AUDIT_TRAIL.md
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from .modes import check_dod


def evidence_md(registry, findings, items, bundle: str = "evidence",
                covers: str = "", verified_by: str = "") -> str:
    """The evidence bundle's front page: what is settled, and what is not.

    The two sections exist to answer the question a fresh reader actually has,
    which is not "what is in here" but "what may I trust without re-running
    it". So the frozen table comes first and the rule is stated before the
    data.
    """
    by_id = {f.claim_id: f for f in findings}
    frozen = [c for c in sorted(registry.claims, key=lambda c: c.id)
              if c.frozen and by_id.get(c.id) and by_id[c.id].settled]
    broken = [c for c in sorted(registry.claims, key=lambda c: c.id)
              if c.frozen and by_id.get(c.id) and not by_id[c.id].settled]
    open_ = [c for c in sorted(registry.claims, key=lambda c: c.id)
             if not c.frozen]

    today = _dt.date.today().isoformat()
    out = [
        f"# Evidence bundle — Thaw",
        "",
        f"- **Frozen:** {today}",
        f"- **Covers:** {covers or 'see the table below'}",
        f"- **Verified by:** {verified_by or 'thaw_audit verify'}",
        f"- **Method:** `semview.results_io check` on each run, then "
        f"`thaw_audit verify` field-by-field against `audit/claims.json`",
        f"- **Manifest:** `MANIFEST.sha256`",
        "",
        "## The rule",
        "",
        "**Do not re-measure anything in the frozen table.** If you doubt it:",
        "",
        "```bash",
        f"python -m semview.results_io verify {bundle}/MANIFEST.sha256",
        f"python -m thaw_audit --runs-root {bundle} verify",
        "```",
        "",
        "If both pass, the numbers are the ones the paper states and the files "
        "that produced them are intact; there is nothing a re-run would tell "
        "you. If either fails it names the file, and only then is re-running "
        "the right move. A new round goes in a new bundle — never overwrite "
        "this one.",
        "",
        "## What this bundle does NOT cover",
        "",
        f"It covers **{len(registry.claims)} registered claims**, not every "
        f"number in the paper. A reader who takes "
        f"\"{len(frozen)} frozen, 0 blocking\" as \"the paper's numbers are "
        f"settled\" has read more into it than it says: the audit is clean "
        f"**within the claims that are registered**, which is a statement "
        f"about those claims and not about the paper.",
        "",
    ]
    if items:
        out += [
            f"Still open, and outside this bundle ({len(items)}) — see "
            f"`PROJECT_STATUS.md`:",
            "",
        ]
        out += [f"- **{it['id']}** {it['title']}" for it in items]
        out += [""]
    out += [
        "Numbers in the paper that no claim covers are in the same position: "
        "unregistered is not verified. `thaw_audit reconcile` lists the runs "
        "on disk that no claim uses, which is where to look for them.",
        "",
        "## What \"frozen\" does and does not mean",
        "",
        "Frozen means **the source file has not changed since the claim was "
        "settled** — the check is a sha256 comparison and nothing else. It "
        "does **not** mean the value is right. A claim frozen with a value "
        "that was wrong when it was recorded stays settled forever, because "
        "nothing re-reads the field once the hash matches. What stands behind "
        "the values is that each was read out of its file by "
        "`collect_evidence` rather than typed, and verified field-by-field "
        "before freezing; what the freeze adds on top is only that the file "
        "is the same file.",
        "",
        f"## Frozen — do not re-run ({len(frozen)})",
        "",
        "| claim | value | paper | run | file | lesson |",
        "|---|---|---|---|---|---|",
    ]
    for c in frozen:
        src = c.source.split("__")[-1] if "__" in c.source else Path(c.source).name
        out.append(f"| `{c.id}` | {c.value} | {c.paper or '—'} | "
                   f"`{c.run_id or '—'}` | `{src}` | {c.lesson or '—'} |")
    if broken:
        out += ["", f"## Frozen but NOT intact — needs a person ({len(broken)})",
                "",
                "A settled number whose source changed or vanished. This tool "
                "will not resolve it by re-measuring; decide whether the paper "
                "or the file is wrong.", "",
                "| claim | what the check found |", "|---|---|"]
        for c in broken:
            out.append(f"| `{c.id}` | {by_id[c.id].message} |")
    if open_:
        out += ["", f"## Not settled ({len(open_)})", "",
                "These are the only claims a re-run could help with.", "",
                "| claim | status | says | why it is not settled |",
                "|---|---|---|---|"]
        for c in open_:
            f = by_id.get(c.id)
            out.append(f"| `{c.id}` | {f.status if f else c.status} | "
                       f"{c.says} | {f.message if f else '—'} |")
    if items:
        out += ["", "## Owed runs", "",
                "| item | mode | what it is |", "|---|---|---|"]
        for it in items:
            out.append(f"| **{it['id']}** | {it.get('mode','?')} | {it['title']} |")
    return "\n".join(out) + "\n"


def project_status_md(registry, findings, items, runs_root, ledger=None) -> str:
    """Where the work is, derived rather than remembered.

    Completed, in progress and next are read off the open items' definitions of
    done, so this file cannot claim an item is finished when its DoD says
    otherwise -- which is the failure mode of a hand-written status file.
    """
    by_id = {f.claim_id: f for f in findings}
    done, open_items = [], []
    for it in items:
        unmet = check_dod(it, runs_root, registry)
        (open_items if unmet else done).append((it, unmet))
    counts = registry.counts()
    settled = sum(1 for f in findings if f.settled)
    blocking = [f for f in findings if f.status in {"X", "P", "S"}]

    out = [
        "# Project status — Thaw",
        "",
        f"Generated {_dt.date.today().isoformat()} by "
        f"`python -m thaw_audit project-status`. Do not edit by hand: it is "
        f"derived from `audit/claims.json` and `audit/open_items.json`, which "
        f"are the files to change.",
        "",
        "## Where the evidence stands",
        "",
        f"- **{len(registry.claims)} claims** registered; **{counts['frozen']} "
        f"frozen**, **{settled} settled** on the last audit.",
        f"- **{len(blocking)} of those claims blocking** "
        f"(`X` {counts.get('X',0)}, `P` {counts.get('P',0)}, "
        f"`S` {counts.get('S',0)}). Only `V`, `M` and `A` may ship.",
        "",
        f"\"Blocking\" counts **registered claims whose status is bad**, and "
        f"nothing else. With {len(open_items)} open item(s) below, a zero "
        f"here does not mean the project is ready to submit — it means no "
        f"number that is registered is in a state that forbids shipping. "
        f"Paper numbers that no claim covers are not counted either way.",
        "",
    ]
    if done:
        out += [f"## Finished ({len(done)})", ""]
        for it, _ in done:
            out.append(f"- **{it['id']}** {it['title']}")
        out.append("")
    if open_items:
        out += [f"## Open ({len(open_items)})", ""]
        for it, unmet in open_items:
            out.append(f"- **{it['id']}** [{it.get('mode','?')}] {it['title']}")
            for u in unmet:
                out.append(f"  - {u}")
        out.append("")
    if blocking:
        out += ["## Blocking claims", ""]
        for f in sorted(blocking, key=lambda f: f.claim_id):
            out.append(f"- `{f.claim_id}` ({f.status}) — {f.message}")
        out.append("")
    if ledger is not None:
        rec = ledger.reconcile(registry, [Path(runs_root)])
        if rec["runs_without_a_lesson"]:
            out += ["## Runs with no recorded lesson", "",
                    "Nothing can generate these; add `lesson` to one of the "
                    "run's claims.", ""]
            for rid in rec["runs_without_a_lesson"]:
                out.append(f"- `{rid}`")
            out.append("")
    out += [
        "## Next, for whoever picks this up",
        "",
        "1. `python -m semview.results_io verify evidence/MANIFEST.sha256`",
        "2. `python -m thaw_audit --runs-root evidence verify`",
        "3. If both pass, **do not re-measure anything frozen.** Work only on "
        "the open items above.",
        "4. Before finishing: update `audit/claims.json` and "
        "`audit/open_items.json`, regenerate this file and `EVIDENCE.md`, and "
        "append a section to `CHANGELOG.md`.",
    ]
    return "\n".join(out) + "\n"


def trail_md(registry, findings) -> str:
    """One section per number: where it came from, by what command, and why."""
    by_id = {f.claim_id: f for f in findings}
    out = ["# Audit trail", "",
           f"Generated {_dt.date.today().isoformat()}. One entry per claim: the "
           f"path from a run directory to a sentence in the paper.", ""]
    for c in sorted(registry.claims, key=lambda c: c.id):
        f = by_id.get(c.id)
        out += [f"## `{c.id}` — {c.says}", "",
                f"- **paper:** {c.paper or '—'}",
                f"- **value:** {c.value}",
                f"- **run:** `{c.run_id or '— none recorded'}`",
                f"- **file:** `{c.source or '—'}`",
                f"- **field:** `{c.field or '—'}`",
                f"- **status:** {c.status}"
                f"{' (frozen)' if c.frozen else ''}",
                f"- **last check:** {f.message if f else 'not checked'}"]
        if c.lesson:
            out.append(f"- **lesson:** {c.lesson}")
        if c.note:
            out.append(f"- **note:** {c.note}")
        out.append("")
    return "\n".join(out) + "\n"
