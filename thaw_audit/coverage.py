"""Does every number in the paper have a claim behind it?

`ledger.reconcile` checks claim -> run and run -> claim. Both directions start
from the ledger, so a number typed into `body.tex` that no claim covers is
invisible to the whole audit apparatus -- which is how 0.789 and 0.944 reached
`body.tex:1627` with nothing behind them. They were not an isolated slip: the
first run of this checker found 97 of 266 result numbers uncovered, mostly
whole table bodies whose headline alone had been registered.

The rule this enforces: **every result number in the paper is either registered
as a claim, or declared with a reason.** The second half matters as much as the
first, because not every number is a measurement:

  * `derived`   -- computed from the cost model, not measured (the 69% at
                   D = 10^3 is algebra over a measured coverage, not a run)
  * `cited`     -- someone else's result, from a reference
  * `setting`   -- a configuration value (N, K', ef_construction, a batch size)
  * `prose`     -- a count, a year, a page or section number

A declaration is a sentence a reviewer could check, so a reason alone is not
enough: `why` must say which quantities a derived number comes from or which
reference a cited one is. An empty `why` is refused, because a declaration file
that accepts "derived" with no source is the same hole one level down.

    python -m thaw_audit coverage
    -> exits 1 and lists every number that is neither claimed nor declared
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# A result number is a decimal with three or more places (0.929, 0.9295) or a
# percentage (43\%, 79.5\%). A bare integer is a setting, a count, a year or a
# cross-reference, and is not considered -- scoping this checker to the numbers
# that can be wrong is what keeps its output actionable.
_NUM = re.compile(r"(?<![\d.])(\d+\.\d{3,}|\d{1,3}\.\d(?=\s*\\%)|\d{1,3}(?=\s*\\%))")

# Lines whose numbers are typesetting, not content.
_SKIP_LINE = re.compile(
    r"\\(setlength|tabcolsep|arraystretch|vspace|hspace|includegraphics|label"
    r"|ref|eqref|cite|input|usepackage|definecolor|addplot|pgf|color|scalebox"
    r"|resizebox)")

_REASONS = {"derived", "cited", "setting", "prose"}


def _strip_comments(text: str) -> list[tuple[int, str]]:
    """LaTeX comments are not content. `%` also appears escaped as `\\%`, which
    must survive, so the split is on an unescaped percent only."""
    return [(i, re.sub(r"(?<!\\)%.*$", "", line))
            for i, line in enumerate(text.splitlines(), 1)]


def _candidates(raw: str, is_percent: bool) -> list[tuple[float, int, bool]]:
    """The printed number, and the fraction it would be if it is a percentage,
    each with the number of decimals it is printed to.

    A percentage in the prose is the same quantity as a fraction in a results
    file, so 43\\% must be allowed to match a claim holding 0.43 -- and the
    fraction carries two more decimals of precision than the percentage does,
    which is what stops 43 from matching anything near zero.
    """
    v = float(raw)
    dec = len(raw.split(".")[1]) if "." in raw else 0
    # the third element says whether this candidate is the number read AS a
    # percentage, which `_matches` needs in order to refuse a claim stored as a
    # fraction: 0.1\% is a saving of one part in a thousand, and at one decimal
    # place the rounding slack is +-0.05, so it was matching recall claims of
    # 0.0617 and 0.1267
    out = [(v, dec, is_percent)]
    if is_percent:
        # Only a number the paper actually prints with a percent sign denotes a
        # fraction a hundredth of its size. Adding the /100 candidate to every
        # number instead is unnecessary slack: it lets 0.929 match a claim of
        # 0.00929. The caller knows, because `_NUM`'s percentage alternatives
        # are the ones with a `\%` lookahead -- so it is passed in rather than
        # guessed from the digits. Sub-1 percentages are real ("0.3\%" at the
        # tail of the savings table), so the test cannot be `v >= 1` either.
        out.append((v / 100.0, dec + 2, False))
    return out


def _matches(paper: float, dec: int, claim_value: float,
             pct_form: bool = False) -> bool:
    """Is the paper's number a correct rounding of the claim's value?

    This is the rule `suggest` has always documented -- compare at the paper's
    own precision, on both sides -- and the rule `report` did not use. It used
    the claim's `tol` as an interval instead, with a floor of 5e-4, and a
    tolerance is not an interval: it is the slack a verifier allows between a
    claim and its source. Read as an interval it becomes a net. `PG-CLAIM-TWICE`
    ("units claimed twice (must be 0)", value 0, tol 0.5) was then a net over
    [0, 0.5] and was the only thing covering 74 of the 169 numbers the checker
    called covered -- so the checker under-reported the hole it exists to
    measure, by nearly a factor of two, in the same direction as the defect it
    was built to find. The floor also let a claim's own tighter tolerance be
    overridden: the paper's cross-space recall 0.929 was covered by a pgvector
    window-overlap of 0.9295, which is a different quantity.
    """
    if pct_form and abs(claim_value) < 1:
        return False
    return abs(paper - claim_value) <= 0.5 * 10 ** (-dec) + 1e-12


def scan(body, claims: list[dict], declared: dict) -> dict:
    """One record per result number: where it is, and what covers it.

    `body` is one path or several. It became several when four subsections moved
    to an appendix for length: the gate read `paper/body.tex` alone, so the move
    took 41 result numbers out of its view and the uncovered count FELL while
    nothing had been registered. A gate that a file move can satisfy is not a
    gate, so every source the paper is built from is scanned, and each row
    records the file it came from.

    Every matching claim is recorded, not the first one found. Two claims whose
    values both round to the printed number mean the checker cannot say which
    quantity the paper is reporting, and first-match-wins turned that into a
    silent pass: with an unrelated claim in range, deleting the claim that
    genuinely backed a number changed nothing the gate could see. An ambiguous
    number is resolved by pinning the claim id in `audit/paper_numbers.json`
    (`{"claim": "SOME-ID"}` at `line:value`), which also makes the deletion of
    that claim a failure rather than a coincidence.
    """
    vals = [(c["id"], float(c["value"]))
            for c in claims if isinstance(c.get("value"), (int, float))]
    ids = {cid for cid, _ in vals}
    rows = []
    sources = [body] if isinstance(body, (str, Path)) else list(body)
    for src in sources:
        src = Path(src)
        if not src.is_file():
            continue
        for ln, line in _strip_comments(src.read_text(encoding="utf-8")):
            if _SKIP_LINE.search(line):
                continue
            for m in _NUM.finditer(line):
                raw = m.group(1)
                try:
                    float(raw)
                except ValueError:
                    continue
                pct = line[m.end():].lstrip().startswith("\\%")
                hits = sorted({cid for cid, cv in vals
                               for v, dec, pf in _candidates(raw, pct)
                               if _matches(v, dec, cv, pct_form=pf)})
                # Three lookups, most specific first. A `line:value` key is exact
                # but FRAGILE: editing the paper above it shifts every line below,
                # and three declarations went stale this way when a single line was
                # added -- 326 -> 327 and so on. So a declaration may instead be
                # keyed on the value alone and carry an `anchor`: a substring that
                # must appear on the line. That is stable under edits and still
                # specific to the sentence it was written for, which a bare value
                # key is not.
                dec_entry = declared.get(f"{ln}:{raw}")
                if dec_entry is None:
                    cand = declared.get(raw)
                    if isinstance(cand, dict) and cand.get("anchor"):
                        dec_entry = cand if cand["anchor"] in line else None
                    else:
                        dec_entry = cand
                pin = (dec_entry or {}).get("claim") if isinstance(dec_entry, dict) else None
                if pin:
                    claim = pin if (pin in ids and any(
                        _matches(v, d, next(cv for cid, cv in vals if cid == pin),
                                 pct_form=pf)
                        for v, d, pf in _candidates(raw, pct))) else None
                    hits = [pin] if claim else hits
                    ambiguous = False
                elif isinstance(dec_entry, dict) and f"{ln}:{raw}" in declared:
                    # a declaration keyed to THIS line is a statement about this
                    # number, so it outranks a claim that merely shares its value.
                    # An integer percentage carries half a point of slack, which is
                    # wide enough to collide: the derived 78% at body.tex:920 is a
                    # correct rounding of the measured 77.59% of the 8.8M ladder,
                    # and is not that quantity.
                    claim, ambiguous = None, False
                else:
                    claim = hits[0] if len(hits) == 1 else None
                    ambiguous = len(hits) > 1
                rows.append({"line": ln, "file": src.name, "value": raw, "claim": claim,
                             "claims_matching": hits, "ambiguous": ambiguous,
                             "pinned": bool(pin),
                             "pin_unresolved": bool(pin) and claim is None,
                             "declared": None if pin else dec_entry,
                             "context": line.strip()[:100],
                             "is_percent": pct})
    return {"total": len(rows), "rows": rows}


def report(body, claims_path: Path, declared_path: Path) -> dict:
    claims = json.loads(claims_path.read_text(encoding="utf-8"))["claims"]
    declared = {}
    if declared_path.exists():
        declared = json.loads(declared_path.read_text(encoding="utf-8"))
        # keys beginning with `_` are the file's own metadata (its comment, the
        # ratchet baseline), not declarations about a number
        declared = {k: v for k, v in declared.items() if not k.startswith("_")}
    # a pin names a claim instead of giving a reason, so it is exempt from the
    # reason/why rule -- but it must name one or the other, never neither
    malformed = sorted(
        k for k, v in declared.items()
        if not isinstance(v, dict)
        or not (v.get("claim")
                or (v.get("reason") in _REASONS and str(v.get("why", "")).strip())))
    res = scan(body, claims, declared)
    # A declaration that matches no number in the paper covers nothing, and
    # silently accumulating those is the same hole one level down again: the
    # file would look like it was doing work it is not.
    keys = {f"{r['line']}:{r['value']}" for r in res["rows"]} | {
        r["value"] for r in res["rows"]}
    stale = sorted(k for k in declared if k not in keys)
    # An ambiguous number is NOT covered: the checker cannot say which quantity
    # the paper reports, and a pin is how that is settled. A pin naming a claim
    # that no longer exists or no longer matches is a failure, which is the
    # point of pinning -- it makes a deleted claim visible.
    uncovered = [r for r in res["rows"] if not r["claim"] and not r["declared"]]
    res.update(
        claimed=sum(1 for r in res["rows"] if r["claim"]),
        declared_count=sum(1 for r in res["rows"]
                           if r["declared"] and not r["claim"]),
        ambiguous=[r for r in res["rows"] if r["ambiguous"] and not r["declared"]],
        pins_unresolved=[r for r in res["rows"] if r["pin_unresolved"]],
        uncovered=uncovered,
        malformed_declarations=malformed,
        stale_declarations=stale,
        pass_=not uncovered and not malformed and not stale,
    )
    return res


def _walk(o, pre=""):
    """Every numeric leaf of a results file, with a `paths.resolve`-style path.

    List rows are addressed by a selector rather than an index wherever the row
    carries one, because a positional index breaks whenever a list is reordered
    -- the same reason the ledger's `field` syntax prefers `[kprime=500]`.
    """
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{pre}.{k}" if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, dict):
                sel = next((f"[{kk}={v[kk]}]" for kk in
                            ("kprime", "scheme", "policy", "N", "leg", "mode",
                             "ridge", "batch", "model", "old")
                            if kk in v), f"[{i}]")
                yield from _walk(v, f"{pre}{sel}")
            else:
                yield from _walk(v, f"{pre}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield pre, float(o)


def suggest(uncovered: list[dict], evidence_dir: Path) -> list[dict]:
    """For each uncovered number, where in the evidence bundle it might live.

    Matching is at **the paper's own precision on both sides**: a paper
    printing 0.873 may be an evidence value of 0.8734, so a full-precision
    index keyed on the stored value finds nothing. Getting this wrong the first
    time made 77 numbers look absent from the bundle when 65 were in it, which
    is worth recording -- the near-miss was a tooling bug reported as a
    submission blocker.

    The output is a WORKLIST, not a registration. A value can collide with an
    unrelated field (0.898 matches a `hit_rate_by_rank` entry; 1.000 matches a
    thread count), and a claim frozen against a coincidence verifies forever
    while meaning nothing -- strictly worse than no claim at all. So each row
    carries every candidate and the field must be confirmed before it is
    registered.
    """
    flat = []
    for f in sorted(evidence_dir.glob("*.json")):
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        flat.extend((f.name, p, v) for p, v in _walk(obj))

    out = []
    for row in uncovered:
        raw = row["value"]
        cands = []
        # `_matches`, not a second rule of its own: the original bug survived
        # precisely because the two functions in this file disagreed about
        # matching and the one that was right only printed advice.
        for probe, pdec, pf in _candidates(raw, row.get("is_percent", False)):
            for fn, path, ev in flat:
                if _matches(probe, pdec, ev, pct_form=pf):
                    cands.append({"source": fn, "field": path, "value": ev})
                    if len(cands) >= 4:
                        break
            if cands:
                break
        out.append({**row, "candidates": cands})
    return out


def main(argv=None) -> int:
    import sys as _sys
    root = Path(__file__).resolve().parent.parent
    # every source the paper is built from, not just body.tex: moving a
    # subsection to the appendix must not take its numbers out of the gate
    paper = root / "paper"
    sources = [paper / "body.tex"] + sorted(
        f for f in paper.glob("*.tex")
        if f.name not in {"body.tex", "main.tex"})
    r = report(sources,
               root / "audit" / "claims.json",
               root / "audit" / "paper_numbers.json")
    if "--suggest" in (argv if argv is not None else _sys.argv[1:]):
        rows = suggest(r["uncovered"], root / "evidence")
        have = [x for x in rows if x["candidates"]]
        gone = [x for x in rows if not x["candidates"]]
        print(f"  uncovered {len(rows)}: locatable in the evidence bundle "
              f"{len(have)}, NOT IN THE BUNDLE {len(gone)}\n")
        print("  locatable -- confirm the field, then register "
              "(a value can collide with an unrelated field):")
        for x in have:
            c = x["candidates"][0]
            print(f"    {x.get('file','body.tex')}:{x['line']:<5} {x['value']:<9} "
                  f"{c['source'][:44]:<44} {c['field'][:52]}"
                  + (f"   (+{len(x['candidates']) - 1} other)"
                     if len(x["candidates"]) > 1 else ""))
        print("\n  NOT IN THE BUNDLE -- no artifact in this tree holds these; "
              "each needs a rerun, a recovered run directory, or a declaration:")
        for x in gone:
            print(f"    {x.get('file','body.tex')}:{x['line']:<5} {x['value']:<9} | "
                  f"{x['context'][:70]}")
        return 0 if not rows else 1
    print(f"  result numbers in the paper      {r['total']}")
    print(f"  covered by a registered claim    {r['claimed']}")
    print(f"  declared as non-measurement      {r['declared_count']}")
    print(f"  UNCOVERED                        {len(r['uncovered'])}")
    if r["ambiguous"]:
        print(f"\n  AMBIGUOUS ({len(r['ambiguous'])}): more than one claim's value rounds to "
              f"the printed number, so\n  the checker cannot say which quantity the paper "
              f"reports. Pin one with\n  {{\"claim\": \"SOME-ID\"}} at line:value in "
              f"audit/paper_numbers.json:")
        for row in r["ambiguous"][:12]:
            print(f"    {row.get('file','body.tex')}:{row['line']:<5} {row['value']:<9} "
                  f"{', '.join(row['claims_matching'][:4])}")
    if r["pins_unresolved"]:
        print(f"\n  PINNED BUT UNRESOLVED ({len(r['pins_unresolved'])}): the pinned claim is "
              f"gone or no longer matches:")
        for row in r["pins_unresolved"]:
            print(f"    body.tex:{row['line']:<5} {row['value']}")
    if r["stale_declarations"]:
        print("\n  declarations matching no number in the paper (the number moved "
              "or was reworded):")
        for k in r["stale_declarations"]:
            print(f"    {k}")
    if r["malformed_declarations"]:
        print("\n  declarations missing a reason or a why:")
        for k in r["malformed_declarations"]:
            print(f"    {k}")
    if r["uncovered"]:
        print("\n  each line below needs a claim, or an entry in "
              "audit/paper_numbers.json:")
        for row in r["uncovered"]:
            print(f"    body.tex:{row['line']:<5} {row['value']:<9} | "
                  f"{row['context']}")
    return 0 if r["pass_"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
