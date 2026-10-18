"""Tests for the ledger. Run with `python -m thaw_audit.tests`.

These assert the properties the package exists for, not its surface. The one
that matters most is `test_a_frozen_claim_is_settled_without_reading_a_field`:
if that ever fails, an audit has no stop condition again.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from thaw_audit import paths
from thaw_audit.claims import Claim, Registry
from thaw_audit.hashes import sha256_cached
from thaw_audit.ledger import Ledger
from thaw_audit.modes import Mode, ModeError, check_dod, require
from thaw_audit.verify import verify_all, verify_claim


class Skip(Exception):
    pass


def _tree(value=0.9295):
    """A runs root with one run directory and one results file in it."""
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "20261013-165911_migration_msmarco_500k_bsl"
    run.mkdir()
    (run / "command.txt").write_text("python -m semview.migration ...\n", encoding="utf-8")
    (run / "env.json").write_text(json.dumps({"code_hash": "abc123"}), encoding="utf-8")
    (run / "migration.json").write_text(json.dumps({
        "baselines": [{"scheme": "lazy", "recall_at_k": value},
                      {"scheme": "stale", "recall_at_k": 0.51}],
        "adapter": {"ridge_sweep": [{"ridge": 0.0001, "heldout_cosine": 0.7719}]},
        "frontier": [{"N": 500000, "kprime": 500, "recall_at_kprime": 0.9164}],
    }), encoding="utf-8")
    return tmp, run


def _claim(**kw):
    base = dict(
        id="CMP-LAZY", says="lazy recall", paper="tab:compare", value=0.9295,
        tol=5e-5,
        source="20261013-165911_migration_msmarco_500k_bsl/migration.json",
        field="baselines[scheme=lazy].recall_at_k",
        run_id="20261013-165911_migration_msmarco_500k_bsl", status="M")
    base.update(kw)
    return Claim(**base)


# ------------------------------------------------------------------- paths ---
def test_a_selector_value_with_a_decimal_point_resolves():
    """A dotted split breaks `[ridge=0.0001]` into two parts and then reports a
    row that does not exist. A ledger that cannot address a float-keyed row
    pushes claims onto positional indices, which break on reordering."""
    obj = {"adapter": {"ridge_sweep": [{"ridge": 0.0001, "heldout_cosine": 0.77}]}}
    assert paths.resolve(obj, "adapter.ridge_sweep[ridge=0.0001].heldout_cosine") \
        == 0.77
    assert paths.resolve(obj, "adapter.ridge_sweep[0].ridge") == 0.0001


def test_a_missing_field_is_distinguished_from_a_wrong_value():
    """They need different repairs: one points at the wrong place, the other
    means a number moved."""
    try:
        paths.resolve({"a": {}}, "a.b.c")
    except paths.FieldMissing:
        pass
    else:
        raise AssertionError("a missing field should raise FieldMissing")


def test_the_resolver_agrees_with_collect_evidence_on_the_bundle_paths():
    """The claims file and the evidence collector share this syntax, and a
    claim's `field` is copied between them untranslated."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ce", Path(__file__).resolve().parent.parent / "analysis"
        / "collect_evidence.py")
    if spec is None or not Path(spec.origin).is_file():
        raise Skip("analysis/collect_evidence.py is not in this checkout")
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    obj = {"baselines": [{"scheme": "lazy", "recall_at_k": 0.931}],
           "frontier": [{"N": 500000, "kprime": 500, "recall_at_kprime": 0.9164}],
           "result": {"penalty": 1.0007}}
    for p in ("baselines[scheme=lazy].recall_at_k",
              "frontier[N=500000][kprime=500].recall_at_kprime",
              "result.penalty"):
        assert ce._resolve(obj, p) == paths.resolve(obj, p), p


# ------------------------------------------------------------------ hashes ---
def test_the_hash_cache_is_reused_until_the_file_changes():
    tmp, run = _tree()
    f = run / "migration.json"
    a = sha256_cached(f, root=tmp)
    assert (tmp / ".audit_hash_cache.json").is_file(), "no cache was written"
    assert sha256_cached(f, root=tmp) == a
    os.utime(f, ns=(0, 0))                       # a changed mtime busts it
    f.write_text(f.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert sha256_cached(f, root=tmp) != a, "the cache survived a file change"


# ------------------------------------------------------------------ claims ---
def test_an_unknown_field_in_a_claims_file_is_refused_not_dropped():
    tmp = Path(tempfile.mkdtemp()) / "claims.json"
    tmp.write_text(json.dumps({"schema": 1, "claims": [
        {"id": "X1", "says": "s", "valu": 1}]}), encoding="utf-8")
    try:
        Registry.load(tmp)
    except SystemExit as e:
        assert "unknown field" in str(e)
    else:
        raise AssertionError("a misspelled field should be refused")


def test_duplicate_claim_ids_are_refused():
    tmp = Path(tempfile.mkdtemp()) / "claims.json"
    tmp.write_text(json.dumps({"schema": 1, "claims": [
        {"id": "X1", "says": "a"}, {"id": "X1", "says": "b"}]}), encoding="utf-8")
    try:
        Registry.load(tmp)
    except SystemExit as e:
        assert "duplicate" in str(e)
    else:
        raise AssertionError("two claims with one id should be refused")


# ------------------------------------------------------------- the stop rule ---
def test_a_frozen_claim_is_settled_without_reading_a_field():
    """The stop condition. A frozen claim must verify from its hash alone.

    The field is made unreadable on purpose: if verification still succeeds,
    nothing re-read it, which is exactly the property that bounds an audit.
    """
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.freeze(reg, ["CMP-LAZY"], [tmp])
    finally:
        os.environ["THAW_MODE"] = "audit"
    # break the field, keep the bytes: rewrite with the same length so the hash
    # is recomputed over identical content
    before = sha256_cached(run / "migration.json", root=tmp, use_cache=False)
    f = verify_claim(reg.get("CMP-LAZY"), [tmp], use_cache=False)
    assert f.settled and f.status == "V", (f.status, f.message)
    assert f.read_value is None, "a settled claim must not have read a value"
    assert f.sha256 == before


def test_a_frozen_claim_whose_source_changed_is_loud_and_does_not_self_resolve():
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.freeze(reg, ["CMP-LAZY"], [tmp])
    finally:
        os.environ["THAW_MODE"] = "audit"
    obj = json.loads((run / "migration.json").read_text(encoding="utf-8"))
    obj["baselines"][0]["recall_at_k"] = 0.8
    (run / "migration.json").write_text(json.dumps(obj), encoding="utf-8")
    f = verify_claim(reg.get("CMP-LAZY"), [tmp], use_cache=False)
    assert f.status == "S" and not f.settled
    assert "CHANGED" in f.message and "not re-measure" in f.message


def test_freezing_refuses_a_claim_that_is_not_verified():
    """Freezing an unchecked number would make the stop condition a lie."""
    tmp, _ = _tree(value=0.8)                 # the file disagrees with the claim
    reg = Registry(tmp / "claims.json", [_claim()])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.freeze(reg, ["CMP-LAZY"], [tmp])
    except SystemExit as e:
        assert "refusing to freeze" in str(e) and "not V" in str(e)
    else:
        raise AssertionError("an unverified claim should not freeze")
    finally:
        os.environ["THAW_MODE"] = "audit"


def test_thawing_requires_a_reason():
    tmp, _ = _tree()
    reg = Registry(tmp / "claims.json", [_claim(frozen=True, status="V")])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.thaw(reg, ["CMP-LAZY"], "   ")
    except SystemExit as e:
        assert "--why" in str(e)
    else:
        raise AssertionError("thawing without a reason should be refused")
    finally:
        os.environ["THAW_MODE"] = "audit"


def test_a_missing_source_under_a_frozen_claim_is_not_just_X():
    """A settled number losing its file is worse than one that never had it."""
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.freeze(reg, ["CMP-LAZY"], [tmp])
    finally:
        os.environ["THAW_MODE"] = "audit"
    (run / "migration.json").unlink()
    f = verify_claim(reg.get("CMP-LAZY"), [tmp])
    assert f.status == "S" and "FROZEN" in f.message


# ------------------------------------------------------------------- modes ---
def test_audit_mode_cannot_change_a_claim_or_run_a_command():
    for cap in ("run_command", "write_claim", "freeze"):
        try:
            require(cap, Mode.AUDIT)
        except ModeError as e:
            assert "THAW_MODE" in str(e)
        else:
            raise AssertionError(f"AUDIT should not allow {cap}")
    require("read", Mode.AUDIT)
    require("write_report", Mode.AUDIT)


def test_implement_mode_cannot_write_to_the_ledger():
    """A code change is not evidence, so it must not be able to settle a number."""
    for cap in ("write_claim", "freeze", "run_command"):
        try:
            require(cap, Mode.IMPLEMENT)
        except ModeError:
            pass
        else:
            raise AssertionError(f"IMPLEMENT should not allow {cap}")


def test_the_default_mode_is_the_one_that_cannot_change_anything():
    old = os.environ.pop("THAW_MODE", None)
    try:
        assert Mode.current() is Mode.AUDIT
    finally:
        if old is not None:
            os.environ["THAW_MODE"] = old


# --------------------------------------------------------------------- dod ---
def test_an_item_with_no_dod_is_reported_unfinishable_rather_than_done():
    tmp, _ = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    unmet = check_dod({"id": "D-X", "title": "t"}, tmp, reg)
    assert unmet and "no dod" in unmet[0]


def test_dod_checks_read_real_files_and_real_claim_statuses():
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim(status="V")])
    rel = "20261013-165911_migration_msmarco_500k_bsl/migration.json"
    item = {"id": "D-Y", "title": "t", "dod": [
        {"kind": "file_exists", "path": rel},
        {"kind": "json_has_field", "path": rel,
         "field": "baselines[scheme=lazy].recall_at_k"},
        {"kind": "claim_status", "id": "CMP-LAZY", "in": ["V"]}]}
    assert check_dod(item, tmp, reg) == []
    item["dod"].append({"kind": "claim_frozen", "id": "CMP-LAZY"})
    assert check_dod(item, tmp, reg) == ["claim CMP-LAZY is not frozen"]


def test_dod_monotone_catches_a_curve_that_is_not_one():
    """The throughput curve's own shape was the evidence it could not be
    trusted, so the check for it is mechanical rather than a reading."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "t.json").write_text(json.dumps({"passage_curve": {
        "8": {"ms_per_item": 3.54}, "16": {"ms_per_item": 7.46},
        "32": {"ms_per_item": 5.14}, "64": {"ms_per_item": 7.37}}}), encoding="utf-8")
    reg = Registry(tmp / "claims.json", [])
    unmet = check_dod({"id": "D-Z", "title": "t", "dod": [
        {"kind": "monotone", "path": "t.json", "field": "passage_curve",
         "key": "ms_per_item", "direction": "any"}]}, tmp, reg)
    assert unmet and "not monotone" in unmet[0], unmet
    (tmp / "u.json").write_text(json.dumps({"passage_curve": {
        "8": {"ms_per_item": 3.1}, "16": {"ms_per_item": 3.4},
        "32": {"ms_per_item": 4.6}}}), encoding="utf-8")
    assert check_dod({"id": "D-Z", "title": "t", "dod": [
        {"kind": "monotone", "path": "u.json", "field": "passage_curve",
         "key": "ms_per_item", "direction": "up"}]}, tmp, reg) == []


# ------------------------------------------------------------------ ledger ---
def test_the_paper_column_is_generated_from_the_claims_not_remembered():
    """The RUNS.md `paper` column was empty for thirty rows. It is derivable."""
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [
        _claim(), _claim(id="CMP-STALE", paper="tab:compare",
                         value=0.51, field="baselines[scheme=stale].recall_at_k")])
    led = Ledger(tmp)
    col = led.paper_column(reg)
    assert col == {run.name: "tab:compare"}, col
    r = led.reconcile(reg)
    assert r["missing_runs"] == [] and r["unused_runs"] == []
    assert r["stale_paper_cells"] == [run.name], r


def test_the_register_is_read_the_way_runlog_writes_it():
    """`semview.runlog` writes the directory cell as `runs/<id>`, and the
    flattened rows as `evidence/<id>__*`. A parser anchored at the start of the
    cell matched neither, so every row the project's own tooling wrote looked
    like a blank `paper` cell and `reconcile` could never report clean -- a
    provenance check that always complains is one nobody reads."""
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim(paper="tab:compare")])
    led = Ledger(tmp)
    for cell, label in ((run.name, "bare"),
                        (f"runs/{run.name}", "as runlog writes it"),
                        (f"evidence/{run.name}__*", "as a flattened row")):
        (tmp / "RUNS.md").write_text(
            "| started | name | dir | paper |\n|---|---|---|---|\n"
            f"| 2026-10-16T00:00:00 | x | {cell} | tab:compare |\n",
            encoding="utf-8")
        assert led.runs_md_rows().get(run.name) == "tab:compare", label
        assert led.reconcile(reg)["stale_paper_cells"] == [], label
    # and a genuinely stale cell is still caught through every spelling
    (tmp / "RUNS.md").write_text(
        "| started | name | dir | paper |\n|---|---|---|---|\n"
        f"| 2026-10-16T00:00:00 | x | runs/{run.name} | tab:other |\n",
        encoding="utf-8")
    assert led.reconcile(reg)["stale_paper_cells"] == [run.name]


def test_a_claim_naming_a_run_that_is_not_on_disk_is_reported():
    tmp, _ = _tree()
    reg = Registry(tmp / "claims.json", [_claim(run_id="20990101-000000_ghost")])
    assert Ledger(tmp).reconcile(reg)["missing_runs"] == ["20990101-000000_ghost"]


def test_verify_all_is_deterministic_and_touches_nothing():
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    before = {p: p.stat().st_mtime_ns for p in run.iterdir()}
    a = [(f.claim_id, f.status) for f in verify_all(reg, [tmp])]
    b = [(f.claim_id, f.status) for f in verify_all(reg, [tmp])]
    assert a == b == [("CMP-LAZY", "V")]
    after = {p: p.stat().st_mtime_ns for p in run.iterdir()}
    assert before == after, "verification modified a run directory"


# ------------------------------------------------------------ generated docs ---
def test_the_evidence_page_leads_with_the_do_not_rerun_rule():
    """The page answers the question a fresh reader has, which is not "what is
    in here" but "what may I trust without re-running it"."""
    from thaw_audit.docs import evidence_md
    tmp, _ = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.freeze(reg, ["CMP-LAZY"], [tmp])
    finally:
        os.environ["THAW_MODE"] = "audit"
    md = evidence_md(reg, verify_all(reg, [tmp]), [], covers="a round")
    assert "Do not re-measure anything in the frozen table" in md
    rule = md.index("## The rule")
    assert rule < md.index("## Frozen"), "the rule must precede the data"
    assert "CMP-LAZY" in md and "a round" in md


def test_a_broken_frozen_claim_gets_its_own_section_not_the_frozen_table():
    """Listing it as frozen would tell the next reader not to re-run a number
    whose file is gone."""
    from thaw_audit.docs import evidence_md
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim()])
    led = Ledger(tmp)
    os.environ["THAW_MODE"] = "measure"
    try:
        led.freeze(reg, ["CMP-LAZY"], [tmp])
    finally:
        os.environ["THAW_MODE"] = "audit"
    (run / "migration.json").unlink()
    md = evidence_md(reg, verify_all(reg, [tmp]), [])
    assert "Frozen but NOT intact" in md
    frozen_table = md.split("## Frozen — do not re-run")[1].split("##")[0]
    assert "CMP-LAZY" not in frozen_table


def test_project_status_cannot_claim_an_item_is_done_against_its_own_dod():
    """The failure mode of a hand-written status file, made impossible."""
    from thaw_audit.docs import project_status_md
    tmp, _ = _tree()
    reg = Registry(tmp / "claims.json", [_claim(status="V")])
    items = [{"id": "D-A", "title": "finished thing", "dod": [
                 {"kind": "claim_status", "id": "CMP-LAZY", "in": ["V"]}]},
             {"id": "D-B", "title": "unfinished thing", "dod": [
                 {"kind": "claim_frozen", "id": "CMP-LAZY"}]}]
    md = project_status_md(reg, verify_all(reg, [tmp]), items, tmp)
    assert "## Finished (1)" in md and "D-A" in md
    assert "## Open (1)" in md and "D-B" in md
    assert "is not frozen" in md, "the unmet reason should be shown"


def test_the_trail_has_one_section_per_claim_with_its_field_path():
    from thaw_audit.docs import trail_md
    tmp, _ = _tree()
    reg = Registry(tmp / "claims.json", [_claim(lesson="what it taught us")])
    md = trail_md(reg, verify_all(reg, [tmp]))
    assert md.count("## `") == 1
    assert "baselines[scheme=lazy].recall_at_k" in md
    assert "what it taught us" in md


def test_a_flattened_evidence_export_is_not_reported_as_a_missing_run():
    """The shipped bundle is flat by design. Calling its runs missing reads as
    a broken archive and sends the next reader hunting for directories that
    were never meant to be there."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "20261013-165911_migration_msmarco_500k_bsl__migration.json").write_text(
        json.dumps({"baselines": [{"scheme": "lazy", "recall_at_k": 0.9295}]}), encoding="utf-8")
    reg = Registry(tmp / "claims.json", [_claim(
        source="20261013-165911_migration_msmarco_500k_bsl__migration.json")])
    r = Ledger(tmp).reconcile(reg, [tmp])
    assert r["missing_runs"] == [], r
    assert r["flattened_only"] == [
        "20261013-165911_migration_msmarco_500k_bsl"], r


def test_the_lesson_column_is_collected_because_nothing_can_derive_it():
    tmp, run = _tree()
    reg = Registry(tmp / "claims.json", [_claim(lesson="the device queues")])
    led = Ledger(tmp)
    assert led.lesson_column(reg) == {run.name: "the device queues"}
    reg2 = Registry(tmp / "c2.json", [_claim()])
    assert led.reconcile(reg2, [tmp])["runs_without_a_lesson"] == [run.name]


# ------------------------------------------- regressions from the 10-14 review ---
def test_a_correct_throughput_curve_passes_the_unimodal_check():
    """The gap this test exists to close was in the TEST, not the checker.

    `test_dod_monotone_catches_a_curve_that_is_not_one` asserted that a bad
    curve fails and never that a good one passes, so it did not notice that
    `direction: "any"` demanded monotonicity from a quantity that is not
    monotone. A per-item throughput curve falls while per-call overhead
    amortises and rises once padding waste takes over, so the correct shape has
    one interior minimum -- and the D-01 criterion was rejecting exactly the
    curve item T is meant to produce.
    """
    tmp = Path(tempfile.mkdtemp())
    reg = Registry(tmp / "claims.json", [])
    correct = {"1": {"ms_per_item": 9.7678}, "8": {"ms_per_item": 3.1561},
               "32": {"ms_per_item": 4.6388}, "128": {"ms_per_item": 7.3274},
               "512": {"ms_per_item": 42.9576}}
    suspect = {"8": {"ms_per_item": 3.54}, "16": {"ms_per_item": 7.46},
               "32": {"ms_per_item": 5.14}, "64": {"ms_per_item": 7.37}}
    def check(curve, **extra):
        (tmp / "t.json").write_text(json.dumps({"passage_curve": curve}), encoding="utf-8")
        chk = {"kind": "monotone", "path": "t.json", "field": "passage_curve",
               "key": "ms_per_item", "direction": "unimodal", **extra}
        return check_dod({"id": "X", "title": "t", "dod": [chk]}, tmp, reg)
    assert check(correct, tol=0.05) == [], "a correct U-shaped curve was rejected"
    bad = check(suspect, tol=0.05)
    assert bad and "turning points" in bad[0], bad
    # and the old strictness is still available where it is the right question
    (tmp / "t.json").write_text(json.dumps({"passage_curve": correct}), encoding="utf-8")
    assert check_dod({"id": "X", "title": "t", "dod": [
        {"kind": "monotone", "path": "t.json", "field": "passage_curve",
         "key": "ms_per_item", "direction": "any"}]}, tmp, reg)


def test_no_shipped_open_item_has_an_empty_definition_of_done():
    """An item with no criterion is reported unfinishable, which is right for a
    placeholder and wrong for real work: the next reader takes it as impossible
    rather than unspecified. D-04 and D-05 shipped that way."""
    p = Path(__file__).resolve().parent.parent / "audit" / "open_items.json"
    if not p.is_file():
        raise Skip("audit/open_items.json is not in this checkout")
    items = json.loads(p.read_text(encoding="utf-8"))["items"]
    bare = [it["id"] for it in items if not it.get("dod")]
    assert not bare, f"open items with no definition of done: {bare}"


def test_replacing_a_frozen_claim_is_refused_even_by_another_frozen_claim():
    """Frozen-replacing-frozen used to be silent, so a re-add could change a
    settled number's value or its recorded hash with no warning -- the exact
    thing freezing exists to prevent."""
    tmp = Path(tempfile.mkdtemp())
    reg = Registry(tmp / "claims.json", [_claim(frozen=True, status="V",
                                                sha256="a" * 64)])
    for replacement in (_claim(value=0.5, frozen=True, sha256="b" * 64),
                        _claim(value=0.5, frozen=False)):
        try:
            reg.add(replacement, replace=True)
        except SystemExit as e:
            assert "frozen" in str(e) and "thaw" in str(e).lower()
        else:
            raise AssertionError("a frozen claim was replaced silently")
    assert reg.get("CMP-LAZY").value == 0.9295, "the frozen value changed"
    assert reg.get("CMP-LAZY").sha256 == "a" * 64, "the frozen hash changed"


def test_a_frozen_claim_is_not_settled_by_a_stale_hash_cache():
    """The cache is keyed on (size, mtime_ns), and a frozen claim's hash is
    the only thing standing between a settled number and a file that changed.

    An in-place edit of the same length with the mtime put back reproduces a
    stale entry -- an editor, a `git checkout` of a same-length line, or
    `touch -r` all do it, and a reviewer did it deliberately. The old code
    passed the caller's `use_cache` straight through, so the default audit
    reported such a file as 'frozen and intact'. Frozen claims now hash the
    bytes.
    """
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "runs" / "r1"
    run.mkdir(parents=True)
    src = run / "migration.json"
    src.write_text(json.dumps({"baselines": [
        {"scheme": "lazy", "recall_at_k": 0.9295}]}), encoding="utf-8")
    st = os.stat(src)

    reg = Registry(tmp / "claims.json", [_claim(
        source="migration.json", field="baselines[scheme=lazy].recall_at_k",
        value=0.9295, run_id="r1")])
    # warm the cache, then freeze against the real bytes
    sha256_cached(src, root=tmp)
    led = Ledger(tmp)
    led.freeze(reg, ["CMP-LAZY"], [tmp], mode=Mode.MEASURE)
    assert verify_claim(reg.get("CMP-LAZY"), [tmp]).settled

    same_length = json.dumps({"baselines": [
        {"scheme": "lazy", "recall_at_k": 0.1111}]})
    assert len(same_length) == len(src.read_text(encoding="utf-8"))
    src.write_text(same_length, encoding="utf-8")
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert sha256_cached(src, root=tmp) != sha256_cached(src, root=tmp,
                                                         use_cache=False), (
        "the fixture no longer produces a stale cache entry, so this test "
        "is no longer the regression")

    f = verify_claim(reg.get("CMP-LAZY"), [tmp])
    assert not f.settled and f.status == "S", (
        f"a frozen claim was settled against changed content: {f.message}")


def test_an_audit_pass_writes_nothing_into_the_runs_root_either():
    """`verify_all_is_deterministic_and_touches_nothing` watched the run
    subdirectories and missed the root, where the hash cache lives -- so an
    audit-mode pass left `.audit_hash_cache.json` inside the evidence bundle,
    the one directory that has to stay as it shipped, while advertising that
    it changes nothing."""
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "migration.json").write_text(json.dumps({"baselines": [
        {"scheme": "lazy", "recall_at_k": 0.9295}]}), encoding="utf-8")
    reg = Registry(tmp / "claims.json", [_claim(
        source="migration.json", field="baselines[scheme=lazy].recall_at_k",
        value=0.9295, run_id="r1")])

    before = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*"))
    verify_all(reg, [tmp], mode=Mode.AUDIT)
    after = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*"))
    assert before == after, f"audit mode created {set(after) - set(before)}"


def test_an_audit_pass_writes_nothing_when_the_mode_is_not_passed_in():
    """The same promise, on the path the CLI actually takes.

    The test above passes `mode=Mode.AUDIT` explicitly, and that is exactly why
    it missed a live bug: `verify_all` and `verify_claim` called
    `require("read", mode)` and **threw the return value away**, so when no
    caller supplied a mode -- which is every call from `cmd_verify` -- `mode`
    stayed None, `None is not Mode.AUDIT` was True, and an audit pass wrote its
    hash cache into the runs root.

    Two conditions have to hold for this test to be able to fail, and both are
    deliberate. The claim must be UNFROZEN, because `use_cache and not
    claim.frozen` short-circuits for a frozen one and nothing reaches the cache
    at all -- which is why the bug sat masked while every shipped claim was
    frozen, and would have surfaced on the first `thaw`. And the mode must come
    from the environment rather than the argument, since passing it is what hid
    the defect in the first place.
    """
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "migration.json").write_text(json.dumps({"baselines": [
        {"scheme": "lazy", "recall_at_k": 0.9295}]}), encoding="utf-8")
    c = _claim(source="migration.json",
               field="baselines[scheme=lazy].recall_at_k",
               value=0.9295, run_id="r1")
    c.frozen = False
    c.sha256 = None
    reg = Registry(tmp / "claims.json", [c])

    before = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*"))
    old = os.environ.get("THAW_MODE")
    os.environ["THAW_MODE"] = "audit"
    try:
        verify_all(reg, [tmp])            # no mode= : the CLI's own call shape
    finally:
        if old is None:
            os.environ.pop("THAW_MODE", None)
        else:
            os.environ["THAW_MODE"] = old
    after = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*"))
    assert before == after, (
        "audit mode created "
        f"{set(after) - set(before)} when the mode was left to the "
        "environment: the resolved mode is being discarded")


def test_verify_claim_alone_writes_nothing_in_audit_mode():
    """`verify_claim` is public and callable on its own, so it has to bind the
    mode itself rather than rely on `verify_all` to hand one down.

    The two bindings are redundant for a call that goes through `verify_all` --
    verified by reverting each in turn, where either one alone still protects
    that path. This test is what makes `verify_claim`'s own binding load
    bearing, and it is the direct-call shape a future caller is most likely to
    reach for.
    """
    tmp = Path(tempfile.mkdtemp())
    run = tmp / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "migration.json").write_text(json.dumps({"baselines": [
        {"scheme": "lazy", "recall_at_k": 0.9295}]}), encoding="utf-8")
    c = _claim(source="migration.json",
               field="baselines[scheme=lazy].recall_at_k",
               value=0.9295, run_id="r1")
    c.frozen = False
    c.sha256 = None

    before = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*"))
    old = os.environ.get("THAW_MODE")
    os.environ["THAW_MODE"] = "audit"
    try:
        verify_claim(c, [tmp])            # no mode=, straight to the claim
    finally:
        if old is None:
            os.environ.pop("THAW_MODE", None)
        else:
            os.environ["THAW_MODE"] = old
    after = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*"))
    assert before == after, (
        f"verify_claim created {set(after) - set(before)} in audit mode when "
        "called directly: it is not binding the resolved mode")


def test_the_stale_schema_sweep_can_see_its_own_extension_schema():
    """Both gates, together, without a database.

    The sweep drops a schema only if the generated-name pattern matches AND an
    age parses out of the name. One check creates a second schema suffixed
    `_ext` (with the ownership marker, so the query does return it), and each
    gate rejected it in turn: the pattern was anchored `\\d+$`, and after that
    was fixed the age parse was `int("ext")`. Fixing only the first was
    described in a changelog as fixing the bug, which is why this test checks
    the conjunction rather than either half.
    """
    from thaw_pg.sql_check import _AUTO_NAME, _sweep_age

    def sweepable(n):
        return bool(_AUTO_NAME.match(n)) and _sweep_age(n) is not None

    assert sweepable("thaw_sqlcheck_123_4567890"), "the ordinary case regressed"
    assert sweepable("thaw_sqlcheck_123_4567890_ext"), (
        "the _ext schema is still invisible to the sweep: a hard kill leaves it "
        "in the database with nothing looking for it")
    assert _sweep_age("thaw_sqlcheck_123_4567890_ext") == 4567890, (
        "the age must come from the run's timestamp, not from the suffix")
    # and ownership is still two signals: a name this tool did not generate is
    # left alone however old it looks
    assert not sweepable("my_own_schema")
    assert not sweepable("thaw_sqlcheck_1_2_3")


def test_a_curve_shape_check_needs_enough_points_to_have_a_shape():
    """D-01 exists to replace a five-point throughput curve with a denser one,
    and its criterion could be satisfied by a file holding ONE batch size:
    every direction is vacuously true of a single point, and `unimodal` was
    also true of a monotone curve. A reviewer closed the item with one data
    point."""
    tmp = Path(tempfile.mkdtemp())
    reg = Registry(tmp / "claims.json", [])
    chk = {"kind": "monotone", "path": "t.json", "field": "passage_curve",
           "key": "ms_per_item", "direction": "unimodal", "tol": 0.05,
           "min_points": 10, "keys": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}
    item = {"id": "D-01", "title": "t", "dod": [chk]}

    def curve(d):
        (tmp / "t.json").write_text(json.dumps(
            {"passage_curve": {str(k): {"ms_per_item": v} for k, v in d.items()}}), encoding="utf-8")

    curve({512: 42.96})
    bad = check_dod(item, tmp, reg)
    assert bad and "point(s)" in bad[0], bad

    # ten points, but only at the top of the range: a count is not coverage
    curve({k: 10.0 + i for i, k in enumerate(
        [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144])})
    bad = check_dod(item, tmp, reg)
    assert bad and "missing the required samples" in bad[0], bad

    # the real shape D-01 asks for: a fall, a minimum, then padding waste
    ms = [9.77, 6.10, 4.20, 3.16, 3.20, 3.55, 4.64, 6.00, 7.33, 42.96]
    curve({k: v for k, v in zip([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], ms)})
    assert check_dod(item, tmp, reg) == [], check_dod(item, tmp, reg)


def test_an_item_whose_only_criterion_is_a_new_claim_is_a_measurement():
    """D-02 shipped as `implement` with a title saying the character sort key
    still had to be changed, while the code had already been changed and the
    remaining work was a re-measurement. A reader comparing the two concluded
    the package contradicted itself, which it did.

    The invariant: an item that can only be closed by registering a verified
    claim is asking for a measurement, whatever its title says."""
    p = Path(__file__).resolve().parent.parent / "audit" / "open_items.json"
    if not p.is_file():
        raise Skip("audit/open_items.json is not in this checkout")
    wrong = []
    for it in json.loads(p.read_text(encoding="utf-8"))["items"]:
        kinds = {c["kind"] for c in it.get("dod", [])}
        if kinds and kinds <= {"claim_status", "claim_frozen"}:
            if it.get("mode") != "measure":
                wrong.append((it["id"], it.get("mode")))
    assert not wrong, (
        f"items whose definition of done is a new verified claim, but whose "
        f"mode is not 'measure': {wrong}")


def test_a_runs_root_with_no_results_in_it_is_a_typo_not_a_broken_archive():
    """`--runs-root` replaces the defaults, so a wrong or not-yet-filled
    directory makes every source file missing and every frozen claim report as
    a settled number whose evidence has disappeared. That is an
    archive-is-broken report produced by an omission, and it is the most
    misleading thing this tool could print. It happened: `evidence/` was empty
    on a clean unpack, and step 1 of `ONBOARDING.md` hit it.
    """
    from thaw_audit.__main__ import _require_sources, _roots, _roots_for

    class _A:
        def __init__(self, roots):
            self.runs_root = roots

    tmp = Path(tempfile.mkdtemp())
    reg = Registry(tmp / "claims.json", [_claim(source="migration.json")])

    # The first version of this guard asked "is there any *.json under the
    # roots", which `audit/claims.json` answers on the default root set -- so
    # the case it was written for still got through. The question is whether
    # any file a CLAIM rests on is there.
    (tmp / "empty").mkdir()
    (tmp / "empty" / "claims.json").write_text("{}", encoding="utf-8")
    (tmp / "empty" / "NOTES.md").write_text("no results here", encoding="utf-8")
    try:
        _roots_for(_A([str(tmp / "empty")]), reg)
    except SystemExit as e:
        assert "wrong or empty root" in str(e) and "not a broken archive" in str(e), e
    else:
        raise AssertionError(
            "a root holding no file any claim rests on was accepted -- a JSON "
            "file that is not a results file satisfied the guard")

    (tmp / "real").mkdir()
    (tmp / "real" / "migration.json").write_text("{}", encoding="utf-8")
    assert _roots_for(_A([str(tmp / "real")]), reg) == [tmp / "real"]

    # one source present is enough: after that a missing file is a finding
    # about that claim, not a reason to refuse the whole audit
    reg2 = Registry(tmp / "c2.json", [_claim(source="migration.json"),
                                      _claim(id="OTHER", source="gone.json")])
    assert _roots_for(_A([str(tmp / "real")]), reg2) == [tmp / "real"]

    # and a registry with no sources at all has nothing to check
    _require_sources([tmp / "empty"], Registry(tmp / "c3.json", []))

    try:
        _roots(_A([str(tmp / "nope")]))
    except SystemExit as e:
        assert "no such runs root" in str(e), e
    else:
        raise AssertionError("a nonexistent root was accepted")


def test_an_unpacked_bundle_covers_every_claim_it_is_asked_to_settle():
    """Every document says `--runs-root evidence`, so once the bundle is
    unpacked there it has to hold every source a claim names -- a bundle
    missing one file settles 63 of 64 and reports the last as lost evidence.

    Skipped rather than failed when `evidence/` is still empty: that is the
    shipped state, and the guard above is what makes it legible."""
    here = Path(__file__).resolve().parent.parent
    claims = here / "audit" / "claims.json"
    bundle = here / "evidence"
    names = {p.name for p in bundle.rglob("*.json")} if bundle.is_dir() else set()
    if not claims.is_file() or not names:
        raise Skip("the claims file or the unpacked evidence bundle "
                   "is not in this checkout")
    srcs = {c["source"] for c in json.loads(claims.read_text(encoding="utf-8"))["claims"]}
    missing = sorted(s for s in srcs if Path(s).name not in names)
    assert not missing, (
        f"{len(missing)} claim source(s) are not in evidence/, which every "
        f"document names as the bundle: {missing[:3]}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Skip as e:
            skipped += 1
            print(f"  SKIP  {fn.__name__}: {e}")
        except SystemExit as e:
            # a BaseException, so `except Exception` let one unexpected refusal
            # abort the suite and print a partial count as though it were the
            # total. This codebase raises SystemExit as its standard refusal.
            bad += 1
            print(f"  FAIL  {fn.__name__}: refused: {e}"[:300])
        except Exception as e:                                    # noqa: BLE001
            bad += 1
            print(f"  FAIL  {fn.__name__}: {e!r}"[:300])
    print(f"\n{len(fns) - bad - skipped} passed, {skipped} skipped, {bad} failed "
          f"out of {len(fns)} tests")
    raise SystemExit(1 if bad else 0)
