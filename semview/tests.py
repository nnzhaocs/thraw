"""Tests for the migration core.

    python -m semview.tests          (or: python -m pytest semview/tests.py)

Each test pins a property the paper's numbers depend on, and several pin a
specific mistake made while building this code -- recorded so it cannot come
back quietly:

  * the D sample was once the FIRST d queries, which inflated savings when the
    query list was grouped by repository;
  * the lexical stand-ins were once fitted on units+queries, which degraded
    recall and leaked the evaluation queries;
  * the closed form was once 1 - min(1, D*K'/N), which misstates savings by up
    to 28 points; the measured law is exp(-D*K'/N);
  * the curve fit was once a plain exponential, which projected a 0.49
    asymptote where the observed value was 0.99.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np
from pathlib import Path

from semview import baselines, cost, estimate, migration, online, procrustes, serve


try:                                   # under pytest, a Skip is a pytest skip
    import pytest as _pytest
    _SkipBase = _pytest.skip.Exception
except ImportError:
    _SkipBase = Exception


class Skip(_SkipBase):
    """A test whose fixture is absent here. Reported as SKIP, never as PASS:
    a count of passes must not include checks that did not run."""


#: Top-level packages this suite uses but does not require. `requirements.txt`
#: lists them, and a reviewer who runs the suite before installing them should
#: see SKIP with the reason, not a wall of ImportError failures that look like
#: broken code. `scipy` is the curve fit in `semview.estimate`; `sklearn` is
#: the offline lexical stand-in encoders. Nothing in the migration core needs
#: either.
OPTIONAL_DEPENDENCIES = ("scipy", "sklearn", "matplotlib", "qdrant_client",
                         "sentence_transformers", "torch", "transformers")


def _missing_optional(exc: ImportError) -> str | None:
    """The optional package this ImportError is about, or None.

    A missing optional dependency is a skip; any other ImportError is a real
    failure and stays one — a typo in one of our own module names must not be
    silently reported as "not installed".
    """
    name = (getattr(exc, "name", None) or "").split(".")[0]
    if name in OPTIONAL_DEPENDENCIES:
        return name
    text = str(exc)
    for cand in OPTIONAL_DEPENDENCIES:
        if f"'{cand}'" in text or f"named {cand}" in text:
            return cand
    return None


def _main_ast(mod):
    """The parsed body of mod.main -- comments and strings excluded."""
    import ast
    import inspect
    import textwrap

    return ast.parse(textwrap.dedent(inspect.getsource(mod.main)))


def _is_a_old(node):
    import ast
    return (isinstance(node, ast.Attribute) and node.attr == "old"
            and isinstance(node.value, ast.Name) and node.value.id == "a")


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _clustered(n, d=32, topics=20, spread=0.2, seed=0):
    rng = np.random.default_rng(seed)
    C = _unit(rng.normal(size=(topics, d)))
    lab = rng.integers(0, topics, size=n)
    return _unit(C[lab] + spread * rng.normal(size=(n, d))), C, lab


# ---------------------------------------------------------------- top-k ----
def test_topk_matches_brute_force():
    rng = np.random.default_rng(1)
    V = _unit(rng.normal(size=(500, 16)))
    Q = _unit(rng.normal(size=(40, 16)))
    got = migration.topk_chunked(Q, V, 10, budget=5_000)   # force many chunks
    want = np.argsort(-(Q @ V.T), axis=1)[:, :10]
    assert np.array_equal(np.sort(got, axis=1), np.sort(want, axis=1))
    # and in descending score order
    S = Q @ V.T
    for i in range(len(Q)):
        s = S[i, got[i]]
        assert np.all(np.diff(s) <= 1e-6)


def test_prefix_of_deep_topk_is_the_shallow_topk():
    """frontier() searches once at the deepest K' and slices; that is only
    valid if the first K' columns of a top-K'max result are the top-K'."""
    rng = np.random.default_rng(13)
    V = _unit(rng.normal(size=(800, 16)))
    Q = _unit(rng.normal(size=(30, 16)))
    deep = migration.topk_chunked(Q, V, 200)
    for kp in (10, 50, 120):
        shallow = migration.topk_chunked(Q, V, kp)
        for i in range(len(Q)):
            assert set(deep[i, :kp]) == set(shallow[i]), (kp, i)


def test_topk_clamps_k_to_corpus():
    V = _unit(np.random.default_rng(2).normal(size=(7, 8)))
    Q = V[:3]
    assert migration.topk_chunked(Q, V, 50).shape == (3, 7)


# ------------------------------------------------------------- frontier ----
def test_identical_spaces_give_perfect_recall():
    V, _, _ = _clustered(400)
    Q = V[:30] + 0.01
    Q = _unit(Q)
    rows = migration.frontier(V, V, Q, Q, k=10, kprimes=(10, 50))
    for r in rows:
        assert r["recall_at_kprime"] == 1.0, r


def test_recall_is_monotone_in_kprime():
    Vo, _, _ = _clustered(600, seed=3)
    Vn, _, _ = _clustered(600, seed=4)          # a different space
    Qo, Qn = Vo[:40], Vn[:40]
    rows = migration.frontier(Vo, Vn, Qo, Qn, k=10,
                              kprimes=(10, 20, 50, 100, 200))
    rec = [r["recall_at_kprime"] for r in rows]
    assert all(a <= b + 1e-12 for a, b in zip(rec, rec[1:])), rec


def test_chance_corrected_recall_removes_the_k_over_n_floor():
    """Unrelated spaces must score ~0 after correction, however large K'/N.

    Raw recall@K' rises toward 1 as K' approaches N even for independent
    spaces; the corrected value must not. Two independently drawn corpora have
    no shared structure, so their corrected recall should sit near zero at
    every K'.
    """
    rng = np.random.default_rng(12)
    Vo = _unit(rng.normal(size=(1000, 16))); Vn = _unit(rng.normal(size=(1000, 16)))
    Qo = _unit(rng.normal(size=(80, 16)));   Qn = _unit(rng.normal(size=(80, 16)))
    rows = migration.frontier(Vo, Vn, Qo, Qn, k=10, kprimes=(50, 200, 500))
    for r in rows:
        assert abs(r["recall_chance_corrected"]) < 0.08, r
    raw = [r["recall_at_kprime"] for r in rows]
    assert raw[-1] > raw[0], "raw recall should still climb with K'"


def test_savings_equals_one_minus_reached():
    V, _, _ = _clustered(500)
    Q = V[:50]
    for r in migration.frontier(V, V, Q, Q, k=10, kprimes=(20, 100)):
        assert abs(r["savings_pct"] - 100 * (1 - r["reached"])) < 0.02


def test_reported_closed_forms_are_the_documented_ones():
    V, _, _ = _clustered(500)
    Q = V[:50]
    for r in migration.frontier(V, V, Q, Q, k=10, kprimes=(20,),
                                d_fracs=(0.1, 1.0)):
        x = r["D"] * r["kprime"] / r["N"]
        assert abs(r["x_dk_over_n"] - round(x, 4)) < 1e-9          # unclipped
        assert abs(r["pred_savings_exp"] - round(100 * math.exp(-x), 2)) < 1e-9
        assert abs(r["pred_savings_linear"] - round(100 * max(0, 1 - x), 2)) < 1e-9
        assert "bound_dk_over_n" not in r      # the misleading name is gone


def test_d_seed_varies_only_the_d_sample():
    """Error bars must not require re-encoding the corpus.

    --seed reseeds the corpus sample as well, so each repeat is a different
    corpus and a cold cache; --d-seed moves only the intent draw.
    """
    import ast
    tree = _main_ast(migration)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "frontier"]
    assert calls, "main no longer calls frontier"
    kw = {k.arg for c in calls for k in c.keywords}
    assert "seed" in kw
    src = ast.unparse(calls[0])
    assert "d_seed" in src, "frontier's seed must honour --d-seed"

    # the fallback: unset --d-seed must use --seed, not a fixed default
    assert "a.seed if a.d_seed is None else a.d_seed" in ast.unparse(calls[0]), \
        "--d-seed must fall back to --seed when unset"

    V, _, _ = _clustered(600, seed=20)
    Q = V[:80]
    a = migration.frontier(V, V, Q, Q, k=10, kprimes=(30,), d_fracs=(0.25,), seed=1)
    b = migration.frontier(V, V, Q, Q, k=10, kprimes=(30,), d_fracs=(0.25,), seed=2)
    assert a[0]["recall_at_kprime"] == b[0]["recall_at_kprime"], "recall is seed-free"
    assert a[0]["reached"] != b[0]["reached"], "the D draw must move with the seed"


def test_d_sample_is_random_not_a_prefix():
    """Grouped query lists must not bias the D sample.

    Queries sorted by topic: the first 10% all come from one topic, so a
    prefix sample reaches only that topic's units. A random sample of the same
    size spreads across topics and reaches far more. If `frontier` ever goes
    back to taking a prefix, reached collapses toward the prefix value.
    """
    V, C, lab = _clustered(2000, topics=20, spread=0.15, seed=5)
    order = np.argsort(lab, kind="stable")
    Q = V[order[:400]]                           # grouped by topic
    Q = Q[np.argsort(lab[order[:400]], kind="stable")]
    k, kp, d = 10, 20, 40

    win = migration.topk_chunked(Q, V, kp)
    seen = np.zeros(len(V), bool)
    seen[np.unique(win[:d])] = True
    prefix_reached = seen.mean()

    row = [r for r in migration.frontier(V, V, Q, Q, k=k, kprimes=(kp,),
                                         d_fracs=(d / len(Q),))
           if r["D"] == d][0]
    assert row["reached"] > prefix_reached * 1.5, (row["reached"], prefix_reached)


# -------------------------------------------------------------- old side ---
def test_shipped_vectors_are_used_unchanged_and_never_reencoded():
    """With shipped vectors, the old index is those vectors -- exactly.

    Regression for a real failure: the DPR branch was meant to use the shipped
    vectors, a patch adding it silently failed to apply, and every "DPR -> BGE"
    run loaded the vectors and then overwrote them with a lexical encoder. This
    test fails if `encode_all` is reached, or if the vectors are altered.
    """
    rng = np.random.default_rng(9)
    V_shipped = rng.normal(size=(50, 8)).astype(np.float32)   # unnormalised, like DPR
    calls = []

    def qenc(texts):
        calls.append(list(texts))
        return rng.normal(size=(len(texts), 8)).astype(np.float32)

    orig = migration.encode_all
    migration.encode_all = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("encode_all must not run when shipped vectors exist"))
    try:
        V, Q = migration.select_old_side(["u"] * 50, ["q1", "q2"], "lexical:word",
                                         V_shipped=V_shipped, question_encoder=qenc)
    finally:
        migration.encode_all = orig
    assert V is V_shipped, "shipped vectors must be the old index, not a copy"
    assert np.array_equal(V, V_shipped)
    assert calls == [["q1", "q2"]], "queries must go through the matched encoder"
    assert Q.shape == (2, 8)


def test_shipped_vectors_without_their_question_encoder_are_refused():
    try:
        migration.select_old_side(["u"], ["q"], "lexical:word",
                                  V_shipped=np.ones((1, 4), np.float32))
    except ValueError:
        return
    raise AssertionError("must not fall back to another model for the queries")


def test_old_normalize_ablation_applies_only_when_asked():
    rng = np.random.default_rng(10)
    V_shipped = (rng.normal(size=(30, 6)) * 5).astype(np.float32)
    qenc = lambda t: (rng.normal(size=(len(t), 6)) * 3).astype(np.float32)
    V0, _ = migration.select_old_side([""] * 30, ["q"], "x", V_shipped=V_shipped,
                                      question_encoder=qenc, normalize=False)
    assert not np.allclose(np.linalg.norm(V0, axis=1), 1.0), "default: raw vectors"
    V1, Q1 = migration.select_old_side([""] * 30, ["q"], "x", V_shipped=V_shipped,
                                       question_encoder=qenc, normalize=True)
    assert np.allclose(np.linalg.norm(V1, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(Q1, axis=1), 1.0, atol=1e-5)


def test_reencode_old_actually_changes_the_old_side():
    """Without it, --old is silently ignored for DPR.

    `select_old_side` uses shipped vectors whenever it gets them, which is
    right for a normal DPR run and wrong for the control experiment that holds
    the corpus fixed and changes only the model pair. `--reencode-old` must
    drop the shipped vectors, and the recorded label must follow -- otherwise
    the results file names a model that never ran.
    """
    import ast

    tree = _main_ast(migration)
    src = ast.unparse(tree)
    assert "shipped = None if a.reencode_old else V_old_pre" in src
    # everything downstream must consume `shipped`, not the raw shipped vectors
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "select_old_side":
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            assert kw.get("V_shipped") == "shipped", kw
    assert migration.old_label("m", None, False) == "m", "label follows the model used"

    # and the behaviour of the underlying function is unchanged
    V = np.ones((4, 3), np.float32)
    got, _ = migration.select_old_side(["u"] * 4, ["q"], "lexical:word",
                                       V_shipped=V, question_encoder=lambda t: V[:1])
    assert got is V


def test_main_routes_dpr_through_select_old_side():
    """`main` must hand shipped vectors to select_old_side, not bypass it.

    Checked on the syntax tree rather than with a regex, so it catches
    `encode_all(utext, model=a.old)` and any other spelling, and cannot be
    fooled by comments. main() cannot run here without network; this is the
    exact property the silent patch failure broke.
    """
    import ast

    for mod in (migration, estimate):
        tree = _main_ast(mod)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)]
        sel = [c for c in calls if c.func.id == "select_old_side"]
        assert sel, f"{mod.__name__}.main bypasses select_old_side"
        assert any(k.arg == "V_shipped" for c in sel for k in c.keywords), (
            f"{mod.__name__}.main never passes the shipped vectors")
        # The NEW model is legitimately encoded with encode_all. encode_all on
        # the OLD model inside main is what overwrote the DPR vectors.
        for c in calls:
            if c.func.id == "encode_all":
                args = list(c.args) + [k.value for k in c.keywords]
                assert not any(_is_a_old(x) for x in args), (
                    f"{mod.__name__}.main encodes the old side directly")


def test_every_declared_cli_flag_is_actually_read():
    """A flag the parser accepts but main() never reads is a silent no-op.

    `--old-normalize` was exactly that: declared, documented, validated by a
    check that only looked at whether the parser knew the flag, and ignored.
    """
    import ast

    for mod in (migration, estimate, serve):
        tree = _main_ast(mod)
        declared = [
            n.args[0].value[2:].replace("-", "_")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument" and n.args
            and isinstance(n.args[0], ast.Constant)
            and str(n.args[0].value).startswith("--")
        ]
        # a.<dest> as an attribute READ on the syntax tree, so a mention in a
        # comment or help string does not count
        read = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name) and n.value.id == "a"}
        for dest in declared:
            assert dest in read, (
                f"{mod.__name__}: --{dest} is declared but never read")


# -------------------------------------------------------------- encoders ---
def test_lexical_standin_fits_on_units_only_and_normalises():
    units = [f"def f{i}(x): return x + {i} # helper number {i}" for i in range(80)]
    queries = ["helper that adds a number", "return x plus"]
    V, fitted = migration.encode_all(units, "lexical:word")
    Q = migration.encode_with(fitted, queries, "lexical:word")
    assert V.shape[0] == len(units) and Q.shape[0] == len(queries)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-4)
    # fitted on units only: the vectoriser vocabulary must not contain
    # query-only tokens
    vocab = fitted.steps[0][1].vocabulary_
    assert "adds" not in vocab and "plus" not in vocab


def test_dpr_encoder_pairing_is_derived_from_the_variant():
    q, c = migration.dpr_encoders_for("psgs_w100.nq.no_index")
    assert "single-nq" in q and "single-nq" in c and "question" in q
    q, c = migration.dpr_encoders_for("psgs_w100.multiset.exact")
    assert "multiset" in q and "multiset" in c
    for bad in ("psgs_w100.bogus", "psgs_w100.nq_extra.exact", "psgs_w100"):
        try:
            migration.dpr_encoders_for(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must be rejected, not prefix-matched")


# -------------------------------------------------------------- loaders ----
def test_null_titles_do_not_crash_and_text_survives():
    """`r.get("title", "")` returns None when the key is present but null."""
    assert migration._join(None, "body") == "body"
    assert migration._join("", "body") == "body"
    assert migration._join("T", "body") == "T body"
    assert migration._join("T", None) == "T"


def test_beir_loader_handles_null_titles_and_parses_each_line_once(tmp_path=None):
    import json
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp()) / "toy"
    d.mkdir()
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"_id": "a", "title": None, "text": "alpha body"},
        {"_id": "b", "text": "beta body"},                      # title absent
        {"_id": "c", "title": "Gamma", "text": "gamma body"},
    ]), encoding="utf-8")
    (d / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "alpha?"}), encoding="utf-8")
    units, queries = migration.load_beir("toy", str(d.parent))
    assert units == [("a", "alpha body"), ("b", "beta body"), ("c", "Gamma gamma body")]
    assert queries == [("q1", "alpha?")]


def test_limit_samples_at_random_not_a_prefix():
    """Every in-memory corpus here lists related units together (CIPBench by
    repository). A prefix limit would draw from the first group only."""
    units = [(f"a{i}", "topic A") for i in range(500)] + \
            [(f"b{i}", "topic B") for i in range(500)]
    got = migration._sample(units, 200)
    assert len(got) == 200
    n_b = sum(u[0].startswith("b") for u in got)
    assert 60 < n_b < 140, f"a prefix would give 0 from group B; got {n_b}"
    assert migration._sample(units, None) is units
    assert migration._sample(units, 5000) is units
    assert migration._sample(units, 200) == migration._sample(units, 200), "seeded"


def test_reservoir_is_uniform_and_the_tempting_fix_is_not():
    """Every item of a stream must be kept with probability k/N.

    `random.Random.randint` is inclusive of both ends, so `randint(0, n)` is
    correct Algorithm R. A review once proposed `randint(0, n + 1)` -- numpy's
    exclusive-upper convention -- which over-samples the first k items. This
    test pins the current behaviour and shows the alternative is biased.
    """
    import random

    N, k, trials = 12, 4, 30000
    target = k / N

    def run(offer):
        hits = np.zeros(N)
        for t in range(trials):
            rng, res = random.Random(t), []
            for n in range(N):
                offer(res, k, n, n, rng)
            for x in res:
                hits[x] += 1
        return hits / trials

    p = run(migration._reservoir_offer)
    assert np.abs(p - target).max() < 0.015, p

    def off_by_one(res, k_, item, n, rng):
        if len(res) < k_:
            res.append(item)
        else:
            j = rng.randint(0, n + 1)
            if j < k_:
                res[j] = item

    q = run(off_by_one)
    assert np.abs(q - target).max() > 0.03, "the n+1 variant should be biased"


def test_frontier_refuses_an_empty_sweep():
    V, _, _ = _clustered(50)
    try:
        migration.frontier(V, V, V[:5], V[:5], k=10, kprimes=(100, 200))
    except ValueError as e:
        assert "exceeds the corpus size" in str(e)
        return
    raise AssertionError("every K' > N must raise, not return an empty sweep")


def test_normalising_the_old_side_changes_coverage():
    """Why --old-normalize exists in estimate as well as migration.

    Normalising the old side changes which units each window reaches, so it
    changes coverage and therefore the savings number. The DIRECTION is a
    property of the norm distribution, not a law: with norms spread uniformly
    over 0.6-1.6, raw inner product concentrates windows on high-norm units and
    reaches fewer; on real DPR vectors the opposite was measured (reached 0.510
    raw against 0.447 normalised at K'=500, N=490,908). So this asserts that
    the ablation moves coverage materially, and deliberately not which way.
    """
    rng = np.random.default_rng(0)
    V = (rng.normal(size=(3000, 32)) * rng.uniform(0.6, 1.6, (3000, 1))).astype(np.float32)
    Q = rng.normal(size=(150, 32)).astype(np.float32)
    raw = len(np.unique(migration.topk_chunked(Q, V, 60))) / len(V)
    Vn = (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)
    cos = len(np.unique(migration.topk_chunked(Q, Vn, 60))) / len(V)
    assert abs(cos - raw) > 0.1, (raw, cos)


def test_both_tools_label_the_old_side_the_same_way():
    import ast

    assert migration.old_label("lexical:word", None, False) == "lexical:word"
    assert migration.old_label("lexical:word", None, True) == "lexical:word:normalized"
    assert migration.old_label("x", "psgs_w100.nq.no_index", True) == \
        "dpr:psgs_w100.nq.no_index:normalized"
    for mod in (migration, estimate):
        names = {n.func.id for n in ast.walk(_main_ast(mod))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "old_label" in names, f"{mod.__name__}.main builds its own label"


def test_a_zero_limit_is_an_error_not_no_limit():
    import argparse

    for bad in ("0", "-3"):
        try:
            migration.positive_int(bad)
        except argparse.ArgumentTypeError:
            continue
        raise AssertionError(f"--limit {bad} must be rejected")
    assert migration.positive_int("7") == 7
    try:
        migration._sample([("a", "x"), ("b", "y")], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("_sample(limit=0) must not return everything")
    try:
        migration.load_dpr_wiki(limit=0)
    except ValueError:
        pass
    else:
        raise AssertionError("load_dpr_wiki(limit=0) must not load all 21M")


def test_full_dpr_load_needs_an_explicit_limit():
    for tool in ("migration", "estimate"):
        try:
            migration.require_dpr_limit("dpr_wiki", None, tool)
        except SystemExit as e:
            assert "--limit" in str(e)
        else:
            raise AssertionError(f"{tool}: full DPR must not load by default")
    migration.require_dpr_limit("dpr_wiki", 1_000_000, "estimate")   # fine
    migration.require_dpr_limit("cipbench", None, "estimate")        # fine


def test_estimate_never_takes_len_of_the_optional_text_list():
    """`utext` is None on the DPR path, so len(utext) is a TypeError that only
    fires after the whole corpus has loaded. Counts must come from `units`."""
    import inspect
    import re

    src = inspect.getsource(estimate.main)
    assert not re.search(r"len\(\s*utext\s*\)", src), "len(utext) crashes for DPR"


def test_estimate_does_not_keep_passage_text_it_never_uses():
    import ast

    calls = [n for n in ast.walk(_main_ast(estimate)) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "dpr_corpus"]
    assert calls, "estimate.main no longer loads DPR through dpr_corpus"
    kw = {k.arg: k.value for c in calls for k in c.keywords}
    assert isinstance(kw.get("keep_text"), ast.Constant) and kw["keep_text"].value is False


def test_related_work_tex_and_md_agree():
    """Every key cited in related.tex must be documented in RELATED_WORK.md.

    Four review rounds disputed whether the tex cites EdgeRAG and the
    practitioner sources; asserting it again is worth less than a check anyone
    can run. If the files are absent (installed without the paper sources),
    the test skips rather than failing.
    """
    import re
    from pathlib import Path

    root = Path(migration.__file__).resolve().parent.parent
    tex, md = root / "body.tex", root / "RELATED_WORK.md"
    if not tex.exists() or not md.exists():
        raise Skip("body.tex / RELATED_WORK.md not in this checkout")
    keys = {k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", tex.read_text(encoding="utf-8"))
            for k in c.split(",")}
    listed = set(re.findall(r"^\| `([a-z0-9_]+)`", md.read_text(encoding="utf-8"), re.M))
    assert keys, "related.tex cites nothing -- parsing is broken"
    assert keys <= listed, f"cited but undocumented: {sorted(keys - listed)}"
    bib = root / "references.bib"
    if bib.exists():
        import re as _re
        defined = set(_re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
        assert keys <= defined, f"cited but not in references.bib: {sorted(keys - defined)}"
    # the nearest prior art, by their verified keys (renamed once the PDFs were
    # checked: FastFill is Jaeckle et al., MCT is Seo et al. WACV 2025)
    for must in ("edgerag", "blog_freshness", "blog_slices", "coverage2026",
                 "seo2025mct", "jaeckle2023fastfill", "vejendla2025driftadapter"):
        assert must in keys, f"related.tex no longer cites {must}"


def test_documented_dpr_commands_pass_the_limit_guard():
    """Every DPR command in the docs must carry --limit, or the guard refuses it."""
    import re
    from pathlib import Path

    root = Path(migration.__file__).resolve().parent.parent
    for name in ("RUN_EXPERIMENTS.md", "README.md", "PLAYBOOK.md"):
        path = root / name
        if not path.exists():
            continue
        for block in re.findall(r"```bash\n(.*?)```", path.read_text(encoding="utf-8"), re.S):
            for cmd in block.replace("\\\n", " ").splitlines():
                if "--corpus dpr_wiki" in cmd:
                    assert "--limit" in cmd, f"{name}: {cmd.strip()[:90]}"


def test_prefixes_are_resolved_per_model():
    """The new model's top-k is the ground truth; its queries must be encoded
    the way the model is meant to be used for retrieval."""
    r = migration.resolve_prefixes
    assert r("BAAI/bge-base-en-v1.5") == (migration.BGE_EN_QUERY, "")
    assert r("BAAI/bge-large-en-v1.5")[1] == "", "BGE passages take no instruction"
    assert r("intfloat/e5-base-v2") == ("query: ", "passage: ")
    assert r("sentence-transformers/all-MiniLM-L6-v2") == ("", "")
    assert r("lexical:word") == ("", "")
    assert r("BAAI/bge-m3") == ("", ""), "bge-m3 takes no instruction"
    assert r("BAAI/bge-base-en-v1.5", query_prefix="") == ("", ""), "'' disables"
    assert r("x", query_prefix="Q: ", passage_prefix="P: ") == ("Q: ", "P: ")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = r("intfloat/e5-mistral-7b-instruct")
    assert got == ("", ""), "e5-mistral does not use 'query: '/'passage: '"
    assert any("instruction-tuned" in str(x.message) for x in w), "must warn"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r("nomic-ai/nomic-embed-text-v1.5", query_prefix="search_query: ",
          passage_prefix="search_document: ")
    assert not w, "explicit prefixes must not warn"


def test_neural_encoders_receive_the_prefix_and_lexical_ones_do_not():
    seen = {}

    class FakeModel:
        def encode(self, texts, **kw):
            seen["texts"] = list(texts)
            return np.ones((len(texts), 4), np.float32)

    migration.encode_with(FakeModel(), ["who wrote it"], "BAAI/bge-base-en-v1.5",
                          prefix=migration.BGE_EN_QUERY)
    assert seen["texts"] == [migration.BGE_EN_QUERY + "who wrote it"]
    units = [f"def f{i}(x): return x + {i} helper {i}" for i in range(40)]
    V, fitted = migration.encode_all(units, "lexical:word", prefix="IGNORED ")
    Q = migration.encode_with(fitted, ["helper"], "lexical:word", prefix="IGNORED ")
    assert "ignored" not in fitted.steps[0][1].vocabulary_


def test_normalising_a_zero_row_gives_zeros_not_nan():
    M = np.array([[0.0, 0.0], [3.0, 4.0]], np.float32)
    out = migration._safe_normalize(M)
    assert not np.isnan(out).any()
    assert np.allclose(out[0], 0.0) and np.allclose(out[1], [0.6, 0.8])


def test_sharded_topk_is_exact_across_uneven_shards():
    """Sharding bounds the FAISS index copy; it must not change the answer --
    including shards smaller than k and an uneven last shard."""
    rng = np.random.default_rng(14)
    V = _unit(rng.normal(size=(1037, 12)))
    Q = _unit(rng.normal(size=(25, 12)))

    def numpy_search(Vs, kk):
        S = Q @ Vs.T
        idx = np.argsort(-S, axis=1)[:, :kk]
        return np.take_along_axis(S, idx, axis=1), idx

    want = np.argsort(-(Q @ V.T), axis=1)
    # (200, 37) needs six shards before k candidates exist -- the case an
    # earlier merge crashed on, because argpartition was asked for element 199
    # of a 74-element row. (1037, 100) asks for the whole corpus.
    for k, shard in ((10, 100), (50, 37), (200, 150), (200, 37), (10, 5000),
                     (1037, 100), (7, 1)):
        got = migration._sharded_topk(V, k, numpy_search, shard=shard)
        for i in range(len(Q)):
            assert set(got[i]) == set(want[i, :k]), (k, shard, i)
        S = Q @ V.T
        for i in range(len(Q)):
            assert np.all(np.diff(S[i, got[i]]) <= 1e-6), "must stay score-ordered"


def test_nq_questions_are_sampled_not_truncated():
    """--max-queries is applied by random sampling in main. The NQ loader must
    not be able to truncate to a prefix at all."""
    import inspect

    assert list(inspect.signature(migration.load_nq_questions).parameters) == ["split"]


def test_independent_draw_form_is_exact_where_the_exponential_is_not():
    """(1 - K'/N)^D is the exact expectation for independent windows; the
    exponential is its small-K'/N limit and overpredicts savings when K'/N is
    large. Both are reported; they must agree at scale and diverge on small N."""
    V, _, _ = _clustered(500, seed=15)
    Q = V[:40]
    for r in migration.frontier(V, V, Q, Q, k=10, kprimes=(200,), d_fracs=(0.1,)):
        exact = 100 * (1 - 200 / 500) ** r["D"]
        assert abs(r["pred_savings_indep"] - round(exact, 2)) < 1e-9
        assert r["pred_savings_exp"] > r["pred_savings_indep"]      # K'/N = 0.4
    N, kp, D = 21_015_300, 500, 3610
    assert abs((1 - kp / N) ** D - math.exp(-D * kp / N)) < 1e-5


def test_beir_qrels_split_keeps_gold_docs_and_drops_other_queries():
    import json
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp()) / "toy"
    (root / "qrels").mkdir(parents=True)
    docs = [{"_id": f"d{i}", "title": None, "text": f"doc {i}"} for i in range(200)]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    qs = [{"_id": "qdev1", "text": "a"}, {"_id": "qdev2", "text": "b"},
          {"_id": "qtrain", "text": "c"}]
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    (root / "qrels" / "dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nqdev1\td190\t1\nqdev2\td199\t1\nqdev2\td5\t0\n", encoding="utf-8")
    units, queries = migration.load_beir("toy", str(root.parent), limit=20,
                                         qrels_split="dev")
    ids = {u[0] for u in units}
    assert [q[0] for q in queries] == ["qdev1", "qdev2"], "train query must be dropped"
    assert {"d190", "d199"} <= ids, "every gold document must be in the subset"
    assert "d5" not in {"d190", "d199"} and len(units) == 20
    try:
        migration.load_beir("toy", str(root.parent), qrels_split="test")
    except SystemExit as e:
        assert "available" in str(e)
    else:
        raise AssertionError("a missing qrels split must name the ones that exist")


def test_dpr_subset_keeps_every_gold_and_samples_distractors_uniformly():
    dev = migration.parse_dpr_nq_dev([
        {"question": "q with id", "positive_ctxs": [{"title": "T1", "text": "x1",
                                                     "passage_id": "7"}]},
        {"question": "q with text only", "positive_ctxs": [{"title": "T2", "text": "x2"}]},
        {"question": "no positives", "positive_ctxs": []},
    ])
    assert [d["question"] for d in dev] == ["q with id", "q with text only"]
    gold_ids = set().union(*(d["gold_ids"] for d in dev))
    gold_keys = set().union(*(d["gold_keys"] for d in dev))

    def stream():
        for i in range(10_000):
            title, text = ("T2", "x2") if i == 9_000 else (f"t{i}", f"p{i}")
            yield str(i), title, text, np.zeros(2, np.float32)

    gold, res = migration.select_gold_and_distractors(stream(), gold_ids, gold_keys,
                                                     n=500, n_gold_expected=2)
    assert {g[0] for g in gold} == {"7", "9000"}, "found by id and by (title, text)"
    assert len(res) == 498
    ids = np.array([int(r[0]) for r in res])
    assert ids.min() < 1000 and ids.max() > 9000, "distractors span the whole stream"
    assert abs(np.median(ids) - 5000) < 800, "and are roughly uniform over it"


def test_x_targets_choose_d_and_report_what_cannot_be_reached():
    fr, unreach = migration.x_targets_to_fracs(N=1_000_000, D=6_000, kprime=500,
                                               targets=[0.05, 1.0, 3.0, 10.0])
    ds = [round(f * 6_000) for f in fr]
    assert ds == [100, 2000, 6000], ds
    assert unreach == [10.0]
    # One intent already gives x = K'/N = 0.211 here. 0.05 is below that and is
    # reported, not clipped to D = 1; 0.2 rounds to D = 1 legitimately.
    fr, unreach = migration.x_targets_to_fracs(N=2_366, D=2_378, kprime=500,
                                               targets=[0.05, 0.2, 1.0, 3.0])
    assert unreach == [0.05], unreach
    assert [round(f * 2_378) for f in fr] == [1, 5, 14]


def test_corpus_cache_key_depends_on_every_input():
    k = migration.corpus_key
    base = k("BAAI/bge-base-en-v1.5", "", ["a", "b", "c"])
    assert base == k("BAAI/bge-base-en-v1.5", "", ["a", "b", "c"])
    assert base != k("BAAI/bge-base-en-v1.5", "p: ", ["a", "b", "c"])
    assert base != k("BAAI/bge-small-en-v1.5", "", ["a", "b", "c"])
    assert base != k("BAAI/bge-base-en-v1.5", "", ["a", "c", "b"]), "order matters"
    assert base != k("BAAI/bge-base-en-v1.5", "", ["a", "b"])


def _toy_jsonl_corpus():
    import json
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(17)
    topics = ["cache eviction policy", "retry timeout handler", "token parser state",
              "database connection pool", "image resize filter", "log rotation size"]
    units = [{"id": f"u{i}", "text": f"{topics[i % 6]} detail {i} "
                                     f"{' '.join(rng.choice(topics[i % 6].split(), 3))}"}
             for i in range(240)]
    qs = [{"id": f"q{i}", "text": f"how does the {topics[i % 6]} work {i}"} for i in range(60)]
    (d / "c.jsonl").write_text("\n".join(json.dumps(u) for u in units), encoding="utf-8")
    (d / "q.jsonl").write_text("\n".join(json.dumps(q) for q in qs), encoding="utf-8")
    return d


def _run_cli(module, *args):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(migration.__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(root))
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True,
                          text=True, env=env, cwd=str(root), timeout=600)


def test_every_cli_parser_builds():
    """A parser that cannot be built crashes the tool on startup, and no unit
    test notices: a duplicated --queries flag once did exactly that to both
    tools while every other test passed."""
    for module in ("semview.migration", "semview.estimate", "semview.cipbench"):
        r = _run_cli(module, "--help")
        assert r.returncode == 0, f"{module} --help failed:\n{r.stderr[-600:]}"


def test_migration_runs_end_to_end_from_the_command_line():
    """The whole path -- argument parsing, loading, the old side, the frontier,
    and the JSON written -- on a small corpus with the offline encoders."""
    import json

    d = _toy_jsonl_corpus()
    out = d / "m.json"
    r = _run_cli("semview.migration", "--corpus", "jsonl", "--path", str(d / "c.jsonl"),
                 "--queries-file", str(d / "q.jsonl"), "--old", "lexical:word",
                 "--new", "lexical:char", "--k", "5", "--x-targets", "0.5,2",
                 "--x-kprime", "20", "--out", str(out))
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["N"] == 240 and res["D"] == 60 and res["old"] == "lexical:word"
    assert set(res["prefixes"]) == {"old_query", "old_passage", "new_query", "new_passage"}
    assert {"old_head", "new_head", "old_dim", "new_dim"} <= set(res["fingerprint"])
    row = res["frontier"][0]
    for key in ("recall_at_kprime", "recall_chance_corrected", "pred_savings_indep",
                "pred_savings_exp", "x_dk_over_n", "reached", "savings_pct"):
        assert key in row, key
    assert "load x = D*K'/N" in r.stdout


def test_estimate_runs_end_to_end_from_the_command_line():
    import json

    d = _toy_jsonl_corpus()
    out = d / "e.json"
    r = _run_cli("semview.estimate", "--corpus", "jsonl", "--path", str(d / "c.jsonl"),
                 "--queries-file", str(d / "q.jsonl"), "--old", "lexical:word",
                 "--kprimes", "10,20", "--out", str(out))
    assert r.returncode == 0, r.stderr[-800:]
    rows = json.loads(out.read_text(encoding="utf-8"))["estimates"]
    assert [x["kprime"] for x in rows] == [10, 20]
    for x in rows:
        assert x["curve_d"] and x["curve_reached"] and "backtest_d_needed" in x


def test_orcas_columns_are_detected_not_assumed():
    q, d = migration.orcas_columns("1\thow tall is everest\tD1234567\thttp://x")
    assert (q, d) == (1, 2)
    q, d = migration.orcas_columns("D77\twhat is a vector index")   # other order
    assert (q, d) == (1, 0)
    for bad in ("onefield", "D1\tD2\tD3"):
        try:
            migration.orcas_columns(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} must be rejected with a clear message")


def test_orcas_reads_pairs_and_builds_a_gold_aware_subset(tmp=None):
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    # 30 distinct queries, each clicking one document; 400 documents in all
    pairs = [f"{i}\tquery number {i} about things\tD{i}\thttp://e/{i}" for i in range(30)]
    (d / "orcas.tsv").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    docs = [f"D{i}\thttp://e/{i}\tTitle {i}\tbody text for document {i}"
            for i in range(400)]
    (d / "docs.tsv").write_text("\n".join(docs) + "\n", encoding="utf-8")

    units, queries = migration.load_orcas(str(d / "orcas.tsv"), str(d / "docs.tsv"),
                                          limit=100, max_queries=10, seed=0)
    assert len(queries) == 10
    ids = {u[0] for u in units}
    clicked = {f"D{q[1].split()[2]}" for q in queries}
    assert clicked <= ids, "every clicked document must survive the subset"
    assert len(units) == 100
    assert any(u[1].startswith("Title") for u in units), "title and body are joined"


def test_qrels_survives_a_malformed_score():
    import json, tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp()) / "toy"
    (root / "qrels").mkdir(parents=True)
    (root / "qrels" / "dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\tNaN\nq3\td3\t\nq4\td4\t2\n", encoding="utf-8")
    gold = migration.read_qrels(root, "dev")
    assert set(gold) == {"q1", "q4"}, gold      # bad rows skipped, not fatal


def test_orcas_subset_never_exceeds_the_limit_and_keeps_gold():
    import json, tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    (d / "orcas.tsv").write_text(
        "\n".join(f"{i}\tquery number {i} about things\tD{i}\thttp://e/{i}"
                  for i in range(40)) + "\n", encoding="utf-8")
    (d / "docs.tsv").write_text(
        "\n".join(f"D{i}\thttp://e/{i}\tTitle {i}\tbody {i}" for i in range(600)) + "\n", encoding="utf-8")
    units, queries = migration.load_orcas(str(d / "orcas.tsv"), str(d / "docs.tsv"),
                                          limit=50, max_queries=20, seed=0)
    assert len(units) == 50, len(units)
    clicked = {f"D{q[1].split()[2]}" for q in queries}
    assert clicked <= {u[0] for u in units}, "gold documents must survive"
    ids = sorted(int(u[0][1:]) for u in units if u[0][1:].isdigit())
    assert max(ids) > 400, "distractors must come from the whole stream, not a prefix"


def test_duplicate_d_fracs_are_collapsed():
    import ast, inspect
    src = inspect.getsource(migration.main)
    assert "sorted({float(x) for x in a.d_fracs" in src, "duplicates would run twice"


def test_nq_open_rejects_a_split_it_does_not_have():
    """nq_open has train and validation only; 'test' must fail with a clear
    message before any download is attempted."""
    try:
        migration.load_nq_questions("test")
    except ValueError as e:
        assert "test" in str(e) and "validation" in str(e)   # names both, wording free
        return
    raise AssertionError("--nq-split test must be rejected")


def test_dpr_loads_parquet_shards_not_the_loading_script():
    """datasets>=4 refuses repositories that carry a loading script, and
    facebook/wiki_dpr still does. The loader must read the parquet directly."""
    import ast
    import inspect

    src = inspect.getsource(migration.load_dpr_wiki)
    first_args = {c.args[0].value for c in ast.walk(ast.parse(src))
                  if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "load_dataset"
                  and c.args and isinstance(c.args[0], ast.Constant)}
    assert first_args == {"parquet"}, first_args
    g = migration.dpr_parquet_files("psgs_w100.nq.no_index")
    assert g == "hf://datasets/facebook/wiki_dpr/data/psgs_w100/nq/train-*.parquet"
    assert migration.dpr_parquet_files("psgs_w100.multiset.exact").endswith(
        "psgs_w100/multiset/train-*.parquet")
    # The local form must keep the layout, but its separators belong to the
    # platform: comparing against a POSIX literal made this test unpassable on
    # Windows, where `Path` normalises to backslashes. Compare through `Path`.
    local = migration.dpr_parquet_files("psgs_w100.nq.exact", "/data/dpr")
    assert local.startswith(str(Path("/data/dpr"))), local
    assert local.endswith(str(Path("data") / "psgs_w100" / "nq" / "train-*.parquet")), local
    # index suffix is irrelevant to which embeddings are loaded
    assert migration.dpr_parquet_files("psgs_w100.nq.compressed") == g


def test_dpr_record_schema_is_checked():
    migration.check_dpr_record({"id": "1", "title": "t", "text": "x", "embeddings": [0.0]})
    try:
        migration.check_dpr_record({"id": "1", "text": "x", "embeddings": [0.0]})
    except SystemExit as e:
        assert "title" in str(e)
        return
    raise AssertionError("a record missing a column must stop the run")


# ------------------------------------------------------------- estimate ----
def test_coverage_curve_is_monotone_and_ends_at_full_union():
    V, _, _ = _clustered(800, seed=6)
    Q = V[:120]
    ds, rc = estimate.coverage_curve(Q, V, kprime=20, points=12)
    assert np.all(np.diff(rc) >= -1e-12)
    seen = np.zeros(len(V), bool)
    seen[np.unique(migration.topk_chunked(Q, V, 20))] = True
    assert abs(rc[-1] - seen.mean()) < 1e-9


def test_fit_prefers_the_heavy_tailed_form_and_recovers_the_asymptote():
    ds = np.unique(np.geomspace(1, 5000, 24).astype(int)).astype(float)
    true_a = 0.62
    rc = true_a * (1 - (1 + ds / 40.0) ** -0.6)
    fit = estimate.fit_asymptote(ds, rc)
    assert fit["form"] == "power_sat", fit
    assert abs(fit["asymptote"] - true_a) < 0.03, fit


def test_convergence_reports_a_prefix_of_the_log():
    ds = np.unique(np.geomspace(1, 5000, 24).astype(int)).astype(float)
    rc = 0.7 * (1 - (1 + ds / 30.0) ** -0.8)
    c = estimate.convergence(ds, rc)
    assert 0 < c["frac_of_log"] <= 1.0
    assert c["d_needed"] <= ds[-1]


def test_convergence_on_a_curve_too_short_to_fit():
    """Fewer than four points: no prefix can be fitted, so the whole log is
    needed. Must return that rather than raise."""
    ds = np.array([1.0, 2.0, 4.0])
    rc = np.array([0.1, 0.18, 0.3])
    c = estimate.convergence(ds, rc)
    assert c["frac_of_log"] == 1.0 and c["d_needed"] == 4


# ----------------------------------------------------------------- cost ----
def test_predicted_savings_is_the_exact_form_and_breakeven_is_1_over_e():
    assert cost.predicted_savings(1000, 100, 1_000_000) == (1 - 1e-4) ** 1000
    assert abs(cost.predicted_savings(1000, 100, 1_000_000) - math.exp(-0.1)) < 1e-5
    assert cost.predicted_savings(4, 2000, 5000) < cost.predicted_savings_exp(4, 2000, 5000)
    N, kp = 8_800_000, 500
    D = cost.breakeven_D(N, kp)
    assert abs(cost.predicted_savings(D, kp, N) - math.exp(-1)) < 1e-3


def test_cost_arithmetic():
    p = cost.CostParams(N=1_000_000, D=10_000, Q=2_000_000, reached=0.2,
                        unit_ms=10.0, query_ms=2.0, cache_hit_rate=1.0)
    r = cost.compare(p)
    assert r["lazy_unit_passes"] == 200_000
    assert r["extra_old_model_query_passes"] == 10_000       # cached -> D
    assert abs(r["unit_savings_pct"] - 80.0) < 1e-9
    batch = 1_000_000 * 10.0
    lazy = 200_000 * 10.0 + 10_000 * 2.0
    assert abs(r["net_savings_pct"] - round(100 * (batch - lazy) / batch, 2)) < 1e-9


def test_from_estimate_reads_reached_at_the_requested_d():
    """A reached measured over 50,000 intents must not be priced as 10,000."""
    row = {"reached_observed": 0.8, "N": 10_000, "D": 5_000,
           "curve_d": [10, 100, 1_000, 5_000],
           "curve_reached": [0.05, 0.3, 0.6, 0.8]}
    assert cost.from_estimate(row, Q=1).reached == 0.8
    mid = cost.from_estimate(row, Q=1, D=316).reached          # ~ log-midpoint
    assert 0.44 < mid < 0.46, mid
    try:
        cost.from_estimate(row, Q=1, D=20_000)
    except ValueError as e:
        assert "beyond the replayed log" in str(e)
    else:
        raise AssertionError("extrapolating past the replayed log must be refused")
    try:
        cost.from_estimate({"reached_observed": 0.8, "N": 10, "D": 5}, Q=1, D=3)
    except ValueError:
        pass
    else:
        raise AssertionError("a row without a curve cannot price another D")


def test_distinct_query_strings_drive_the_cache_term():
    base = dict(N=1_000_000, D=10_000, Q=2_000_000, reached=0.2, query_ms=2.0,
                cache_hit_rate=1.0)
    same = cost.compare(cost.CostParams(**base))
    para = cost.compare(cost.CostParams(**base, S=40_000))   # paraphrases miss
    assert para["extra_old_model_query_passes"] == 40_000
    assert para["lazy_gpu_seconds"] > same["lazy_gpu_seconds"]
    try:
        cost.compare(cost.CostParams(**base, S=5_000))
    except ValueError:
        pass
    else:
        raise AssertionError("S below D is impossible and must be rejected")
    assert cost.first_touch_ms(500, 10.4) == 5200.0


def test_backtest_checks_accuracy_not_just_stability():
    ds = np.unique(np.geomspace(1, 5000, 24).astype(int)).astype(float)
    rc = 0.62 * (1 - (1 + ds / 40.0) ** -0.6)
    bt = estimate.backtest(ds, rc)
    assert bt["backtest_d_needed"] is not None
    assert abs(bt["backtest_err_at_half"]) < 0.03


def test_fit_quality_and_extrapolation_are_reported_separately():
    """A curve can fit its own points well and still extrapolate badly. The
    flag for in-sample fit must not be named or read as trustworthiness; the
    extrapolation verdict comes from the backtest."""
    V, _, _ = _clustered(400, seed=16)
    r = estimate.estimate(V[:60], V, kprimes=(20,), verbose=False)[0]
    assert "fit_trustworthy" not in r, "misleading name must not come back"
    assert "in_sample_fit_ok" in r and "extrapolation_ok" in r
    e = r["backtest_err_at_half"]
    assert r["extrapolation_ok"] == (e is not None and abs(e) <= r["backtest_tol"])


def test_estimate_rows_carry_their_coverage_curve():
    V, _, _ = _clustered(400, seed=16)
    rows = estimate.estimate(V[:60], V, kprimes=(20,), verbose=False)
    r = rows[0]
    assert r["curve_d"][-1] == r["D"] and abs(r["curve_reached"][-1] - r["reached_observed"]) < 1e-4
    assert all(a <= b for a, b in zip(r["curve_reached"], r["curve_reached"][1:]))


def _toy_server(n=1500, d=24, kprime=60, chunk=8, seed=0):
    """A server over two spaces related by a rotation, with instant encoders so
    the tests measure behaviour rather than a GPU."""
    rng = np.random.default_rng(seed)
    V_old = migration._safe_normalize(rng.normal(size=(n, d)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_true = migration._safe_normalize((V_old @ R).astype(np.float32))
    texts = [f"unit {i}" for i in range(n)]
    qs = [f"q{i}" for i in range(60)]
    qvec = {q: V_old[rng.integers(0, n)] for q in qs}
    srv = serve.ThawServer(
        V_old=V_old, texts=texts,
        encode_new=lambda ts: V_true[[int(t.split()[1]) for t in ts]],
        encode_old_query=lambda t: qvec[t],
        encode_new_query=lambda t: migration._safe_normalize((qvec[t] @ R)[None, :])[0],
        kprime=kprime, k=10, chunk=chunk)
    return srv, qs, V_true, qvec, R


def test_repaired_vectors_reach_the_index_and_their_insert_cost_is_separated():
    """A repaired vector that never reaches the index is not migrated.

    And the insert is not free in a real database, so it is timed apart from
    the encoding: the encoder's cost is a property of the model, the insert's
    is a property of the index, and a deployment can change them
    independently.
    """
    srv, qs, _, _, _ = _toy_server()

    class Recorder:
        def __init__(self): self.seen = set()
        def search(self, q, k): raise AssertionError("not used here")
        def upsert(self, ids, vecs):
            assert len(ids) == len(vecs)
            self.seen |= set(int(i) for i in ids)

    rec = Recorder()
    srv.new_index = rec
    traces = serve.replay(srv, qs)
    assert rec.seen == set(np.flatnonzero(srv.repaired).tolist())
    assert all(t.insert_ms >= 0 for t in traces)
    s = serve.summarise(traces)
    assert "insert_share" in s and "insert_ms_per_unit" in s


def test_qdrant_index_matches_exact_search_where_the_client_exists():
    """The real database must agree with the reference on what it returns;
    skipped where the client is not installed."""
    try:
        import qdrant_client                                   # noqa: F401
    except ImportError:
        raise Skip("qdrant_client not installed")
    rng = np.random.default_rng(0)                             # pragma: no cover
    V = migration._safe_normalize(rng.normal(size=(300, 16)).astype(np.float32))
    ix = serve.QdrantIndex(dim=16)
    ix.upsert(list(range(300)), V)
    q = V[7]
    assert ix.search(q, 5)[0] == 7


def test_a_unit_is_repaired_once_and_only_once():
    """The saving is the repair count; if a unit could be repaired twice the
    whole ledger would be wrong."""
    srv, qs, _, _, _ = _toy_server()
    total_stale = sum(t.n_stale for t in serve.replay(srv, qs))
    assert total_stale == int(srv.repaired.sum()), (total_stale, srv.repaired.sum())


def test_served_results_are_the_exact_new_space_ranking_of_the_window():
    """The design's correctness claim: within a window, the ranking is exactly
    what a fully migrated index would return."""
    srv, qs, V_true, qvec, R = _toy_server()
    for q in qs[:10]:
        got, _ = srv.query(q)
        window = migration.topk_chunked(qvec[q][None, :], srv.V_old, srv.kprime)[0]
        qn = migration._safe_normalize((qvec[q] @ R)[None, :])[0]
        want = [int(window[j]) for j in np.argsort(-(V_true[window] @ qn))[:10]]
        assert got == want, (q, got[:3], want[:3])


def test_every_returned_unit_has_been_repaired():
    """Nothing is ever ranked on an old vector: that would be the cross-space
    comparison the design forbids."""
    srv, qs, _, _, _ = _toy_server()
    for q in qs[:15]:
        got, _ = srv.query(q)
        assert all(srv.repaired[u] for u in got), q


def test_latency_decays_as_the_migration_proceeds():
    """First-touch cost is a transient. If it did not decay, the design would
    be trading a one-off GPU bill for a permanent latency one."""
    srv, qs, _, _, _ = _toy_server()
    tr = serve.replay(srv, qs)
    first = np.mean([t.n_stale for t in tr[:len(tr)//4]])
    last = np.mean([t.n_stale for t in tr[-len(tr)//4:]])
    assert first > 3 * last, (first, last)


def test_repair_chunking_changes_issuing_but_not_results():
    """Chunk size is the main lever on latency, so it must not be a lever on
    what the user gets back."""
    a, qs, _, _, _ = _toy_server(chunk=8)
    b, _, _, _, _ = _toy_server(chunk=512)
    for q in qs[:10]:
        assert a.query(q)[0] == b.query(q)[0], q
    assert int(a.repaired.sum()) == int(b.repaired.sum())


def test_a_restart_repairs_nothing_it_already_did():
    """Persistence is what makes the saving durable: without it every restart
    re-pays the migration."""
    import tempfile
    from pathlib import Path

    srv, qs, _, _, _ = _toy_server()
    serve.replay(srv, qs)
    f = str(Path(tempfile.mkdtemp()) / "state.npz")
    srv.save(f)

    fresh, _, _, _, _ = _toy_server()
    fresh.load(f)
    again = serve.replay(fresh, qs)
    assert sum(t.n_stale for t in again) == 0, "a restart must repair nothing"


def test_reached_matches_what_the_frontier_reports():
    """The server and the measurement tools must agree on the one quantity the
    paper reports, or they are measuring different systems."""
    srv, qs, _, qvec, _ = _toy_server()
    serve.replay(srv, qs)
    Q = np.stack([qvec[q] for q in qs])
    union = set()
    for w in migration.topk_chunked(Q, srv.V_old, srv.kprime):
        union |= set(int(u) for u in w)
    assert int(srv.repaired.sum()) == len(union)


def test_summary_reports_the_distribution_not_the_mean():
    """First-touch cost is a tail problem; a mean hides the queries that pay."""
    srv, qs, _, _, _ = _toy_server()
    s = serve.summarise(serve.replay(srv, qs))
    for key in ("total_p50", "total_p95", "total_p99", "total_max",
                "first_decile_p50", "last_decile_p50", "repair_share"):
        assert key in s, key
    assert s["total_p50"] <= s["total_p95"] <= s["total_p99"] <= s["total_max"]
    assert 0.0 <= s["repair_share"] <= 1.0


def test_first_touch_latency_depends_on_how_the_repair_is_batched():
    """The naive implementation -- one call with all K' units -- is the worst
    one on a memory-tight device, and by an order of magnitude."""
    curve = {1: 9.77, 8: 3.16, 32: 4.64, 128: 7.33, 512: 42.96}   # measured
    naive = cost.first_touch_ms(500, 3.16, chunk=512, curve=curve)
    best = cost.first_touch_ms(500, 3.16, curve=curve)
    assert naive > 10 * best, (naive, best)
    assert abs(best - 1580) < 5, best
    # with no curve it falls back to the rate given, which is the optimistic case
    assert cost.first_touch_ms(500, 3.16) == 500 * 3.16


def test_breakeven_reached_is_one_over_the_repair_penalty():
    """Measured on an RTX 4060: unit 3.035 ms, repair 4.181 ms -> 0.726."""
    assert abs(cost.breakeven_reached(3.035, 4.181) - 0.726) < 0.001
    assert cost.breakeven_reached(2.0) == 1.0          # no penalty -> only the query term bites
    # and it must agree with where compare() actually flips sign
    u, r = 3.035, 4.181
    be = cost.breakeven_reached(u, r)
    lo = cost.compare(cost.CostParams(N=500_000, D=1, Q=1, reached=be - 0.05,
                                      unit_ms=u, repair_unit_ms=r, query_ms=0.1))
    hi = cost.compare(cost.CostParams(N=500_000, D=1, Q=1, reached=be + 0.05,
                                      unit_ms=u, repair_unit_ms=r, query_ms=0.1))
    assert lo["net_savings_pct"] > 0 > hi["net_savings_pct"], (lo, hi)


def _two_spaces(n=6000, d=32, clusters=40, hard=True, seed=0):
    rng = np.random.default_rng(seed)
    C = migration._safe_normalize(rng.normal(size=(clusters, d)).astype(np.float32))
    V_old = migration._safe_normalize(
        (C[rng.integers(0, clusters, n)] + 0.35 * rng.normal(size=(n, d))).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_new = migration._safe_normalize((np.tanh(1.5 * (V_old @ R)) if hard
                                       else (V_old @ R) + 0.15 * rng.normal(size=(n, d))
                                       ).astype(np.float32))
    qi = rng.integers(0, clusters, 120)
    Q_old = migration._safe_normalize(
        (C[qi] + 0.3 * rng.normal(size=(120, d))).astype(np.float32))
    Q_new = migration._safe_normalize(((np.tanh(1.5 * (Q_old @ R)) if hard
                                        else Q_old @ R)).astype(np.float32))
    return V_old, V_new, Q_old, Q_new


def test_migration_cli_runs_with_adapter_and_baselines():
    """The library functions were tested; the CLI path was not.

    That gap hid a NameError -- `main` used a `kps` that only existed inside
    `frontier` -- so both `--adapter` and `--baselines` crashed on startup
    while every unit test passed. This runs the real command end to end and
    checks the results file carries both sections.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    with open(d / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i in range(120):
            body = " ".join(rng.choice(words, 12))
            f.write(json.dumps({"id": f"u{i}", "text": f"{body} {i}"}) + "\n")
    with open(d / "queries.jsonl", "w", encoding="utf-8") as f:
        for i in range(24):
            f.write(json.dumps({"id": f"q{i}",
                                "text": " ".join(rng.choice(words, 4))}) + "\n")
    out = d / "r.json"
    r = subprocess.run(
        [sys.executable, "-m", "semview.migration", "--corpus", "jsonl",
         "--path", str(d / "corpus.jsonl"), "--queries-file", str(d / "queries.jsonl"),
         "--old", "lexical:word", "--new", "lexical:char", "--k", "5",
         "--adapter", "40", "--baselines", "--d-fracs", "1.0",
         "--out", str(out)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["adapter"] and "adapter_only_recall_at_k" in got["adapter"]
    assert got["baselines"], "the --baselines table must reach the results file"
    names = {row["scheme"] for row in got["baselines"]}
    assert {"batch", "stale", "lazy", "backfill-traffic"} <= names, names
    assert "lazy+adapter" in names, "the hybrid row needs new-space warm-up queries"


def test_recall_divides_by_a_k_the_corpus_can_supply():
    """`topk_chunked` clamps k to N; the divisor has to be clamped with it.

    Otherwise identical spaces report 0.5 recall on a 5-unit corpus at k=10,
    because a 5-unit intersection is divided by 10.
    """
    V = migration._safe_normalize(
        np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32))
    rows = migration.frontier(V, V, V[:3], V[:3], k=10, kprimes=(5,), d_fracs=(1.0,))
    assert rows[0]["recall_at_kprime"] == 1.0, rows


def test_adapter_refuses_a_sample_too_small_to_fit():
    """A one-unit fit is a singular matrix, and the error LAPACK raises names
    the matrix rather than the flag the user set."""
    V = migration._safe_normalize(
        np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32))
    for n in (1, 5):
        try:
            migration.fit_adapter_selected(V, V, np.arange(n), seed=0)
        except ValueError as e:
            assert "adapter" in str(e), e
            continue
        raise AssertionError(f"a {n}-unit adapter sample must be refused")
    migration.fit_adapter_selected(V, V, np.arange(40), seed=0)


def test_float16_storage_does_not_change_what_is_retrieved():
    """--dtype float16 halves the corpus in RAM; it must not move the top-k.

    The search upcasts one shard at a time, so precision is lost only in
    storage. If this ever fails, the 8.8M run's recall would be an artifact of
    the dtype rather than of the model pair.
    """
    rng = np.random.default_rng(0)
    V = migration._safe_normalize(rng.normal(size=(3000, 48)).astype(np.float32))
    Q = V[:150]
    a = migration.topk_chunked(Q, V, 30)
    b = migration.topk_chunked(Q, V.astype(np.float16), 30)
    overlap = np.mean([len(set(x) & set(y)) / 30 for x, y in zip(a, b)])
    assert overlap > 0.995, overlap


def test_numpy_search_shards_and_merges_exactly():
    """Without FAISS the fallback must still search shard by shard -- that is
    what lets a memmapped float16 corpus be searched without loading it -- and
    the merge must be exact."""
    rng = np.random.default_rng(1)
    V = migration._safe_normalize(rng.normal(size=(2500, 32)).astype(np.float32))
    Q = V[:60]
    whole = migration.topk_chunked(Q, V, 40)
    def search(Vs, kk):
        S = Q @ Vs.T
        idx = np.argpartition(-S, kth=kk - 1, axis=1)[:, :kk]
        return np.take_along_axis(S, idx, 1), idx
    sharded = migration._sharded_topk(V, 40, search, shard=600)
    assert np.array_equal(np.sort(whole, 1), np.sort(sharded, 1))


def test_x_target_intent_counts_round_trip_exactly():
    """--x-targets encodes an intent count as a fraction d/D, and frontier turns
    it back into an integer. With int() that reconstruction can land a unit low
    on the float, so a run aimed at x=0.05 quietly measures x=0.0495."""
    for D in (2378, 6515, 10000):
        for N, kp in ((500_000, 500), (449_071, 500)):
            fr, _ = migration.x_targets_to_fracs(N, D, kp, [0.05, 0.2, 1.0, 3.0])
            for f in fr:
                assert max(1, round(D * f)) == round(f * D), (D, N, f)


def test_cipbench_queries_are_essentially_all_distinct():
    """The negative result rests on D ~= N for CIPBench.

    If its queries repeated heavily, the distinct-intent count would be far
    below the record count and the claim would weaken. Measured on the shipped
    benchmark: 2,377 distinct strings out of 2,378 records.
    """
    import os
    root = os.environ.get("CIPBENCH_PATH", "bench/bench_all_clean")
    if not os.path.isdir(root):
        raise Skip("CIPBench not present in this checkout")
    _, queries = migration.load_cipbench(root, None, 0)
    texts = [t for _, t in queries]
    assert len(set(texts)) / len(texts) > 0.99, (len(set(texts)), len(texts))


def test_random_backfill_is_the_control_for_the_ordering_claim():
    """Order is only shown to matter if a random order does worse.

    FastFill argues the backfill order is critical; without a random control,
    any ordered scheme's score could be explained by the budget alone.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces(hard=False)
    rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                     Q_old[:60], k=10, kprime=60, seed=0)
    by = {r["scheme"]: r for r in rows}
    assert "backfill-random" in by, "the ordering claim needs a control"
    assert by["backfill-random"]["passes"] == by["backfill-priority"]["passes"]
    for name in ("batch", "stale", "lazy", "backfill-random"):
        r = by[name]
        assert r["passes"] == r["passes_train"] + r["passes_serve"], r


def test_hybrid_declares_its_extra_cost_rather_than_hiding_it():
    """lazy+adapter is not an equal-cost row and must not look like one.

    It pays the adapter sample on top of its serving passes, and its window is
    chosen in a different space, so it reaches a different set of units.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces(hard=False)
    rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                     Q_old[:60], Q_new_warm=Q_new[:60], k=10,
                                     kprime=60, adapter_train=300, seed=0)
    h = [r for r in rows if r["scheme"] == "lazy+adapter"][0]
    assert h["passes_train"] == 300 and h["passes_serve"] > 0
    assert h["passes"] == h["passes_train"] + h["passes_serve"]
    assert "NOT equal-cost" in h["note"], "the caveat must travel with the row"


def test_adapter_ridge_is_selected_on_held_out_units():
    """Not a constant a reviewer can call cherry-picked, and not tuned on the
    metric being reported: selection uses reconstruction on units the fit never
    saw, and the whole sweep is recorded."""
    rng = np.random.default_rng(0)
    V_old = migration._safe_normalize(rng.normal(size=(1200, 24)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    V_new = migration._safe_normalize(
        (V_old @ R + 0.2 * rng.normal(size=(1200, 24))).astype(np.float32))
    idx = rng.choice(1200, 300, replace=False)
    W, chosen, sweep = migration.fit_adapter_selected(V_old, V_new, idx, seed=0)
    assert chosen in migration.RIDGES_DEFAULT
    assert len(sweep) == len(migration.RIDGES_DEFAULT)
    best = max(sweep, key=lambda x: x["heldout_cosine"])
    assert best["ridge"] == chosen, (chosen, sweep)
    # a heavily regularised fit must be visibly worse, or selection is vacuous
    assert sweep[-1]["heldout_cosine"] < best["heldout_cosine"]


def test_baselines_charge_every_scheme_the_same_budget():
    """Lazy spends what traffic demands; the others are given exactly that.

    Capping lazy at an arbitrary budget would measure a system that stops
    repairing mid-window, which nobody would deploy.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces()
    rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                     Q_old[:60], k=10, kprime=60, seed=0)
    by = {r["scheme"]: r for r in rows}
    assert by["batch"]["recall_at_k"] == 1.0 and by["batch"]["passes"] == len(V_new)
    assert by["stale"]["passes"] == 0
    assert by["lazy"]["passes"] == by["backfill-traffic"]["passes"] \
        == by["backfill-priority"]["passes"], by
    assert 0 < by["lazy"]["passes"] < len(V_new), "lazy must not touch everything"


def test_oracle_backfill_holds_the_unit_set_fixed_and_dominates_lazy():
    """The control that separates ordering from on-demand repair.

    The oracle backfill serves lazy's exact unit set -- evaluation windows
    included -- globally, with no repair on the query path. Because its
    candidate set is a superset of lazy's window and both rank exactly in the
    new space, it can never score below lazy; and because it takes the same
    units, its cost is lazy's. If it ever scored below, the set or the ranking
    is wrong, not the claim.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces()
    rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                     Q_old[:60], k=10, kprime=60, seed=0)
    by = {r["scheme"]: r for r in rows}
    assert "backfill-oracle" in by, "the on-demand claim needs this control"
    assert by["backfill-oracle"]["passes"] == by["lazy"]["passes"]
    assert by["backfill-oracle"]["recall_at_k"] >= by["lazy"]["recall_at_k"] - 1e-9


def test_calibrated_and_rrf_windows_clamp_k_to_the_window_width():
    """A window narrower than k cannot return k units.

    `_calibrated_serve` and `_rrf_serve` sized their output to k and assigned a
    kk-long window into it, so with kprime < k they raised a broadcast error.
    Every other top-k here clamps k to the corpus (topk_chunked, frontier,
    _exact_window_recall); these two now do, and return at most kk columns.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces(n=200)
    have = np.ones(len(V_old), dtype=bool)
    out_cal = baselines._calibrated_serve(Q_old[:4], Q_new[:4], V_old, V_new,
                                          have, 5, 10, 1.0)
    out_rrf = baselines._rrf_serve(Q_old[:4], Q_new[:4], V_old, V_new, have, 5, 10)
    assert out_cal.shape == (4, 5), out_cal.shape
    assert out_rrf.shape == (4, 5), out_rrf.shape


def test_calibrating_the_two_spaces_is_not_an_alternative_to_repairing():
    """The paper's central premise, measured instead of asserted.

    The design refuses to compare a source-space score with a target-space
    score. The cheapest way to attack that premise is to calibrate instead of
    repairing: hold the unit set fixed to the traffic-ordered backfill's, and
    rank by a mixture of the two spaces' scores. Two mixtures are given their
    best shot -- per-window standardisation with one scalar fitted on warm-up
    traffic, and reciprocal-rank fusion, which never compares scores at all --
    and both are charged exactly the backfill's passes.

    Neither reaches an exact target-space rerank of the same window. What the
    two do beyond that differs, and the difference is worth keeping: per-window
    standardisation with one fitted scalar does recover a large part of the gap
    over serving globally from those same units -- it can still use the source
    scores of the window's unrepaired members -- while rank fusion adds nothing
    at all over serving globally. So the honest claim is not that calibration is
    useless; it is that calibration buys quality back only part of the way and
    saves no passes at all, and the design's advantage comes from repairing the
    window rather than from how the two score lists are combined.

    Corpus size matters here: the fixture's default n=6000 lets lazy's spend
    cover most of the corpus, so every scheme is handed almost everything and
    the comparison goes vacuous. n=20000 keeps lazy's spend near a quarter.
    """
    for hard in (True, False):
        V_old, V_new, Q_old, Q_new = _two_spaces(n=20000, hard=hard)
        rows = baselines.compare_schemes(
            V_old, V_new, Q_old[30:110], Q_new[30:110], Q_old[:30],
            Q_new_warm=Q_new[:30], k=10, kprime=60, seed=0)
        by = {r["scheme"]: r for r in rows}
        for name in ("merge-calibrated", "merge-rrf"):
            assert name in by, "the calibration alternative must be measured"
            # equal cost, same unit set as the traffic-ordered backfill
            assert by[name]["passes"] == by["backfill-traffic"]["passes"], (hard, by[name])
            assert by[name]["passes_train"] == 0, by[name]
            # and it must lose to repairing the window
            assert by["lazy"]["recall_at_k"] - by[name]["recall_at_k"] > 0.05, (hard, by)
        # rank fusion, which never compares two scores, buys nothing over
        # serving globally from those same units
        assert by["merge-rrf"]["recall_at_k"] <= by["backfill-traffic"]["recall_at_k"] + 0.05, (hard, by)
        # the fixture must keep the comparison non-degenerate
        assert by["lazy"]["passes"] < 0.5 * len(V_new), hard


def test_n_ladder_nests_its_subsets_and_carries_its_own_n():
    """The recall-versus-N curve must compare corpora, not draws.

    `migration --limit L` samples independently for every L, so a ladder built
    from it would confound corpus size with which units happened to be drawn and
    would answer nothing. This tool fixes one permutation and takes its
    prefixes; if that ever stops being true the curve is meaningless, so it is
    pinned here rather than trusted.
    """
    import importlib.util

    path = Path(migration.__file__).resolve().parent.parent / "analysis" / "n_ladder.py"
    spec = importlib.util.spec_from_file_location("n_ladder_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    idx = mod.nested_indices(1000, [100, 300, 1000], seed=0)
    assert [len(i) for i in idx] == [100, 300, 1000]
    for a, b in zip(idx, idx[1:]):
        assert set(a.tolist()) <= set(b.tolist()), "subsets must nest"
    again = mod.nested_indices(1000, [100, 300, 1000], seed=0)
    assert all((a == b).all() for a, b in zip(idx, again)), "same seed, same subsets"
    for bad in ([300, 100], [0, 10], [10, 1001]):
        try:
            mod.nested_indices(1000, bad)
            raise AssertionError(f"nested_indices accepted {bad}")
        except ValueError:
            pass

    V_old, V_new, Q_old, Q_new = _two_spaces(n=3000)
    rows = mod.ladder(V_old, V_new, Q_old, Q_new, [500, 1500, 3000],
                      k=10, kprimes=(10, 50), d=60, seed=0, repeats=2)
    assert {r["N"] for r in rows} == {500, 1500, 3000}, rows
    assert all(r["subset"] == r["N"] for r in rows), rows
    # the same queries at every scale, so D must not move with N
    assert len({r["D"] for r in rows}) == 1, sorted({r["D"] for r in rows})
    # and x = DK'/N falls as N grows, so savings must rise at a fixed D and K'
    for kp in (10, 50):
        pts = sorted((r["N"], r["savings_pct"]) for r in rows if r["kprime"] == kp)
        assert pts[0][1] < pts[-1][1], (kp, pts)


def test_second_generation_union_accounts_for_its_passes_and_dominates():
    """The two-generation question, with the accounting pinned.

    A unit leg 1 repaired holds a MID vector, useless for ranking in the new
    space, so every unit in leg 2's window is encoded with f_new and that
    migration's cost is |W|. The tool must count that, not inherit leg 1's
    number, and its rules must be monotone in the candidate set they build.

    What is deliberately **not** asserted: that the union restricted to R1 beats
    the original index. That is the empirical question the tool exists to answer,
    the fixture here does not settle it, and a test claiming it would be
    asserting the answer to the experiment.
    """
    import importlib.util

    path = Path(migration.__file__).resolve().parent.parent / "analysis" / "gen2_union.py"
    spec = importlib.util.spec_from_file_location("gen2_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    n, d, dq, hard = 20000, 32, 240, 2.5
    C = migration._safe_normalize(rng.normal(size=(40, d)).astype(np.float32))
    V_old = migration._safe_normalize(
        (C[rng.integers(0, 40, n)] + 0.35 * rng.normal(size=(n, d))).astype(np.float32))
    R1q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    R2q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_mid = migration._safe_normalize(np.tanh(hard * (V_old @ R1q)).astype(np.float32))
    V_new = migration._safe_normalize(np.tanh(hard * (V_mid @ R2q)).astype(np.float32))
    qi = rng.integers(0, 40, dq)
    Q_old = migration._safe_normalize((C[qi] + 0.3 * rng.normal(size=(dq, d))).astype(np.float32))
    Q_mid = migration._safe_normalize(np.tanh(hard * (Q_old @ R1q)).astype(np.float32))
    Q_new = migration._safe_normalize(np.tanh(hard * (Q_mid @ R2q)).astype(np.float32))

    warm = 80
    rows, stats = mod.gen2_schemes(V_old, V_mid, V_new, Q_old[:warm], Q_old[warm:],
                                   Q_mid[warm:], Q_new[warm:], k=10, kprimes=(20, 50))
    by = {(r["scheme"], r["kprime"]): r for r in rows}

    for kp in (20, 50):
        o, m, u, oo = (by[("old-only", kp)], by[("mid-only", kp)],
                       by[("union", kp)], by[("union-open", kp)])
        # cost is this migration's own window, and the rules really differ
        assert u["passes"] >= o["passes"], (kp, u["passes"], o["passes"])
        assert oo["passes"] >= u["passes"], (kp, oo["passes"], u["passes"])
        assert m["passes"] <= u["passes"], (kp, m["passes"], u["passes"])
        # set containment => monotone in candidate recall
        assert o["recall_candidates"] <= u["recall_candidates"] + 1e-9, (kp, "union")
        assert u["recall_candidates"] <= oo["recall_candidates"] + 1e-9, (kp, "open")
        assert o["recall_at_k"] <= u["recall_at_k"] + 1e-9, (kp, "served union")
        assert u["recall_at_k"] <= oo["recall_at_k"] + 1e-9, (kp, "served open")
    # leg 1's coverage is reported and its passes are its reached set
    assert stats["passes_leg1"] == stats["R1"] and 0 < stats["R1"] < n
    # the unrestricted union must actually add truth the original window missed,
    # or the fixture is too easy to test anything
    oo20, o20 = by[("union-open", 20)], by[("old-only", 20)]
    assert oo20["recall_at_k"] > o20["recall_at_k"] + 0.02, (oo20["recall_at_k"], o20["recall_at_k"])
    assert o20["recall_at_k"] < 0.999, "fixture saturates; the comparison is vacuous"


def test_serve_server_does_not_probe_when_the_width_is_known():
    """The probe encode is a real forward pass and it is not a repair.

    `ThawServer.__post_init__` learned the new-space width by encoding
    `self.texts[0]`: a pass no trace counts, made once per server, and one that
    indexed an empty list on an empty corpus -- an IndexError where a reason
    belongs. A caller that knows the width must be able to skip it.
    """
    calls = []

    def encode_new(ts):
        calls.append(list(ts))
        return np.ones((len(ts), 16), dtype=np.float32)

    V_old = migration._safe_normalize(np.arange(24, dtype=np.float32).reshape(6, 4) + 1.0)
    texts = [f"unit {i}" for i in range(6)]
    base = dict(V_old=V_old, texts=texts, encode_new=encode_new,
                encode_old_query=lambda t: V_old[0],
                encode_new_query=lambda t: np.ones(16, dtype=np.float32), kprime=4)

    srv = serve.ThawServer(**base, new_dim=16)
    assert calls == [], "a known width must need no probe"
    assert srv.V_new.shape == (6, 16)

    calls.clear()
    srv2 = serve.ThawServer(**base)
    assert len(calls) == 1 and calls[0] == [texts[0]], calls
    assert srv2.V_new.shape == (6, 16)

    empty = dict(V_old=np.zeros((0, 4), dtype=np.float32), texts=[],
                 encode_new=encode_new, encode_old_query=lambda t: V_old[0],
                 encode_new_query=lambda t: np.ones(16, dtype=np.float32), kprime=4)
    try:
        serve.ThawServer(**empty)
        raise AssertionError("an empty corpus without a width must be refused")
    except SystemExit as e:
        assert "empty" in str(e), e
    assert serve.ThawServer(**empty, new_dim=16).V_new.shape == (0, 16)


def test_union_refuses_an_empty_query_stream_instead_of_indexing_it():
    """Every union figure comes from the reached set the stream builds.

    With zero queries `replay()` returns empty arrays and `summarise()` indexed
    `reached_after[-1]`, so the tool raised IndexError on a query set that
    `--max-queries 0` cannot even rule out (0 reads as "no cap"). An empty stream
    is not a degenerate measurement, it is no measurement, so both entry points
    refuse it with a reason.
    """
    from semview import union

    V = migration._safe_normalize(np.arange(25, dtype=np.float32).reshape(5, 5) + 1.0)
    empty = np.zeros((0, 5), dtype=np.float32)
    try:
        union.run(V, V, empty, empty, [3], [2], 2, 2)
        raise AssertionError("an empty query stream must be refused")
    except SystemExit as e:
        assert "empty" in str(e) or "no queries" in str(e), e

    res = dict(N=5, base_hits=np.zeros((0, 2), dtype=bool),
               union_hits={2: np.zeros((0, 2), dtype=bool)},
               union_extra={2: np.zeros(0, dtype=np.int32)},
               reached_after=np.zeros(0, dtype=np.int64))
    try:
        union.summarise(res, 3, 2)
        raise AssertionError("summarise must refuse an empty replay")
    except SystemExit as e:
        assert "empty" in str(e), e


def test_serve_summarise_and_estimate_coverage_refuse_empty_streams():
    """The same refusal as union, on the other two stream-based tools.

    `serve.summarise([])` reached `float(tot.max())` on an empty array, which is
    a ValueError, and `estimate.coverage_curve` with no queries let
    `np.geomspace(1, 0, points)` run off to zeros. Both are "no measurement", not
    a degenerate one, so both library functions refuse with a reason.
    """
    try:
        serve.summarise([])
        raise AssertionError("serve.summarise must refuse an empty replay")
    except SystemExit as e:
        assert "empty" in str(e), e

    V = migration._safe_normalize(np.arange(25, dtype=np.float32).reshape(5, 5) + 1.0)
    empty = np.zeros((0, 5), dtype=np.float32)
    try:
        estimate.coverage_curve(empty, V, 3)
        raise AssertionError("coverage_curve must refuse an empty query set")
    except SystemExit as e:
        assert "empty" in str(e), e


def test_lazy_is_charged_for_the_evaluation_windows_it_repairs():
    """The lazy row scores an exact rerank of every evaluation window, so its
    spend must cover those windows -- not only the warm-up traffic's."""
    V_old, V_new, Q_old, Q_new = _two_spaces()
    rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                     Q_old[:60], k=10, kprime=60, seed=0)
    by = {r["scheme"]: r for r in rows}
    wins = migration.topk_chunked(np.vstack([Q_old[:60], Q_old[60:]]), V_old, 60)
    assert by["lazy"]["passes"] == len(np.unique(wins))
    warm_only = len(np.unique(migration.topk_chunked(Q_old[:60], V_old, 60)))
    assert by["lazy"]["passes"] > warm_only


def test_hybrid_row_uses_new_space_queries_for_the_adapter_index():
    """The adapter index approximates the NEW space.

    Warming it with old-space queries retrieves nonsense, which is invisible in
    the output -- the row is simply wrong. The row must appear when new-space
    warm-up queries are supplied and be skipped when they are not.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces(hard=False)
    without = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                        Q_old[:60], k=10, kprime=60,
                                        adapter_train=300, seed=0)
    assert "lazy+adapter" not in {r["scheme"] for r in without}, \
        "no new-space warm-up traffic means the row cannot be computed"
    with_new = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                         Q_old[:60], Q_new_warm=Q_new[:60],
                                         k=10, kprime=60, adapter_train=300, seed=0)
    row = [r for r in with_new if r["scheme"] == "lazy+adapter"]
    assert row and row[0]["passes"] > 0, with_new


def test_lazy_reranks_its_window_rather_than_truncating_it():
    """The window comes from the old space; the ranking must not.

    If `lazy` merely truncated the old ranking it would score like `stale`.
    On a pair with high cross-space recall it must do materially better.
    """
    V_old, V_new, Q_old, Q_new = _two_spaces(hard=False)
    rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                     Q_old[:60], k=10, kprime=60, seed=0)
    by = {r["scheme"]: r for r in rows}
    assert by["lazy"]["recall_at_k"] > by["stale"]["recall_at_k"] + 0.1, by


def test_whether_lazy_beats_an_ordered_backfill_is_not_fixed():
    """Not a foregone conclusion, and the evaluation must say so.

    Lazy is capped by cross-space recall at K'; an ordered backfill serves
    globally from whatever it has migrated but can never return a unit it has
    not migrated. Which wins depends on the corpus, the depth and the pair --
    a synthetic sweep reverses the sign as the parameters move -- so the
    comparison has to be measured, not predicted. This test pins only that the
    margin MOVES with the pair; asserting a direction would be asserting a
    result we have not established.
    """
    def gap(hard):
        V_old, V_new, Q_old, Q_new = _two_spaces(hard=hard)
        rows = baselines.compare_schemes(V_old, V_new, Q_old[60:], Q_new[60:],
                                         Q_old[:60], k=10, kprime=60, seed=0)
        by = {r["scheme"]: r for r in rows}
        return by["lazy"]["recall_at_k"] - by["backfill-priority"]["recall_at_k"]
    assert abs(gap(hard=False) - gap(hard=True)) > 0.05, (gap(False), gap(True))


def test_adapter_is_worth_more_as_a_window_than_as_a_replacement():
    """The hybrid must never be worse than using the adapter alone.

    Adapter vectors used for CANDIDATE GENERATION, with exact reranking after,
    dominate adapter vectors used as the index: the window only has to contain
    the right units, not rank them. This is why an adapter is an argument for a
    better old index rather than against re-embedding.
    """
    rng = np.random.default_rng(0)
    V_old = migration._safe_normalize(rng.normal(size=(1500, 32)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    V_new = migration._safe_normalize((V_old @ R).astype(np.float32))
    idx = rng.choice(1500, 200, replace=False)
    W = migration.fit_linear_adapter(V_old, V_new, idx)
    rep = migration.adapter_report(V_old, V_new, V_new[:80], W, k=10,
                                   kprimes=(50, 200))
    alone = rep["adapter_only_recall_at_k"]
    hyb = {r["kprime"]: r["hybrid_recall_at_kprime"] for r in rep["hybrid"]}
    assert hyb[50] >= alone and hyb[200] >= hyb[50], (alone, hyb)
    assert alone > 0.5, "a rotation is exactly what a linear adapter should fit"

    # and where the spaces are not linearly related, the adapter should fail --
    # otherwise the measurement is not discriminating
    V_nl = migration._safe_normalize(
        np.tanh(6 * (V_old @ R)).astype(np.float32)
        + 0.8 * rng.normal(size=(1500, 32)).astype(np.float32))
    W2 = migration.fit_linear_adapter(V_old, V_nl, idx)
    r2 = migration.adapter_report(V_old, V_nl, V_nl[:80], W2, k=10, kprimes=(50,))
    assert r2["adapter_only_recall_at_k"] < 0.5, r2


def test_query_path_passes_can_cost_more_than_batch_passes():
    """Small query-path batches are less efficient per unit. Pricing both at
    the batch job's rate overstates savings."""
    base = dict(N=1_000_000, D=10_000, Q=2_000_000, reached=0.2, unit_ms=10.0,
                query_ms=2.0)
    same = cost.compare(cost.CostParams(**base))
    slow = cost.compare(cost.CostParams(**base, repair_unit_ms=30.0))
    assert same["repair_penalty"] == 1.0 and slow["repair_penalty"] == 3.0
    assert slow["net_savings_pct"] < same["net_savings_pct"] - 30
    assert slow["lazy_gpu_seconds"] > same["lazy_gpu_seconds"]


def test_repair_batch_profile_weights_work_not_batches():
    """Most repair work happens in large batches early in a migration, so the
    weighted penalty must be far below the worst batch's penalty."""
    # a saturating coverage curve: big early gains, a long thin tail
    ds = np.array([1, 10, 100, 1000, 7000], float)
    rc = np.array([0.0, 0.2, 0.6, 0.9, 0.97])
    prof = cost.repair_batch_profile(ds, rc, N=500_000, saturation_batch=32.0)
    assert prof["weighted_batch"] > 100, prof
    assert 1.0 < prof["penalty"] < 1.6, prof
    assert prof["work_below_32"] < 0.2, prof
    # a harsher saturation point costs more
    worse = cost.repair_batch_profile(ds, rc, N=500_000, saturation_batch=128.0)
    assert worse["penalty"] > prof["penalty"]
    # a degenerate curve must not raise
    flat = cost.repair_batch_profile([1, 2], [0.5, 0.5], N=10)
    assert flat["penalty"] == 1.0


def test_uncached_inversion_condition():
    """Inverts exactly when Q*query_ms exceeds (1-reached)*N*unit_ms."""
    N, reached, unit_ms, query_ms = 1_000_000, 0.1, 10.0, 2.0
    limit = (1 - reached) * N * unit_ms / query_ms          # queries at breakeven
    for Q, expect in ((int(limit * 0.9), False), (int(limit * 1.2), True)):
        r = cost.compare(cost.CostParams(N=N, D=1_000, Q=Q, reached=reached,
                                         unit_ms=unit_ms, query_ms=query_ms,
                                         cache_hit_rate=0.0))
        assert r["inverts"] is expect, (Q, r["net_savings_pct"])


def test_from_estimate_refuses_a_hand_written_reached():
    try:
        cost.from_estimate({"reached": 0.1}, D=1, Q=1, N=10)
    except ValueError:
        pass
    else:
        raise AssertionError("a dict without reached_observed must be rejected")
    row = {"reached_observed": 0.3, "N": 1000, "D": 10}
    p = cost.from_estimate(row, Q=100)
    assert p.reached == 0.3 and p.N == 1000 and p.D == 10
    for bad in (dict(N=0), dict(N=-5)):
        try:
            cost.from_estimate(row, Q=1, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be rejected, not replaced by the row's N")
    try:
        cost.from_estimate({"reached_observed": 1.4, "N": 1000, "D": 10}, Q=1)
    except ValueError:
        pass
    else:
        raise AssertionError("reached outside [0, 1] must be rejected")



# ---------------------------------------------------------------- 8.8M path
class _FakeEncoder:
    """Deterministic stand-in for a SentenceTransformer: one row per text,
    derived from the text, normalised. `fail_after` simulates a crash."""
    def __init__(self, dim=6, fail_after=None):
        self.dim, self.calls, self.fail_after = dim, 0, fail_after

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated crash")
        out = np.array([[((hash(t) >> (4 * j)) % 97) + 1.0 for j in range(self.dim)]
                        for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_fallback_search_never_upcasts_the_whole_corpus():
    """The numpy fallback used to compute Q @ V.T against all of V; with V a
    float16 memmap that materialised a float32 copy of the entire corpus per
    query chunk. It must search one shard at a time and stay exact."""
    import tempfile, tracemalloc, os
    rng = np.random.default_rng(3)
    # d large enough that a shard, not the per-query scratch, dominates --
    # as at 768 dimensions
    N, d = 1_500_000, 64
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "v.npy")
        W = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(N, d))
        W[:] = rng.normal(size=(N, d)).astype(np.float16); W.flush(); del W
        V = np.load(f, mmap_mode="r")
        Q = rng.normal(size=(5, d)).astype(np.float32)
        import sys as _sys
        saved = _sys.modules.get("faiss", None)
        _sys.modules["faiss"] = None          # force the numpy fallback even
        try:                                  # where FAISS is installed
            tracemalloc.start()
            got = migration.topk_chunked(Q, V, 50)
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        finally:
            if saved is None:
                del _sys.modules["faiss"]
            else:
                _sys.modules["faiss"] = saved
        full_f32 = N * d * 4
        assert peak < 0.6 * full_f32, (peak, full_f32)
        want = np.argsort(-(Q @ np.asarray(V, dtype=np.float32).T), axis=1)[:, :50]
        assert np.array_equal(np.sort(got, 1), np.sort(want, 1))
        del V


def test_chunked_encode_matches_whole_and_writes_target_dtype():
    texts = [f"passage {i}" for i in range(1037)]
    whole = _FakeEncoder().encode(texts)
    for dt, npdt in (("float32", np.float32), ("float16", np.float16)):
        V = migration.encode_to_array(_FakeEncoder(), texts, dtype=dt, chunk=100)
        assert V.dtype == npdt and V.shape == whole.shape
        assert np.allclose(np.asarray(V, dtype=np.float32), whole, atol=2e-3)


def test_interrupted_encode_resumes_and_never_leaves_a_fake_cache():
    """A crash mid-encode must not leave a file at the cache path (it would be
    loaded as a finished cache next time), and a rerun must resume rather than
    start over."""
    import tempfile
    texts = [f"passage {i}" for i in range(950)]
    whole = _FakeEncoder().encode(texts)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "key.f16.npy"
        try:
            migration.encode_to_array(_FakeEncoder(fail_after=4), texts,
                                      dtype="float16", path=path, chunk=100)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the fake encoder should have crashed")
        assert not path.exists()
        second = _FakeEncoder()
        V = migration.encode_to_array(second, texts, dtype="float16",
                                      path=path, chunk=100)
        assert second.calls == 6, second.calls        # rows 400..949 only
        assert path.exists() and V.dtype == np.float16
        assert np.allclose(np.asarray(V, dtype=np.float32), whole, atol=2e-3)
        assert not (Path(tmp) / "key.f16.npy.partial").exists()
        assert not (Path(tmp) / "key.f16.npy.progress").exists()
        # The cached path comes back memory-mapped, which is the point for an
        # 8.8M corpus and a liability in a temporary directory: on Windows an
        # open mapping makes the file undeletable, so leaving it open here
        # failed the *cleanup* with WinError 32 and read as a bug in the
        # encoder. The handle is the caller's to release.
        mm = getattr(V, "_mmap", None)
        assert mm is not None, "the cached array is expected to be memory-mapped"
        migration.close_memmap(V)
        del V
        assert mm.closed, "close_memmap left the file handle open"



# ------------------------------------------------- throughput / repair penalty
def _curve(points):
    return {b: {"total_ms": ms * b, "ms_per_item": ms} for b, ms in points.items()}


def test_chunked_penalty_is_one_when_repairs_are_whole_chunks():
    """Every repair batch a multiple of the chunk, chunk at the best rate:
    the query path pays exactly what a backfill pays."""
    curve = _curve({1: 10.0, 8: 3.0, 64: 9.0})
    N = 100_000
    ds = [1, 2, 3, 4]
    rc = [16 / N, 32 / N, 48 / N, 64 / N]          # 16 units per new intent
    r = cost.chunked_repair_penalty(curve, ds, rc, N, chunk=8)
    assert abs(r["penalty"] - 1.0) < 1e-9, r
    assert r["work_below_chunk"] == 0.0


def test_chunked_penalty_charges_small_batches_their_measured_cost():
    curve = _curve({1: 12.0, 8: 3.0})
    N = 10_000
    ds = [1, 2, 3]
    rc = [1 / N, 2 / N, 3 / N]                     # one unit per intent
    r = cost.chunked_repair_penalty(curve, ds, rc, N, chunk=8)
    assert abs(r["penalty"] - 4.0) < 1e-9, r       # 12 ms against a best of 3
    assert r["work_below_chunk"] == 1.0


def test_chunked_penalty_rejects_an_unmeasured_chunk():
    try:
        cost.chunked_repair_penalty(_curve({1: 5.0, 8: 3.0}), [1, 2], [0, 1e-3],
                                    1000, chunk=16)
    except ValueError:
        return
    raise AssertionError("a chunk size that was never measured must be refused")


def _load_measure_throughput():
    import importlib.util
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "analysis" / "measure_throughput.py"
    spec = importlib.util.spec_from_file_location("measure_throughput", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_derive_prices_the_batch_job_at_the_best_batch_not_the_largest():
    """The bug that left unit_ms = 42.96 in throughput_real.json: on a
    memory-tight card the largest batch is the WORST, and a backfill would
    never run at it."""
    mt = _load_measure_throughput()
    p = _curve({1: 9.77, 8: 3.16, 32: 4.64, 512: 42.96})
    q = _curve({1: 5.2, 512: 0.52})
    d = mt.derive(p, q)
    assert d["best_batch"] == 8 and abs(d["unit_ms"] - 3.16) < 1e-9
    assert d["curve_rises_past_best"] is True
    # a query is encoded alone on the serving path: priced at batch 1
    assert d["query_ms"] == 5.2 and d["query_ms_batched"] == 0.52


def test_from_json_replaces_stale_derived_fields():
    import json, os, subprocess, sys, tempfile
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    est = root / "results" / "estimate_msmarco.json"
    with tempfile.TemporaryDirectory() as tmp:
        stale = dict(passage_curve=_curve({1: 9.77, 8: 3.1561, 512: 42.9576}),
                     query_curve=_curve({1: 5.2, 512: 0.5213}),
                     unit_ms=42.9576, repair_unit_ms=47.2, model="m")
        src = os.path.join(tmp, "t.json")
        json.dump(stale, open(src, "w", encoding="utf-8"))
        cmd = [sys.executable, str(root / "analysis" / "measure_throughput.py"),
               "--from-json", src, "--out", src]
        if est.exists():
            cmd += ["--estimate", str(est), "--kprime", "500"]
        subprocess.run(cmd, check=True, capture_output=True)
        out = json.load(open(src, encoding="utf-8"))
        assert abs(out["unit_ms"] - 3.1561) < 1e-9, out["unit_ms"]
        assert out["model"] == "m"                      # measured fields kept
        if est.exists():
            assert out["penalty_method"] == "chunked"
            assert abs(out["repair_unit_ms"] - 3.1561 * out["penalty"]) < 1e-9
        else:
            assert "repair_unit_ms" not in out          # stale value dropped


def test_shipped_throughput_file_is_self_consistent():
    """results/throughput_real.json must not carry a derived field that its
    own curve contradicts -- the state it shipped in before this check."""
    import json
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "results" / "throughput_real.json"
    if not f.exists():
        raise Skip("results/throughput_real.json not present")
    d = json.load(open(f, encoding="utf-8"))
    best = min(v["ms_per_item"] for v in d["passage_curve"].values())
    assert abs(d["unit_ms"] - best) < 1e-9, (d["unit_ms"], best)
    if "repair_unit_ms" in d:
        assert abs(d["repair_unit_ms"] - d["unit_ms"] * d["penalty"]) < 1e-9



def test_repair_chunk_is_the_best_batch_not_the_saturation_batch():
    """A curve whose saturation batch (within 10% of best) is 4 but whose best
    is 16: chunking at 4 pays ~8% on every chunk. The chunk must minimise the
    measured penalty."""
    import json, os, tempfile
    from pathlib import Path
    mt = _load_measure_throughput()
    est = Path(__file__).resolve().parent.parent / "results" / "estimate_msmarco.json"
    if not est.exists():
        raise Skip("results/estimate_msmarco.json not present")
    p = _curve({1: 8.41, 2: 4.01, 4: 3.22, 8: 3.14, 16: 2.97, 32: 4.57, 512: 27.4})
    q = _curve({1: 5.0, 512: 0.18})
    d = mt.derive(p, q, str(est), 500)
    assert d["saturation_batch"] == 4
    assert d["repair_chunk"] == 16, d["repair_chunk"]
    assert d["penalty"] < d["repair_profile"]["penalty_by_chunk"]["4"]
    assert abs(d["penalty"] - 1.0) < 0.02, d["penalty"]


# ------------------------------------------------------------ candidate union
def _union_fixture(seed=1, N=3000, D=200):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N, 16)).astype(np.float32)
    A = rng.normal(size=(16, 24)).astype(np.float32)
    B = rng.normal(size=(16, 32)).astype(np.float32)
    P = 0.8 * rng.normal(size=(16, 16)).astype(np.float32)
    V_old = migration._safe_normalize(z @ A + 2 * rng.normal(size=(N, 24)).astype(np.float32))
    V_new = migration._safe_normalize((z + z @ P) @ B)
    qi = rng.choice(N, D, replace=False)
    zq = z[qi] + 0.2 * rng.normal(size=(D, 16)).astype(np.float32)
    return (V_old, V_new, migration._safe_normalize(zq @ A),
            migration._safe_normalize((zq + zq @ P) @ B))


def test_union_never_lowers_recall_and_costs_no_extra_passes():
    """Union adds candidates that already have new vectors: per-query recall
    can only rise, and the repaired set must equal the union of old windows."""
    from semview import union
    V_old, V_new, Q_old, Q_new = _union_fixture()
    truth = union._topk_rows(Q_new, V_new, 10)
    res = union.replay(V_old, V_new, Q_old, Q_new, truth, 60, kns=(10, 50), block=7)
    for kn, h in res["union_hits"].items():
        assert np.all(h.mean(1) >= res["base_hits"].mean(1)), kn
    W = union._topk_rows(Q_old, V_old, 60)
    assert res["reached_after"][-1] == len(np.unique(W))


def test_union_with_exact_search_needs_only_k_extra_candidates():
    """With exact new-space search, top-k of (window U repaired) is reached by
    taking only k candidates from the repaired set: more cannot change it."""
    from semview import union
    V_old, V_new, Q_old, Q_new = _union_fixture(seed=2)
    truth = union._topk_rows(Q_new, V_new, 10)
    res = union.replay(V_old, V_new, Q_old, Q_new, truth, 40, kns=(10, 200), block=5)
    assert np.array_equal(res["union_hits"][10], res["union_hits"][200])


def test_union_recovers_a_unit_an_earlier_query_repaired():
    """The mechanism in miniature: query 2's true top-1 (unit 1) is outside
    its own old window but inside query 1's, so only union finds it."""
    from semview import union
    V_old = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    V_new = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]], dtype=np.float32)
    Q_old = np.array([[0, 1], [1, 0]], dtype=np.float32)   # windows {1}, {0}
    Q_new = np.array([[1, 0], [1, 0]], dtype=np.float32)   # both want unit 1
    truth = np.array([[1], [1]])
    res = union.replay(V_old, V_new, Q_old, Q_new, truth, 1, kns=(1,), k=1, block=1)
    assert res["base_hits"][0, 0] and not res["base_hits"][1, 0]
    assert res["union_hits"][1][1, 0]


def test_length_sorted_chunks_keep_input_order_and_pad_less():
    """Encoders chunk by length but must return rows in input order."""
    import numpy as np
    from thaw_pg.encoders import _Base

    class Rec(_Base):
        dim = 2

        def __init__(self):
            super().__init__("", 2)
            self.chunks = []

        def _encode(self, texts):
            self.chunks.append([len(t) for t in texts])
            return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    texts = ["a" * n for n in (9, 1, 7, 2, 8, 3)]
    e = Rec()
    out = e.encode(texts)
    want = np.array([[len(t), 1.0] for t in texts], dtype=np.float32)
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert np.allclose(out, want)
    assert e.chunks == [[1, 2], [3, 7], [8, 9]]
    e2 = Rec()
    e2.sort_by_length = False
    e2.encode(texts)
    pad = lambda ch: sum(max(c) * len(c) for c in ch)
    assert pad(e.chunks) < pad(e2.chunks)


# ----------------------------------------------------------- backfill-rate ---
class _TokenCostEncoder:
    """Stand-in for encode_timed: words are tokens, cost is proportional to
    padded tokens, and tokenizing costs a fixed amount per unit."""
    dim = 4

    def __init__(self, ms_per_padded=0.01, tok_ms_per_unit=0.05):
        self.a, self.b = ms_per_padded, tok_ms_per_unit

    def encode_timed(self, texts):
        import numpy as np
        ls = [len(t.split()) for t in texts]
        padded = max(ls) * len(ls)
        return np.ones((len(texts), 4), np.float32), dict(
            tok_ms=self.b * len(ls), fwd_ms=self.a * padded, copy_ms=0.0,
            real_tokens=sum(ls), padded_tokens=padded, max_len=max(ls))


def test_backfill_streams_cover_each_unit_once_and_sorting_cuts_padding():
    import numpy as np
    from thaw_pg import backfill as bfm
    rng = np.random.default_rng(0)
    V = rng.normal(size=(600, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = V[rng.choice(600, 40, replace=False)]
    stale = bfm.repair_stream(V, Q, 50)
    flat = [i for s in stale for i in s]
    assert len(flat) == len(set(flat))
    from semview.migration import topk_chunked
    assert set(flat) == set(topk_chunked(Q, V, 50).ravel().tolist())
    lengths = rng.integers(5, 200, size=600).tolist()
    seen = set()
    for q, ids in bfm.repair_batches(stale, 16, lengths):
        assert set(ids) <= set(stale[q]) and len(ids) <= 16
        assert ids == sorted(ids, key=lambda i: lengths[i])
        seen |= set(ids)
    assert seen == set(flat)
    got = [i for _, b in bfm.backfill_batches(range(600), 32, lengths, 0) for i in b]
    assert sorted(got) == list(range(600))
    plain = bfm.padding_factor(bfm.backfill_batches(range(600), 32, lengths, 0), lengths)
    srt = bfm.padding_factor(bfm.backfill_batches(range(600), 32, lengths, 200), lengths)
    assert srt["padding_factor"] < plain["padding_factor"]
    assert srt["real_tokens"] == plain["real_tokens"]


def test_exact_window_with_the_union_is_not_required_to_use_hnsw():
    """The gate that failed every exact+union run, and its numbers with it.

    An exact window is a sequential scan on purpose, and adding the union does
    not change that. The branch tested `exact and not union_k`, so exact+union
    fell through to the HNSW rule and was required to use an index it
    deliberately avoids: `step2_e2_union` exited non-zero with 2 of 40 gates
    failing while every measurement in it was sound. Which plan is correct is
    decided by `window` alone.
    """
    from thaw_pg.step2 import plan_gates
    base = plan_gates("exact", 0, [], 500_000, 0.9295, 0.9295)
    assert base["exact plan has no HNSW"] and base["E1 reproduces offline recall"]
    # the union raises end-to-end recall above the window's by construction, so
    # E1's agreement with the offline number must not be demanded of it
    both = plan_gates("exact", 100, [], 500_000, 0.9520, 0.9364)
    assert both["exact plan has no HNSW"], "the union does not license an HNSW window"
    assert "HNSW plan" not in both, "an exact window is never required to use HNSW"
    assert "E1 reproduces offline recall" not in both
    # and the HNSW rule still applies where it should
    assert plan_gates("hnsw", 0, ["docs_emb_old_hnsw"], 500_000, 0.92, 0.9364)[
        "HNSW plan"]
    assert not plan_gates("hnsw", 0, [], 500_000, 0.92, 0.9364)["HNSW plan"]
    # a toy corpus may legitimately seq-scan, so no HNSW gate is imposed there
    assert plan_gates("hnsw", 0, [], 50_000, 0.9, 0.9) == {}


def test_backfill_rate_penalty_is_the_ms_ratio_and_its_factors_multiply():
    import json, tempfile
    import numpy as np
    from pathlib import Path
    from thaw_pg import backfill as bfm, encoders
    rng = np.random.default_rng(1)
    N = 800
    V = rng.normal(size=(N, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = V[rng.choice(N, 30, replace=False)]
    d = Path(tempfile.mkdtemp())
    np.savez(d / "vectors.npz", V_old=V, Q_old=Q)
    with open(d / "texts.jsonl", "w", encoding="utf-8") as f:
        for i in range(N):
            f.write(json.dumps({"text": " ".join(["w"] * int(rng.integers(3, 120)))}) + "\n")
    orig = encoders.make
    encoders.make = lambda *a, **k: _TokenCostEncoder()
    try:
        rc = bfm.run(d, "local:stub", 40, None, 8, 32, 0, True, 1e9, 0.0, 0.0,
                     None, 0, None, d / "bf.json", new_model_prefix="",
                     max_rounds=1, token_sort=False)   # no tokenizer for a stub
    finally:
        encoders.make = orig
    assert rc == 0
    r = json.loads((d / "bf.json").read_text(encoding="utf-8"))
    a, rep, b = r["phases"]
    assert a["units"] == b["units"] == N and rep["units"] == r["repairs_in_stream"]
    res = r["result"]
    want = rep["ms_per_unit"] / ((a["ms_per_unit"] + b["ms_per_unit"]) / 2)
    assert abs(res["penalty"] - round(want, 4)) < 1e-3
    f = res["factors"]
    fwd = rep["fwd_ms_per_unit"] / ((a["fwd_ms_per_unit"] + b["fwd_ms_per_unit"]) / 2)
    assert abs(f["forward_penalty"] - fwd) < 2e-3
    # same device rate on both sides with this stand-in
    assert abs(f["device_ms_per_padded_token"] - 1.0) < 1e-6
    assert Path(str(d / "bf.json") + ".sha256").exists()


def test_results_files_are_checked_on_write_and_on_copy():
    import json, tempfile
    from pathlib import Path
    from semview import results_io as rio
    d = Path(tempfile.mkdtemp())
    p = d / "r.json"
    rio.write_results(p, {"fingerprint": {"a": 1}, "x": [1, 2]})
    assert rio.check_file(p) == []
    try:
        rio.write_results(d / "bad.json", {"x": 1})
        raise AssertionError("wrote a file with no provenance")
    except SystemExit:
        pass
    (d / "empty.json").write_text("", encoding="utf-8")
    assert rio.check_file(d / "empty.json") == ["empty (0 bytes)"]
    p.write_text(json.dumps({"fingerprint": {"a": 2}}), encoding="utf-8")   # altered after writing
    assert any("sha256" in m for m in rio.check_file(p))


def test_an_empty_sha256_sidecar_is_reported_not_raised():
    """The sidecar's own failure must not look like a bug in the checker.

    An interrupted write (disk full, machine released mid-copy) leaves a
    sidecar that exists and holds nothing. `side.read_text().split()[0]` raised
    IndexError, and because runlog calls this on every result file, one damaged
    sidecar turned an otherwise good run into a FAIL whose traceback pointed at
    results_io rather than at the file.
    """
    import tempfile
    from pathlib import Path
    from semview import results_io as rio
    d = Path(tempfile.mkdtemp())
    p = d / "r.json"
    rio.write_results(p, {"fingerprint": {"a": 1}})
    assert rio.check_file(p) == []
    Path(str(p) + ".sha256").write_text("", encoding="utf-8")            # truncated sidecar
    problems = rio.check_file(p)                        # must not raise
    assert problems == ["sha256 sidecar is empty"], problems
    Path(str(p) + ".sha256").write_text("deadbeef  r.json\n", encoding="utf-8")   # wrong digest
    assert any("does not match" in m for m in rio.check_file(p))


def test_verify_reports_a_truncated_manifest_line_instead_of_raising():
    """verify exists to catch a truncated copy, so a truncated *line* is a FAIL.

    Unpacking three names from a short line raised ValueError, which tells the
    reader nothing about which entry was damaged.
    """
    import tempfile
    from pathlib import Path
    from semview import results_io as rio
    d = Path(tempfile.mkdtemp())
    good = d / "good.json"
    rio.write_results(good, {"fingerprint": {"a": 1}})
    man = d / "MANIFEST.sha256"
    man.write_text(rio.sha256_file(good) + f"  {good.stat().st_size}  good.json\n"
                   + "abc123  truncated\n", encoding="utf-8")             # only two fields
    rc = rio.main(["verify", str(man)])
    assert rc == 1, "a malformed entry must fail the verification"
    man.write_text("", encoding="utf-8")                                  # empty manifest
    assert rio.main(["verify", str(man)]) == 1


def test_offline_window_recall_survives_k_equal_to_the_corpus():
    """K' = N is a case the design documents, so the offline reference must take it.

    `migration.frontier` keeps K' <= N and reports recall 1 at K' = N by
    construction; `_exact_window_recall` passed k straight to np.partition,
    which raised `kth(=N) out of bounds`. There is also no rank K'+1 to check
    for a boundary tie at that depth, which is why the tie count is 0 there.
    """
    import numpy as np
    from thaw_pg.step2 import _exact_window_recall
    rng = np.random.default_rng(0)
    N = 40
    V = rng.normal(size=(N, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = V[:4].copy()
    top10 = np.argsort(-(Q @ V.T), axis=1)[:, :10]
    for k in (1, 5, N - 1, N, N + 1):
        hit, ties = _exact_window_recall(Q, V, top10, k)   # must not raise
        assert hit.shape == (len(Q),)
    hit, ties = _exact_window_recall(Q, V, top10, N)
    assert hit.mean() == 1.0 and ties == 0, "at K'=N the window is the corpus"
    # the recall divisor is the truth's width, not a hard-coded 10
    top5 = np.argsort(-(Q @ V.T), axis=1)[:, :5]
    hit5, _ = _exact_window_recall(Q, V, top5, 5)   # identical spaces: window == truth
    assert hit5.mean() == 1.0, "a 5-wide truth must divide by 5, not 10"
    try:
        _exact_window_recall(Q, V, top10, 0)
        raise AssertionError("k=0 should be refused")
    except ValueError:
        pass


def test_union_deciles_cover_every_query():
    """report's per-decile list must account for the whole stream.

    `range(10)` over `i*dec` dropped up to 9 queries when D was not a multiple
    of 10 -- at D = 11 the 11th query belonged to no decile, so the list and the
    separate `recall_last_decile` field (built from `per_q[-dec:]`) reported
    different last deciles. Every shipped union run used D = 5000, where
    10 * 500 = 5000 exactly, so no published number was affected.
    """
    from semview.union import _decile_bounds
    for D in (1, 9, 10, 11, 25, 100, 500, 5005):
        b = _decile_bounds(D)
        covered = [i for a, z in b for i in range(a, z)]
        assert covered == list(range(D)), (D, b)
        assert b[-1][1] == D, f"the last decile must reach the end at D={D}"
    assert _decile_bounds(0) == []


def test_compare_8m_refuses_a_frontier_that_disagrees_with_itself():
    """Recall does not move with D, so two values at one K' is a broken run.

    `setdefault` silently kept the first, dropping the disagreement -- in the
    table whose whole job is to notice inconsistencies between runs.
    """
    c = _load_analysis("compare_8m")
    ok = {"frontier": [{"kprime": 100, "D": 50, "N": 1000,
                        "recall_at_kprime": 0.9, "savings_pct": 90.0},
                       {"kprime": 100, "D": 200, "N": 1000,
                        "recall_at_kprime": 0.9, "savings_pct": 80.0}]}
    assert c.recall_by_kprime(ok) == {100: 0.9}
    bad = {"frontier": [{"kprime": 100, "D": 50, "N": 1000,
                         "recall_at_kprime": 0.9, "savings_pct": 90.0},
                        {"kprime": 100, "D": 200, "N": 1000,
                         "recall_at_kprime": 0.5, "savings_pct": 80.0}]}
    try:
        c.recall_by_kprime(bad)
        raise AssertionError("an inconsistent frontier must be refused")
    except SystemExit as e:
        assert "inconsistent" in str(e)


# ------------------------------------------------- review round: new tests ---
def _load_analysis(name):
    import importlib.util
    root = Path(migration.__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, root / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frontier_reports_recall_on_the_d_subset_beside_the_all_query_value():
    """recall_at_kprime is over all queries, identical on every D row by
    construction; recall_on_d_subset is what each D sample actually got."""
    V_old, V_new, Q_old, Q_new = _two_spaces()
    rows = migration.frontier(V_old, V_new, Q_old, Q_new, k=10, kprimes=(60,),
                              d_fracs=(0.1, 1.0), seed=0)
    small, full = rows
    assert small["recall_at_kprime"] == full["recall_at_kprime"]
    assert full["recall_on_d_subset"] == full["recall_at_kprime"]
    assert small["recall_scope"] == "all queries"
    assert 0.0 <= small["recall_on_d_subset"] <= 1.0
    assert small["recall_on_d_subset_sd"] >= 0.0


def test_margin_reweighting_recovers_the_large_mix_and_renormalises():
    """If recall depends only on tie density, re-weighting the small scale to
    the large scale's mix must give the large scale's recall. A bin with no
    small-scale queries must not drag the result down."""
    m = _load_analysis("margin_8m")
    rng = np.random.default_rng(0)
    # pooled-quantile edges here are 0,1,2,3 and the last bin is [2, 3], so
    # recall is made constant within each bin (H at the resolution of the bins)
    f = {0: 0.95, 1: 0.9, 2: 0.7, 3: 0.7}
    st = rng.choice(4, 4000, p=[.1, .2, .3, .4])
    lt = rng.choice(4, 4000, p=[.4, .3, .2, .1])
    sr = {10: np.array([f[t] for t in st])}
    lr = {10: np.array([f[t] for t in lt])}
    bins = m.tie_bins(st, lt, sr, lr, [10], quantiles=(0, .3, .55, .8, 1))
    adj, cov, by_kp = m.reweight(bins, [10])
    assert abs(adj[10] - lr[10].mean()) < 1e-3, (adj, lr[10].mean())
    assert abs(cov - 1.0) < 1e-9
    assert by_kp == {10: 1.0}, by_kp
    # drop every small-scale query from the first bin: coverage < 1, and the
    # renormalised value stays a weighted mean of the covered bins
    keep = st != 0
    sr2 = {10: sr[10][keep]}
    bins2 = m.tie_bins(st[keep], lt, sr2, lr, [10], quantiles=(0, .3, .55, .8, 1))
    adj2, cov2, by_kp2 = m.reweight(bins2, [10])
    assert cov2 < 1.0
    covered = lt != 0
    assert abs(adj2[10] - lr[10][covered].mean()) < 1e-3, (adj2, lr[10][covered].mean())
    # the reported coverage is the WORST depth, not the last one evaluated. Two
    # depths are given different unusable bins by hand, because today they are
    # always the same bins -- which is exactly why `covered = w` inside the loop
    # went unnoticed, and why the test has to construct the case rather than
    # wait for it.
    # The degraded depth is the FIRST one asked for and the complete one is
    # last, so `min` and "last evaluated" give different answers -- a case built
    # the other way round passes under both and proves nothing. (Checked by
    # reintroducing last-wins: this is the ordering that catches it.)
    split = [dict(b) for b in bins]
    for b in split:
        b["small_recall@20"] = b["small_recall@10"]
    split[1]["small_recall@10"] = None
    _, cov3, by_kp3 = m.reweight(split, [10, 20])
    assert by_kp3[10] < by_kp3[20] == 1.0, by_kp3
    assert cov3 == by_kp3[10], (cov3, by_kp3)


def test_margin_spearman_handles_ties_and_constants():
    m = _load_analysis("margin_8m")
    assert m._spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert m._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert m._spearman([1, 1, 2, 2], [1, 1, 2, 2]) == 1.0
    assert m._spearman([1, 2, 3], [5, 5, 5]) is None


def test_tab_form_applies_the_caption_rule_to_both_kinds_of_file():
    """The paper's Table 3 caption states a rule: measured minus predicted
    savings, in points, over x = D*K'/N in [0.05, 3]. `analysis/tab_form.py` is
    what regenerates the table from result files, so the rule has to live in one
    place and be checked -- two of the table's rows could not be reproduced from
    any archived file before this tool existed."""
    t = _load_analysis("tab_form")

    # A migration file contributes its frontier rows; an estimate file every
    # point of every coverage curve. Neither is pooled across files.
    N, kp = 500_000, 500
    D_in = 1000                      # x = 1.0, inside the range
    D_out = 10                       # x = 0.01, outside it
    mig = dict(N=N, frontier=[
        dict(kprime=kp, D=D_in, N=N, savings_pct=43.0, recall_at_kprime=0.93),
        dict(kprime=kp, D=D_out, N=N, savings_pct=99.0, recall_at_kprime=0.51),
    ])
    rows = t.residuals(mig)
    assert len(rows) == 1, "a point outside x in [0.05, 3] must be dropped"
    k_, d_, x, meas, pred, resid = rows[0]
    assert (k_, d_) == (kp, D_in) and abs(x - 1.0) < 1e-9
    assert abs(pred - 100 * (1 - kp / N) ** D_in) < 1e-9
    assert abs(resid - (meas - pred)) < 1e-9
    # +6.23 is what the shipped 500k run gives at K'=500, D=1000 (paper: 43.0
    # measured against 36.77 predicted)
    assert abs(resid - 6.23) < 0.01, resid

    est = dict(N=N, estimates=[dict(kprime=kp, N=N,
                                    curve_d=[1, D_in, D_out],
                                    curve_reached=[0.001, 0.57, 0.9])])
    rows = t.residuals(est)
    assert len(rows) == 1, "estimate curves are filtered by the same rule"
    assert abs(rows[0][5] - (100 * (1 - 0.57) - 100 * (1 - kp / N) ** D_in)) < 1e-9

    # A file with nothing in range must say so rather than print an empty row:
    # that is the failure mode that made the table untraceable.
    assert t.residuals(dict(N=N, frontier=[
        dict(kprime=kp, D=D_out, N=N, savings_pct=99.0, recall_at_kprime=0.51)])) == []
    assert t.residuals(dict(N=N)) == []


def test_compare_8m_accepts_like_for_like_and_rejects_the_rest():
    c = _load_analysis("compare_8m")

    def run(N, rec, sav, prefix="P", q="abc"):
        return dict(N=N, D=10, k=10, old="o", new="n", prefixes={"new_query": prefix},
                    fingerprint={"query_ids_sha256": q},
                    frontier=[dict(kprime=kp, D=d, N=N, recall_at_kprime=r,
                                   savings_pct=sv)
                              for (kp, d, r, sv) in zip((100, 200), (500, 500), rec, sav)])
    # predicted savings at x = 0.1 and 0.2: 100*(1-K/N)^D
    good_s = run(1_000_000, (0.9, 0.95), (95.2, 90.6))
    good_l = run(1_000_000, (0.92, 0.96), (95.3, 90.7))
    assert c.acceptance(good_s, good_l) == []
    assert any("prefixes" in f for f in c.acceptance(good_s, run(1_000_000, (0.92, 0.96),
                                                               (95.3, 90.7), prefix="X")))
    assert any("different queries" in f for f in c.acceptance(
        good_s, run(1_000_000, (0.92, 0.96), (95.3, 90.7), q="zzz")))
    assert any("falls with K'" in f for f in c.acceptance(
        good_s, run(1_000_000, (0.96, 0.92), (95.3, 90.7))))
    assert any("below" in f for f in c.acceptance(
        good_s, run(1_000_000, (0.92, 0.96), (90.0, 90.7))))


def test_padding_tool_runs_offline_and_sorted_backfill_pads_least():
    import json, sys, tempfile
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(2)
    N = 1500
    V = rng.normal(size=(N, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.savez(d / "vectors.npz", V_old=V, Q_old=V[:40])
    with open(d / "texts.jsonl", "w", encoding="utf-8") as f:
        for _ in range(N):
            f.write(json.dumps({"text": " ".join(["w"] * int(rng.integers(5, 150)))}) + "\n")
    m = _load_analysis("padding")
    argv = sys.argv
    sys.argv = ["padding", "--data-dir", str(d), "--tokenizer", "words", "--kprime", "50",
                "--chunks", "16", "--backfill-batches", "64", "--sort-windows", "0,500",
                "--backfill-sample", "1000", "--out", str(d / "p.json")]
    try:
        m.main()
    finally:
        sys.argv = argv
    rows = json.loads((d / "p.json").read_text(encoding="utf-8"))["rows"]
    bf = {r["sort"]: r["padding_factor"] for r in rows if r["side"] == "backfill"}
    assert bf[500] < bf[0]
    assert all(r["padding_factor"] >= 1.0 for r in rows)


def test_tei_encoder_sends_the_prefix_and_normalises():
    """The step-2 headline path goes through TEI over HTTP; a stub server
    checks the request shape and the response handling."""
    import json, threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from thaw_pg.encoders import TEIEncoder
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = [[float(len(t)), 3.0, 4.0] for t in body["inputs"]]
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        enc = TEIEncoder(f"http://127.0.0.1:{srv.server_port}", prefix="passage: ", batch=2)
        V = enc.encode(["a", "bbb", "cc"])
    finally:
        srv.shutdown()
    assert enc.dim == 3 and V.shape == (3, 3)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0)
    # chunks are formed in length order; rows come back in input order
    assert seen[-2]["inputs"] == ["passage: a", "passage: cc"]
    assert seen[-1]["inputs"] == ["passage: bbb"]
    want = np.array([[len("passage: " + t), 3.0, 4.0] for t in ["a", "bbb", "cc"]])
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert np.allclose(V, want)
    assert seen[-1]["truncate"] is True


def test_plan_walker_finds_every_index_in_a_nested_plan():
    from thaw_pg.checks import _index_names
    plan = [{"Plan": {"Node Type": "Limit", "Plans": [
        {"Node Type": "Index Scan", "Index Name": "docs_emb_old_hnsw"},
        {"Node Type": "Nested Loop", "Plans": [
            {"Node Type": "Index Scan", "Index Name": "docs_new_pkey"},
            {"Node Type": "Seq Scan"}]}]}}]
    assert _index_names(plan) == ["docs_emb_old_hnsw", "docs_new_pkey"]
    assert _index_names([{"Plan": {"Node Type": "Seq Scan"}}]) == []


def test_every_thaw_pg_cli_flag_is_actually_read():
    """The same check as for the semview tools, for thaw_pg's CLI -- where
    backfill.run takes many positional arguments and a dropped flag would be
    silent."""
    import ast, inspect, textwrap
    import thaw_pg.__main__ as tm
    tree = ast.parse(textwrap.dedent(inspect.getsource(tm.main)))
    declared = [n.args[0].value[2:].replace("-", "_") for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)
                and str(n.args[0].value).startswith("--")]
    read = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "a"}
    # --force is read through getattr inside the guard
    src = inspect.getsource(tm.main)
    missing = [d for d in declared if d not in read and f'"{d}"' not in src]
    assert not missing, missing


def test_load_jsonl_is_called_with_its_arguments_in_order():
    """serve once passed (path, limit, 0, queries_file) to a function whose
    signature is (path, qpath, limit, seed): it crashed on every jsonl run."""
    import ast, inspect, textwrap
    params = list(inspect.signature(migration.load_jsonl).parameters)
    want = {"path": "path", "qpath": "queries_file", "limit": "limit"}
    for mod in (migration, estimate, serve):
        tree = ast.parse(textwrap.dedent(inspect.getsource(mod.main)))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_jsonl":
                for p, arg in zip(params, n.args):
                    if p in want:
                        assert isinstance(arg, ast.Attribute) and arg.attr == want[p], \
                            (mod.__name__, p, ast.unparse(arg))


def test_first_touch_accepts_both_curve_shapes():
    flat = {1: 9.77, 8: 3.16, 512: 42.96}
    rich = {str(b): {"ms_per_item": v, "total_ms": v * b} for b, v in flat.items()}
    for curve in (flat, rich):
        assert abs(cost.first_touch_ms(500, 3.0, chunk=8, curve=curve) - 1580.0) < 1e-6
        assert abs(cost.first_touch_ms(500, 3.0, curve=curve) - 1580.0) < 1e-6


def test_orcas_single_token_query_is_not_read_from_the_url_or_id():
    q, d = migration.orcas_columns("123\tparis\tD5\thttp://example.com/long/url/path")
    assert (q, d) == (1, 2)


def test_backfill_phases_run_equal_wall_time_when_the_repair_stream_is_short():
    import json, tempfile, time as _t
    from thaw_pg import backfill as bfm, encoders

    class Slow(_TokenCostEncoder):
        def encode_timed(self, texts):
            _t.sleep(0.002)
            return super().encode_timed(texts)

    rng = np.random.default_rng(4)
    N = 400
    V = rng.normal(size=(N, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    d = Path(tempfile.mkdtemp())
    np.savez(d / "vectors.npz", V_old=V, Q_old=V[:3])        # a tiny repair stream
    with open(d / "texts.jsonl", "w", encoding="utf-8") as f:
        for _ in range(N):
            f.write(json.dumps({"text": "w " * int(rng.integers(3, 50))}) + "\n")
    orig = encoders.make
    encoders.make = lambda *a, **k: Slow()
    try:
        # token_sort=False: this measures the phase machinery, not the key,
        # and "local:stub" has no tokenizer. Asking for a token sort and
        # getting characters is what run() now refuses to do silently.
        bfm.run(d, "local:stub", 20, None, 8, 8, 0, True, 0.4, 0.0, 0.0,
                None, 0, None, d / "bf.json", new_model_prefix="",
                token_sort=False)
    finally:
        encoders.make = orig
    r = json.loads((d / "bf.json").read_text(encoding="utf-8"))["result"]
    assert r["phases_equal_length"], r["phase_wall_s"]


def test_backfill_rate_needs_a_model_name_for_tei():
    import tempfile
    from thaw_pg import backfill as bfm
    d = Path(tempfile.mkdtemp())
    V = np.eye(4, dtype=np.float32)
    np.savez(d / "vectors.npz", V_old=V, Q_old=V[:1])
    (d / "texts.jsonl").write_text("\n".join('{"text": "x"}' for _ in range(4)) + "\n", encoding="utf-8")
    try:
        # token_sort=False so the refusal under test is the PREFIX one: a
        # tei: spec also has no tokenizer, and that refusal's message happens
        # to contain "--new-model", which would have satisfied a bare "--new"
        # check and quietly stopped testing the prefix at all.
        bfm.run(d, "tei:http://127.0.0.1:9", 2, None, 2, 2, 0, True, 0.1, 0.0, 0.0,
                None, 0, None, d / "bf.json", token_sort=False)
        raise AssertionError("must refuse to guess a TEI model's prefix")
    except SystemExit as e:
        assert "passage prefix" in str(e), e


def test_runlog_keeps_everything_and_marks_a_bad_run_as_failed():
    """Every result-producing run goes through runlog: its own directory, the
    exact command, the log, a manifest, and a line in RUNS.md -- and a run
    whose results file is missing or empty is marked FAIL, not ok."""
    import json, sys, tempfile
    from semview import runlog
    res = Path(tempfile.mkdtemp())
    good = ("import json,sys; json.dump({'fingerprint': {'a': 1}, 'x': 2}, "
            "open(sys.argv[1], 'w')); print('hello')")
    code = runlog.run([sys.executable, "-c", good, "{run}/out.json"], "unit", res, tag="t")
    assert code == 0
    (rd,) = list((res / "runs").iterdir())
    names = {f.name for f in rd.iterdir()}
    assert {"command.txt", "env.json", "log.txt", "MANIFEST.sha256", "out.json"} <= names
    assert "hello" in (rd / "log.txt").read_text(encoding="utf-8")
    assert str(rd) in (rd / "command.txt").read_text(encoding="utf-8")
    env = json.loads((rd / "env.json").read_text(encoding="utf-8"))["environment"]
    assert len(env["code_hash"]) == 16 and env["tag"] == "t"
    reg = (res / "RUNS.md").read_text(encoding="utf-8").splitlines()
    assert reg[-1].split("|")[4].strip() == "ok"
    # an empty results file, and a run that writes nothing: both FAIL
    bad = "import sys; open(sys.argv[1], 'w').close()"
    assert runlog.run([sys.executable, "-c", bad, "{run}/out.json"], "unit", res) != 0
    assert runlog.run([sys.executable, "-c", "pass"], "unit", res) != 0
    rows = (res / "RUNS.md").read_text(encoding="utf-8").splitlines()[-2:]
    assert all(r.split("|")[4].strip() == "FAIL" for r in rows)
    assert len(list((res / "runs").iterdir())) == 3          # nothing overwritten


def test_code_compiles_on_the_oldest_supported_python():
    """f"{d["k"]}" (same quotes nested) is legal only from Python 3.12 (PEP
    701); on 3.10/3.11 it is a SyntaxError that takes the whole thaw_pg package
    down. Compile everything with an older interpreter when one is installed."""
    import shutil, subprocess
    py = next((shutil.which(v) for v in ("python3.10", "python3.11")
               if shutil.which(v)), None)
    if py is None:
        raise Skip("no python3.10/3.11 on PATH")
    root = Path(migration.__file__).resolve().parent.parent
    files = [str(p) for d in ("semview", "thaw_pg", "analysis")
             for p in sorted((root / d).rglob("*.py")) if "__pycache__" not in p.parts]
    r = subprocess.run([py, "-m", "py_compile", *files], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]


def test_runlog_checks_results_in_subdirectories():
    import sys, tempfile
    from semview import runlog
    res = Path(tempfile.mkdtemp())
    bad = ("import os,sys; os.makedirs(os.path.dirname(sys.argv[1])); "
           "open(sys.argv[1], 'w').close()")
    assert runlog.run([sys.executable, "-c", bad, "{run}/sub/out.json"], "u", res) != 0
    (rd,) = list((res / "runs").iterdir())
    assert "sub/out.json" in (rd / "MANIFEST.sha256").read_text(encoding="utf-8")


# ------------------------------------------------ documented commands parse ---
_PARSE_ONLY = r"""
import argparse, runpy, sys
def _parse(self, args=None, namespace=None):
    ns, extra = self.parse_known_args(args, namespace)
    if extra:
        print("UNRECOGNISED", extra)
        raise SystemExit(3)
    raise SystemExit(0)          # parsed cleanly: stop before doing any work
argparse.ArgumentParser.parse_args = _parse
target, sys.argv = sys.argv[1], [sys.argv[1]] + sys.argv[2:]
if target.endswith(".py"):
    runpy.run_path(target, run_name="__main__")
else:
    runpy.run_module(target, run_name="__main__", alter_sys=True)
print("NO PARSE")                # main() never called parse_args
raise SystemExit(4)
"""


def _documented_commands():
    """Every `python ...` command in a ```bash block of every README, with
    line continuations joined and trailing comments dropped.

    Leading environment assignments are stripped, so `THAW_MODE=measure python
    ...` is checked like any other command. It was not, for a while: the filter
    was `line.startswith("python ")`, and the eight commands it skipped were
    precisely the measurement runs -- the ones that go through `runlog` and
    produce the paper's evidence. The rule this test enforces most loudly, that
    a results-producing command must be wrapped in runlog, was therefore only
    ever applied to commands that do not use runlog, and a renamed flag in any
    of RUNBOOK's main measurement commands could not fail here.
    """
    import re
    root = Path(migration.__file__).resolve().parent.parent
    out = []
    env = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*=\S*\s+")
    for md in sorted(list(root.glob("*.md")) + list((root / "thaw_pg").glob("*.md"))):
        for block in re.findall(r"```bash\n(.*?)```", md.read_text(encoding="utf-8"), re.S):
            for line in block.replace("\\\n", " ").splitlines():
                line = line.strip()
                while env.match(line):
                    line = env.sub("", line, count=1)
                if line.startswith("python "):
                    out.append((md.relative_to(root).as_posix(), line))
    return root, out


_RESULT_TOOLS = {"semview.migration", "semview.estimate", "semview.union",
                 "semview.serve", "analysis/padding.py", "analysis/margin_8m.py",
                 "analysis/measure_throughput.py", "analysis/compare_8m.py",
                 "analysis/n_ladder.py", "analysis/gen2_union.py",
                 "analysis/chain.py", "analysis/drift_sweep.py"}
_RESULT_THAW = {"step2", "check", "backfill-rate"}


def _thaw_subcommands() -> set:
    """thaw_pg's subcommand names, read from its parser definitions (AST), so
    a new subcommand is known here without editing this list."""
    import ast, inspect, textwrap
    import thaw_pg.__main__ as tm
    tree = ast.parse(textwrap.dedent(inspect.getsource(tm.main)))
    return {n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_parser" and n.args
            and isinstance(n.args[0], ast.Constant)}


def _produces_results(tokens) -> bool:
    """Commands whose output can reach the paper (docs/history/README_RESULTS.md's rule).
    `tokens` starts with 'python'. For thaw_pg the subcommand is the first
    token that IS a subcommand name -- not the first non-flag token, which may
    be the value of --data-dir or --dsn."""
    t = tokens[1:]
    target = t[1] if t[:1] == ["-m"] else t[0]
    if target in _RESULT_TOOLS:
        return True
    if target == "thaw_pg":
        # thaw_pg's top-level options come before the subcommand; skip them and
        # their values, so a --data-dir named like a subcommand can't be taken
        # for one. The first remaining token is the subcommand.
        rest, sub = t[2:], None
        i = 0
        while i < len(rest):
            x = rest[i]
            if x in ("--dsn", "--data-dir"):
                i += 2
                continue
            if x.startswith("--"):           # --dsn=... or a flag
                i += 1
                continue
            sub = x
            break
        assert sub is None or sub in _thaw_subcommands(), f"unknown subcommand {sub!r}"
        return sub in _RESULT_THAW
    return False


def test_thaw_subcommand_detection_ignores_option_values():
    assert _RESULT_THAW <= _thaw_subcommands()
    for cmd, want in (
            ("python -m thaw_pg --data-dir mydata check", True),
            ("python -m thaw_pg --dsn postgresql://x/thaw --data-dir d step2 --enc-new y", True),
            ("python -m thaw_pg --data-dir check_dir quick", False),
            ("python -m thaw_pg --data-dir prepare step2 --enc-new y", True),
            ("python -m thaw_pg --dsn=postgresql://x/thaw check", True),
            ("python -m thaw_pg prepare --source synthetic", False),
            ("python -m thaw_pg backfill-rate --enc local:m", True)):
        assert _produces_results(cmd.split()) is want, cmd


def _check_command(root, tokens):
    """Problems with one tokenised command (after 'python'); [] if it parses."""
    import subprocess, sys
    if tokens[:2] == ["-m", "semview.tests"]:
        return []
    if tokens[:2] == ["-m", "semview.runlog"]:
        from semview import runlog
        if "--" not in tokens:
            return ["runlog without -- before the command"]
        i = tokens.index("--")
        try:
            a = runlog.parser().parse_args(tokens[2:i])
        except SystemExit:
            return [f"runlog options do not parse: {tokens[2:i]}"]
        inner = tokens[i + 1:]
        if not inner or inner[0] != "python":
            return ["runlog wraps something that is not a python command"]
        probs = []
        if not a.allow_no_json and not any("{run}" in t for t in inner):
            probs.append("wrapped command writes nowhere under {run}")
        return probs + _check_command(root, inner[1:])
    if tokens[0] == "-m":
        target, args = tokens[1], tokens[2:]
    else:
        target, args = str(root / tokens[0]), tokens[1:]
    r = subprocess.run([sys.executable, "-c", _PARSE_ONLY, target, *args],
                       capture_output=True, text=True, cwd=root,
                       env={**__import__("os").environ, "PYTHONPATH": str(root)})
    if r.returncode != 0:
        return [f"exit {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}"]
    return []


_PLACEHOLDER = __import__("re").compile(
    r"<(?! )(?=[^<>]*[A-Za-z])[A-Za-z0-9_./\- ]+(?<! )>")
"""A documentation placeholder, which is what `<cache>`, `<hash>` and
`<data directory>` are. No shell redirection has this shape: `2>&1`, `&>`,
`>&2` and `<<'SQL'` all fail to match, and so does a comparison like `1 < 2`,
because the rule needs a closing `>` with only placeholder characters between,
at least one letter among them, and no space against either bracket. That last
clause came from this rule's own self-check: the first attempt allowed a space
anywhere inside, so `print(1 < 2 and 3 > 2)` matched as the placeholder
`< 2 and 3 >`. The checker caught the checker, which is the only reason that
one did not ship.

The first version of this rule flagged tokens not in `{"<", ">", ">>", "2>"}`,
which makes `... > log 2>&1` -- an ordinary redirection -- a failure. A review
caught it, and the point stands beyond the one case: five rounds of findings in
this project were checks that did not check, and the fix for that is not a
longer exemption list, it is a rule that describes what it is looking FOR
rather than enumerating what to let through. `_check_the_placeholder_rule`
below asserts both directions.
"""


def _shell_command_lines(root):
    """(where, line) for every shell command this project documents.

    ```bash blocks in markdown, and the usage examples in python docstrings --
    the second because `semview/procrustes.py`, `semview/runlog.py`,
    `analysis/make_figures.py`, `analysis/collect_evidence.py` and
    `analysis/scaling_mechanism.py` all carried `<dir>`-style placeholders in
    their docstrings while the markdown was clean. A reader copies the command
    out of `--help` as readily as out of a README.

    Lines beginning `$ ` are transcripts -- a record of what a shell printed,
    including deliberate failures -- and are not commands to copy.
    """
    import ast as _ast
    import re as _re

    for md in sorted(list(root.glob("*.md")) + list((root / "thaw_pg").glob("*.md"))):
        for block in _re.findall(r"```bash\n(.*?)```", md.read_text(encoding="utf-8"), _re.S):
            for line in block.splitlines():
                s = line.strip()
                if s and not s.startswith("#") and not s.startswith("$ "):
                    yield md.name, line
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        tree = _ast.parse(py.read_text(encoding="utf-8"), str(py))
        for node in _ast.walk(tree):
            if not isinstance(node, (_ast.Module, _ast.FunctionDef,
                                     _ast.AsyncFunctionDef, _ast.ClassDef)):
                continue
            doc = _ast.get_docstring(node)
            if not doc:
                continue
            # continuations joined first: scaling_mechanism.py's placeholders
            # sat on a `\`-continuation line whose own first word is `--v-new`,
            # so a per-line filter walked straight past them
            for line in doc.replace("\\\n", " ").splitlines():
                s = line.strip()
                # a command, not prose about one: prose mentions it in
                # backticks mid-sentence, a usage example starts with it
                if s.startswith(("python ", "python3 ", "docker ", "psql ",
                                 "pg_ctl ", "export ")):
                    yield py.name, line


def _check_the_placeholder_rule():
    """The rule is checked before it is trusted, in both directions."""
    must_flag = ["--v-new <cache>/<hash>.f16.npy", "--old cache/<bge>.npy",
                 "docker restart <container>", "pg_ctl -D <data directory> -w",
                 "export TEI=ghcr.io/x/y:<tag-for-your-gpu>",
                 "python -m semview.procrustes --from-run results/runs/<dir>"]
    must_pass = ["python x.py --out out.json > log.txt 2>&1",
                 "python x.py 2>&1 | tee log", "python x.py &> log",
                 "python x.py >> log 1>&2", "psql \"$DSN\" <<'SQL'",
                 "python -c 'print(1 < 2 and 3 > 2)'",
                 "python x.py --v-new cache/NEWMODELHASH.f16.npy",
                 "git log --format='%an <%ae>' | head"]
    for s in must_flag:
        assert _PLACEHOLDER.search(s), f"rule misses a placeholder: {s}"
    for s in must_pass:
        hit = _PLACEHOLDER.search(s)
        assert not hit, f"rule false-positives on {s!r}: matched {hit.group(0)!r}"


def test_no_call_site_unpacks_the_wrong_number_of_return_values():
    """A function's return arity and its callers must agree.

    `margin_8m.reweight` went from `(out, covered)` to `(out, covered, per_kp)`
    this round. Its unit test was updated and passed, and `main` was updated
    too -- but nothing in the suite would have noticed if it had not been:
    `main` needs the BEIR corpus and both encoders, so it cannot run here, and
    a review correctly pointed out that the unit test alone leaves
    `ValueError: too many values to unpack` to be discovered on the machine
    holding the 8.8M matrices, after the expensive part.

    An end-to-end run is not available, but the defect is static. Any function
    whose every `return` is a tuple of the same length has a known arity, and
    any caller in the same module that unpacks it into a different number of
    names is wrong before it runs. Starred targets are skipped, since those
    absorb any length.
    """
    import ast as _ast

    root = Path(migration.__file__).resolve().parent.parent

    def fixed_arity(tree):
        out = {}
        for node in tree.body:
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            sizes, only_tuples = set(), True
            for sub in _ast.walk(node):
                if (isinstance(sub, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                        and sub is not node):
                    continue
                if isinstance(sub, _ast.Return):
                    if isinstance(sub.value, _ast.Tuple):
                        sizes.add(len(sub.value.elts))
                    else:
                        only_tuples = False
            if only_tuples and len(sizes) == 1 and next(iter(sizes)) > 1:
                out[node.name] = next(iter(sizes))
        return out

    bad, scanned = [], 0
    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        tree = _ast.parse(py.read_text(encoding="utf-8"), str(py))
        arity = fixed_arity(tree)
        scanned += len(arity)
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Assign) or len(node.targets) != 1:
                continue
            tgt, val = node.targets[0], node.value
            if not isinstance(tgt, (_ast.Tuple, _ast.List)):
                continue
            if not isinstance(val, _ast.Call):
                continue
            name = (getattr(val.func, "id", None)
                    or getattr(val.func, "attr", None))
            if name not in arity:
                continue
            if any(isinstance(e, _ast.Starred) for e in tgt.elts):
                continue
            if len(tgt.elts) != arity[name]:
                bad.append(
                    f"{py.relative_to(root).as_posix()}:{node.lineno} unpacks "
                    f"{len(tgt.elts)} values from {name}(), which returns "
                    f"{arity[name]}")
    assert scanned > 10, f"only {scanned} fixed-arity functions found; the scan broke"
    assert not bad, "\n".join(bad)


def test_a_documented_claim_registration_points_at_a_field_that_exists():
    """`--field` is a string to argparse, so a wrong path parses and fails at
    run time.

    RUNBOOK's `add` example was written as `--field rows.3.recall@500` and
    described in the same breath as "a real `add`, not a sketch". It is not a
    real one: `n_ladder` writes `frontier[...]` rows carrying
    `recall_at_kprime`, there is no `rows` key, and `thaw_audit.paths.resolve`
    raises `FieldMissing`. The command-parsing test could not see it -- argparse
    was satisfied -- which is the same gap as a flag that parses against a file
    that does not exist.

    So the example is now checked the way a claim is: the path must resolve
    against the source it names, and reach the value the example claims. That
    makes the documented command an executable example rather than a
    plausible-looking one.
    """
    import json as _json
    import re as _re
    import shlex

    root = Path(migration.__file__).resolve().parent.parent
    from thaw_audit.paths import FieldMissing, resolve

    checked = 0
    for md in sorted(root.glob("*.md")):
        for block in _re.findall(r"```bash\n(.*?)```",
                                 md.read_text(encoding="utf-8"), _re.S):
            for line in block.replace("\\\n", " ").splitlines():
                if "thaw_audit add" not in line:
                    continue
                tok = shlex.split(line.split("python", 1)[1], comments=True)
                opt = {t2: tok[i + 1] for i, t2 in enumerate(tok)
                       if t2.startswith("--") and i + 1 < len(tok)}
                src, field = opt.get("--source"), opt.get("--field")
                if not src or not field:
                    continue
                path = root / src
                assert path.is_file(), (
                    f"{md.name}: the documented `add` names --source {src}, "
                    f"which is not in this tree; name a file that is")
                try:
                    got = resolve(_json.loads(path.read_text(encoding="utf-8")), field)
                except FieldMissing as e:
                    raise AssertionError(
                        f"{md.name}: --field {field!r} does not resolve in "
                        f"{src}: {e}") from None
                if "--value" in opt:
                    assert abs(float(got) - float(opt["--value"])) < 1e-9, (
                        f"{md.name}: --field {field} is {got} in {src}, but the "
                        f"example says --value {opt['--value']}")
                checked += 1
    assert checked, (
        "no documented `thaw_audit add` example was found to check; if the "
        "example moved, point this test at it rather than deleting it")


def test_no_document_or_help_string_names_a_directory_nothing_creates():
    """`arrays/` was in four places and is created by nothing.

    `semview.migration --dump-queries DIR` saves `q_old.npy` and `q_new.npy`
    straight into DIR. No tool makes an `arrays/` subdirectory, and the real
    chain legs used `dump/leg0` ... `dump/leg3`. But a scaling-mechanism
    docstring, two runbook entries and `procrustes`'s `--from-run` help all
    told a reader to look under `arrays/`, so copying any of them gives a
    file-not-found whose message says nothing about this project.

    It survived because the one tool that reads those files does so with
    `rglob`, which finds them at any depth -- the code was tolerant and only
    the documentation was wrong, which is the combination that keeps a wrong
    path alive. Checked as a literal across code and documents, since it is
    cheap and the component is distinctive.
    """
    import re as _re

    root = Path(__file__).resolve().parent.parent
    writes_it = []
    mentions = []
    for p in sorted(list(root.rglob("*.py")) + list(root.rglob("*.md"))):
        if "__pycache__" in str(p) or "docs/history" in p.as_posix():
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            # a USAGE names a file under the directory, as in
            # `arrays` + `/q_new.npy`; prose about
            # the mistake, including this test's own docstring, does not. The
            # first version of this rule flagged its own explanation, which is
            # the same self-reference the non-ASCII identifier test hit.
            if not _re.search(r"arrays/[\w.-]+\.\w+", line):
                continue
            where = f"{p.relative_to(root).as_posix()}:{n}"
            # a line that CREATES the directory would make it legitimate
            if any(k in line for k in ("mkdir", "makedirs")):
                writes_it.append(where)
            else:
                mentions.append(f"{where}  {line.strip()[:70]}")
    if writes_it:
        raise Skip(f"something now creates arrays/ ({writes_it}); this test is "
                   "about a path nothing creates, so re-point it")
    assert not mentions, (
        "these name an `arrays/` path that no tool creates; the dump layout is "
        f"<--dump-queries DIR>/q_new.npy: {mentions}")


def test_no_documented_command_uses_an_angle_bracket_placeholder():
    """`--v-new <cache>/<hash>.npy` is not a placeholder in a shell. It is a
    redirection, and the command fails before it starts:

        $ echo --v-new <cache>/<hash>.f16.npy
        bash: cache: No such file or directory

    Verified against bash, including the `Is a directory` variant you get when
    a directory of that name happens to exist -- which it does here, so the
    error names `/` and looks like nothing to do with this project. The flag
    parser could not catch it: `shlex.split` does no shell expansion, so
    `<cache>` tokenises cleanly, and the eight env-prefixed commands were not
    scanned at all (see `_documented_commands`).

    The rule covers markdown and python docstrings alike, and `docker restart
    <container>` fails the same way as a python command. Substitute an
    uppercase word, which no shell touches.
    """
    _check_the_placeholder_rule()
    root = Path(migration.__file__).resolve().parent.parent
    bad = []
    for where, line in _shell_command_lines(root):
        for hit in _PLACEHOLDER.findall(line):
            bad.append(f"{where}: {hit}  in: {line.strip()[:70]}")
    assert not bad, (
        "these documented commands contain angle brackets a shell will read as "
        f"redirection: {bad}. Use an uppercase placeholder word.")


def test_every_documented_command_parses_and_results_go_through_runlog():
    """Each README command is parsed by the real argument parser of the tool it
    calls (nothing is run): a renamed or removed flag in any README fails here.
    And every command that writes a results JSON goes through runlog, as
    docs/history/README_RESULTS.md requires."""
    import shlex
    root, cmds = _documented_commands()
    assert len(cmds) > 20, len(cmds)
    bad = []
    for md, line in cmds:
        tokens = shlex.split(line, comments=True)
        if tokens[:3] != ["python", "-m", "semview.runlog"] and _produces_results(tokens):
            bad.append(f"{md}: a results-producing run not wrapped in runlog: {line[:90]}")
        for p in _check_command(root, tokens[1:]):
            bad.append(f"{md}: {p}: {line[:90]}")
    assert not bad, "\n".join(bad)



# --------------------------------------------------------------------- online ---
def _online_toy(n=1200, d=24, kprime=60, budget_m=8, seed=0, **kw):
    """An online server over two spaces related by a rotation, instant encoders.

    The same shape as `_toy_server`, so a difference between the synchronous and
    the online path is a difference in policy rather than in the fixture.
    """
    rng = np.random.default_rng(seed)
    V_old = migration._safe_normalize(rng.normal(size=(n, d)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_true = migration._safe_normalize((V_old @ R).astype(np.float32))
    texts = [f"unit {i}" for i in range(n)]
    qs = [f"q{i}" for i in range(80)]
    qvec = {q: V_old[rng.integers(0, n)] for q in qs}

    def search(q, kp):
        s = V_old @ q
        idx = np.argsort(-s)[:kp]
        return [int(i) for i in idx], [float(x) for x in s[idx]]

    srv = online.OnlineServer(
        search=search,
        encode_new=lambda ts: V_true[[int(t.split()[1]) for t in ts]],
        encode_old_query=lambda t: qvec[t],
        encode_new_query=lambda t: migration._safe_normalize((qvec[t] @ R)[None, :])[0],
        texts=texts, new_dim=d, kprime=kprime, k=10, budget_m=budget_m, **kw)
    return srv, qs, V_true, qvec, R


def test_online_never_encodes_a_unit_twice_even_with_the_repairer_running():
    """The cost guarantee the online path must keep.

    Dropping the wait does not license dropping the claim: a unit encoded by an
    inline repair and again by the background repairer would be charged twice,
    which is the one thing that would invalidate the ledger.
    """
    srv, qs, _, _, _ = _online_toy(budget_m=8)
    calls = []
    inner = srv.encode_new
    srv.encode_new = lambda ts: (calls.extend(int(t.split()[1]) for t in ts),
                                 inner(ts))[1]
    online.replay(srv, qs, background=True)
    srv.drain_now()
    assert len(calls) == len(set(calls)), (
        f"{len(calls) - len(set(calls))} units were encoded more than once")
    assert len(set(calls)) == int(srv.repaired.sum()) == srv.encoded_units


def test_online_ranks_only_units_that_actually_have_a_vector():
    """An unrepaired unit must never be ranked, because its vector is zeros.

    This is the failure the publish order in `_encode_and_store` exists to
    prevent, and it would read as a quality regression rather than a
    visibility bug.
    """
    srv, qs, _, _, _ = _online_toy(budget_m=4)
    for q in qs:
        res, tr = srv.query(q)
        assert all(srv.repaired[u] for u in res), "ranked an unrepaired unit"
        assert tr.n_visible <= tr.n_window
        assert len(res) <= srv.k


def test_online_inline_budget_is_never_exceeded():
    """The budget is the latency bound; a query that exceeds it has none."""
    for m in (0, 1, 8, 32):
        srv, qs, _, _, _ = _online_toy(budget_m=m)
        traces = online.replay(srv, qs, background=False)
        assert max(t.n_inline for t in traces) <= m, m
        if m == 0:
            assert all(t.n_inline == 0 for t in traces)


def test_online_window_coverage_rises_with_the_budget_and_with_traffic():
    """More inline budget buys more of the window; so does more traffic.

    These are the two axes the policy trades, and if neither held the budget
    would be a knob with no effect.
    """
    cov = {}
    for m in (0, 4, 16):
        srv, qs, _, _, _ = _online_toy(budget_m=m)
        traces = online.replay(srv, qs, background=False)
        cov[m] = float(np.mean([t.window_coverage for t in traces]))
    assert cov[0] < cov[4] < cov[16], cov

    srv, qs, _, _, _ = _online_toy(budget_m=4)
    traces = online.replay(srv, qs, background=False)
    n = len(traces)
    first = np.mean([t.window_coverage for t in traces[:n // 4]])
    last = np.mean([t.window_coverage for t in traces[-n // 4:]])
    assert last > first, (first, last)


def test_online_sheds_the_inline_budget_when_the_queue_is_deep():
    """Admission control is what bounds the tail, so it has to actually fire."""
    srv, qs, _, _, _ = _online_toy(budget_m=16, shed_at=0)
    traces = online.replay(srv, qs, background=False)
    assert all(t.shed for t in traces[1:]), "shedding never engaged"
    assert all(t.n_inline == 0 for t in traces[1:])
    srv2, qs2, _, _, _ = _online_toy(budget_m=16, shed_at=10 ** 9)
    traces2 = online.replay(srv2, qs2, background=False)
    assert not any(t.shed for t in traces2)
    assert sum(t.n_inline for t in traces2) > 0


def test_online_at_a_full_window_budget_reproduces_the_synchronous_answer():
    """The policy's endpoint must be the design it generalises.

    With a budget as deep as the window and no contention, every member is
    repaired before ranking, so the answer is the exact whole-window rerank
    `serve.ThawServer` produces. If this drifts, the online path is a different
    system rather than a scheduling choice.
    """
    srv_a, qs, V_true, qvec, R = _online_toy(kprime=40, budget_m=10 ** 6)
    online_res = [srv_a.query(q)[0] for q in qs]

    rng = np.random.default_rng(0)
    srv_b, qs_b, _, _, _ = _online_toy(kprime=40, budget_m=10 ** 6)
    sync = serve.ThawServer(
        V_old=srv_b.V_new * 0 + _old_of(srv_b), texts=srv_b.texts,
        encode_new=srv_b.encode_new, encode_old_query=srv_b.encode_old_query,
        encode_new_query=srv_b.encode_new_query,
        kprime=40, k=10, chunk=8, new_dim=srv_b.new_dim)
    sync_res = [sync.query(q)[0] for q in qs]
    assert online_res == sync_res, "the full-budget policy is not the sync design"


def _old_of(srv):
    """Recover the old matrix the fixture's search closure captured."""
    probe = np.eye(srv.new_dim, dtype=np.float32)
    # the fixture searches V_old @ q; recover it by searching with basis vectors
    cols = []
    for e in probe:
        ids, sc = srv.search(e, len(srv.texts))
        col = np.zeros(len(srv.texts), dtype=np.float32)
        col[np.asarray(ids)] = np.asarray(sc, dtype=np.float32)
        cols.append(col)
    return np.stack(cols, axis=1)


def test_repair_queue_orders_by_misses_and_drops_rather_than_blocks():
    """The queue's two deliberate behaviours, both of which are policy.

    Ordering by miss count is what makes the repairer converge on the hot set
    faster than arrival order. Dropping at capacity is what keeps a traffic
    spike from becoming unbounded memory, and it is safe only because a
    dropped unit returns on its next miss.
    """
    q = online.RepairQueue(capacity=3)
    q.offer([1, 2, 3])
    q.offer([2, 3])
    q.offer([3])
    assert q.offer([99]) == 0 and q.dropped == 1, "capacity was not enforced"
    assert q.take(2) == [3, 2], "the queue did not drain by miss count"
    assert q.depth == 1
    assert q.offer([99]) == 1, "a drop did not free room for the next miss"
    q.forget([99, 1])
    assert q.depth == 0


def test_claim_set_hands_a_unit_to_exactly_one_caller_and_never_waits():
    """The claim is what keeps the ledger honest once the wait is gone."""
    c = online.ClaimSet()
    a = c.claim([1, 2, 3])
    b = c.claim([2, 3, 4])
    assert a == [1, 2, 3] and b == [4], (a, b)
    c.release(a)
    assert c.claim([1]) == [1], "a released unit could not be reclaimed"


def test_online_summary_reports_coverage_beside_latency():
    """A latency summary without coverage would make the policy look free."""
    srv, qs, _, _, _ = _online_toy(budget_m=8)
    s = online.summarise(online.replay(srv, qs, background=False))
    for key in ("total_p50", "total_p99", "window_coverage_mean",
                "inline_per_query_mean", "queued_total", "shed_queries"):
        assert key in s, key
    assert 0.0 <= s["window_coverage_mean"] <= 1.0
    try:
        online.summarise([])
    except SystemExit:
        pass
    else:
        raise AssertionError("an empty replay should be refused, not summarised")


def test_async_and_budget_rows_cost_exactly_what_lazy_costs():
    """The claim that the savings analysis covers both variants, as a test.

    Deferring a repair changes when a vector exists, not whether it is
    computed: the background repairer encodes whatever the inline budget did
    not, so the set re-embedded is the same union of windows. If this fails,
    the latency rows are not equal-cost rows and cannot sit in the same table.
    """
    rng = np.random.default_rng(3)
    n, d = 700, 16
    V_old = migration._safe_normalize(rng.normal(size=(n, d)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_new = migration._safe_normalize((V_old @ R).astype(np.float32))
    Q_old = V_old[rng.integers(0, n, 40)]
    Q_new = migration._safe_normalize(Q_old @ R)
    rows = baselines.compare_schemes(
        V_old, V_new, Q_old[20:], Q_new[20:], Q_old[:20], Q_new_warm=Q_new[:20],
        k=10, kprime=50, budgets=(4, 16))
    by = {r["scheme"]: r for r in rows}
    lazy = by["lazy"]["passes"]
    for name in ("lazy-async", "lazy-budget-4", "lazy-budget-16"):
        assert name in by, name
        assert by[name]["passes"] == lazy, (
            f"{name} spends {by[name]['passes']} against lazy's {lazy}")
        assert by[name]["passes_inline"] + by[name]["passes_queued"] + \
            int(by["backfill-traffic"]["passes"] >= 0) >= 0   # fields present


def test_a_bigger_inline_budget_never_lowers_quality_and_lazy_bounds_it():
    """The Pareto curve has to be a curve: quality monotone in the budget, and
    the synchronous design its ceiling.

    The whole value of the budget row is that an operator can read a quality
    cost off a latency choice. If quality were not monotone in the budget there
    would be nothing to read.
    """
    rng = np.random.default_rng(5)
    n, d = 700, 16
    V_old = migration._safe_normalize(rng.normal(size=(n, d)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_new = migration._safe_normalize((V_old @ R).astype(np.float32))
    Q_old = V_old[rng.integers(0, n, 40)]
    Q_new = migration._safe_normalize(Q_old @ R)
    rows = baselines.compare_schemes(
        V_old, V_new, Q_old[20:], Q_new[20:], Q_old[:20], Q_new_warm=Q_new[:20],
        k=10, kprime=50, budgets=(2, 8, 32))
    by = {r["scheme"]: r["recall_at_k"] for r in rows}
    curve = [by["lazy-async"], by["lazy-budget-2"], by["lazy-budget-8"],
             by["lazy-budget-32"], by["lazy"]]
    assert curve == sorted(curve), curve
    assert by["lazy-async-cold"] <= by["lazy-async"] + 1e-12, (
        "a repairer that never runs cannot beat one that keeps up")
    assert by["lazy"] >= by["lazy-budget-32"], "lazy is not the ceiling"
    inl = {r["scheme"]: r.get("inline_per_query_mean")
           for r in rows if "inline_per_query_mean" in r}
    assert inl["lazy-async"] == 0.0
    assert inl["lazy-budget-2"] <= 2 and inl["lazy-budget-8"] <= 8


def test_latency_helper_turns_inline_counts_into_a_budgeted_latency():
    """Counts are what the simulator measures; milliseconds belong to a device.

    The helper must take the device figure rather than carry one, or the table
    silently inherits whichever GPU the throughput curve last ran on.
    """
    row = dict(scheme="lazy-budget-16", inline_per_query_mean=12.0,
               inline_per_query_p95=16)
    out = baselines.latency_ms(row, unit_ms=3.78, window_ms=8.3, rank_ms=4.3)
    assert abs(out["mean_ms"] - (8.3 + 12.0 * 3.78 + 4.3)) < 0.05
    assert out["p95_ms"] > out["mean_ms"]
    slow = baselines.latency_ms(row, unit_ms=42.96)
    assert slow["mean_ms"] > out["mean_ms"], "unit_ms must drive the result"



# ----------------------------------------------------------------- adapters ---
def _rotation_fixture(n=4000, d=32, nq=120, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    V_old = migration._safe_normalize(rng.normal(size=(n, d)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_new = migration._safe_normalize(
        (V_old @ R + noise * rng.normal(size=(n, d))).astype(np.float32))
    Q_old = migration._safe_normalize(
        V_old[rng.integers(0, n, nq)]
        + 0.05 * rng.normal(size=(nq, d)).astype(np.float32))
    Q_new = migration._safe_normalize(Q_old @ R)
    return Q_old, Q_new, V_old, V_new


def test_procrustes_recovers_a_pure_rotation_exactly():
    """The implementation check. If the two spaces differ by a rotation, an
    orthogonal map is the exact answer, so the query-side adapter must lose
    nothing. A fit that cannot do this is wrong, and a negative result from a
    wrong fit says nothing about adapters."""
    Q_old, Q_new, V_old, V_new = _rotation_fixture(noise=0.0)
    res = procrustes.evaluate(Q_old, Q_new, V_old, V_new, k=10, ks=(200,))
    by = {r["scheme"]: r["recall_at_k"] for r in res["schemes"]}
    proc = by["procrustes (orthogonal)"]
    assert proc > 0.99, f"Procrustes lost a pure rotation: {proc}"


def test_procrustes_is_scored_on_queries_it_was_not_fitted_on():
    """An adapter scored on its own fitting sample measures memorisation."""
    Q_old, Q_new, V_old, V_new = _rotation_fixture()
    res = procrustes.evaluate(Q_old, Q_new, V_old, V_new, train_frac=0.5)
    assert res["queries_fit"] + res["queries_eval"] == len(Q_old)
    assert res["queries_fit"] > 1 and res["queries_eval"] > 1
    try:
        procrustes.evaluate(Q_old[:4], Q_new[:4], V_old, V_new)
    except SystemExit:
        pass
    else:
        raise AssertionError("too few queries to split should be refused")


def test_the_adapter_rows_carry_no_depth_because_they_have_no_window():
    """An adapter returns its top-k from the old index. Reporting it per K'
    would imply a knob it does not have, and invite a reader to compare it
    against the design at 'the same depth', which is not a thing."""
    Q_old, Q_new, V_old, V_new = _rotation_fixture(noise=0.3)
    res = procrustes.evaluate(Q_old, Q_new, V_old, V_new, ks=(10, 200))
    for r in res["schemes"]:
        if r["scheme"].startswith("thaw"):
            assert r["kprime"] is not None
        else:
            assert r["kprime"] is None, r
    thaw = [r for r in res["schemes"] if r["scheme"].startswith("thaw")]
    assert len(thaw) == 2 and thaw[1]["kprime"] > thaw[0]["kprime"]


def test_a_deeper_window_helps_the_design_where_an_adapter_cannot_be_helped():
    """The shape of the argument the paper makes, as a property.

    When the two spaces genuinely disagree, a query-side map has one shot and
    no depth to spend; the design can buy accuracy with the window. If this
    ever failed, the paper's answer to Drift-Adapter would not hold even in
    principle.
    """
    Q_old, Q_new, V_old, V_new = _rotation_fixture(noise=0.35, n=6000)
    res = procrustes.evaluate(Q_old, Q_new, V_old, V_new, ks=(10, 500))
    thaw = sorted([r for r in res["schemes"] if r["scheme"].startswith("thaw")],
                  key=lambda r: r["kprime"])
    assert thaw[-1]["recall_at_k"] > thaw[0]["recall_at_k"], thaw
    adapters = [r["recall_at_k"] for r in res["schemes"]
                if r["kprime"] is None and "no adapter" not in r["scheme"]]
    assert max(adapters) < thaw[-1]["recall_at_k"], (adapters, thaw[-1])


def test_a_unit_taken_from_the_queue_but_not_claimed_goes_back():
    """Otherwise it is in no queue and claimable by nobody.

    `take()` removes units from the queue and `claim()` may then win only some
    of them. The rest are normally published by whoever holds them -- but a
    holder that dies without releasing leaves a unit that no queue offers and
    no claim can be taken on, so it could never be repaired. Returning the
    unclaimed ones costs one bounce of churn and removes that hole.
    """
    srv, qs, _, _, _ = _online_toy(budget_m=0)
    srv.query(qs[0])                               # fills the queue
    while srv.queue.take(256):                     # empty it, so the scenario
        pass                                       # is only the held units
    assert srv.queue.depth == 0
    held = [int(u) for u in range(5)]
    srv.queue.offer(held)
    other = online.ClaimSet()
    srv.claims = other
    assert other.claim(held) == held                # someone else owns them all
    before = int(srv.repaired.sum())
    assert srv.drain_now(max_batches=3) == 0, "it encoded units it did not own"
    assert int(srv.repaired.sum()) == before
    assert srv.queue.depth == len(held), (
        f"unclaimed units were dropped: depth {srv.queue.depth} of {len(held)}")


def test_the_event_trace_records_the_interleaving_a_log_cannot_show():
    """A JSONL trace of the concurrent path, with the race visible in it.

    The double-encode this module once had -- a claim won after its previous
    owner published and released, on a stale absence -- reads as `claim`
    followed by `skip_already_present` for the same unit. It was found by a
    test; this is the artifact that would have shown it without one.
    """
    import json as _json
    import tempfile
    path = Path(tempfile.mkdtemp()) / "events.jsonl"
    srv, qs, _, _, _ = _online_toy(budget_m=8, trace=online.Tracer(path))
    online.replay(srv, qs, background=True)
    srv.drain_now()
    srv.trace.close()
    lines = [_json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert lines, "no events were written"
    kinds = {l["event"] for l in lines}
    for want in ("window", "claim", "encode_start", "encode_done", "publish",
                 "claim_released"):
        assert want in kinds, f"{want} never traced; got {sorted(kinds)}"
    # every line is addressable: offset, thread, and the units involved
    for l in lines:
        assert "t" in l and "thread" in l and isinstance(l["t"], float)
    enc = [l for l in lines if l["event"] == "encode_done"]
    assert enc and all("ms" in l for l in enc)
    # and the invariant the trace exists to witness: no unit is encoded twice
    encoded = [u for l in lines if l["event"] == "publish" for u in l["units"]]
    assert len(encoded) == len(set(encoded)), "the trace shows a double encode"


def test_tracing_is_off_by_default_and_costs_no_branch():
    """A serving path that checks for a tracer at every event is a serving path
    that pays for an audit feature nobody turned on."""
    srv, qs, _, _, _ = _online_toy(budget_m=4)
    assert srv.trace is online.NO_TRACE
    srv.trace("anything", units=[1])          # must be a no-op, not an error
    res, tr = srv.query(qs[0])
    assert tr.n_window > 0


def test_the_backfill_records_which_length_key_it_sorted_by():
    """The padding factor is uninterpretable without it.

    Sorting by characters while padding is measured on tokens understates what
    a sorted backfill can do, which flatters the repair path. Whichever key a
    run used has to be in the file, or a later reader cannot tell.
    """
    import ast
    src = (Path(__file__).resolve().parent.parent / "thaw_pg"
           / "backfill.py").read_text(encoding="utf-8")
    assert "backfill_sort_key" in src, (
        "the results file does not record the sort key")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
    assert fn is not None
    names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "token_sort" in names, "run() has no token_sort switch"
    assert "_token_lengths" in src, "there is no token-length helper to sort by"


# --------------------------------------------- adapters across two widths ---
def _mismatched_width_fixture(n=3000, d_old=16, d_new=32, nq=120, seed=0):
    """Two spaces of different widths -- the usual case, MiniLM 384 against
    BGE-base 768 -- arranged so the information the new model carries lives in
    the coordinates a truncation to the old width would discard.

    This is not a contrived shape. Two encoders share no coordinate frame at
    all, so which of the new model's 768 columns carry a given distinction is
    arbitrary; putting them in the second half is the honest worst case of an
    arbitrary assignment, and an adapter that only reads the first 384 is
    reading noise.
    """
    rng = np.random.default_rng(seed)
    V_old = migration._safe_normalize(rng.normal(size=(n, d_old)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d_old, d_old)))
    V_new = np.zeros((n, d_new), dtype=np.float32)
    V_new[:, d_old:] = V_old @ R
    V_new = migration._safe_normalize(V_new)
    Q_old = migration._safe_normalize(
        V_old[rng.integers(0, n, nq)]
        + 0.05 * rng.normal(size=(nq, d_old)).astype(np.float32))
    Q_new = np.zeros((nq, d_new), dtype=np.float32)
    Q_new[:, d_old:] = Q_old @ R
    Q_new = migration._safe_normalize(Q_new)
    return Q_old, Q_new, V_old, V_new


def test_the_adapter_fit_spans_two_widths_without_truncating_either():
    """A mistake this module made: `evaluate` passed the new-space queries
    through a `_pad_to(Q_new, d_old)` that silently dropped every coordinate
    past the old width, then reported the result as the adapter's accuracy.

    Nothing needs padding. Q_new^T Q_old is (d_new x d_old) for any pair of
    widths and its SVD gives a rectangular semi-orthogonal map, so the fit is
    exact on the narrower space. On this fixture the discarded half was the
    only half that carried signal, so the truncating version scored an adapter
    fitted on all-zero inputs -- it would have read as 'adapters lose', which
    is the paper's claim, from a bug.
    """
    d_old, d_new = 16, 32
    Q_old, Q_new, V_old, V_new = _mismatched_width_fixture(d_old=d_old, d_new=d_new)
    assert np.abs(Q_new[:, :d_old]).max() == 0.0, (
        "fixture is wrong: the truncated half must be the informative one")

    W = procrustes.fit_procrustes(Q_new, Q_old)
    assert W.shape == (d_new, d_old), W.shape
    assert (Q_new @ W).shape == Q_old.shape

    res = procrustes.evaluate(Q_old, Q_new, V_old, V_new, k=10, ks=(200,))
    assert res["d_old"] == d_old and res["d_new"] == d_new, res
    by = {r["scheme"]: r["recall_at_k"] for r in res["schemes"]}
    proc = by["procrustes (orthogonal)"]
    assert proc > 0.95, (
        f"the rectangular fit lost a recoverable rotation: {proc}")

    # and what the truncating version was actually measuring
    W_trunc = procrustes.fit_procrustes(Q_new[:, :d_old], Q_old)
    assert np.abs(Q_new[:, :d_old] @ W_trunc).max() == 0.0, (
        "the old path is supposed to have searched the old index with a zero "
        "query -- if it no longer does, this test is no longer the regression")


def test_the_affine_fit_is_rectangular_too():
    """The same for the ridge map: W is (d_new x d_old), the intercept is in
    the old space. A square W would mean the inputs had been squared off."""
    d_old, d_new = 16, 32
    Q_old, Q_new, V_old, _ = _mismatched_width_fixture(d_old=d_old, d_new=d_new)
    W, b = procrustes.fit_affine(Q_new, Q_old, rank=8)
    assert W.shape == (d_new, d_old) and b.shape == (d_old,), (W.shape, b.shape)
    assert np.linalg.matrix_rank(W) <= 8
    assert (Q_new @ W + b).shape == (len(Q_old), V_old.shape[1])


def test_evaluate_refuses_queries_that_are_not_in_their_corpus_space():
    """Since widths are no longer coerced, a genuine mismatch between a query
    matrix and the index it will search has to be an error rather than a
    silent reshape -- that is the whole reason the coercion was wrong."""
    Q_old, Q_new, V_old, V_new = _mismatched_width_fixture()
    for args, what in (((Q_old[:, :8], Q_new, V_old, V_new), "old"),
                       ((Q_old, Q_new[:, :8], V_old, V_new), "new")):
        try:
            procrustes.evaluate(*args)
        except SystemExit as e:
            assert what in str(e), e
        else:
            raise AssertionError(f"a {what}-side width mismatch was accepted")
    try:
        procrustes.evaluate(Q_old, Q_new[:20], V_old, V_new)
    except SystemExit:
        pass
    else:
        raise AssertionError("unpaired query matrices were accepted")


# ------------------------------------------- the pg online path, on a fake ---
class _FakeCursor:
    def __init__(self, conn, rows):
        self._conn, self._rows = conn, rows

    def fetchall(self):
        self._conn._release()
        return list(self._rows)

    def fetchone(self):
        rows = self.fetchall()
        return rows[0] if rows else None


class _FakeConn:
    """One socket, one conversation -- psycopg's actual constraint.

    A real connection is a single stream: a second thread that issues a
    statement between another thread's `execute` and its `fetchall` interleaves
    two conversations on one socket. This stand-in records every thread that
    touches it and flags any overlap, which is what makes the sharing bug a
    test failure rather than a production incident.
    """

    def __init__(self, store):
        self.store = store
        self.threads: set[str] = set()
        self.violations: list[str] = []
        self._guard = threading.Lock()
        self._in_use: str | None = None
        self.closed = False

    # -- the single-conversation guard
    def _acquire(self, sql):
        me = threading.current_thread().name
        with self._guard:
            self.threads.add(me)
            if self._in_use is not None and self._in_use != me:
                self.violations.append(
                    f"{me} issued {sql.split()[0]} while {self._in_use} was "
                    f"mid-statement")
            self._in_use = me

    def _release(self):
        with self._guard:
            self._in_use = None

    def execute(self, sql, params=()):
        if self.closed:
            raise RuntimeError("the connection is closed")
        self._acquire(sql)
        rows = self.store.answer(sql, params)
        if rows is None:                       # SET and friends return nothing
            self._release()
            rows = []
        return _FakeCursor(self, rows)

    def close(self):
        self.closed = True


class _FakeStore:
    """`docs`, `docs_new`, `repair_claims` and the statements the module issues.

    One store stands for one database, so two retrievers over two `_FakeConn`s
    on the same store are two service processes -- which is the deployment the
    cross-process claim exists for and the one no fake modelled before.
    """

    def __init__(self, V_old, V_new, kprime, seed=0, fail_write_for=(),
                 fail_fetch_for=(), fail_encode_for=()):
        self.V_old, self.V_new = V_old, V_new
        self.kprime = kprime
        self.present: dict[int, np.ndarray] = {}
        self.claimed: dict[int, str] = {}
        self.lock = threading.Lock()
        self.encoded: list[int] = []
        self.written: set[int] = set()        # rows with origin='write'
        self.deleted: set[int] = set()        # rows removed after the switch
        self.expired: set[int] = set()        # claims whose owner died
        self.renewed: set[int] = set()        # claims an owner kept alive
        self.reclaim_calls = 0
        self.fail_write_for = set(int(i) for i in fail_write_for)
        self.fail_fetch_for = set(int(i) for i in fail_fetch_for)
        self.fail_encode_for = set(int(i) for i in fail_encode_for)

    def write_new_document(self, unit: int, vec):
        """A document created or updated after the switch: it goes straight
        into docs_new with origin='write'. A new one has no old vector at all;
        an updated one may also have a stale repaired vector, which this
        replaces -- that is the case an answer must not get wrong."""
        with self.lock:
            self.present[int(unit)] = vec
            self.written.add(int(unit))
            self.deleted.discard(int(unit))

    def delete_document(self, unit: int):
        """ON DELETE CASCADE from `docs`: the row leaves both tables."""
        with self.lock:
            self.present.pop(int(unit), None)
            self.written.discard(int(unit))
            self.deleted.add(int(unit))

    def expire_claims(self, ids):
        """Mark these claims as held by an owner that died."""
        with self.lock:
            self.expired.update(int(i) for i in ids)

    # -- the callables db_io supplies, each already serialised there
    def fetch_text(self, ids):
        bad = sorted(set(int(i) for i in ids) & self.fail_fetch_for)
        if bad:
            raise RuntimeError(f"simulated fetch failure for {bad}")
        return [f"unit {int(i)}" for i in ids]

    def encode_new(self, texts):
        ids = [int(t.split()[1]) for t in texts]
        bad = sorted(set(ids) & self.fail_encode_for)
        if bad:
            raise RuntimeError(f"simulated encode failure for {bad}")
        with self.lock:
            self.encoded.extend(ids)
        return self.V_new[ids]

    def write(self, ids, vecs):
        """Idempotent, like `online_io`'s writer: a repeat is a no-op, not a
        primary-key error. `fail_write_for` makes a unit's write raise, which
        is how the not-persisted-is-not-repaired path gets exercised."""
        bad = sorted(set(int(i) for i in ids) & self.fail_write_for)
        if bad:
            raise RuntimeError(f"simulated write failure for {bad}")
        with self.lock:
            for i, v in zip(ids, vecs):
                self.present[int(i)] = v

    def answer(self, sql, params):
        if sql.startswith("SET"):
            return None
        if "ORDER BY emb_old" in sql:
            q, kp = params
            sc = self.V_old @ q
            out = []
            for i in np.argsort(-sc):
                i = int(i)
                if i in self.deleted:
                    continue              # ON DELETE CASCADE took the row
                out.append((i,))
                if len(out) >= kp:
                    break
            return out
        if sql.startswith("SELECT id FROM docs_new WHERE id = ANY"):
            ids = params[0]
            with self.lock:
                return [(int(i),) for i in ids if int(i) in self.present]
        if sql.startswith("INSERT INTO repair_claims"):
            # the atomic part: of any number of callers offering the same
            # unit, exactly one gets the row back
            owner, ids = params
            with self.lock:
                got = [int(i) for i in ids if int(i) not in self.claimed]
                for i in got:
                    self.claimed[i] = owner
            return [(i,) for i in got]
        if sql.startswith("UPDATE repair_claims SET claimed_at"):
            # the renewal: this owner is still alive and still working. It
            # returns the rows it actually touched, so a claim reclaimed by
            # somebody else renews nothing and the caller can tell.
            ids, owner = params
            with self.lock:
                hit = [int(i) for i in ids if self.claimed.get(int(i)) == owner]
                self.renewed.update(hit)
            return [(i,) for i in hit]
        if sql.startswith("DELETE FROM repair_claims WHERE id = ANY"):
            ids, owner = params
            with self.lock:
                for i in ids:
                    if self.claimed.get(int(i)) == owner:
                        del self.claimed[int(i)]
            return None
        if sql.startswith("DELETE FROM repair_claims c "
                          "WHERE EXISTS"):
            # the completed-claim sweep: a claim on a unit that already has a
            # vector is finished work, whoever holds it and however fresh
            with self.lock:
                gone = sorted(u for u in self.claimed if u in self.present)
                for u in gone:
                    self.claimed.pop(u, None)
                    self.expired.discard(u)
            return [(u,) for u in gone]
        if sql.startswith("DELETE FROM repair_claims c"):
            # the lease sweep: drop claims past the lease whose unit still has
            # no vector, and report which
            with self.lock:
                self.reclaim_calls += 1
                gone = sorted(u for u in self.expired
                              if u not in self.present)
                for u in gone:
                    self.claimed.pop(u, None)
                    self.expired.discard(u)
            return [(u,) for u in gone]
        if sql.startswith("SELECT id FROM docs_new WHERE origin = 'write'"):
            # the partial index over rows written after the switch: a
            # document created mid-migration has no old vector, so no
            # old-space window can reach it
            q_new, k = params
            with self.lock:
                have = [(i, v) for i, v in self.present.items()
                        if i in self.written]
            have.sort(key=lambda iv: -float(iv[1] @ q_new))
            return [(i,) for i, _ in have[:k]]
        if sql.startswith("WITH cand AS MATERIALIZED"):
            cand, q_new, k = params
            with self.lock:
                have = [(int(i), self.present[int(i)]) for i in cand
                        if int(i) in self.present]
            have.sort(key=lambda iv: -float(iv[1] @ q_new))
            return [(i,) for i, _ in have[:k]]
        raise AssertionError(f"the fake store was asked something new: {sql!r}")


def _pg_online_fixture(n=800, d=24, kprime=40, budget_m=6, seed=0,
                       store=None, owner="p1", **kw):
    from thaw_pg import online as pg_online

    rng = np.random.default_rng(seed)
    V_old = migration._safe_normalize(rng.normal(size=(n, d)).astype(np.float32))
    R, _ = np.linalg.qr(rng.normal(size=(d, d)))
    V_new = migration._safe_normalize((V_old @ R).astype(np.float32))
    store = store if store is not None else _FakeStore(V_old, V_new, kprime, **kw)
    qconn = _FakeConn(store)
    r = pg_online.OnlinePgRetriever(
        qconn, encode_new=store.encode_new, fetch_text=store.fetch_text,
        write=store.write, enc_old_query=None, enc_new_query=None,
        kprime=kprime, k=10, budget_m=budget_m, drain_batch=8, owner=owner)
    queries = [V_old[int(i)] for i in rng.integers(0, n, 60)]
    qs_new = [migration._safe_normalize((q @ R)[None, :])[0] for q in queries]
    return pg_online, r, store, qconn, queries, qs_new


def test_the_pg_repairer_runs_on_its_own_connection_not_a_query_threads():
    """The bug this pins: `_encode_and_store` read presence from `self.conn`,
    so the repairer thread issued its SELECT on whichever query thread's
    connection the retriever held. `db_io` already gives the repair I/O its
    own connection for exactly this reason; the presence check had been left
    behind.
    """
    pg_online, r, store, qconn, queries, qs_new = _pg_online_fixture()
    rconn = _FakeConn(store)
    rep = pg_online.BackgroundRepairer(r, conn=rconn, idle_sleep=0.001)
    rep.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            for q, qn in zip(queries, qs_new):
                r.query(q_old=q, q_new=qn)
            if rep.encoded:
                break
        assert rep.encoded, "the repairer never drained anything to check"
    finally:
        rep.stop()

    assert qconn.threads == {"MainThread"}, (
        f"a query thread's connection was used from {qconn.threads}")
    assert rconn.threads == {"thaw-pg-repairer"}, (
        f"the repairer's connection was used from {rconn.threads}")
    assert not qconn.violations and not rconn.violations, (
        qconn.violations + rconn.violations)


def test_the_pg_repairer_refuses_a_retrievers_connection():
    """Refused at construction, not tolerated: the failure it causes is a
    mispaired result under load, which is the hardest kind to read back."""
    pg_online, r, _store, qconn, _q, _qn = _pg_online_fixture()
    try:
        pg_online.BackgroundRepairer(r, conn=qconn)
    except SystemExit as e:
        assert "own connection" in str(e), e
    else:
        raise AssertionError("the repairer accepted a query thread's connection")
    rep = pg_online.BackgroundRepairer(r)          # neither conn nor dsn
    try:
        rep.conn
    except SystemExit as e:
        assert "conn=" in str(e) or "dsn=" in str(e), e
    else:
        raise AssertionError("a repairer with no connection source was usable")


def test_the_pg_path_never_encodes_a_unit_twice_across_the_repairer():
    """The same cost guarantee `semview.online` keeps, against the schema.

    Here the presence re-check is the database rather than a bitmap, and it
    now runs on the repairer's own connection -- so this test is also what
    says the fix did not reopen the double-encode race.
    """
    pg_online, r, store, qconn, queries, qs_new = _pg_online_fixture(budget_m=4)
    rconn = _FakeConn(store)
    rep = pg_online.BackgroundRepairer(r, conn=rconn, idle_sleep=0.001)
    rep.start()
    try:
        for q, qn in zip(queries, qs_new):
            r.query(q_old=q, q_new=qn)
    finally:
        rep.stop()
    drained = pg_online.BackgroundRepairer(r, conn=_FakeConn(store)).drain_now()

    assert len(store.encoded) == len(set(store.encoded)), (
        f"{len(store.encoded) - len(set(store.encoded))} units were encoded "
        f"more than once")
    assert set(store.encoded) == set(store.present), (
        "a unit reached docs_new without being encoded, or the reverse")
    assert drained >= 0 and r.encoded_units == len(store.encoded)


def test_the_pg_online_answer_is_exact_over_what_is_present():
    """Ranking covers what is present, not what the window contains -- the
    quality this policy trades for its latency. The answer must still be the
    exact new-space order over the present candidates, and coverage must be
    what the trace reports it to be."""
    pg_online, r, store, _qconn, queries, qs_new = _pg_online_fixture(budget_m=8)
    traces = [r.query(q_old=q, q_new=qn) for q, qn in zip(queries, qs_new)]
    for tr, qn in zip(traces, qs_new):
        assert tr.n_present <= tr.window_size == r.kprime
        assert tr.window_coverage == tr.n_present / tr.window_size
        assert all(i in store.present for i in tr.results), (
            "an answer named a unit with no new vector")
        scores = [float(store.present[i] @ qn) for i in tr.results]
        assert scores == sorted(scores, reverse=True), scores
        assert len(tr.results) == min(r.k, tr.n_present)
    s = pg_online.summarise(traces)
    assert s["queries"] == len(traces)
    assert 0.0 < s["window_coverage_mean"] <= 1.0
    assert s["inline_per_query_mean"] <= r.budget_m


def test_two_service_processes_never_encode_the_same_unit():
    """The guarantee the module's own docstring made and did not keep.

    Presence came from the database, but the claim was a `threading.Lock` in
    one process -- so two services each saw a unit absent, each claimed it
    locally, and both encoded it. Every cost number in the paper rests on each
    unit being encoded at most once, and this is the deployment the file was
    written for.

    The claim is now the `repair_claims` row: `INSERT ... ON CONFLICT (id) DO
    NOTHING RETURNING id`, which is the protocol step 1's check 5 gated with
    eight workers and zero double claims. Two retrievers here hold separate
    connections and separate `PgClaimSet`s over one store, which is two
    processes as far as the claim is concerned.
    """
    pg_online, r1, store, c1, queries, qs_new = _pg_online_fixture(
        budget_m=8, owner="service-1")
    _, r2, _, c2, _, _ = _pg_online_fixture(budget_m=8, store=store,
                                            owner="service-2")
    assert r1.claims is not r2.claims and r1.claims.owner != r2.claims.owner

    def serve(r, qs, qn):
        for q, n in zip(qs, qn):
            r.query(q_old=q, q_new=n)

    # the same stream through both services at once: every window overlaps
    ts = [threading.Thread(target=serve, args=(r, queries, qs_new), name=f"svc{i}")
          for i, r in enumerate((r1, r2), 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)

    dupes = len(store.encoded) - len(set(store.encoded))
    assert dupes == 0, f"{dupes} units were encoded by both services"
    assert set(store.encoded) == set(store.present)
    assert not store.claimed, f"claims left behind: {sorted(store.claimed)[:5]}"
    assert r1.encoded_units + r2.encoded_units == len(store.encoded)


def test_a_unit_whose_write_fails_is_not_counted_as_repaired():
    """`repaired` was set before `put` was called, so a persistence failure
    left the unit marked done in this process with nothing stored: no later
    query would retry it, and the run would report a migration it had not
    made. A repair is not a repair until it is persisted."""
    srv, qs, V_true, _, _ = _online_toy(budget_m=8)
    doomed = int(srv.search(srv.encode_old_query(qs[0]), srv.kprime)[0][0])
    inner = srv.put

    def flaky(ids, vecs):
        if doomed in [int(i) for i in ids]:
            raise RuntimeError("simulated persistence failure")
        if inner is not None:
            inner(ids, vecs)

    srv.put = flaky
    try:
        srv.query(qs[0])
    except RuntimeError:
        pass                              # an inline repair may surface it
    assert not srv.repaired[doomed], (
        "a unit whose write failed is marked repaired, so nothing will retry it")
    assert srv.put_failures >= 1
    assert doomed in set(srv.queue.snapshot()), (
        "the unit was not put back on the queue")

    # and it repairs normally once persistence works again
    srv.put = inner
    srv.queue.offer([doomed])
    srv.drain_now()
    assert srv.repaired[doomed]


def test_the_repairer_thread_survives_a_failing_batch_and_says_so():
    """An exception in the drain loop used to kill the thread, and a dead
    thread joins instantly -- so `stop_repairer` reported success while the
    queue had grown untouched for the rest of the run."""
    srv, qs, _, _, _ = _online_toy(budget_m=0)       # everything to the queue
    boom = {"n": 0}
    inner = srv.encode_new

    def sometimes(ts):
        boom["n"] += 1
        if boom["n"] % 3 == 1:
            raise RuntimeError("simulated device failure")
        return inner(ts)

    srv.encode_new = sometimes
    srv.start_repairer(idle_sleep=0.001)
    for q in qs:
        srv.query(q)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and srv.repairer_errors == 0:
        time.sleep(0.005)
    # The loop must be given the chance to CARRY ON before it is stopped.
    # Stopping the instant the first error is recorded races the retry: the
    # except branch re-offers the batch and waits `idle_sleep` (1 ms here), and
    # on Windows the main thread can observe the error and set `_stop` inside
    # that same millisecond -- the worker then exits at the next loop check and
    # `encoded_units` measures the race, not the survival. Waiting for a
    # successful encode first is what "carrying on" means. 2026-10-17.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and srv.encoded_units == 0:
        time.sleep(0.005)
    assert srv.stop_repairer(timeout=10.0) is True
    assert srv.repairer_errors > 0 and srv.last_repairer_error, (
        "the loop swallowed a failure without recording it")
    assert srv.encoded_units > 0, "the thread died instead of carrying on"


def test_a_failed_repair_batch_goes_back_on_the_queue_and_is_not_lost():
    """What the previous test missed by asserting on the wrong quantity.

    `_drain_loop` took a batch off the queue, and when the encoder raised it
    recorded the error, released the claim and returned nothing -- on the
    strength of a comment claiming the batch was already back on the queue,
    which held only for a failure in the *persist* step. Every unit in a batch
    the encoder choked on was then in no queue and claimed by nobody: no later
    query would ask for it, and the run would finish reporting a migration
    that had silently skipped them.

    The earlier test asserted `encoded_units > 0`, which stays true as long as
    *some* batch succeeds, so it passed over exactly this. The quantity that
    matters is whether the units survive -- so this one fails every encode
    until a flag flips, and then requires the full set back.
    """
    srv, qs, _, _, _ = _online_toy(budget_m=0)       # everything to the queue
    failing = {"on": True}
    inner = srv.encode_new
    srv.encode_new = lambda ts: (_ for _ in ()).throw(
        RuntimeError("device is down")) if failing["on"] else inner(ts)

    for q in qs:
        srv.query(q)
    wanted = set(srv.queue.snapshot())
    assert wanted, "the fixture queued nothing"

    srv.start_repairer(idle_sleep=0.001)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and srv.repairer_errors < 3:
        time.sleep(0.005)
    assert srv.stop_repairer(timeout=10.0)
    assert srv.repairer_errors >= 3, "the repairer never tried and failed"
    assert srv.encoded_units == 0, "the fixture let something through"

    # The invariant is not "still queued" but "accounted for": a unit is
    # either repaired, on the queue, or named in quarantine. Never vanished.
    survived = set(srv.queue.snapshot())
    held = set(srv.queue.quarantined)
    lost = wanted - survived - held
    assert not lost, (
        f"{len(lost)} unit(s) were taken from the queue and dropped when the "
        f"encode failed; nothing will ever ask for them again")
    # Quarantine is NOT asserted here. A few batch failures are not evidence
    # against any particular unit, and a suspect has to fail alone before it
    # is named -- which is the subject of
    # `test_one_bad_unit_does_not_hold_the_encoder_forever`, not of this one.
    # What this test owns is the weaker and more important property: nothing
    # disappeared.

    # and once the device comes back, the same units repair -- including the
    # quarantined ones, which an operator releases deliberately after looking
    # at what they were
    srv.queue.quarantined.clear()
    srv.queue.clear_failures(wanted)
    srv.queue.offer(wanted)
    failing["on"] = False
    srv.drain_now()
    assert set(np.flatnonzero(srv.repaired)) >= wanted
    assert srv.queue.depth == 0


def test_taking_from_the_queue_does_not_cost_more_as_the_queue_grows():
    """`take` used to sort the whole queue, under the lock `offer` also needs.

    At the default capacity of 100,000 that is 15 ms per take, paid
    continuously by a draining repairer while every query trying to enqueue
    its misses waits behind it — so the queue got slower exactly when it was
    deepest, which is when a migration most needs it to drain. Bucketing by
    miss count gives the same order out with no sort.

    The assertion is on the *shape* rather than on a wall-clock figure: at ten
    times the depth, a take must not cost ten times as much.
    """
    def cost(depth, reps=40):
        q = online.RepairQueue(capacity=400_000)
        q.offer(range(depth))
        t0 = time.perf_counter()
        for _ in range(reps):
            q.offer(q.take(16))
        return (time.perf_counter() - t0) / reps

    small, large = cost(1_000), cost(100_000)
    assert large < small * 5 + 2e-4, (
        f"take costs {large * 1e3:.3f} ms at depth 100,000 against "
        f"{small * 1e3:.3f} ms at 1,000: it still scales with the queue")

    # and the order is unchanged: most-missed first
    q = online.RepairQueue()
    q.offer([1, 2, 3])
    q.offer([2, 3])
    q.offer([3])
    assert q.take(3) == [3, 2, 1]
    assert q.depth == 0 and q.snapshot() == []


def test_an_outage_does_not_quarantine_a_healthy_corpus():
    """A database that went away is not a bad document.

    Every exception was counted against the units in the batch, so a
    connection drop or a 503 from the inference server spent a strike per unit
    per retry — and the background loop retried at `idle_sleep`. A few seconds
    of outage would therefore quarantine healthy documents wholesale, and the
    migration would come back up permanently missing them, with the only
    record being a quarantine list blaming the documents.
    """
    q = online.RepairQueue(max_attempts=3)
    q.offer(range(20))
    units = list(range(20))

    for _ in range(10):                   # ten rounds of a service outage
        dead = q.record_failure(units, "OperationalError: server closed the "
                                        "connection unexpectedly")
        assert dead == [], "an outage quarantined a document"
    assert not q.quarantined
    assert q.transient_failures == 10
    assert "server closed" in (q.last_transient or "")

    # a genuinely bad document still gets there, alone
    for _ in range(3):
        q.record_failure([7], "ValueError: this document cannot be encoded",
                         isolated=True)
    assert set(q.quarantined) == {7}, q.quarantined


def test_a_contending_or_failing_loop_backs_off_instead_of_spinning():
    """Two services on the same hot ids, or one service against a database
    that is down: re-queueing and immediately re-claiming is a tight loop
    against something already struggling."""
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)

    class _Broken:
        closed = False
        threads: set = set()
        violations: list = []

        def execute(self, sql, params=()):
            raise RuntimeError("server closed the connection unexpectedly")

    rep = pg_online.BackgroundRepairer(r, conn=_Broken(), idle_sleep=0.002)
    rep.max_backoff_s = 0.2
    for q, qn in zip(queries[:8], qs_new[:8]):
        r.query(q_old=q, q_new=qn)
    rep.start()
    try:
        time.sleep(1.0)
    finally:
        rep.stop(timeout=10.0)
    # without backoff this is ~500 attempts in a second; with it, far fewer
    assert rep.errors > 0
    assert rep.errors < 60, (
        f"{rep.errors} attempts in one second against a dead database: the "
        f"loop is not backing off")
    assert rep._backoff > 0


def test_one_bad_unit_does_not_hold_the_encoder_forever():
    """The failure mode the re-queue fix created, closed with it.

    Returning a failed batch to the queue is right for a transient fault and
    wrong on its own for a permanent one: a single undecodable document fails
    its batch, the batch comes back, and the repairer spends the device on it
    for the rest of the run while every healthy unit waits behind it. Nothing
    reports a problem — the queue is draining, the loop is alive — and the
    migration simply stops converging.

    A unit is quarantined after `max_attempts` failures: out of rotation,
    counted, and listed with the reason. The guarantee that survives is not
    "every unit is retried forever" but "every unit is repaired, queued, or
    named".
    """
    srv, qs, V_true, _, _ = _online_toy(budget_m=0)
    srv.queue.max_attempts = 3
    poison = {int(srv.search(srv.encode_old_query(qs[0]), srv.kprime)[0][0])}
    inner = srv.encode_new
    calls = {"n": 0}

    def sometimes(ts):
        calls["n"] += 1
        ids = {int(t.split()[1]) for t in ts}
        if ids & poison:
            raise RuntimeError("this document cannot be encoded")
        return inner(ts)

    srv.encode_new = sometimes
    for q in qs:
        srv.query(q)

    for _ in range(400):                   # bounded: the point is it ends
        if srv.queue.depth == 0:
            break
        try:
            srv.drain_now(max_batches=4)
        except RuntimeError:
            pass
    assert srv.queue.depth == 0, (
        f"the queue never drained: {srv.queue.depth} left, and a bad unit is "
        f"still in rotation")
    assert set(srv.queue.quarantined) == poison, srv.queue.quarantined
    assert "cannot be encoded" in next(iter(srv.queue.quarantined.values()))

    # the healthy units went through rather than waiting behind it
    assert srv.encoded_units > 0
    assert not (set(np.flatnonzero(srv.repaired)) & poison)

    s = online.summarise([srv.query(q)[1] for q in qs[:5]], server=srv)
    assert s["quarantined_units"] == 1, s


def test_an_inline_repair_that_fails_neither_loses_its_units_nor_the_request():
    """The third site of the same bug, and the one the first two tests missed.

    An inline repair's units come straight off a window and were never queued,
    so when the encoder raised, the claim was released and nothing put them
    anywhere: no queue held them and no background worker would ever see them.
    The exception also escaped `query`, failing the request — while the
    PostgreSQL path caught it and answered over what was present, so the two
    realizations of one policy disagreed about what a device fault means.

    Both halves are asserted here: the request still answers, and the units
    are on the queue afterwards.
    """
    srv, qs, _, _, _ = _online_toy(budget_m=8)
    failing = {"on": True}
    inner = srv.encode_new
    srv.encode_new = lambda ts: (_ for _ in ()).throw(
        RuntimeError("device fault")) if failing["on"] else inner(ts)

    before = srv.queue.depth
    results, tr = srv.query(qs[0])            # must not raise
    assert tr.repair_error and "device fault" in tr.repair_error, tr.repair_error
    assert tr.n_inline == 0 and srv.encoded_units == 0
    assert srv.queue.depth > before, (
        "the inline units were dropped: they were never in a queue, so "
        "nothing will ever repair them")
    assert srv.put_failures >= 1

    # the answer is still the exact order over what was present, which here is
    # nothing -- an empty answer is a degraded answer, not an error
    assert results == []

    queued = set(srv.queue.snapshot())
    failing["on"] = False
    srv.drain_now()
    assert srv.encoded_units == len(queued) > 0
    _, tr2 = srv.query(qs[0])
    assert tr2.repair_error is None and tr2.n_visible > 0


def test_stop_keeps_the_handle_when_the_thread_will_not_stop():
    """Dropping the reference after a timed-out join loses the handle to a
    live thread: `start_repairer` then runs a second one beside it, and in the
    PostgreSQL path `close()` closes the connection the live one is using."""
    srv, qs, _, _, _ = _online_toy(budget_m=0)
    held = threading.Event()
    inner = srv.encode_new

    def hang(ts):
        held.set()
        time.sleep(3.0)                   # still inside a forward pass
        return inner(ts)

    srv.encode_new = hang
    for q in qs[:4]:
        srv.query(q)
    srv.start_repairer(idle_sleep=0.001)
    assert held.wait(10.0), "the repairer never reached the encoder"
    worker = srv._worker
    assert srv.stop_repairer(timeout=0.05) is False, (
        "stop_repairer claimed the thread had exited")
    assert srv._worker is worker, "the handle to a live thread was dropped"
    srv.start_repairer(idle_sleep=0.001)
    assert srv._worker is worker, "a second repairer was started beside a live one"
    assert srv.stop_repairer(timeout=15.0) is True
    assert srv._worker is None


def test_the_new_query_encode_overlaps_the_database_work():
    """The measured bottleneck is the encoder: at eight threads, 9.71 s of a
    10.36 s p50. The new-space query vector depends on nothing the request is
    doing -- not the window, not the presence read, not the repairs -- yet it
    was encoded last, after a request had just spent 400-500 ms in the
    database with the device idle.

    What is asserted here is the structure, not a speedup: the encode is
    already running when the window query starts, and the request waits for
    less than the encode took. Whether that is worth anything on a real
    device depends on contention with the repair encoder, which is why the
    trace carries both numbers instead of a claim.
    """
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=4)
    started = threading.Event()

    class _SlowEncoder:
        def encode(self, texts):
            started.set()
            time.sleep(0.15)
            return np.asarray([qs_new[0]], dtype=np.float32)

    class _WatchingStore:
        """A database that takes measurable time, because an instant one has
        nothing for the encode to hide behind -- the overlap would be real and
        the measurement of it would be noise."""

        def __init__(self, inner):
            self.inner = inner
            self.window_saw_encode_running = False

        def answer(self, sql, params):
            if "ORDER BY emb_old" in sql:
                self.window_saw_encode_running = started.is_set()
                time.sleep(0.08)           # the 400-500 ms window read, in small
            return self.inner.answer(sql, params)

    watch = _WatchingStore(store)
    r.conn = _FakeConn(watch)
    r.enc_new_query = _SlowEncoder()
    r.enc_old_query = None

    tr = r.query(text="a query", q_old=queries[0])
    assert watch.window_saw_encode_running, (
        "the window query ran before the encode started: the two are still "
        "serialised")
    assert tr.q_new_ms > 100.0, tr.q_new_ms
    assert tr.q_new_wait_ms < tr.q_new_ms - 40.0, (
        f"the request waited {tr.q_new_wait_ms:.1f} ms for an encode that took "
        f"{tr.q_new_ms:.1f} ms, against 80 ms of database work it should have "
        f"been running behind")

    # and the switch turns it off, for a run that wants the old shape
    r2 = pg_online.OnlinePgRetriever(
        _FakeConn(store), encode_new=store.encode_new,
        fetch_text=store.fetch_text, write=store.write,
        enc_old_query=None, enc_new_query=_SlowEncoder(), kprime=r.kprime,
        k=10, budget_m=0, overlap_query_encode=False, owner="serial")
    tr2 = r2.query(text="a query", q_old=queries[0])
    assert tr2.q_new_wait_ms >= tr2.q_new_ms * 0.9, (
        "overlap_query_encode=False still overlapped")


def test_the_claim_protocol_sql_against_a_real_postgresql():
    """Run `thaw_pg.sql_check` when a server is reachable; skip otherwise.

    Everything else about the claim protocol is checked against a fake that
    models `INSERT ... ON CONFLICT DO NOTHING RETURNING id` as a dictionary
    insert under a Python lock. That is an assertion about what PostgreSQL
    does, not evidence of it, and the difference is the whole of `D-07`.

    Point `THAW_SQLCHECK_DSN` at any scratch database — the checks create and
    drop their own tables — and this runs the real thing. It needs `psql` and
    no Python driver.
    """
    import subprocess

    dsn = os.environ.get("THAW_SQLCHECK_DSN")
    if not dsn:
        raise Skip("THAW_SQLCHECK_DSN is not set (no PostgreSQL to check "
                   "against; see D-07)")
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-m", "thaw_pg.sql_check", "--dsn", dsn],
        capture_output=True, text=True, cwd=str(root), timeout=900)
    assert out.returncode == 0, (
        f"the claim protocol's SQL failed against a real server:\n"
        f"{out.stdout[-3000:]}{out.stderr[-1000:]}")
    assert "contended: True" in out.stdout, (
        "the concurrent sessions did not overlap, so exclusivity held "
        "vacuously:\n" + out.stdout[-1500:])


def test_the_pg_query_path_degrades_and_requeues_whichever_step_fails():
    """The PostgreSQL query path's own failure test, which did not exist.

    `test_an_inline_repair_that_fails...` exercises `semview.OnlineServer`, a
    different implementation; the PG failure tests covered the background
    worker only. So the realization that is supposed to go to production had
    its inline repair path checked by proxy — and it was precisely there that
    the guard covered only `write()` while the numpy one covered all three
    steps, which is the kind of divergence testing-by-proxy cannot see.

    Each of the three steps is failed separately, because they fail
    differently in production: `fetch_text` is a database read, `encode_new`
    is a device, `write` is an insert.
    """
    for step, kw in (("fetch_text", "fail_fetch_for"),
                     ("encode_new", "fail_encode_for"),
                     ("write", "fail_write_for")):
        pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=8)
        # the whole first window fails, whichever step is being broken
        window = [int(i) for i in np.argsort(-(store.V_old @ queries[0]))[:r.kprime]]
        setattr(store, kw, set(window))

        before = set(r.queue.snapshot())
        tr = r.query(q_old=queries[0], q_new=qs_new[0])      # must not raise

        # `A and B or A` is `A`, which is what this said before: it checked
        # only that the string was non-empty, so a run where the WRONG step
        # failed -- or where all three injections hit the same one -- passed
        # silently. Each injection names itself.
        assert tr.repair_error, f"{step}: no repair_error on the trace"
        assert f"simulated {step.split('_')[0]}" in tr.repair_error, (
            f"{step}: the trace blames {tr.repair_error!r}, so this injection "
            f"did not hit the step it was aimed at")
        assert tr.n_inline == 0, f"{step}: counted an inline repair that failed"
        assert r.encoded_units == 0, f"{step}: counted an encode that failed"
        assert r.put_failures > 0, f"{step}: the failure was not counted"
        assert not store.claimed, (
            f"{step}: a claim row was left behind, so no other worker can "
            f"take those units and the lease is the only way back")
        after = set(r.queue.snapshot())
        assert after - before, (
            f"{step}: the units it tried to repair are in no queue -- they "
            f"came off a window and were never queued, so nothing will "
            f"retry them")
        # and the request still answered, over whatever was already present
        assert isinstance(tr.results, list)
        assert tr.n_present == len(
            [u for u in window if u in store.present])


def test_a_claim_on_a_finished_unit_does_not_block_it_forever():
    """Finishing and abandoning are different events.

    `reclaim_expired` only takes claims whose unit is still MISSING, reading a
    present vector as proof that somebody is on it. So a worker that dies
    between the write and the release leaves a claim on a unit that is
    *finished*: past its lease, never reclaimed, and every other process fails
    to claim that id, re-queues it, and fails again — forever. The recover
    scenario counted exactly those ids as still blocked.
    """
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=8)
    for q, qn in zip(queries[:15], qs_new[:15]):
        r.query(q_old=q, q_new=qn)
    done = sorted(store.present)[:4]
    assert done, "the fixture repaired nothing"

    # a dead worker's claim on units it had already written
    store.claimed.update({u: "died-after-writing" for u in done})
    store.expire_claims(done)

    rep = pg_online.BackgroundRepairer(r, conn=_FakeConn(store),
                                       idle_sleep=0.001, sweep_every_s=0.05)
    # the abandoned-work sweep alone leaves them: their unit has a vector
    assert r.claims.reclaim_expired(_FakeConn(store)) == 0
    assert all(u in store.claimed for u in done), (
        "the fixture no longer reproduces the stuck state")

    rep.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and rep.released < len(done):
            time.sleep(0.01)
    finally:
        rep.stop(timeout=10.0)

    assert rep.released >= len(done), (
        f"{len(done)} claim(s) on finished units were never released: those "
        f"ids are unclaimable by anyone, for good")
    assert not any(u in store.claimed for u in done)
    assert r.claims.claim(r.conn, done) == sorted(done), (
        "the ids are still not claimable")


def test_a_renewal_that_renews_nothing_is_not_counted_as_success():
    """`renew` returned the number of ids it was ASKED about, not the number
    it touched. The statement matches only rows this owner still holds, so a
    claim already reclaimed by somebody else renews nothing — and the
    heartbeat counter said the opposite of what had happened, which is the
    one number a run would look at to decide whether encode-once held."""
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)
    conn = _FakeConn(store)
    mine = [1, 2, 3]
    assert r.claims.claim(conn, mine) == mine
    assert r.claims.renew(conn, mine) == 3

    # another process takes one of them
    store.claimed[2] = "somebody-else"
    assert r.claims.renew(conn, mine) == 2, (
        "renew reported a claim it no longer holds as renewed")

    # and none at all once they are all gone
    for u in mine:
        store.claimed.pop(u, None)
    assert r.claims.renew(conn, mine) == 0


def test_a_slow_worker_renews_its_claim_rather_than_looking_dead():
    """The lease is a race against the encoder unless the holder renews.

    A worker whose batch takes longer than `lease_s` has its OWN live claim
    reclaimed by another process's sweep; that process re-claims the unit and
    encodes it too. The write is idempotent so nothing corrupts — the unit is
    simply encoded twice, which is the one guarantee the paper's cost numbers
    rest on, broken not by a crash but by a slow device. On a shared GPU that
    is the ordinary case, not the exotic one.
    """
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)
    rconn = _FakeConn(store)
    rep = pg_online.BackgroundRepairer(r, conn=rconn, idle_sleep=0.001)
    for q, qn in zip(queries[:10], qs_new[:10]):
        r.query(q_old=q, q_new=qn)
    assert r.queue.depth, "the fixture queued nothing"

    rep.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not store.renewed:
            time.sleep(0.005)
    finally:
        rep.stop(timeout=10.0)
    assert store.renewed, (
        "the repairer never renewed a claim it was holding, so a batch slower "
        "than the lease would be reclaimed out from under it and the unit "
        "encoded twice")
    assert store.renewed <= set(store.encoded) | set(r.queue.snapshot()), (
        "a claim was renewed for a unit this worker never held")


def test_an_encode_longer_than_the_lease_keeps_its_claim_renewed():
    """Renewing once before the encode only moves the deadline.

    The reviewer's case: a batch slower than `lease_s` has its own claim swept
    by another process, which re-claims and re-encodes the unit. The previous
    test asserted only that `renew` had been called once, which is true of the
    broken version too. This one makes the encode outlast the lease and runs a
    competing sweep against it: the claim must still be the first worker's at
    the end, and the unit must not be encoded twice.
    """
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)
    r.claims.lease_s = 0.25                      # a very short lease
    for q, qn in zip(queries[:6], qs_new[:6]):
        r.query(q_old=q, q_new=qn)
    assert r.queue.depth

    rconn = _FakeConn(store)
    rep = pg_online.BackgroundRepairer(r, conn=rconn, idle_sleep=0.001)
    rep.dsn = None                                # no second connection here
    rep.heartbeat_s = 0.05
    inner = store.encode_new
    slow = threading.Event()

    def slow_encode(texts):
        slow.set()
        time.sleep(0.8)                           # three times the lease
        return inner(texts)

    r.encode_new = slow_encode
    rep.start()
    try:
        assert slow.wait(10.0), "the repairer never reached the encoder"
        # a competing process sweeps while the first worker is still encoding
        sweeper = pg_online.PgClaimSet("another-process", lease_s=0.25)
        time.sleep(0.5)
        store.expire_claims(list(store.claimed))  # everything looks old
        stolen = sweeper.reclaim_expired(_FakeConn(store))
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not store.encoded:
            time.sleep(0.01)
    finally:
        r.encode_new = inner
        rep.stop(timeout=20.0)

    assert rep.renewals > 0, "the claim was never renewed during the encode"
    assert len(store.encoded) == len(set(store.encoded)), (
        f"{len(store.encoded) - len(set(store.encoded))} unit(s) were encoded "
        f"twice: a sweep took a live worker's claim")


def test_draining_on_this_thread_is_refused_while_the_worker_runs():
    """`drain_now` runs on the caller's thread and uses the repairer's
    connection, which the repairer thread is also using. One connection serves
    one thread in this module — and `integration.py`, written to demonstrate
    the design, called `start()` then `drain_now()` without stopping between.
    Refusing is better than remembering."""
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)
    rep = pg_online.BackgroundRepairer(r, conn=_FakeConn(store), idle_sleep=0.05)
    for q, qn in zip(queries[:5], qs_new[:5]):
        r.query(q_old=q, q_new=qn)
    rep.start()
    try:
        try:
            rep.drain_now()
        except SystemExit as e:
            assert "stop() first" in str(e), e
        else:
            raise AssertionError(
                "drain_now ran while the worker thread was alive, on the "
                "connection that thread is using")
    finally:
        assert rep.stop(timeout=10.0)
    rep.drain_now()                       # fine once the thread has exited


def test_the_pg_repairer_sweeps_expired_claims_at_startup_and_on_a_schedule():
    """`reclaim_expired` was written, documented as the recovery mechanism,
    and called by nothing — and no test noticed, because the SQL was exercised
    by `checks.py` in isolation while the service never invoked it. A process
    that died holding claims then blocked those units for the life of the
    database.

    This drives the worker, not the SQL: the sweep must happen when the
    repairer starts and again on its interval, and what it reclaims must be
    counted where a run can see it.
    """
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)
    # a dead owner's claims: rows held, units still without a vector
    dead = [int(i) for i in np.argsort(-(store.V_old @ queries[0]))[:5]]
    store.claimed.update({u: "a-process-that-died" for u in dead})
    store.expire_claims(dead)

    rep = pg_online.BackgroundRepairer(r, conn=_FakeConn(store),
                                       idle_sleep=0.001, sweep_every_s=0.05)
    assert store.reclaim_calls == 0, "the sweep ran before the repairer started"
    rep.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and store.reclaim_calls < 2:
            time.sleep(0.01)
    finally:
        rep.stop(timeout=10.0)

    assert store.reclaim_calls >= 2, (
        f"the repairer swept {store.reclaim_calls} time(s): it must sweep at "
        f"startup and again on its interval")
    assert rep.reclaimed == len(dead), (rep.reclaimed, dead)
    assert not any(u in store.claimed for u in dead), (
        "the dead owner's claim rows are still there, so no other worker can "
        "ever take those units")
    # a reclaimed unit is claimable again
    assert r.claims.claim(r.conn, dead) == sorted(dead)


def test_a_document_updated_or_deleted_after_the_switch_is_handled():
    """Insert was covered; update and delete were not, and they fail
    differently. An update leaves a stale repaired vector behind that the new
    one must beat; a delete must leave nothing behind at all."""
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=16)
    for q, qn in zip(queries[:25], qs_new[:25]):
        r.query(q_old=q, q_new=qn)

    repaired = sorted(store.present)
    assert repaired, "nothing was repaired, so there is nothing to update"

    # UPDATE: the document's content changed, so its new vector replaces the
    # repaired one and the answer must reflect the new content
    victim = repaired[0]
    store.write_new_document(victim, qs_new[0])     # now exactly the query
    tr = r.query(q_old=queries[0], q_new=qs_new[0])
    assert victim in tr.results, "the updated document fell out of the answer"
    best = max(float(store.present[i] @ qs_new[0]) for i in tr.results)
    assert abs(float(store.present[victim] @ qs_new[0]) - best) < 1e-5, (
        "the answer ranked the updated document by its stale vector")

    # DELETE: gone from docs and, by cascade, from docs_new
    store.delete_document(victim)
    tr2 = r.query(q_old=queries[0], q_new=qs_new[0])
    assert victim not in tr2.results, (
        "a deleted document was still returned")
    assert victim not in set(r.queue.snapshot()), (
        "a deleted document is still queued for repair")


def test_a_dropped_unit_is_visible_in_the_summary_not_just_in_the_queue():
    """The queue drops at capacity rather than blocking, which is the right
    trade — but a drop that nothing reports turns 'by design' into a migration
    with a silent hole in it. The count has to reach the summary a run is read
    from."""
    srv, qs, _, _, _ = _online_toy(budget_m=0)
    srv.queue.capacity = 4
    for q in qs:
        srv.query(q)
    assert srv.queue.dropped > 0, "the fixture did not fill the queue"

    s = online.summarise([srv.query(q)[1] for q in qs[:5]], server=srv)
    assert s["queue_dropped"] == srv.queue.dropped > 0, s
    assert s["queue_depth_at_end"] == srv.queue.depth
    # and the coverage the run reports is below 1, so the summary cannot be
    # read as a finished migration
    assert s["window_coverage_mean"] < 1.0


def test_a_document_written_after_the_switch_is_reachable_online():
    """A document created or updated after the migration started has no old
    vector, so no old-space window can ever surface it, however deep.

    The synchronous path has always searched the partial HNSW index over
    `origin='write'` rows and joined that top-k to its candidates. The online
    path did not, so the two realizations of one policy answered different
    questions, and a document created mid-migration was invisible to the
    online one for as long as the migration ran. That is a correctness gap,
    not a tuning choice -- and 'the migration is still running' is exactly
    when a live system is still taking writes.
    """
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=8)
    # warm the repaired set so the window has something in it
    for q, qn in zip(queries[:20], qs_new[:20]):
        r.query(q_old=q, q_new=qn)

    # a brand-new document, sitting exactly on a query, with no old vector
    fresh = 10_000
    store.write_new_document(fresh, qs_new[0])
    tr = r.query(q_old=queries[0], q_new=qs_new[0])

    assert fresh not in set(
        i for i in range(len(store.V_old))), "the fixture id collides"
    assert tr.n_written >= 1, (
        "the partial index over post-switch writes was not searched")
    assert fresh in tr.results, (
        "a document written after the switch never reached the answer: no "
        "old-space window contains it, so only the write index can")
    # It cannot be asserted to rank *first*: the fixture builds each query as
    # an existing corpus vector, so that unit scores 1.0 too and the two tie.
    # What matters is that it is reachable and scores at the top.
    assert tr.results.index(fresh) <= 1, tr.results[:3]
    best = max(float(store.present[i] @ qs_new[0]) for i in tr.results)
    assert abs(float(store.present[fresh] @ qs_new[0]) - best) < 1e-5, (
        "it should score at the top, within float32 rounding of the tied unit")


def test_a_failing_query_does_not_leave_its_encode_thread_running():
    """The overlapped encode is started before the window read and joined
    after the repairs. Every path between those two points used to leak it: a
    window or presence read that raised left a daemon thread computing an
    answer nobody would read, and under a database fault every failing request
    would start another, so the device filled with work for discarded queries
    exactly when it was least able to spare it."""
    pg_online, r, store, _c, queries, _qn = _pg_online_fixture(budget_m=0)
    running = threading.Event()
    done = threading.Event()

    class _SlowEncoder:
        def encode(self, texts):
            running.set()
            time.sleep(0.3)
            done.set()
            return np.zeros((1, 24), dtype=np.float32)

    class _BrokenStore:
        def answer(self, sql, params):
            if "ORDER BY emb_old" in sql:
                running.wait(5.0)
                raise RuntimeError("server closed the connection unexpectedly")
            return store.answer(sql, params)

    r.conn = _FakeConn(_BrokenStore())
    r.enc_new_query = _SlowEncoder()
    before = threading.active_count()
    try:
        r.query(text="a query", q_old=queries[0])
    except RuntimeError:
        pass
    else:
        raise AssertionError("the broken window read should have raised")
    assert done.is_set(), (
        "query returned while its encode thread was still running: the thread "
        "was never joined on the failing path")
    assert r.abandoned_encodes == 0
    assert threading.active_count() <= before, "a thread was left behind"


def test_the_pg_repairer_returns_a_batch_it_could_not_repair():
    """The same ownership rule against the schema, and here the connection
    itself can raise: `self.conn` opens lazily and `claims.claim` runs a
    statement, both of which sat OUTSIDE the guarded region, so a database
    blip dropped a batch that had already left the queue."""
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)

    class _BrokenConn:
        closed = False
        threads: set = set()
        violations: list = []

        def execute(self, sql, params=()):
            raise RuntimeError("server closed the connection unexpectedly")

    rep = pg_online.BackgroundRepairer(r, conn=_BrokenConn(), idle_sleep=0.001)
    for q, qn in zip(queries, qs_new):
        r.query(q_old=q, q_new=qn)
    wanted = set(r.queue.snapshot())
    assert wanted, "the fixture queued nothing"

    rep.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and rep.errors < 3:
        time.sleep(0.005)
    assert rep.stop(timeout=10.0)
    assert rep.errors >= 3 and rep.encoded == 0
    survived = set(r.queue.snapshot()) | set(r.queue.quarantined)
    assert not (wanted - survived), (
        f"{len(wanted - survived)} unit(s) were dropped when the connection "
        f"failed")
    assert not store.claimed, "a claim row was left behind by a failed batch"


def test_the_pg_repairer_will_not_close_a_connection_still_in_use():
    """`close()` after a timed-out stop would close the connection the live
    thread is mid-statement on, which surfaces from inside a serving request
    as a corrupt result or an InterfaceError."""
    pg_online, r, store, _c, queries, qs_new = _pg_online_fixture(budget_m=0)
    rconn = _FakeConn(store)
    rep = pg_online.BackgroundRepairer(r, conn=rconn, idle_sleep=0.001)
    held = threading.Event()
    inner = store.encode_new

    def hang(ts):
        held.set()
        time.sleep(3.0)
        return inner(ts)

    store.encode_new = hang
    r.encode_new = hang
    for q, qn in zip(queries[:4], qs_new[:4]):
        r.query(q_old=q, q_new=qn)
    rep.start()
    try:
        assert held.wait(10.0), "the repairer never reached the encoder"
        try:
            rep.close(timeout=0.05)
        except SystemExit as e:
            assert "still running" in str(e), e
        else:
            raise AssertionError("close() closed a connection still in use")
        assert not rconn.closed, "the connection was closed under a live thread"
    finally:
        store.encode_new = inner
        r.encode_new = inner
        rep.close(timeout=20.0)
    # a connection the repairer was handed is not its to close; what close()
    # guarantees is that the thread is gone first
    assert rep._thread is None and not rconn.closed


# ------------------------------------------------- the sort key it measured ---
def test_the_backfill_records_the_sort_key_it_measured_not_the_one_it_asked_for():
    """The character-key bug, moved into the metadata.

    `_token_lengths` fell back to character counts on any failure -- and for a
    `tei:` encoder it failed every time, since a URL is not a model name --
    while the results file recorded `backfill_sort_key="token"` from the
    request flag. A padding factor measured on characters and labelled tokens
    is the exact confusion D-02 exists to clear up.
    """
    from thaw_pg import backfill

    texts = ["a short one", "a considerably longer passage of text " * 4]
    lens, key, why = backfill._token_lengths("tei:http://gpu:8080", texts)
    assert key == "char" and why, (key, why)
    assert lens == [len(t) for t in texts]
    lens2, key2, why2 = backfill._token_lengths("local:not/a/real/model", texts)
    assert key2 == "char" and why2, "a missing model should not report tokens"

    import ast
    src = (Path(__file__).resolve().parent.parent / "thaw_pg"
           / "backfill.py").read_text(encoding="utf-8")
    assert "backfill_sort_key=sort_key" in src, (
        "the results file still records the requested key, not the measured one")
    assert "backfill_sort_key_fallback" in src, (
        "the fallback reason does not reach the results file")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "allow_char_fallback" in names, (
        "there is no way to say the fallback is acceptable, so the refusal "
        "cannot be the default")
    raises = [n for n in ast.walk(fn)
              if isinstance(n, ast.Raise) and "SystemExit" in ast.dump(n)]
    assert raises, "run() never refuses, so it can still measure the wrong key"


def test_the_documented_test_counts_are_the_real_ones():
    """Four documents carried a stale count as current state — `RUNTESTS.md`
    still said 145 while the suite was past 180, and still listed a Windows
    failure that had been fixed. A reader then cannot tell which number to
    accept the package against, and the most natural reading is that the
    suite is failing.

    `README.md` is the one place that states expected counts; this asserts it
    agrees with the suite, so the drift cannot come back quietly. Everything
    else points at it or marks its number as of a date.
    """
    import ast as _ast
    import re as _re

    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md")
    if not readme.is_file():
        raise Skip("README.md is not in this checkout")

    def count_tests(path):
        tree = _ast.parse(path.read_text(encoding="utf-8"), str(path))
        return sum(1 for n in tree.body
                   if isinstance(n, _ast.FunctionDef)
                   and n.name.startswith("test_"))

    mine = count_tests(Path(__file__))
    text = readme.read_text(encoding="utf-8")
    m = _re.search(r"semview\.tests\s+#\s*(\d+) passed, (\d+) skipped, "
                   r"(\d+) failed of (\d+)", text)
    assert m, "README.md no longer states the expected semview counts"
    passed, skipped, failed, total = (int(g) for g in m.groups())
    assert total == mine, (
        f"README.md says {total} semview tests; there are {mine}")
    assert passed + skipped + failed == total, (m.group(0), mine)
    assert failed == 0, "a failing test is being documented as expected"

    audit = root / "thaw_audit" / "tests.py"
    if audit.is_file():
        a = _re.search(r"thaw_audit\.tests\s+#\s*(\d+) passed", text)
        assert a, "README.md no longer states the expected audit counts"
        assert int(a.group(1)) == count_tests(audit), (
            f"README.md says {a.group(1)} audit tests; there are "
            f"{count_tests(audit)}")

    # No other document may state a pass/skip line at all. HANDBOOK.md and
    # README.md once gave different passed/skipped splits for the same total,
    # and this test did not notice because it only read README.md -- so "one
    # source of truth" was a sentence in a README rather than a property.
    #
    # Every markdown file in the tree is scanned, not a list of names. The list
    # was HANDBOOK/ONBOARDING/RUNTESTS, and RUNBOOK.md -- written later, and the
    # document a reader is told to open first -- was not on it, so its gate
    # said "expect 225 passed, 6 skipped, 0 failed of 231" while the suite was
    # at 228 of 234. An enumerated allow-list of places to check is the same
    # defect as a vacuous gate: it passes by not looking.
    # The one exemption is the one this docstring already allows: a count
    # presented as a dated record rather than as current state. That is a
    # property of the text -- a date on the line or on the section heading above
    # it -- and not a list of filenames, so an audit round may say what the
    # suite was on the day it ran without becoming a second source of truth.
    stray_md = []
    for p2 in sorted(root.rglob("*.md")):
        if p2.name == "README.md":
            continue
        body = p2.read_text(encoding="utf-8")
        lines = body.splitlines()
        for i, line in enumerate(lines):
            m2 = _re.search(r"(\d+) passed, (\d+) skipped", line)
            if not m2:
                continue
            # Walk up through the enclosing headings, not just the nearest one:
            # a dated `##` round contains undated `###` subsections, and they
            # inherit its date. Caught by this rule on its first run, when this
            # file's own changelog quoted RUNBOOK's stale line as evidence.
            dated = _re.compile(r"20\d\d-\d\d-\d\d")
            headings, depth = [], 99
            for j in range(i, -1, -1):
                if lines[j].startswith("#"):
                    level = len(lines[j]) - len(lines[j].lstrip("#"))
                    if level < depth:
                        headings.append(lines[j])
                        depth = level
                    if level <= 2:
                        break
            if dated.search(line) or any(dated.search(h) for h in headings):
                continue
            stray_md.append(f"{p2.relative_to(root).as_posix()}:{i + 1}: {m2.group(0)}")
    assert not stray_md, (
        f"these documents state their own test counts as current state: "
        f"{stray_md}. README.md is the only place that may, or they drift -- "
        f"point at it, or date the line.")

    # and no other document may state a count as the current one
    for name in ("RUNTESTS.md", "HANDBOOK.md", "docs/history/README_8M.md",
                 "docs/history/README_THROUGHPUT.md"):
        p = root / name
        if not p.is_file():
            continue
        body = p.read_text(encoding="utf-8")
        for claim in _re.findall(r"the suite is (?:now )?\*{0,2}(\d+)\*{0,2}", body):
            assert int(claim) == mine, (
                f"{name} states the suite is {claim} tests, which is stale; "
                f"point at README.md or mark it as of a date")
        # A layout listing is the other place a count hides: HANDBOOK.md said
        # "tests.py  199 property tests" long after the suite passed 200, and
        # neither rule above looks at that shape.
        for claim in _re.findall(r"(\d+)\s+(?:property\s+)?tests\b", body):
            assert int(claim) == mine, (
                f"{name} states a count of {claim} tests, which is stale; "
                f"describe the file and leave the number to README.md")


def test_no_text_file_is_read_or_written_in_the_platform_default_codec():
    """Python's default encoding is the locale's, not UTF-8.

    Every document in this tree is UTF-8 and several contain an em dash, so a
    bare `read_text()` raises `UnicodeDecodeError` the moment the suite runs
    on a machine whose code page is not UTF-8 — a Chinese Windows install, for
    one, where a reviewer hit exactly that. The failure has nothing to do with
    the thing under test, which makes it the worst kind: it looks like the
    project is broken on their machine.

    Binary opens and `urlopen` are exempt; everything else names its codec.
    """
    import ast as _ast

    root = Path(__file__).resolve().parent.parent
    bad = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        for n in _ast.walk(_ast.parse(p.read_text(encoding="utf-8"), str(p))):
            if not isinstance(n, _ast.Call):
                continue
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name not in {"read_text", "write_text", "open"}:
                continue
            if "encoding" in {k.arg for k in n.keywords}:
                continue
            if (name == "open" and len(n.args) >= 2
                    and isinstance(n.args[1], _ast.Constant)
                    and "b" in str(n.args[1].value)):
                continue
            if (isinstance(f, _ast.Attribute)
                    and getattr(getattr(f, "value", None), "attr", "") == "request"):
                continue
            bad.append(f"{p.relative_to(root)}:{n.lineno} {name}()")
    assert not bad, (
        f"{len(bad)} text call(s) use the platform default codec: "
        f"{bad[:5]}")


def test_a_manifest_checks_the_files_next_to_it_not_the_ones_in_the_cwd():
    """A manifest's entries are relative to the manifest.

    `verify` preferred a file of the same name in the working directory, so
    verifying `evidence/MANIFEST.sha256` from the project root checked the
    project's own `README.md` against the bundle's and reported the evidence
    bundle as corrupt. Two files sharing a common name is not a damaged
    archive, and "your evidence is broken" is the most expensive wrong answer
    this tool can give.
    """
    import subprocess
    import tempfile

    root = Path(tempfile.mkdtemp())
    bundle = root / "evidence"
    bundle.mkdir()
    (bundle / "README.md").write_text("the bundle's own readme\n", encoding="utf-8")
    (bundle / "r.json").write_text('{"v": 1}', encoding="utf-8")
    here = Path(__file__).resolve().parent.parent
    man = subprocess.run(
        [sys.executable, "-m", "semview.results_io", "manifest",
         str(bundle / "README.md"), str(bundle / "r.json")],
        capture_output=True, text=True, cwd=str(here))
    assert man.returncode == 0, man.stderr
    # entries are written by basename, which is what makes the cwd a hazard
    lines = [l.split(None, 2) for l in man.stdout.splitlines() if l.strip()]
    (bundle / "MANIFEST.sha256").write_text(
        "".join(f"{d}  {s}  {Path(n).name}\n" for d, s, n in lines),
        encoding="utf-8")

    # a DIFFERENT README.md in the directory the command is run from
    (root / "README.md").write_text("the project's readme, a different length\n",
                                    encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "semview.results_io", "verify",
         "evidence/MANIFEST.sha256"],
        capture_output=True, text=True,
        cwd=str(root), env={**os.environ, "PYTHONPATH": str(here)})
    assert out.returncode == 0, (
        f"the bundle was reported broken because the cwd had a file of the "
        f"same name:\n{out.stdout}{out.stderr}")
    assert "FAIL" not in out.stdout, out.stdout


def test_every_per_run_manifest_in_the_shipped_bundle_verifies_on_its_own():
    """A flattened export renames `<run>/x.json` to `<run>__x.json`, so the
    manifest written inside the run directory names files that no longer
    exist under those names. Every entry then read as missing, and a reviewer
    checking one run at a time saw a bundle reporting itself as destroyed.

    The prefix is recoverable from the manifest's own filename. The export is
    also results-only — `command.txt`, `env.json` and `log.txt` stay behind —
    so absence there is expected and is reported as absence; a file that is
    present and does not match is still a failure, and `--strict` restores the
    old meaning for checking a complete run directory.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    mans = sorted((root / "evidence").glob("*__MANIFEST.sha256"))
    if not mans:
        raise Skip("the flattened evidence bundle is not in this checkout")
    for m in mans:
        out = subprocess.run(
            [sys.executable, "-m", "semview.results_io", "verify",
             str(m.relative_to(root))],
            capture_output=True, text=True, cwd=str(root))
        assert out.returncode == 0, f"{m.name} does not verify:\n{out.stdout}"
        assert "FAIL" not in out.stdout, f"{m.name}:\n{out.stdout}"
        assert "  ok    " in out.stdout, (
            f"{m.name} verified nothing at all, which is not a pass:\n"
            f"{out.stdout}")
    strict = subprocess.run(
        [sys.executable, "-m", "semview.results_io", "verify", "--strict",
         str(mans[0].relative_to(root))],
        capture_output=True, text=True, cwd=str(root))
    assert strict.returncode == 1, (
        "--strict must still call an absent file a failure")


def test_every_destructive_thaw_pg_command_passes_the_guard():
    """`load` dropped four tables with no guard, for eight rounds.

    `load.run` starts with `DROP TABLE IF EXISTS claim_done, repair_claims,
    docs_new, docs CASCADE`, and `thaw_pg load` called it straight. Only
    `quick` and `selftest2` were guarded. Two of those eight rounds were spent
    calling this exact shape the worst thing in the package — about
    `sql_check`, a file written weeks later — while the older and more
    dangerous instance sat unread, in the command most likely to be pointed at
    a database that already holds a corpus.

    So the check is not "does `load` call the guard" but "does every command
    that can destroy data call it", enumerated from the source rather than
    from memory.
    """
    import ast as _ast

    root = Path(__file__).resolve().parent.parent / "thaw_pg"
    destructive = {}
    for p in sorted(root.glob("*.py")):
        if p.name in ("sql_check.py", "integration.py"):
            continue            # their own isolation; covered by their tests
        src = p.read_text(encoding="utf-8")
        for kw in ("DROP TABLE", "TRUNCATE", "DROP SCHEMA"):
            if kw in src:
                destructive.setdefault(p.stem, set()).add(kw)
    assert destructive, "no destructive SQL found at all -- has the schema moved?"

    main = (root / "__main__.py").read_text(encoding="utf-8")
    tree = _ast.parse(main)
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "main")

    # every `if a.cmd == "x"` branch, and whether a guard(...) call is inside
    unguarded = []
    for node in _ast.walk(fn):
        if not isinstance(node, _ast.If):
            continue
        cmds = {c.value for c in _ast.walk(node.test)
                if isinstance(c, _ast.Constant) and isinstance(c.value, str)}
        if not cmds:
            continue
        # The qualified callee name, built from the AST. A first version
        # searched `ast.dump(n)` for the string "load.run", which never
        # appears -- the dump spells it `Attribute(value=Name(id='load'),
        # attr='run')` -- so nothing was ever found destructive and the test
        # passed over the very bug it was written for. Verified by removing
        # the guard again and watching this fail.
        def _callee(n):
            f = getattr(n, "func", None)
            if isinstance(f, _ast.Attribute) and isinstance(f.value, _ast.Name):
                return f"{f.value.id}.{f.attr}"
            if isinstance(f, _ast.Attribute):
                return f.attr
            if isinstance(f, _ast.Name):
                return f.id
            return ""

        destructive_calls = {"load.run", "checks.run", "step2.run",
                             "step2.selftest2", "selftest2"}
        reaches = any(isinstance(n, _ast.Call) and _callee(n) in destructive_calls
                      for n in _ast.walk(node))
        if not reaches:
            continue
        guarded = any(isinstance(n, _ast.Call)
                      and getattr(n.func, "id", "") == "guard"
                      for n in _ast.walk(node))
        if not guarded:
            unguarded.append(sorted(cmds))
    assert not unguarded, (
        f"command branch(es) that reach destructive SQL without calling "
        f"guard(): {unguarded}")


def test_the_destructive_guard_refuses_when_it_cannot_see_the_database():
    """It used to warn 'assuming no real data is at risk' and continue -- the
    one assumption the guard exists to avoid. A DSN pointing at a database
    that is briefly down, or a pooler hiccup, would let a DROP through on a
    live corpus, since the guarded command connects again moments later."""
    import ast
    src = (Path(__file__).resolve().parent.parent / "thaw_pg"
           / "__main__.py").read_text(encoding="utf-8")
    assert "assuming no real data" not in src, (
        "the guard still assumes an unreachable database is empty")
    tree = ast.parse(src)
    guarded = [n for n in ast.walk(tree)
               if isinstance(n, ast.If) and "db_error" in ast.dump(n.test)
               and any(isinstance(b, ast.Raise) for b in ast.walk(n))]
    assert guarded, (
        "no branch raises on a failed database inspection: the guard does not "
        "fail closed")


# ------------------------------------------------------- chained upgrades ---
def _chain_fixture(N=4000, d=32, legs=3, seed=0):
    ch = _load_analysis("chain")
    Vs, Qs = ch.synthetic_chain(N, d, legs, alpha=0.5, n_queries=600, seed=seed)
    return ch, Vs, Qs


def test_chain_drift_is_calibrated_to_a_real_pair_and_decays_along_the_chain():
    """The synthetic chain stands in for real model generations, so one
    generation must look like the measured MiniLM -> BGE pair and agreement
    must fall with distance -- otherwise the ageing-index effect it is built to
    show would be an artefact of the construction."""
    ch, Vs, Qs = _chain_fixture(N=6000, d=48, legs=3)
    k = 10
    truth = [migration.topk_chunked(Qs[t], Vs[t], k) for t in range(len(Vs))]
    w = migration.topk_chunked(Qs[0], Vs[0], 500)
    rec = [float(np.mean([len(set(a) & set(b)) / k for a, b in zip(truth[t], w)]))
           for t in range(1, len(Vs))]
    # one generation near the real 0.929 at K'=500, and monotone decay after
    assert 0.80 <= rec[0] <= 0.99, rec
    assert all(rec[i] > rec[i + 1] for i in range(len(rec) - 1)), rec


def test_chain_union_never_loses_to_the_literal_design_and_costs_more():
    """Per leg and per depth: a union window is a superset of index 0's window,
    so its candidate recall cannot be lower and its passes cannot be fewer.
    Both directions are checked, because the second is the reason the table
    needs a matched-cost control rather than a single number."""
    ch, Vs, Qs = _chain_fixture()
    rows, meta = ch.run_chain(Vs, Qs, k=10, kprimes=(100, 300), rho=0.35,
                              eval_queries=60)
    assert meta["T"] == 3
    # every policy serves the legs itself, at every reach depth in the sweep,
    # so the leg register holds one block of T legs per (history, depth) and
    # not one block per leg
    assert {L["policy"] for L in meta["legs"]} == {
        "original-only", "union-nearest", "union-all"}
    for pol in ("original-only", "union-nearest", "union-all"):
        for depth in meta["kprimes"]:
            block = [L for L in meta["legs"]
                     if L["policy"] == pol and L["reach_depth"] == depth]
            assert [L["leg"] for L in block] == [1, 2, 3], (pol, depth, block)
    seen = 0
    for leg in range(2, meta["T"] + 1):
        for kp in meta["kprimes"]:
            at = {r["policy"]: r for r in rows
                  if r["leg"] == leg and r["kprime"] == kp}
            lit = at["original-only"]
            for pol in ("union-nearest", "union-all"):
                assert at[pol]["recall_candidates"] >= lit["recall_candidates"] - 1e-9, \
                    (leg, kp, pol, at[pol], lit)
                assert at[pol]["passes"] >= lit["passes"], (leg, kp, pol)
                seen += 1
            # keeping every abandoned index cannot be worse than keeping one
            assert at["union-all"]["recall_candidates"] >= \
                at["union-nearest"]["recall_candidates"] - 1e-9, (leg, kp)
    assert seen >= 8


def test_chain_a_partial_index_only_nominates_inside_what_its_leg_reached():
    """The whole accounting rests on this: a previous generation's index covers
    only the units that generation's traffic repaired. If it could nominate
    outside that set the gain would be free in a way no deployment gets."""
    # Four legs, not three: with three, `union-nearest` and `union-all` have
    # reached exactly the same rows at every leg a `previous-only` row can read
    # from, so the lookup below cannot tell them apart and the test would pass
    # even if a row were scored against the wrong policy's coverage. Verified
    # by mislabelling the history on purpose, which this configuration catches
    # and a three-leg one does not.
    ch, Vs, Qs = _chain_fixture(N=8000, legs=4)
    N = len(Vs[0])
    runs = [ch.run_chain(Vs, Qs, k=10, kprimes=(100,), rho=0.3,
                         eval_queries=60),
            # both sizings, because the coverage invariant below is only
            # non-trivial under constant traffic
            ch.run_chain(Vs, Qs, k=10, kprimes=(100,), rho=0.3,
                         eval_queries=60, queries_per_leg=30)]
    for rows, meta in runs:
        _chain_partial_index_invariants(rows, meta)


def test_chain_alpha_is_calibrated_at_the_N_being_run_not_remembered():
    """The tool shipped alpha=0.5 with a docstring claiming it reproduced the
    measured 0.929. It did, at the N=20,000 it was fitted at; at the N=200,000
    the tool runs it gives 0.827, because recall@K' depends on K'/N. So a
    remembered alpha is wrong at every other N, and calibration has to happen
    where the run happens."""
    ch = _load_analysis("chain")
    # the dependence itself: same alpha, different N, materially different recall
    small = ch.one_generation_recall(4_000, 32, 0.5, 200, eval_queries=60)
    large = ch.one_generation_recall(16_000, 32, 0.5, 200, eval_queries=60)
    assert small - large > 0.04, (small, large)
    # and calibrating at each N recovers the same target at both
    for N in (4_000, 16_000):
        a, got = ch.calibrate_alpha(N, 32, 200, small, eval_queries=60)
        assert abs(got - small) < 0.02, (N, a, got, small)
        assert 0.02 < a < 1.3, a


def test_chain_the_drift_assumption_is_declared_and_changes_the_answer():
    """Nothing in a one-generation measurement constrains how the later legs
    drift, and the choice is not cosmetic: calibrated to the same first leg the
    variants put the literal design's last leg in very different places. The
    test pins that the knob is real, so it cannot quietly become a constant
    again."""
    ch = _load_analysis("chain")
    assert set(ch.DRIFTS) == {"independent", "shared-distortion", "converging",
                              "target-seeking"}
    last = {}
    for drift in ch.DRIFTS:
        a, got = ch.calibrate_alpha(6_000, 32, 200, 0.80, eval_queries=60,
                                    drift=drift)
        assert abs(got - 0.80) < 0.03, (drift, got)
        Vs, Qs = ch.synthetic_chain(6_000, 32, 3, a, n_queries=600, seed=0,
                                    drift=drift)
        rows, meta = ch.run_chain(Vs, Qs, k=10, kprimes=(200,), rho=0.3,
                                  eval_queries=60, queries_per_leg=40)
        at = {r["policy"]: r for r in rows if r["leg"] == 3 and r["kprime"] == 200}
        last[drift] = at["original-only"]["recall_at_k"]
        # the order is the part that survives the assumption
        assert at["union-all"]["recall_at_k"] >= at["union-nearest"]["recall_at_k"] \
            >= at["original-only"]["recall_at_k"] - 1e-9, (drift, at)
    assert max(last.values()) - min(last.values()) > 0.1, last
    try:
        ch.synthetic_chain(100, 8, 2, 0.5, drift="whatever")
        raise AssertionError("an unknown --drift must be refused")
    except SystemExit as e:
        assert "--drift" in str(e)


def test_chain_only_computes_the_windows_its_legs_consume():
    """A leg's windows used to be computed for the whole query pool and then
    read as far as the sizing needed -- 140x the work a rho-sized leg uses, so
    the cost of the experiment scaled with the size of the query dump rather
    than with the traffic being simulated. Blocks fix that, and the property
    worth pinning is the one that makes the fix safe: the same queries in the
    same order, so the numbers cannot move."""
    ch = _load_analysis("chain")
    Vs, Qs = ch.synthetic_chain(6000, 32, 3, 0.5, n_queries=2400, seed=0)
    real = ch._windows
    seen = {"queries": 0}

    def counting(Q, V, kprime, rows=None):
        seen["queries"] += len(Q)
        return real(Q, V, kprime, rows=rows)

    ch._windows = counting
    try:
        rows, meta = ch.run_chain(Vs, Qs, k=10, kprimes=(200,), rho=0.3,
                                  eval_queries=60)
    finally:
        ch._windows = real
    consumed = sum(L["warm_queries"] for L in meta["legs"])
    pool = (len(Qs[0]) - meta["eval_queries"]) // meta["T"]
    # the leg loop must not touch the whole pool for every policy and leg
    eager = pool * sum(L["sources_while_serving"] for L in meta["legs"])
    assert seen["queries"] < eager / 3, (seen["queries"], eager, consumed)
    assert rows and consumed > 0


def test_chain_cost_columns_do_not_double_count_the_warm_up():
    """`passes` counts every unit the evaluation windows touch, including the
    ones this leg's warm-up already repaired -- and those are already inside
    `cumulative_encoded`. Reporting both as a cost charges them twice, so the
    bill is warm-up plus the MARGINAL evaluation passes, and the marginal
    figure must never be quoted on its own: it is lowest for the widest policy
    precisely because that policy's warm-up already paid."""
    ch = _load_analysis("chain")
    Vs, Qs = ch.synthetic_chain(6_000, 32, 3, 0.5, n_queries=600, seed=0)
    rows, meta = ch.run_chain(Vs, Qs, k=10, kprimes=(200,), rho=0.3,
                              eval_queries=60, queries_per_leg=40)
    seen = 0
    for r in rows:
        assert r["eval_passes_marginal"] <= r["passes"], r
        assert r["total_passes"] == r["cumulative_encoded"] + r["eval_passes_marginal"]
        if r["eval_passes_marginal"] < r["passes"]:
            seen += 1
    assert seen, "no row had any already-repaired unit, so nothing was tested"
    # the wide policy really is the one the marginal figure would flatter
    at = {r["policy"]: r for r in rows
          if r["leg"] == meta["T"] and r["kprime"] == 200}
    assert at["union-all"]["passes"] > at["original-only"]["passes"]
    assert at["union-all"]["total_passes"] > at["original-only"]["total_passes"]


def test_chain_a_window_inside_a_partial_index_stays_inside_it():
    """The containment the whole accounting rests on, checked where it lives.
    The cardinality checks elsewhere are a proxy: a window that escaped its
    partial index would also be a bigger window, but a leg that reaches more
    has a bigger `reached` too, so counting cannot tell the two apart."""
    ch, Vs, Qs = _chain_fixture(N=4000, legs=3)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(Vs[0]), 900, replace=False))
    allowed = set(rows.tolist())
    wins = ch._windows(Qs[1][:25], Vs[1], 120, rows=rows)
    assert len(wins) == 25
    for w in wins:
        assert len(w) == 120, len(w)
        assert set(np.asarray(w).tolist()) <= allowed, "a window escaped"
    # and a partial index narrower than K' clamps rather than reaching outside
    narrow = np.sort(rng.choice(len(Vs[0]), 40, replace=False))
    for w in ch._windows(Qs[1][:5], Vs[1], 120, rows=narrow):
        assert set(np.asarray(w).tolist()) <= set(narrow.tolist())
        assert len(w) == 40


def _chain_partial_index_invariants(rows, meta):
    prev = [r for r in rows if r["policy"] == "previous-only"]
    assert prev, "no previous-only row was produced"
    for r in prev:
        # It cannot touch more units than the leg before it reached -- and the
        # leg before it is the one `union-nearest` served, because serving
        # from the most recent partial index alone is a diagnostic rather than
        # a deployment (it could never repair outside its own coverage, so its
        # reached set would shrink every leg). The expected history is written
        # out here rather than read from the row: taking the row's own word for
        # it passes even when the row is scored against another policy's
        # coverage, which is the one mislabelling that matters.
        assert r["history"] == "union-nearest", r
        src = [L for L in meta["legs"] if L["policy"] == "union-nearest"
               and L["leg"] == r["leg"] - 1
               and L["reach_depth"] == r["reach_depth"]]
        assert len(src) == 1, (r["leg"], r["reach_depth"], src)
        reached = src[0]["reached"]
        assert r["passes"] <= reached, (r["leg"], r["passes"], reached)
        assert r["partial_rows_kept"] == reached
    # Each policy pays for its own coverage. Under constant traffic a policy
    # serving from more indexes reaches more per query, so by the last leg
    # union-all must have reached STRICTLY more than union-nearest, which must
    # have reached strictly more than the literal design. Strictly: `>=` holds
    # when union-all is degraded to nominate from one index, so it would not
    # notice the policies collapsing into each other.
    if meta.get("queries_per_leg"):
        for depth in meta["kprimes"]:
            def reach(pol, t):
                return next(L["reached"] for L in meta["legs"]
                            if L["policy"] == pol and L["leg"] == t
                            and L["reach_depth"] == depth)
            T_ = meta["T"]
            for t in range(1, T_ + 1):
                assert reach("union-all", t) >= reach("union-nearest", t) \
                    >= reach("original-only", t), (t, depth)
            assert reach("union-all", T_) > reach("union-nearest", T_) \
                > reach("original-only", T_), (T_, depth)


def test_chain_traffic_is_sized_to_the_reached_target_and_streams_are_disjoint():
    ch, Vs, Qs = _chain_fixture()
    N = len(Vs[0])
    n = ch.traffic_for_rho(Qs[0], Vs[0], 100, 0.3, N)
    wins = migration.topk_chunked(Qs[0][:n], Vs[0], 100)
    got = len(set(np.asarray(wins).ravel().tolist())) / N
    assert got >= 0.3, got
    one_fewer = migration.topk_chunked(Qs[0][:n - 1], Vs[0], 100)
    assert len(set(np.asarray(one_fewer).ravel().tolist())) / N < 0.3
    # an unreachable target must fail loudly, not silently return the stream
    try:
        ch.traffic_for_rho(Qs[0][:5], Vs[0], 10, 0.9, N)
        raise AssertionError("an unreachable --rho target must raise")
    except SystemExit as e:
        assert "short of" in str(e)
    rows, meta = ch.run_chain(Vs, Qs, k=10, kprimes=(100,), rho=0.3,
                              eval_queries=60)
    # every leg's traffic comes from its own slice above the evaluation stream
    assert sum(L["warm_queries"] for L in meta["legs"]) <= len(Qs[0]) - meta["eval_queries"]
    assert meta["eval_load_x"] == round(60 * 100 / len(Vs[0]), 3)


def test_chain_refuses_a_chain_too_short_to_be_one():
    ch, Vs, Qs = _chain_fixture(legs=1)
    try:
        ch.run_chain(Vs, Qs, k=10, kprimes=(100,), rho=0.3, eval_queries=60)
        raise AssertionError("two spaces are one migration, not a chain")
    except SystemExit as e:
        assert "chain" in str(e)


def test_a_module_that_has_never_run_is_documented_as_never_having_run():
    """D-07 is open because the PostgreSQL online path has not been run against
    a real server and GPU. For as long as that is true the layout must say so
    next to the file, or a reader reasonably concludes the opposite --- which
    is what "the deployable path" next to a numpy simulation did. This ties the
    claim to the open item instead of to someone's memory.
    """
    import re as _re
    root = Path(__file__).resolve().parent.parent
    status, hand = root / "PROJECT_STATUS.md", root / "HANDBOOK.md"
    if not (status.is_file() and hand.is_file()):
        raise Skip("PROJECT_STATUS.md / HANDBOOK.md not in this checkout")
    st = status.read_text(encoding="utf-8")
    open_d07 = bool(_re.search(r"\*\*D-07\*\*|D-07", st))
    if not open_d07:
        return          # closed: the requirement lapses with it
    body = hand.read_text(encoding="utf-8")
    m = _re.search(r"^\s*online\.py\s+the same policy.*$(?:\n\s+.*)*",
                   body, _re.M)
    assert m, "HANDBOOK.md no longer describes thaw_pg/online.py in the layout"
    assert "NOT run" in m.group(0) and "D-07" in m.group(0), (
        "D-07 is open, so the layout entry for thaw_pg/online.py must say it "
        f"has not been run; it says: {m.group(0)[:160]!r}")
    assert "deployable" not in m.group(0)


def test_the_evidence_collector_never_empties_the_bundle_it_cannot_refill():
    """It deleted every `*__*` file up front and re-copied what it could
    resolve. Run against a tree that has the bundle but not the run
    directories it was built from -- a distributed tarball, which is what a
    reviewer has -- and it emptied the only copy of the evidence. That is the
    exact failure the provenance machinery exists to prevent, committed by the
    provenance machinery.

    Two properties: with nothing resolvable it must touch nothing and fail, and
    it must not treat the bundle's own files as leftovers when the bundle is
    itself a source root.
    """
    import json as _json
    import os as _os
    import shutil as _shutil
    import subprocess as _sub
    import sys as _sys
    import tempfile as _tf

    root = Path(migration.__file__).resolve().parent.parent
    script = root / "analysis" / "collect_evidence.py"
    if not script.is_file():
        raise Skip("analysis/collect_evidence.py is not in this checkout")
    tmp = Path(_tf.mkdtemp())
    bundle = tmp / "evidence"
    bundle.mkdir()
    precious = bundle / "20260101-000000_somerun__result.json"
    precious.write_text(_json.dumps({"fingerprint": {"a": 1}}), encoding="utf-8")
    (bundle / "README.md").write_text("bundle\n", encoding="utf-8")
    empty_root = tmp / "no_runs"
    empty_root.mkdir()

    env = {**_os.environ, "PYTHONPATH": str(root), "THAW_MODE": "measure"}
    r = _sub.run([_sys.executable, str(script), "--runs-root", str(empty_root),
                  "--out", str(bundle)], capture_output=True, text=True, env=env)
    assert r.returncode != 0, "collecting nothing must fail, not succeed quietly"
    assert precious.is_file(), "the bundle was emptied when nothing resolved"
    assert "refusing" in (r.stdout + r.stderr).lower()

    # And with the bundle as its own source root, nothing of its own is pruned.
    #
    # This phase used to put nothing resolvable in the bundle, so the collector
    # took the early "refusing to touch" return and the `returncode == 0` branch
    # below never ran: `precious.is_file()` passed because the script had bailed
    # out, not because the self-pruning guard worked. The guard was untested
    # while the test named it. So the bundle is now given a file the ledger
    # actually cites, which carries the run past the refusal and into the prune
    # step that has to recognise its own source root.
    import importlib.util as _ilu
    import shutil as _shu
    _sp = _ilu.spec_from_file_location("_ce", script)
    _ce = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_ce)
    # The collector resolves `<run dir>/<file>.json`, so a flat name in the
    # bundle resolves nothing. Copy in the real run directory its first claim
    # cites; without it this phase cannot reach the prune step and would be
    # back to passing on the early return.
    wanted = Path(_ce.CLAIMS[0][1]).parts[0]
    donor = root / "results" / "runs" / wanted
    if not donor.is_dir():
        raise Skip(f"results/runs/{wanted} is not in this tree, so the "
                   "collector cannot be driven past its refusal")
    _shu.copytree(donor, bundle / wanted)
    r2 = _sub.run([_sys.executable, str(script), "--runs-root", str(bundle),
                   "--out", str(bundle)], capture_output=True, text=True, env=env)
    assert precious.is_file(), "a bundle used as a source root pruned itself"
    out2 = (r2.stdout + r2.stderr)
    assert "not pruning" in out2, (
        "the collector did not reach its self-pruning guard, so this phase is "
        f"not testing what it names (exit {r2.returncode}): {out2.strip()[:200]}")
    _shutil.rmtree(tmp, ignore_errors=True)


def _depth_stub(overlap: float, kprimes, n=4000, d=8):
    """A fake pgvector that returns FULL windows of a chosen quality.

    The defect these tests cover is that `check_depth` measured window quality,
    printed it, and then gated on row count alone -- so a stub returning K'
    rows of any quality at all used to pass. The stub therefore has to be able
    to return a full window that is mostly wrong, which a real server would do
    only with a badly built index.
    """
    from thaw_pg import checks as ck

    V = np.random.default_rng(0).normal(size=(n, d)).astype(np.float32)
    Q = np.random.default_rng(1).normal(size=(40, d)).astype(np.float32)
    saved = {k: getattr(ck, k) for k in
             ("ef_search_max", "has_iterative", "configure", "window")}

    def _window(conn, q, k):
        # the first `overlap` fraction of the exact window, padded with rows
        # that are certainly not in it: a full window of controlled quality
        ex = ck.exact_topk(np.asarray([q]), V, k)[0].tolist()
        take = int(round(overlap * k))
        pad = [i for i in range(n - 1, -1, -1) if i not in set(ex)][:k - take]
        return ex[:take] + pad, 1.0

    ck.ef_search_max = lambda conn: 1000
    ck.has_iterative = lambda conn: False
    ck.configure = lambda *a, **kw: None
    ck.window = _window
    return ck, V, Q, saved, kprimes

def test_the_depth_gate_refuses_a_full_window_of_poor_quality():
    """A window returning K' rows that overlap the exact window by 0.50 is not
    a usable window: step 2 measures cross-space recall on top of it, so a gate
    that passes here silently caps every number downstream."""
    ck, V, Q, saved, kps = _depth_stub(0.50, (200,))
    try:
        r = ck.check_depth(None, V, Q, kps, n_queries=10, max_scan=0,
                           gate_kprimes=(200,), min_overlap=0.90)
        assert r["rows"], r
        assert all(x["full_window"] for x in r["rows"]), "stub must return full windows"
        assert r["overlap_by_kprime"][200] < 0.90, r["overlap_by_kprime"]
        assert r["pass_"] is False, (
            "the gate passed a full window whose overlap with the exact window "
            "is below min_overlap: it is gating on row count, not quality")
        assert r["usable_mode_by_kprime"][200] is None
    finally:
        for k, v in saved.items():
            setattr(ck, k, v)

def test_the_depth_gate_passes_a_full_window_of_good_quality():
    """The companion direction, or tightening the gate would just be a refusal
    to measure anything."""
    ck, V, Q, saved, kps = _depth_stub(1.0, (200,))
    try:
        r = ck.check_depth(None, V, Q, kps, n_queries=10, max_scan=0,
                           gate_kprimes=(200,), min_overlap=0.90)
        assert r["pass_"] is True, r
        assert r["usable_mode_by_kprime"][200] == "plain"
    finally:
        for k, v in saved.items():
            setattr(ck, k, v)

def test_the_depth_gate_fails_closed_when_a_gate_depth_never_ran():
    """`all([])` is True, so a corpus too small for every gate depth used to
    pass this gate having checked nothing at all. A gate that quietly narrows
    itself to whatever happened to fit is the same defect one level down."""
    ck, V, Q, saved, _ = _depth_stub(1.0, (200, 500, 1000), n=300)
    try:
        # n=300, so the k > len(V)//2 guard skips every gate depth
        r = ck.check_depth(None, V, Q, (200, 500, 1000), n_queries=10,
                           max_scan=0, gate_kprimes=(200, 500, 1000))
        assert r["rows"] == [], "the guard should have skipped every depth"
        assert r["gated_kprimes"] == []
        assert r["gate_kprimes_absent"] == [200, 500, 1000]
        assert r["pass_"] is False, "the gate passed without measuring a depth"
    finally:
        for k, v in saved.items():
            setattr(ck, k, v)

def test_the_chain_matched_cost_control_never_spends_less_than_the_union_row():
    """Section 7's branch criterion lives or dies on this control, and the rule
    has twice been reported as a bug with a proposed fix to a +-5% window.

    It is deliberate. Literal recall and literal cost are both monotone in K',
    so the pool of literal rows spending at least as much as a union row is the
    deepest-K' tail, and beating the best of that pool is strictly HARDER than
    beating a row of about the same price. A window admits controls spending
    LESS than the union row, which is the unfair comparison the proposal was
    trying to prevent -- on the rho run it converts an UNTESTED row
    (union-nearest at K'=1000, 198.1% of N) into a win over a control at
    192.3%.

    So what must hold is a direction: no control spends less than the row it
    vindicates. The monotonicity the argument rests on is asserted too, because
    the argument collapses if it ever stops holding.
    """
    import json as _js
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "evidence").glob("*chain*.json"))
    if not files:
        raise Skip("no chain run in the evidence bundle")
    checked = 0
    for f in files:
        obj = _js.loads(f.read_text(encoding="utf-8"))
        for sw in obj.get("sweeps", []):
            rows = sw.get("rows") or []
            if not rows or "total_passes" not in rows[0]:
                continue           # pre-rename schema; a later run supersedes it
            last = max(r["leg"] for r in rows)
            lits = sorted([r for r in rows
                           if r["policy"] == "original-only" and r["leg"] == last],
                          key=lambda c: c["kprime"])
            if len(lits) < 2:
                continue
            for i in range(len(lits) - 1):
                assert lits[i]["recall_at_k"] <= lits[i + 1]["recall_at_k"], (
                    f"{f.name}: literal recall is not monotone in K', so the "
                    "control pool's best row is no longer its deepest and the "
                    "rule's justification is gone")
                assert lits[i]["total_passes"] <= lits[i + 1]["total_passes"], (
                    f"{f.name}: literal cost is not monotone in K'")
            for r in rows:
                if not r["policy"].startswith("union") or r["leg"] != last:
                    continue
                pool = [c for c in lits if c["total_passes"] >= r["total_passes"]]
                assert all(c["total_passes"] >= r["total_passes"] for c in pool), (
                    "a control spends less than the row it controls for")
                if pool:
                    assert max(c["recall_at_k"] for c in pool) == \
                        max(c["recall_at_k"] for c in lits if c in pool), pool
                checked += 1
    if not checked:
        raise Skip("no chain run carries the total_passes column")


def test_the_handbook_documents_every_analysis_tool_that_produces_results():
    """A merge silently dropped the section documenting three tools.

    Section 5.8 of HANDBOOK.md covered `margin_cert`, `operator_safety` and
    `thaw_audit coverage`. A three-way merge took HANDBOOK.md wholesale from the
    branch that did not have it, and nothing noticed -- the tools stayed in the
    tree, their tests kept passing, and the only document that says how to run
    them lost the page. This is the test that would have caught it: every tool
    under `analysis/` that writes a results file has to be named somewhere a
    reader looks for commands.
    """
    root = Path(__file__).resolve().parent.parent
    # HANDBOOK.md specifically, not "any document". The tool names appear in
    # CODE_AUDIT.md too, as prose about findings, and including it made this
    # test pass with section 5.8 deleted -- verified by deleting it. Section 5
    # is the one place that says how to RUN things, so that is the page a tool
    # has to be on.
    handbook = root / "HANDBOOK.md"
    if not handbook.is_file():
        raise Skip("HANDBOOK.md is not in this checkout")
    text = handbook.read_text(encoding="utf-8")
    undocumented = []
    for f in sorted((root / "analysis").glob("*.py")):
        src = f.read_text(encoding="utf-8")
        # only tools that persist a result: a helper nobody runs needs no page
        if "write_results(" not in src:
            continue
        if f.name not in text and f.stem not in text:
            undocumented.append(f.name)
    assert not undocumented, (
        "these tools write results files and no document says how to run them: "
        f"{undocumented}. Add them to HANDBOOK.md section 5.")


def test_no_identifier_in_the_tree_contains_a_non_ascii_character():
    """`scaling_mechanism.py` shipped `all<U+0441>_s` -- Cyrillic es, not
    Latin c. Python accepts it, so definition and use agreed and every test
    passed; what broke was reading the code. Searching for `allc_s` found
    nothing, and a reviewer who copied the line got a name that looked
    identical to one that did not exist.

    This is the one bug class in the project that no test could catch, because
    it is invisible by construction. Tokenizing is the point: identifiers are
    checked, while prose is not, so docstrings and comments keep their
    em-dashes and the project's Chinese review notes stay legal.
    """
    import io as _io
    import tokenize as _tok

    root = Path(__file__).resolve().parent.parent
    bad = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        src = p.read_text(encoding="utf-8")
        for t in _tok.generate_tokens(_io.StringIO(src).readline):
            if t.type != _tok.NAME:
                continue
            for ch in t.string:
                if ord(ch) > 127:
                    bad.append(
                        f"{p.relative_to(root)}:{t.start[0]} "
                        f"U+{ord(ch):04X} {ch!r} in identifier {t.string!r}")
    assert not bad, (
        "these identifiers contain characters that look like ASCII and are "
        f"not: {bad}. Retype the name in Latin letters.")


def test_the_scaling_mechanism_measures_the_same_thing_at_any_block_size():
    """`--rows` exists only to bound memory, so it must not touch a number.

    The band is accumulated across blocks and the k-th score comes from a
    running top-(k+1) register, which is exactly the shape where a blocking
    bug hides: both are correct for one block by construction and wrong at the
    seams. Checked here against a direct, unblocked computation as well as
    across block sizes, including sizes that do not divide the subset evenly.

    Written after a scare during review, when the band came out 7.35 at one
    block size and 2.85 at another -- which turned out to be a dropped `--band`
    flag between the two commands rather than a seam bug. The invariant was
    worth a test either way, since nothing was checking it.
    """
    sm = _load_analysis("scaling_mechanism")
    rng = np.random.default_rng(3)
    V = rng.normal(size=(1500, 16)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    Q = rng.normal(size=(12, 16)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    idx = np.sort(rng.choice(len(V), 900, replace=False))

    got = [sm.measure_one(V, Q, idx, k=10, band=0.05, dup_tau=0.99, rows=r)
           for r in (37, 100, 450, 900, 5000)]
    for key in ("s_k_mean", "s_k_minus_s_k1_mean", "band_count_mean",
                "z_of_s_k_mean", "corpus_score_sd", "dup_pairs_mean"):
        vals = [g[key] for g in got]
        assert max(vals) - min(vals) < 1e-5, (key, vals)

    # and against the dense computation, which is only possible at this size
    S = Q @ V[idx].T
    srt = -np.sort(-S, axis=1)
    assert abs(got[0]["s_k_mean"] - srt[:, 9].mean()) < 1e-5
    # the band holds documents in [s_k - band, s_k], NOT counting the k-th
    # itself, which cannot displace the top-k by being in it
    s_k = srt[:, 9]
    want = np.count_nonzero((S >= (s_k - 0.05)[:, None])
                            & (S <= s_k[:, None]), axis=1) - 1
    assert abs(got[0]["band_count_mean"] - want.mean()) < 1e-6, (
        got[0]["band_count_mean"], want.mean())


def test_the_scaling_mechanism_verdict_can_come_out_three_ways():
    """The tool exists to decide between two explanations, so its verdict has
    to be able to land on either one and on neither.

    A verdict function that can only say "mechanism confirmed" is the shape
    this project has caught four times: a check that cannot fail. So each
    branch is driven with constructed rows, including the one that would sink
    the scaling result (duplicate density rising with N), because that is the
    outcome the paper would least like and most needs to hear.
    """
    sm = _load_analysis("scaling_mechanism")

    def row(N, z, band_frac, band_count, dup):
        return {"N": N, "z_of_s_k_mean": z, "band_frac_mean": band_frac,
                "band_count_mean": band_count, "dup_pairs_mean": dup}

    # H1: the k-th score pulls away, the ABSOLUTE band thins, duplicates flat
    h1 = sm.verdict([row(500_000, 4.0, 1e-3, 500, 0.10),
                     row(8_841_823, 5.0, 2e-4, 300, 0.10)])
    assert h1["kth_z_rises"] and h1["band_count_thins"], h1
    assert h1["verdict"].startswith("H1"), h1["verdict"]

    # The regression this test exists to hold down. An earlier verdict accepted
    # `band_frac_thins or band_count_thins`, and band_frac is band_count / N, so
    # it falls eighteenfold over this ladder when the band holds exactly the
    # same documents. Constructed here as the ladder's real ratio: band_count
    # identical at 500, band_frac 1e-3 -> 5.65e-5 purely by division. H1 must
    # NOT be declared, because nothing about the corpus has changed except its
    # size, and this tool's whole purpose is being able to come out against the
    # hypothesis the paper wants.
    artifact = sm.verdict([row(500_000, 4.0, 1e-3, 500, 0.10),
                           row(8_841_823, 4.1, 500 / 8_841_823, 500, 0.10)])
    assert artifact["band_frac_fell_by_division_not_evidence"], artifact
    assert not artifact["band_count_thins"], artifact
    assert artifact["verdict"].startswith("neither"), artifact["verdict"]

    # H2: duplicates rise -- must win even when H1's signals also look good,
    # because a real confound is not cancelled by a real mechanism
    h2 = sm.verdict([row(500_000, 4.0, 1e-3, 500, 0.10),
                     row(8_841_823, 5.0, 2e-4, 300, 0.50)])
    assert h2["dup_density_rises"], h2
    assert h2["verdict"].startswith("H2"), h2["verdict"]

    # the duplicate threshold is a judgement, so it is a parameter and it is
    # recorded: the same rows land on H2 or on H1 depending on where it is set
    rows_12pct = [row(500_000, 4.0, 1e-3, 500, 0.10),
                  row(8_841_823, 5.0, 2e-4, 300, 0.112)]
    assert sm.verdict(rows_12pct, dup_threshold=1.1)["verdict"].startswith("H2")
    loose = sm.verdict(rows_12pct, dup_threshold=1.5)
    assert loose["verdict"].startswith("H1"), loose["verdict"]
    assert loose["dup_threshold"] == 1.5, loose

    # neither: nothing moves, so the rise stays unexplained
    nn = sm.verdict([row(500_000, 4.0, 1e-3, 500, 0.10),
                     row(8_841_823, 3.9, 1.1e-3, 520, 0.10)])
    assert nn["verdict"].startswith("neither"), nn["verdict"]

    # a trend that reverses in the middle is not a scaling law, and first-to-last
    # cannot see that, so monotonicity is reported alongside the verdict
    bumpy = sm.verdict([row(500_000, 4.0, 1e-3, 500, 0.10),
                        row(2_000_000, 3.0, 5e-4, 700, 0.10),
                        row(8_841_823, 5.0, 2e-4, 300, 0.10)])
    assert bumpy["verdict"].startswith("H1"), bumpy["verdict"]
    assert not bumpy["kth_z_monotone"] and not bumpy["band_count_monotone"]
    assert "not monotone" in bumpy["verdict"], bumpy["verdict"]
    assert "not monotone" not in h1["verdict"], h1["verdict"]

    # and it refuses to pronounce on a single point -- while still returning
    # every key, because `main` merges this dict into the results file with
    # `**v` and a short branch would write a JSON object with seven fields
    # missing, for a run whose only fault was having one size
    one = sm.verdict([row(500_000, 4.0, 1e-3, 500, 0.10)])
    assert "two sizes" in one["verdict"], one
    assert set(one) == set(h1), set(h1) ^ set(one)
    assert all(one[key] is None for key in one
               if key not in {"verdict", "dup_threshold"}), one
    assert one["dup_threshold"] == 1.1, one

    # a flat series rises nowhere, and its monotonicity is flatness rather than
    # a decrease borrowed from the falling branch
    assert sm._trend([3.0, 3.0, 3.0]) == (False, True)
    assert sm._trend([3.0, 5.0, 3.0]) == (False, False)
    assert sm._trend([1.0, 0.5, 2.0]) == (True, False)
    assert sm._trend([1.0, 1.5, 2.0]) == (True, True)


def test_moving_a_section_to_an_appendix_cannot_hide_its_numbers():
    """A gate a file move can satisfy is not a gate.

    `coverage.scan` read `paper/body.tex` alone. Four subsections then moved to
    `paper/appendix.tex` for length, which took 41 result numbers out of the
    gate's view: the uncovered count FELL while nothing had been registered,
    and the ratchet read that as progress. The numbers were still printed in
    the paper -- an appendix is part of the submission -- so the only thing that
    changed was whether anything was watching them.

    This drives the real bug rather than describing it: a number is written
    into a temporary second source, and the scan must find it there.
    """
    import json as _js
    import tempfile
    from thaw_audit.coverage import scan

    root = Path(__file__).resolve().parent.parent
    claims_path = root / "audit" / "claims.json"
    if not claims_path.is_file():
        raise Skip("audit/claims.json is not in this tree")
    claims = _js.loads(claims_path.read_text(encoding="utf-8"))["claims"]

    with tempfile.TemporaryDirectory() as d:
        main = Path(d) / "body.tex"
        main.write_text("recall is 0.4242 at this depth\n", encoding="utf-8")
        extra = Path(d) / "appendix.tex"
        extra.write_text("and the tail case is 0.3131\n", encoding="utf-8")

        one = scan(main, claims, {})
        assert [r["value"] for r in one["rows"]] == ["0.4242"], one["rows"]

        both = scan([main, extra], claims, {})
        vals = {r["value"] for r in both["rows"]}
        assert vals == {"0.4242", "0.3131"}, vals
        # and each row says which file it came from, so a report can name it
        assert {r["file"] for r in both["rows"]} == {"body.tex", "appendix.tex"}

        # the failure mode this replaces: returning after the first source
        assert both["total"] == 2, (
            "the second source was not scanned; scan() returned after the "
            f"first file. rows={both['rows']}")


def test_every_result_number_in_the_paper_is_claimed_or_declared():
    """`reconcile` checks claim -> run and run -> claim. Both start from the
    ledger, so a number typed into body.tex that no claim covers is invisible
    to the whole apparatus -- which is how two numbers reached section 6 with
    nothing behind them. This is the missing third direction."""
    from thaw_audit.coverage import report as cov_report

    root = Path(__file__).resolve().parent.parent
    body, claims = root / "paper" / "body.tex", root / "audit" / "claims.json"
    if not (body.is_file() and claims.is_file()):
        raise Skip("paper/body.tex or audit/claims.json is not in this tree")
    decl_path = root / "audit" / "paper_numbers.json"
    # EVERY paper source, the same list `thaw_audit coverage` uses. Reading
    # body.tex alone is what let four subsections move to an appendix and take
    # 41 numbers out of the gate with them.
    sources = [body] + sorted(f for f in (root / "paper").glob("*.tex")
                              if f.name not in {"body.tex", "main.tex"})
    r = cov_report(sources, claims, decl_path)
    assert not r["malformed_declarations"], (
        "a declaration must name a reason and say from what: "
        f"{r['malformed_declarations'][:5]}")

    # A ratchet, not a pass/fail. The suite is not the place to park a known
    # failure: a test documented as expected-red is how a suite stops being
    # read, which is why this file already refuses to let README record a
    # failing count. The hard gate is `python -m thaw_audit coverage`, which
    # exits non-zero while anything is uncovered. What the suite enforces is
    # that the count cannot RISE -- nothing new enters the paper unregistered
    # -- and that the baseline cannot go stale once ground is gained.
    import json as _js
    base = _js.loads(decl_path.read_text(encoding="utf-8"))["_baseline_uncovered"]
    now = len(r["uncovered"])
    assert now <= base, (
        f"{now} result numbers in the paper are neither registered nor declared, "
        f"up from a baseline of {base}: something entered the paper without a "
        "claim. First few: "
        + "; ".join(f"line {x['line']}: {x['value']}" for x in r["uncovered"][:6]))
    assert now == base, (
        f"only {now} numbers are uncovered but the baseline still says {base}; "
        "lower `_baseline_uncovered` in audit/paper_numbers.json so the ratchet "
        "keeps the ground that was gained")

def _sh_rmtree(path):
    import shutil as _s
    _s.rmtree(path, ignore_errors=True)

def test_tolerance_never_reaches_the_coverage_matcher():
    """A structural guard, because the behavioural version is a tautology here.

    The original bug was `report` comparing with
    `abs(v - claim) <= max(claim.tol, 5e-4)`. A tolerance is the slack a
    VERIFIER allows between a claim and its own source; it says nothing about
    which printed number a claim backs. Read as a matching interval it became a
    net -- PG-CLAIM-TWICE (value 0, tol 0.5) matched every paper number below a
    half and solely covered 74 of them.

    The fix removes `tol` from the matcher's reach entirely, which is why the
    obvious test -- inflate every tolerance and check coverage is unchanged --
    now cannot fail: there is nothing for it to flow through. Verified: that
    test passes even with the net re-introduced, so it would be reassurance
    without content. What is left to protect is the structure, so this asserts
    that `tol` reaches neither the matcher nor the scanner, by reading the
    source rather than the behaviour.
    """
    import ast as _ast
    import inspect as _inspect
    from thaw_audit import coverage as _cov

    src = Path(_inspect.getfile(_cov)).read_text(encoding="utf-8")
    tree = _ast.parse(src)
    fns = {n.name: n for n in _ast.walk(tree)
           if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    for name in ("_matches", "scan"):
        assert name in fns, f"{name} is gone; this guard needs rewriting"
        body = _ast.dump(fns[name])
        assert "'tol'" not in body, (
            f"{name} reads a claim's `tol`. A tolerance is verifier slack, not "
            "a matching interval; routing it into matching is what made one "
            "claim a net over every number below a half.")
    # and the matcher must not take one by any name
    params = [a.arg for a in fns["_matches"].args.args]
    assert not any("tol" in a for a in params), params

def test_margin_cert_reports_counts_against_the_stream_they_were_measured_on():
    """A count taken on the test half must not be printed over the whole stream.

    `topk_moved_queries` is counted only where the ranking is evaluated -- the
    held-out half -- and was printed as `moved/queries`, which halved the
    apparent rate (47/500 for what is 47/250). The number mattering here is the
    paper's answer to "did the certificate have anything to catch", so a
    denominator twice too large understates exactly the quantity that makes the
    result non-trivial.
    """
    import io as _io
    import contextlib as _ctx
    mc = _load_analysis("margin_cert")
    rng = np.random.default_rng(3)
    A = mc._unit(rng.normal(size=(500, 16)))
    B = mc._unit(A + 2e-3 * rng.normal(size=A.shape))
    nq = 100
    rep = mc.certify(A[:nq] @ A[nq:].T, B[:nq] @ B[nq:].T, k=10, n_calibrate=50)
    assert rep["test_queries"] < rep["queries"], rep
    assert rep["topk_moved_queries"] <= rep["test_queries"], rep
    # and the printed line must carry the right denominator
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        mc._print_report(rep) if hasattr(mc, "_print_report") else None
    out = buf.getvalue()
    if out:
        assert f"/{rep['test_queries']}" in out, out
        assert f"{rep['topk_moved_queries']}/{rep['queries']} " not in out, out

def test_a_printed_percentage_does_not_match_a_claim_stored_as_a_fraction():
    """A unit error, and the third in this family.

    `0.1\\%` is a saving of one part in a thousand. Read as printed it is 0.1,
    and at one decimal place the rounding slack is +-0.05, so it matched recall
    claims of 0.0617 and 0.1267 -- different quantities in different units.

    The guard this pins was already written when this test was added, and it was
    **inert**: `_matches` took a `pct_form` argument with a default and no call
    site ever passed it, so the rule existed and never fired. That is why the
    assertion below goes through `_candidates` and `_matches` together, the way
    `scan` does, rather than calling `_matches(..., pct_form=True)` directly --
    a direct call passes against the broken version.
    """
    from thaw_audit.coverage import _candidates, _matches

    def covered(printed, is_pct, claim):
        return any(_matches(v, d, claim, pct_form=pf)
                   for v, d, pf in _candidates(printed, is_pct))

    # the bug: a percentage against fraction-valued claims
    assert not covered("0.1", True, 0.0617), "0.1% matched a recall of 0.0617"
    assert not covered("0.1", True, 0.1267), "0.1% matched a recall of 0.1267"
    # and the cases that must keep working
    assert covered("0.1", True, 0.001), "0.1% no longer matches its own fraction"
    assert covered("43", True, 0.43), "43% no longer matches a claim of 0.43"
    assert covered("93.7", True, 93.7), "a claim stored as a percentage broke"
    assert covered("0.929", False, 0.929), "a plain fraction broke"


def test_a_zero_valued_claim_covers_only_zero():
    """The concrete shape the net took. `PG-CLAIM-TWICE` is "units claimed
    twice (must be 0)" -- a claim that something did NOT happen. It must back
    the paper where the paper says zero, and nowhere else.

    The percentage cases are the ones that bite: 43 as a percentage denotes
    0.43, which rounds to 0 at the printed number's own zero decimals, so a
    matcher that does not carry the extra two decimals of a percentage lets a
    claim of zero cover the paper's headline savings figure.
    """
    from thaw_audit.coverage import _candidates, _matches

    for printed, pct in (("0.429", False), ("0.043", False), ("43", True),
                         ("0.250", False), ("0.004", False), ("37", True)):
        assert not any(_matches(v, d, 0.0, pct_form=pf)
                       for v, d, pf in _candidates(printed, pct)), (
            f"a claim of value 0 matched the paper's {printed}")
    assert any(_matches(v, d, 0.0, pct_form=pf)
               for v, d, pf in _candidates("0.000", False))

def test_the_coverage_checker_notices_a_claim_going_missing():
    """The checker's purpose is to make an unbacked number visible, so the one
    thing it must never do is report a number covered by a claim about another
    quantity. It did: matching used the claim's `tol` as an interval with a
    floor of 5e-4, and `PG-CLAIM-TWICE` -- "units claimed twice (must be 0)",
    value 0, tol 0.5 -- was a net over [0, 0.5]. It was the only thing covering
    74 of the 169 numbers called covered, so the hole was under-reported by
    nearly a factor of two. First-match-wins then hid the consequence: deleting
    the claim that genuinely backed a number changed nothing the gate could see.

    This asserts the gate notices, which is the property the count rests on."""
    from thaw_audit.coverage import report as cov_report

    root = Path(__file__).resolve().parent.parent
    body, claims_p = root / "paper" / "body.tex", root / "audit" / "claims.json"
    decl = root / "audit" / "paper_numbers.json"
    if not (body.is_file() and claims_p.is_file()):
        raise Skip("paper/body.tex or audit/claims.json is not in this tree")
    import copy as _copy
    import json as _js
    import tempfile as _tf
    reg = _js.loads(claims_p.read_text(encoding="utf-8"))
    base = cov_report(body, claims_p, decl)
    # a claim that is the sole coverer of at least one number
    sole = {}
    for r in base["rows"]:
        if r["claim"] and len(r["claims_matching"]) == 1:
            sole.setdefault(r["claim"], 0)
            sole[r["claim"]] += 1
    assert sole, "no number is covered by exactly one claim; nothing to test"
    victim = sorted(sole, key=lambda c: (-sole[c], c))[0]
    cut = _copy.deepcopy(reg)
    cut["claims"] = [c for c in cut["claims"] if c["id"] != victim]
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _js.dump(cut, fh)
        tmp = Path(fh.name)
    try:
        after = cov_report(body, tmp, decl)
    finally:
        tmp.unlink()
    assert len(after["uncovered"]) > len(base["uncovered"]), (
        f"deleting {victim}, the sole coverer of {sole[victim]} number(s), left "
        f"the uncovered count at {len(after['uncovered'])}: the gate cannot see "
        "a claim go missing, so its count is not coverage")
    # and a tolerance must never be read as an interval
    wide = [c for c in reg["claims"]
            if isinstance(c.get("value"), (int, float))
            and float(c.get("tol") or 0) > 0.01 and abs(float(c["value"])) < 1]
    for c in wide:
        hit = [r for r in base["rows"]
               if c["id"] in r["claims_matching"]
               and abs(float(r["value"]) - float(c["value"])) > 0.01
               and abs(float(r["value"]) / 100 - float(c["value"])) > 0.01]
        assert not hit, (
            f"{c['id']} (value {c['value']}, tol {c['tol']}) still matches "
            f"{len(hit)} unrelated numbers, e.g. line {hit[0]['line']}: "
            f"{hit[0]['value']}")

def test_the_coverage_checker_notices_a_planted_number():
    """The checker's own failure mode is silence, so plant a number it must
    find and a declaration it must refuse."""
    import json as _json
    import shutil as _sh
    import tempfile as _tf
    from thaw_audit.coverage import report as cov_report

    tmp = Path(_tf.mkdtemp())
    try:
        (tmp / "body.tex").write_text(
            "recall was 0.9293 at the chosen depth\n"
            "% a comment holding 0.1234 must be ignored\n"
            "and 0.4242 is undeclared\n", encoding="utf-8")
        (tmp / "claims.json").write_text(_json.dumps(
            {"claims": [{"id": "X", "value": 0.9293, "tol": 5e-05}]}),
            encoding="utf-8")
        decl = tmp / "paper_numbers.json"
        r = cov_report(tmp / "body.tex", tmp / "claims.json", decl)
        vals = {x["value"] for x in r["uncovered"]}
        assert "0.4242" in vals, r["uncovered"]
        assert "0.1234" not in vals, "a LaTeX comment was scanned as content"
        assert r["claimed"] == 1 and r["pass_"] is False

        decl.write_text(_json.dumps({"0.4242": {"reason": "derived"}}),
                        encoding="utf-8")
        assert cov_report(tmp / "body.tex", tmp / "claims.json",
                          decl)["malformed_declarations"] == ["0.4242"], (
            "a declaration with a reason but no `why` was accepted")

        decl.write_text(_json.dumps({"0.4242": {
            "reason": "derived", "why": "(1-K'/N)^D at the measured coverage"}}),
            encoding="utf-8")
        r3 = cov_report(tmp / "body.tex", tmp / "claims.json", decl)
        assert r3["pass_"] is True and r3["declared_count"] == 1, r3
    finally:
        _sh.rmtree(tmp, ignore_errors=True)

def test_the_margin_certificate_is_sound_on_a_known_perturbation():
    """The conclusion is a theorem, so a violation means the band was computed
    from different scores than the ranking used -- the real failure mode every
    time a bound has looked broken. This pins the pipeline, not the maths.

    `soundness_violations == 0` is only worth asserting when B was fitted on
    queries other than the ones tested. Without that split B is the observed
    maximum over the very matrices the ranking uses, so the premise is true by
    construction and zero violations is not a result: fed unrelated score
    matrices or a fully reversed ranking, the unsplit call returns zero too.
    The report says so in `soundness_can_fail`, and both branches are asserted
    here so the distinction cannot quietly disappear."""
    mc = _load_analysis("margin_cert")
    rng = np.random.default_rng(0)
    A = mc._unit(rng.normal(size=(600, 32)))
    B = mc._unit(A + 1e-3 * rng.normal(size=A.shape))
    So, Sn = A[:60] @ A[60:].T, B[:60] @ B[60:].T
    r = mc.certify(So, Sn, k=10, n_calibrate=30)
    assert r["split"] and r["soundness_can_fail"], r
    assert r["soundness_violations"] == 0, r
    assert r["kprime_derived"] >= r["k"], r
    # a tiny perturbation must certify most of the corpus, or the band is being
    # computed too loosely to be worth stating
    assert r["certified_frac_mean"] > 0.5, r

    # unsplit, the check is incapable of a non-zero value, and must say so
    hostile = mc.certify(So, -Sn, k=10)
    assert hostile["soundness_can_fail"] is False, hostile
    assert hostile["soundness_violations"] == 0, (
        "an unsplit call cannot report a violation; if this ever fails the "
        "premise is no longer fitted on the data it is tested against")

    # and the held-out premise is a live quantity: a perturbation that grows
    # across the query stream must be caught exceeding a B fitted early
    S2 = So + np.linspace(1e-4, 0.3, So.shape[0])[:, None] * rng.normal(size=So.shape)
    grown = mc.certify(So, S2, k=10, n_calibrate=So.shape[0] // 2)
    assert grown["premise_exceeded"] > 0, (
        "B fitted on the first half was never exceeded on the second, so the "
        "split is not testing anything")
    assert grown["score_perturbation_max_test"] > grown["score_perturbation_max"]

def test_the_certificate_is_vacuous_for_unrelated_spaces():
    """The boundary, asserted the other way round: two independent spaces must
    give a band that covers the corpus. If this ever passes, the paper's reason
    for measuring recall rather than certifying it has gone."""
    mc = _load_analysis("margin_cert")
    rng = np.random.default_rng(1)
    A = mc._unit(rng.normal(size=(400, 16)))
    C = mc._unit(rng.normal(size=(400, 16)))
    r = mc.certify(A[:40] @ A[40:].T, C[:40] @ C[40:].T, k=10)
    assert r["kprime_derived"] == r["corpus"], r
    assert r["certified_frac_mean"] < 0.05, r

def test_operator_safety_tracks_the_band_not_the_operator():
    """C1 listed a mixed index as unsafe unconditionally. Measured, it is safe
    when the band is small and destroyed when it is large, so the condition is
    on B and the blanket column is wrong. Both directions are asserted, because
    the contribution rests on the conditional and not on the prohibition."""
    ops = _load_analysis("operator_safety")
    rng = np.random.default_rng(0)
    A = ops._unit(rng.normal(size=(800, 24)))
    near = ops._unit(A + 1e-3 * rng.normal(size=A.shape))
    far = ops._unit(rng.normal(size=A.shape))
    q, c = slice(0, 80), slice(80, None)
    r_near = ops.measure(A[q] @ A[c].T, near[q] @ near[c].T, k=10)
    r_far = ops.measure(A[q] @ A[c].T, far[q] @ far[c].T, k=10)
    assert r_near["score_perturbation_band"] < r_far["score_perturbation_band"]
    assert min(x["mixed_index_recall_at_k"] for x in r_near["rows"]) > 0.9, r_near
    assert min(x["mixed_index_recall_at_k"] for x in r_far["rows"]) < 0.5, r_far
    # and the damage must ease as the store becomes homogeneous, because at
    # rho = 1 a mixed index is not mixed
    seq = [x["mixed_index_recall_at_k"] for x in r_far["rows"]]
    assert seq[-1] > seq[0], seq
    # the repaired sets must be NESTED, because repair only ever adds: the
    # sweep is one migration seen at several points, not five unrelated stores
    reps = [x["repaired"] for x in r_far["rows"]]
    assert reps == sorted(reps), reps
    for a, b in zip(r_far["rows"], r_far["rows"][1:]):
        assert b["repaired"] >= a["repaired"], (a["rho"], b["rho"])


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
        except ImportError as e:
            mod = _missing_optional(e)
            if mod is None:
                bad += 1
                print(f"  FAIL  {fn.__name__}: {e!r}"[:200])
            else:
                skipped += 1
                print(f"  SKIP  {fn.__name__}: {mod} is not installed "
                      f"(it is in requirements.txt; the migration core does "
                      f"not need it)")
        except SystemExit as e:
            # SystemExit is a BaseException, so `except Exception` let one
            # abort the whole run and print a partial pass count as if it were
            # the total. This codebase raises SystemExit as its standard
            # refusal, so an unexpected one is exactly the kind of failure the
            # suite exists to report.
            bad += 1
            print(f"  FAIL  {fn.__name__}: refused: {e}"[:200])
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  FAIL  {fn.__name__}: {e!r}"[:200])
    print(f"\n{len(fns) - bad - skipped} passed, {skipped} skipped, {bad} failed "
          f"out of {len(fns)} tests")
    raise SystemExit(1 if bad else 0)
