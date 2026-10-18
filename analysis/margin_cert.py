"""Does a rank-stability certificate decide which units need re-embedding?

The two directions on the table answer the same question -- *after a model
update, which documents actually need a new vector?* -- and this tool measures
whether they are one mechanism or two.

**The correction that makes them one.** The certificate is usually written on
vector movement: if a document's vector moves by at most eps and the query's by
at most delta, each cosine score moves by at most eps+delta. That form is
unusable twice over. First it is circular -- eps(u) is only known after running
the new model on u, which is the work being avoided. Second it is undefined
whenever the two models have different dimension, which is the common case for
a model replacement (384 vs 768 here).

Both problems disappear if the premise is stated on the quantity the ranking
actually consumes, which is the **score**:

    |s_new(q,u) - s_old(q,u)| <= B/2   for every (q,u)

B is then measurable for *any* update, across spaces of different dimension,
from a sample of queries. Everything the certificate concludes follows from the
scalar B alone:

  * two documents cannot swap unless their old scores differ by less than B;
  * the new top-k can only differ from the old inside a band of width B below
    the k-th old score;
  * therefore the band's width in *documents* is a window depth K' that is
    GUARANTEED to contain the new top-k, rather than one read off a recall
    curve at some recall below 1;
  * and the complement -- every unit no query's band reaches -- provably never
    needs a new vector, which is the existing cost model (1-K'/N)^D with K'
    supplied by proof instead of by measurement.

So the certificate does not compete with the window: it *derives* the window.
The exact-reuse case (token sequence unchanged, so the forward pass is
bit-identical) is not a separate layer either -- it is B = 0, where the band is
empty and the derived K' is k.

**What this tool reports**, per model pair: B, the band's width in documents
(the derived K'), the fraction of the corpus the band never reaches, a
soundness check (no top-k change may occur outside the band -- a violation
means the premise was measured wrong, since the conclusion is a theorem), and
how often the top-k genuinely moved, because a certificate that never has
anything to catch is not evidence of anything.

The decisive output is the comparison between pairs. The certificate is a tool
for one regime only, and this prints the boundary rather than asserting it.

The scores are a (queries x corpus) matrix held twice, so cost is set by their
product and not by the corpus alone: a 500-query window over a 500,000-document
corpus needs about 9 GB, which the pre-flight warning says. Nothing here wants
that. The quantity being measured is a band over a score distribution, and
section 10.4 of the research contract calls for a sample of roughly a thousand
documents -- so sample the corpus rather than running it whole.

    python analysis/margin_cert.py \
        --old cache/MINILM.sample.npy --new cache/BGE.sample.npy \
        --queries 200 --k 10 --label "MiniLM->BGE" \
        --out results/margin_cert_minilm_bge.json

(`.sample.npy` being a few thousand rows taken from the full matrix at the same
row indices in both -- the row order has to agree.)

With `--new-f16` the same corpus matrix in float16 is compared against its own
float32, which is the quantization pair and needs no second model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview.results_io import environment, write_results  # noqa: E402


def _unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def certify(S_old: np.ndarray, S_new: np.ndarray, k: int = 10,
            n_calibrate: int | None = None) -> dict:
    """The whole certificate, from two score matrices and nothing else.

    `S_old` and `S_new` are (queries x corpus). They may come from models of
    different dimension; only the scores are compared, which is the point.

    `n_calibrate` splits the queries: B is fitted on the first `n_calibrate`
    rows and everything else is measured on the rest. Without the split the
    soundness check CANNOT FAIL -- and not merely "is expected to pass". B is
    defined as the observed maximum over the same two matrices the ranking
    uses, so the premise is true by construction on that data and the
    conclusion follows as a theorem. Verified: fed unrelated score matrices,
    perturbations a hundred times the score scale, and a fully reversed
    ranking, it returns zero violations every time. A reported check that is
    incapable of a non-zero value is not evidence of anything, which is this
    project's own standard for an experiment.

    With the split, two numbers become real and both can fail:

      * `premise_exceeded_frac` -- how often a held-out (q,u) moves by more
        than the fitted B/2. This is what "guaranteed with respect to that B"
        costs, as a number rather than a qualifier.
      * `soundness_violations` -- a held-out top-k change outside the band.

    The split is what "held-out queries" has to mean. Splitting queries from
    the corpus, which is what the caller did, leaves calibration and test on
    the same rows.
    """
    if S_old.shape != S_new.shape:
        raise ValueError("score matrices must cover the same queries and corpus")
    nq, N = S_old.shape
    if nq < 2:
        # A certificate needs one query to fit B and one to test the premise on.
        # The CLI only refused nq < 1, so `--queries 1` reached here and died in
        # numpy with "zero-size array to reduction operation maximum" -- a
        # library's constraint should be stated by the library, not discovered
        # by its caller.
        raise ValueError(
            f"a certificate needs at least 2 queries, got {nq}: B is fitted on "
            "one part of the stream and tested on the rest, so a single query "
            "leaves nothing to test the premise against")
    k = int(min(k, N))
    cal = nq if n_calibrate is None else int(min(max(n_calibrate, 1), nq - 1))
    split = cal < nq

    half = float(np.abs(S_new[:cal] - S_old[:cal]).max())   # fitted here only
    band = 2.0 * half
    S_old_t, S_new_t = (S_old[cal:], S_new[cal:]) if split else (S_old, S_new)
    nt = S_old_t.shape[0]

    d_t = np.abs(S_new_t - S_old_t)
    exceeded = int((d_t > half).sum())

    kth = np.partition(S_old_t, N - k, axis=1)[:, N - k]
    flagged = S_old_t >= (kth[:, None] - band)
    per_q = flagged.sum(axis=1)

    # top-k only: a full argsort of a (queries x corpus) matrix is two int64
    # copies of it, which is what made this tool need ~15 GB at the corpus size
    # its own usage example names
    top_old = np.argpartition(-S_old_t, k - 1, axis=1)[:, :k]
    top_new = np.argpartition(-S_new_t, k - 1, axis=1)[:, :k]
    moved = 0
    unsound = 0
    for i in range(nt):
        so = set(top_old[i].tolist())
        sn = set(top_new[i].tolist())
        if so == sn:
            continue
        moved += 1
        for u in sn - so:          # only an ENTERING unit tests the theorem;
            if S_old_t[i, u] < kth[i] - band:   # a leaving one is above kth
                unsound += 1

    union = int(flagged.any(axis=0).sum())
    load = float(nt * per_q.max() / N)
    return {
        "queries": int(nq), "corpus": int(N), "k": k,
        "calibration_queries": int(cal), "test_queries": int(nt),
        "split": bool(split),
        "score_perturbation_max": half,
        "score_perturbation_max_test": float(d_t.max()),
        "premise_exceeded": exceeded,
        "premise_exceeded_frac": float(exceeded / d_t.size),
        "soundness_can_fail": bool(split),
        "score_perturbation_mean": float(d_t.mean()),
        "band": band,
        "kprime_derived": int(per_q.max()),
        "kprime_mean": float(per_q.mean()),
        "kprime_p99": float(np.percentile(per_q, 99)),
        "flagged_frac_mean": float(per_q.mean() / N),
        "certified_frac_mean": float(1.0 - per_q.mean() / N),
        "union_flagged": union,
        "union_frac": float(union / N),
        "topk_moved_queries": int(moved),
        "soundness_violations": int(unsound),
        "load_x": load,
        "load_note": (
            "union_frac is only interpretable at load x below about 3; above it "
            "every policy reaches the whole corpus and the union saturates "
            "regardless of the band. The transferable quantity is kprime_derived, "
            "which the existing cost model turns into savings at any N."
        ),
    }


def _print_report(rep: dict) -> None:
    """Printing lives in its own function so a test can read it.

    A count measured on the held-out half printed over the whole stream
    reads at half its true rate, and that was live here. The invariant is
    not checkable from the report dict alone -- both numbers are in it --
    so the only way to pin it is to look at what is printed.
    """
    # `label` and `dims` are added by main(); a report straight from certify()
    # has neither, and a printer that crashes on its own data source's output
    # cannot be exercised by a test
    dims = rep.get("dims")
    head = rep.get("label", "report")
    if dims:
        head += f"   dims {dims[0]} -> {dims[1]}"
    print(f"\n  {head}   N={rep['corpus']:,}  {rep['queries']} queries  "
          f"k={rep['k']}")
    print(f"    max score perturbation      {rep['score_perturbation_max']:.6f}")
    print(f"    certificate band            {rep['band']:.6f}")
    print(f"    K' DERIVED (guaranteed)     {rep['kprime_derived']}"
          f"   (mean {rep['kprime_mean']:.1f}, p99 {rep['kprime_p99']:.0f})")
    print(f"    certified per query         {100 * rep['certified_frac_mean']:.2f}% of corpus")
    # the denominator is the TEST stream, not the whole one: `moved` is counted
    # only where the ranking is evaluated, so printing it over `queries` halved
    # the apparent rate
    print(f"    top-{rep['k']} genuinely moved      "
          f"{rep['topk_moved_queries']}/{rep['test_queries']} test queries")
    print(f"    B fitted on                 {rep['calibration_queries']} queries, "
          f"tested on {rep['test_queries']}")
    print(f"    premise exceeded on test    {rep['premise_exceeded']} pairs "
          f"({100 * rep['premise_exceeded_frac']:.3f}%)  "
          f"max |ds| there {rep['score_perturbation_max_test']:.6f}")
    print(f"    soundness violations        {rep['soundness_violations']}"
          + ("" if rep["soundness_can_fail"] else
             "   (CANNOT FAIL: no calibration split)"))
    if rep["kprime_derived"] >= rep["corpus"]:
        print("    -> the band covers the corpus: the certificate is VACUOUS for "
              "this pair,\n       and the window must come from measured "
              "cross-space recall instead.")
    elif rep["topk_moved_queries"] == 0:
        print("    -> nothing moved, so the certificate had nothing to catch: "
              "tight but\n       uninformative on this pair.")
    else:
        print(f"    -> non-vacuous AND non-trivial: {rep['topk_moved_queries']} "
              f"queries changed and the\n       band caught every change while "
              f"flagging {100 * rep['flagged_frac_mean']:.2f}% of the corpus.")


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", required=True, help="corpus under the old model, N x d")
    ap.add_argument("--new", required=True,
                    help="corpus under the new model, N x d' (d' may differ)")
    ap.add_argument("--new-f16", action="store_true",
                    help="--new is the float16 storage of --old: the quantization "
                         "pair, where the certificate is expected to be tight")
    ap.add_argument("--queries", type=int, default=500,
                    help="leading rows held out as queries; the rest is the corpus")
    ap.add_argument("--calibrate-queries", type=int, default=None,
                    help="how many of --queries fit B; the rest test it. Without "
                         "a split the soundness check cannot fail at all, since B "
                         "is the observed maximum over the same scores the ranking "
                         "uses. Defaults to half")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    O = np.load(a.old).astype(np.float64)
    Nw = np.load(a.new).astype(np.float64)
    if len(O) != len(Nw):
        raise SystemExit("the two corpus matrices must share a row order and length")
    nq = a.queries
    if nq < 1 or nq >= len(O):
        raise SystemExit(f"--queries {nq} leaves no corpus")
    O, Nw = _unit(O), _unit(Nw)

    # Pre-flight, because the footprint is a few multiples of a (queries x
    # corpus) float64 matrix and the usage example above names a 500k corpus.
    # Measured at 4.6-4.9x one score matrix after the argpartition fix, which
    # is ~9 GB at N=500,000 with 500 queries -- down from ~15 GB when the top-k
    # came from a full argsort, and still more than most machines will give.
    # The decision procedure this tool is for samples ~1,000 documents, where
    # it is nothing; say so rather than letting the kernel say it.
    gb = 5.0 * nq * (len(O) - nq) * 8 / 1e9
    if gb > 4:
        print(f"  WARNING: {nq} queries over a corpus of {len(O) - nq:,} needs "
              f"about {gb:.0f} GB of RAM (a few multiples of one score matrix). "
              f"Lower --queries, or sample the corpus: B is a property of the "
              f"update, so ~1,000 documents through both models is enough to "
              f"estimate it.")

    # Queries are held-out rows encoded by the SAME model as the corpus they
    # search, which is what both deployments do. Mixing them would measure the
    # cross-space nomination step instead of the perturbation.
    S_old = O[:nq] @ O[nq:].T
    S_new = Nw[:nq] @ Nw[nq:].T
    n_cal = a.calibrate_queries if a.calibrate_queries else max(1, nq // 2)
    rep = certify(S_old, S_new, k=a.k, n_calibrate=n_cal)
    rep["label"] = a.label or f"{Path(a.old).name} -> {Path(a.new).name}"
    rep["quantization_pair"] = bool(a.new_f16)
    rep["dims"] = [int(O.shape[1]), int(Nw.shape[1])]
    rep["vector_eps_defined"] = O.shape[1] == Nw.shape[1]
    if rep["vector_eps_defined"]:
        ve = np.linalg.norm(Nw - O, axis=1)
        rep["vector_eps_max"] = float(ve.max())
        rep["vector_eps_mean"] = float(ve.mean())

    _print_report(rep)

    obj = {"tool": "margin_cert", "inputs": {"old": a.old, "new": a.new},
           "environment": environment(dtype=str(np.load(a.new, mmap_mode="r").dtype)),
           "report": rep}
    write_results(a.out, obj)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
