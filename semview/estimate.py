"""Will lazy migration pay? Answer it before migrating anything.

The closed form `savings ~= (1 - K'/N)^D` (exp(-D*K'/N) at scale) explains the
effect but, treating intents as independent, ignores the overlap between them,
so it underestimates savings. This measures the thing
directly:

    replay a sample of the query log against the EXISTING old index,
    take the union of the top-K' windows, and that union is what a lazy
    migration would have to embed.

No new-model forward passes are involved. The old vectors are already in the
index and the old query embeddings either exist in the log or cost one cheap
pass in a model already loaded. So this is a decision procedure you can run in
an afternoon, on production data, before committing to anything.

Two things it reports that a point estimate cannot:

  convergence -- how large a log sample is needed before the estimate settles.
      Extrapolating from too few queries is the obvious way to get this wrong,
      because coverage rises steeply at first and then saturates.
  extrapolation -- a saturating fit to the observed coverage curve, so the
      asymptote can be read from a partial sample. The same curve can be
      fitted from repair counts DURING a migration, which makes the decision
      revisable rather than final: if the asymptote comes out worse than
      projected, abort and fall back to a batch job.
"""

from __future__ import annotations

import numpy as np


def coverage_curve(Q_old: np.ndarray, V_old: np.ndarray, kprime: int,
                   points: int = 24, seed: int = 0, budget: float = 6e7):
    """Unique units reached as the number of distinct replayed intents grows.

    Returns (d_values, reached_values). Queries are shuffled first, so the
    curve is not an artifact of how the log happened to be ordered -- a log
    grouped by tenant, topic or repository will otherwise look far more
    favourable than it is.

    Uses the same top-k as `migration` (FAISS when installed, sharded so the
    index never copies the whole corpus). An earlier version carried its own
    numpy-only search: a second implementation of the same thing, and one
    that could not use FAISS at 21M.
    """
    from .migration import topk_chunked

    D, N = Q_old.shape[0], V_old.shape[0]
    if D == 0:
        # np.geomspace(1, 0, points) degenerates to a row of zeros and the curve
        # would claim a single reached value for a stream that never existed.
        raise SystemExit("coverage_curve() was given an empty query set: there "
                         "is no stream to replay, so there is no curve to draw")
    rng = np.random.default_rng(seed)
    order = rng.permutation(D)
    win = topk_chunked(Q_old[order], V_old, min(kprime, N), budget=budget)

    ds = np.unique(np.geomspace(1, D, points).astype(int))
    seen = np.zeros(N, dtype=bool)
    out, done = [], 0
    for d in ds:
        if d > done:
            seen[np.unique(win[done:d])] = True
            done = d
        out.append(seen.mean())
    return ds, np.array(out)


def _power_sat(d, a, tau, b):
    """a * (1 - (1 + d/tau)^-b): saturating with a heavy tail."""
    return a * (1.0 - (1.0 + d / tau) ** -b)


def _exponential(d, a, tau):
    return a * (1.0 - np.exp(-d / tau))


def fit_asymptote(ds: np.ndarray, reached: np.ndarray) -> dict:
    """Fit the coverage curve and report the asymptote `a`.

    1 - a is the permanent saving: units no intent ever surfaces are never
    embedded, because retrieve-then-rerank never compares across spaces and so
    never requires the index to be homogeneous.

    The functional form matters more than it looks. Coverage under a Zipf-like
    intent distribution is heavy-tailed, and an exponential cannot represent
    that: fitted to half of pooled CIPBench's log it projects an asymptote of
    0.49 where the observed value is 0.99, i.e. it would promise 51% permanent
    savings against a real 1%. The power-saturating form projects 0.97 from the
    same prefix. We fit both and use the better residual, reporting the choice.
    """
    from scipy.optimize import curve_fit

    cands = [
        ("power_sat", _power_sat, [min(1.0, float(reached[-1]) * 1.05),
                                   max(float(ds[len(ds) // 2]), 1.0), 1.0],
         ([0.0, 1e-6, 1e-3], [1.0, 1e12, 20.0])),
        ("exponential", _exponential, [min(1.0, float(reached[-1]) * 1.05),
                                       max(float(ds[len(ds) // 2]), 1.0)],
         ([0.0, 1e-6], [1.0, 1e12])),
    ]
    best = None
    for name, fn, p0, bnds in cands:
        try:
            popt, _ = curve_fit(fn, ds, reached, p0=p0, bounds=bnds, maxfev=40000)
        except Exception:  # pragma: no cover
            continue
        rmse = float(np.sqrt(np.mean((fn(ds, *popt) - reached) ** 2)))
        if best is None or rmse < best[2]:
            best = (name, popt, rmse)
    if best is None:  # pragma: no cover
        return dict(asymptote=float(reached[-1]), rmse=float("nan"),
                    fitted=False, form="none")
    name, popt, rmse = best
    a = float(popt[0])
    return dict(asymptote=a, rmse=rmse, fitted=True, form=name,
                params=[float(x) for x in popt],
                # How well the curve fits the points it was fitted to. This is
                # NOT evidence the extrapolation is right: on CIPBench at K'=50
                # the fit has rmse 0.022 yet needs 71% of the log before it
                # predicts the observed end. See backtest / extrapolation_ok.
                in_sample_fit_ok=bool(rmse < 0.03),
                permanent_savings_pct=round(100 * (1 - a), 2))


def convergence(ds: np.ndarray, reached: np.ndarray, tol: float = 0.02) -> dict:
    """Smallest prefix whose fitted asymptote is within `tol` of the full fit.

    This measures STABILITY -- the estimate has stopped moving -- not accuracy.
    A fit can settle confidently on the wrong value; the plain exponential did
    exactly that, projecting 0.49 against an observed 0.99. `backtest` checks
    accuracy against what was actually observed.
    """
    full = fit_asymptote(ds, reached)["asymptote"]
    for i in range(4, len(ds)):
        a = fit_asymptote(ds[:i], reached[:i])["asymptote"]
        if abs(a - full) <= tol:
            return dict(d_needed=int(ds[i - 1]),
                        frac_of_log=float(ds[i - 1] / ds[-1]),
                        asymptote_full=round(full, 4), tol=tol)
    return dict(d_needed=int(ds[-1]), frac_of_log=1.0,
                asymptote_full=round(full, 4), tol=tol)


def backtest(ds: np.ndarray, reached: np.ndarray, tol: float = 0.02) -> dict:
    """Accuracy, not stability: fit each prefix, predict reached at the end of
    the replayed log, and compare with the value actually observed there.

    Returns the smallest prefix whose prediction lands within `tol` of the
    observed end-of-log value, and the error at half the log. Unlike
    `convergence`, a fit that settles on the wrong value fails this.
    """
    target, d_end = float(reached[-1]), float(ds[-1])
    half_err, first_ok = None, None
    for i in range(4, len(ds)):
        f = fit_asymptote(ds[:i], reached[:i])
        if not f.get("fitted"):
            continue
        fn = _power_sat if f["form"] == "power_sat" else _exponential
        pred = float(fn(d_end, *f["params"]))
        err = pred - target
        if half_err is None and ds[i - 1] >= d_end / 2:
            half_err = err
        if first_ok is None and abs(err) <= tol:
            first_ok = int(ds[i - 1])
    # The in-sample proxy for "can this be extrapolated": fitted to half the
    # replayed log, does it predict the observed end within tol? Extrapolating
    # past the end of the log can never be validated in-sample.
    return dict(extrapolation_ok=bool(half_err is not None and abs(half_err) <= tol),
                backtest_d_needed=first_ok,
                backtest_frac_of_log=(first_ok / d_end) if first_ok else None,
                backtest_err_at_half=None if half_err is None else round(half_err, 4),
                backtest_tol=tol)


def estimate(Q_old: np.ndarray, V_old: np.ndarray, kprimes=(50, 100, 200),
             seed: int = 0, verbose: bool = True) -> list[dict]:
    rows = []
    for kp in kprimes:
        ds, rc = coverage_curve(Q_old, V_old, kp, seed=seed)
        fit = fit_asymptote(ds, rc)
        conv = convergence(ds, rc)
        bt = backtest(ds, rc)
        rows.append(dict(kprime=kp, N=int(V_old.shape[0]), D=int(Q_old.shape[0]),
                         reached_observed=round(float(rc[-1]), 4),
                         savings_observed_pct=round(100 * (1 - rc[-1]), 2),
                         curve_d=[int(x) for x in ds],
                         curve_reached=[round(float(x), 6) for x in rc],
                         **fit, **conv, **bt))
        if verbose:
            bt_txt = (f"within {bt['backtest_tol']} after {bt['backtest_d_needed']} "
                      f"intents ({100*bt['backtest_frac_of_log']:.0f}% of log)"
                      if bt["backtest_d_needed"] else "never within tolerance")
            print(f"  K'={kp:<5} observed reached={rc[-1]:.3f} "
                  f"(savings {100*(1-rc[-1]):.1f}%)  "
                  f"asymptote={fit['asymptote']:.3f} ({fit['form']}, "
                  f"rmse={fit['rmse']:.4f})")
            print(f"           stable after {conv['d_needed']} intents "
                  f"({100*conv['frac_of_log']:.0f}% of log); accurate: {bt_txt}")
            if not bt["extrapolation_ok"]:
                e = bt["backtest_err_at_half"]
                print(f"           ! fitted to half the log, the curve misses the "
                      f"observed end by {'n/a' if e is None else f'{e:+.3f}'}: do not "
                      "extrapolate from a partial log; use the observed value")
    return rows


def main() -> None:  # pragma: no cover
    import argparse
    import json
    from pathlib import Path

    from .migration import (NQ_DEV_URL, corpus_key, dpr_corpus, encode_dpr_questions,
                            load_beir, load_cipbench, load_jsonl, load_msmarco,
                            load_orcas,
                            old_label, positive_int,
                            require_dpr_limit, resolve_prefixes,
                            select_old_side, verify_dpr_provenance)

    ap = argparse.ArgumentParser(prog="semview.estimate")
    ap.add_argument("--corpus", default="cipbench",
                    choices=["cipbench", "jsonl", "msmarco", "beir", "dpr_wiki",
                             "orcas"])
    ap.add_argument("--path", default="bench_all_clean")
    ap.add_argument("--queries-file", default=None,
                    help="--corpus jsonl: a {\"id\",\"text\"} query file")
    ap.add_argument("--beir-name", default="nfcorpus")
    ap.add_argument("--limit", type=positive_int, default=None)
    ap.add_argument("--max-queries", type=positive_int, default=20000)
    ap.add_argument("--old", default="lexical:word")
    ap.add_argument("--kprimes", default="50,100,200")
    ap.add_argument("--dpr-variant", default="psgs_w100.nq.no_index")
    ap.add_argument("--queries", default="nq-dev",
                    choices=["nq-dev", "nq-open-validation", "nq-open-train"],
                    help="DPR query set. nq-dev (default) has known gold passages "
                         "and is the only choice with --limit; the nq-open sets "
                         "need the full corpus. nq-open-train is memorised by both "
                         "encoders: coverage only, never recall.")
    ap.add_argument("--nq-dev", default=NQ_DEV_URL,
                    help="path or URL of DPR's biencoder-nq-dev.json(.gz)")
    ap.add_argument("--orcas-docs", default=None)
    ap.add_argument("--orcas-query-col", type=int, default=None)
    ap.add_argument("--orcas-doc-col", type=int, default=None)
    ap.add_argument("--qrels-split", default=None,
                    help="BEIR: restrict queries to this qrels split (use 'dev' for "
                         "msmarco, whose query file includes train queries) and, "
                         "with --limit, keep every gold document")
    ap.add_argument("--dpr-path", default=None)
    ap.add_argument("--old-normalize", action="store_true",
                    help="ablation: L2-normalise the old side. It changes which "
                         "units each window reaches, so it changes coverage: raw "
                         "inner product concentrates windows on high-norm "
                         "vectors. Report DPR savings both ways.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=None,
                    help="cache the old-side corpus embeddings here (ignored for "
                         "DPR, whose vectors ship with the dataset)")
    ap.add_argument("--old-query-prefix", default="auto")
    ap.add_argument("--old-passage-prefix", default="auto")
    ap.add_argument("--skip-provenance", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="estimate.json")
    a = ap.parse_args()

    require_dpr_limit(a.corpus, a.limit, "estimate")
    V_pre = None
    if a.corpus == "dpr_wiki":
        # The shipped DPR vectors are the old index; only DPR's own question
        # encoder is needed, and no new-model pass happens anywhere.
        # estimate never embeds the corpus, so it does not keep passage text
        units, V_pre, dpr_pairs, queries = dpr_corpus(
            a.limit, a.dpr_variant, a.dpr_path, a.queries, a.nq_dev,
            keep_text=False, seed=a.seed)
        if not a.skip_provenance:
            if not verify_dpr_provenance(dpr_pairs, V_pre,
                                         a.dpr_variant, device=a.device)["ok"]:
                raise SystemExit("DPR provenance check failed; aborting")
    elif a.corpus == "orcas":
        if not a.orcas_docs:
            raise SystemExit("--corpus orcas needs --orcas-docs")
        units, queries = load_orcas(a.path, a.orcas_docs, a.limit, a.max_queries,
                                    a.seed, a.orcas_query_col, a.orcas_doc_col)
    elif a.corpus == "cipbench":
        units, queries = load_cipbench(a.path, a.limit, a.seed)
    elif a.corpus == "jsonl":
        units, queries = load_jsonl(a.path, a.queries_file, a.limit, a.seed)
    elif a.corpus == "msmarco":
        units, queries = load_msmarco(a.limit)
    else:
        units, queries = load_beir(a.beir_name, a.path, a.limit, a.seed,
                                   a.qrels_split)
    if a.max_queries and len(queries) > a.max_queries:
        rng = np.random.default_rng(a.seed)
        queries = [queries[i] for i in
                   rng.choice(len(queries), a.max_queries, replace=False)]
    if not units:
        raise SystemExit("the corpus is empty: nothing to estimate coverage over")
    if not queries:
        # Refuse before encoding: the whole output is a coverage curve over the
        # query stream, and an empty stream has no x-axis.
        raise SystemExit("the query set is empty: there is no stream to replay. "
                         "Check --qrels-split, the queries file, and any filter "
                         "that could have selected nothing.")

    # With shipped vectors the old side is never encoded, so there is nothing
    # to build a text list for -- and for DPR those texts are empty strings
    # anyway, since this tool asks the loader not to keep them.
    utext = None if V_pre is not None else [t for _, t in units]
    qtext = [t for _, t in queries]
    old_name = old_label(a.old, a.dpr_variant if V_pre is not None else None,
                         a.old_normalize)
    if V_pre is not None and a.cache_dir:
        print("note: --cache-dir is ignored for DPR; the shipped vectors are "
              "the old index, so nothing is encoded")
    print(f"corpus={a.corpus}  N={len(units)}  D={len(qtext)}  old={old_name}")
    print("NOTE: no new-model forward passes are performed by this tool.")
    qenc = ((lambda t: encode_dpr_questions(t, a.dpr_variant, device=a.device))
            if V_pre is not None else None)
    oq, op = resolve_prefixes(a.old, a.old_query_prefix, a.old_passage_prefix)
    print(f"prefixes  old: query={oq!r} passage={op!r}")
    old_key = (None if V_pre is not None
               else corpus_key(a.old, op, (i for i, _ in units)))
    V_old, Q_old = select_old_side(utext, qtext, a.old, V_shipped=V_pre,
                                   question_encoder=qenc,
                                   normalize=a.old_normalize, device=a.device,
                                   query_prefix=oq, passage_prefix=op,
                                   cache_dir=a.cache_dir, cache_key=old_key)

    rows = estimate(Q_old, V_old,
                    tuple(int(x) for x in a.kprimes.split(",")), seed=a.seed)
    from .migration import _fingerprint, _ids_hash
    from .results_io import environment, write_results
    write_results(a.out,
        dict(corpus=a.corpus, N=len(units), D=len(qtext), old=old_name,
             fingerprint=dict(old_head=_fingerprint(V_old),
                              query_ids_sha256=_ids_hash(q[0] for q in queries),
                              unit_ids_sha256=_ids_hash(u[0] for u in units)),
             environment=environment(device=a.device),
             estimates=rows))


if __name__ == "__main__":
    main()
