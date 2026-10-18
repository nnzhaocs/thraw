"""The schemes a migration could use, measured against each other at equal cost.

Every alternative to lazy migration answers the same question -- what do you do
about vectors the new model has not produced yet -- so they can be compared on
one axis: given a budget of B new-model forward passes, what quality does each
deliver? That is the table the evaluation needs, and all of it runs from
vectors already computed for a `migration` run.

The schemes:

  batch        Re-embed everything. Quality 1.0 by definition, cost N passes.
               The reference, not a competitor.

  stale        Change nothing: keep serving from the old index. Cost 0. Its
               quality is the reason a migration happens at all.

  adapter      Fit old -> new with a learned map on a sample and rank with the
               mapped vectors. Cost: the sample. Nothing is ever exact.
               (Drift-Adapter, and the transfer half of compatible-training
               work.)

  backfill-R   Re-embed B units chosen at random, serve from the new index
               only. Cost B. The control for the ordering claim: if an ordered
               backfill does no better than this, then order is not what
               matters, which is worth knowing either way.

  backfill-P   Re-embed B units chosen by a corpus-side priority, serve from
               the new index only. Cost B. This is the shape of an ordered
               backfill (FastFill's claim is that the order matters), with a
               centrality prior standing in for the priority function.

  backfill-T   The same units lazy chose -- picked by traffic -- but served
               from the new index globally, with no repair on the query path.
               Cost B. This separates the two claims: traffic is a good
               ORDERING signal (shared with lazy) versus repairing on demand is
               worth its latency (lazy only). A query needing an unmigrated
               unit never gets it here; under lazy it is repaired and returned.

  lazy         Re-embed the units queries actually reach, up to B, and rerank
               each window exactly in the new space. Cost B.

  lazy+adapter The hybrid: same as lazy, but candidates come from adapter-mapped
               vectors instead of raw old vectors, so the window starts closer
               to the new geometry. Cost B plus the adapter sample, plus one
               new-model pass per distinct query, since the adapter index is in
               the new space and queries must be too. Needs `Q_new_warm`.

A note on what "quality" means for a partially migrated index. A unit without a
new vector cannot be scored against a new-model query without comparing across
spaces, which is exactly what this design refuses to do. So for every scheme
the served result is the exact new-space ranking over whichever units have new
vectors and are visible to that scheme, and quality is its overlap with the
ranking a fully migrated index would return. No scheme is charged for units it
legitimately cannot see, and none is credited for a cross-space comparison.
"""

from __future__ import annotations

import numpy as np

from .migration import (apply_adapter, fit_linear_adapter, topk_chunked,
                        _safe_normalize)


def _recall(truth, served, k: int) -> float:
    return float(np.mean([len(set(t) & set(s)) / k for t, s in zip(truth, served)]))


def _rerank(Q_new, V_new, windows, k: int):
    """Exact new-space top-k WITHIN each candidate window.

    This is the step the whole design turns on: the old space chose the window,
    the new space ranks it. Truncating the old ranking instead would measure a
    system that never reranks -- which is the `stale` row, not this one.
    """
    out = []
    for q, w in zip(Q_new, windows):
        w = np.asarray(w, dtype=np.int64)
        if w.size == 0:
            out.append([])
            continue
        order = np.argsort(-(V_new[w] @ q))[:k]
        out.append([int(w[j]) for j in order])
    return out


def _served_from_subset(Q_new, V_new, have: np.ndarray, k: int):
    """Exact new-space top-k over the units that have new vectors."""
    idx = np.flatnonzero(have)
    if idx.size == 0:
        return [[] for _ in range(len(Q_new))]
    sub = topk_chunked(Q_new, V_new[idx], min(k, idx.size))
    return [[int(idx[j]) for j in row] for row in sub]


def _touched(Q_warm, V_window, kprime: int, budget: int) -> np.ndarray:
    """Units reached by warm-up traffic, in arrival order, capped at `budget`.

    `Q_warm` must be in the same space as `V_window`: old-space queries for the
    old index, new-space queries for an adapter index.
    """
    have = np.zeros(len(V_window), dtype=bool)
    spent = 0
    for w in topk_chunked(Q_warm, V_window, kprime):
        for u in w:
            if not have[u]:
                if spent >= budget:
                    return have
                have[u] = True
                spent += 1
    return have


def _std(x: np.ndarray) -> np.ndarray:
    """Standardise one query's score list; a constant list maps to zeros.

    Standardising the two spaces' scores separately, per query, is the cheapest
    training-free calibration available, so it is the calibration the ablation
    below has to beat.
    """
    if x.size == 0:
        return x
    sd = float(x.std())
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def _calibrated_serve(Q_old, Q_new, V_old, V_new, have, kprime: int, k: int,
                      lam: float):
    """Rank a window by a calibrated mixture of the two spaces' scores.

    A window member's score is its target score -- standardised within the
    window's repaired members -- if it has one, and `lam` times its source
    score, standardised within the window's unrepaired members, if it does not.
    No unit is repaired on the query path: the target scores of repaired
    members were already paid for by the migration that filled `have`.

    `lam` is the whole calibration: one scalar. A learned cross-space map is
    what prior work needs; this is the cheapest thing an operator could do
    instead, and the comparison is only fair if the alternative is fitted.

    Note which way the fit is biased, because it is the alternative's
    direction: `have` covers the warm-up windows almost entirely, so on the
    warm-up queries this looks like a nearly warm index, while the evaluation
    windows are largely unrepaired. The fitted scalar therefore leans towards
    trusting the target scores more than a cold query could justify.
    """
    have = np.asarray(have, dtype=bool)
    # topk_chunked clamps k to the corpus; K' = N is a supported input and the
    # same clamp is needed here, or argpartition raises for K' > N.
    kk = int(min(kprime, len(V_old)))
    if kk < 1:
        return np.zeros((len(Q_new), 0), dtype=np.int64)
    # A window of kk < k units cannot return k of them; topk_chunked and
    # _exact_window_recall both clamp the same way, and without it the
    # assignment into a (..., k)-wide row raises a broadcast error.
    k = min(k, kk)
    out = np.empty((len(Q_new), k), dtype=np.int64)
    for i, (qo, qn) in enumerate(zip(Q_old, Q_new)):
        s_old = V_old @ qo
        win = np.argpartition(-s_old, kk - 1)[:kk]
        m = have[win]
        score = np.empty(len(win))
        if (~m).any():
            score[~m] = lam * _std(s_old[win][~m])
        if m.any():
            score[m] = _std(V_new[win[m]] @ qn)
        out[i] = win[np.argsort(-score)[:k]]
    return out


def _rrf_serve(Q_old, Q_new, V_old, V_new, have, kprime: int, k: int,
               rrf_k: float = 60.0):
    """Reciprocal-rank fusion of the source window and the repaired set.

    The standard scale-free fusion, and the strongest training-free
    alternative to ranking in one space: it never compares two scores, only two
    ranks. Its known behaviour with a missing list applies here -- a unit with
    no target vector can appear in only one of the two rankings and is
    penalised by construction, which is exactly the quantity being measured.
    """
    have = np.asarray(have, dtype=bool)
    # same clamp as _calibrated_serve: topk_chunked clamps k to the corpus, and
    # K' = N is a supported input, so this must not hand argpartition kth >= N.
    kk = int(min(kprime, len(V_old)))
    if kk < 1:
        return np.zeros((len(Q_new), 0), dtype=np.int64)
    # A window of kk < k units cannot return k of them; topk_chunked and
    # _exact_window_recall both clamp the same way, and without it the
    # assignment into a (..., k)-wide row raises a broadcast error.
    k = min(k, kk)
    out = np.empty((len(Q_new), k), dtype=np.int64)
    for i, (qo, qn) in enumerate(zip(Q_old, Q_new)):
        s_old = V_old @ qo
        win = np.argpartition(-s_old, kk - 1)[:kk]
        m = have[win]
        rank_old = np.empty(len(win))
        rank_old[np.argsort(-s_old[win])] = np.arange(1, len(win) + 1)
        score = 1.0 / (rrf_k + rank_old)
        if m.any():
            s_new = V_new[win[m]] @ qn
            rank_new = np.empty(len(s_new))
            rank_new[np.argsort(-s_new)] = np.arange(1, len(s_new) + 1)
            score[m] += 1.0 / (rrf_k + rank_new)
        out[i] = win[np.argsort(-score)[:k]]
    return out


def _async_serve(Q_old, Q_new, V_old, V_new, have0, kprime: int, k: int,
                 budget_m: int, drain: bool = True):
    """Serve each query without waiting for its whole window to be repaired.

    This is the other end of the choice the paper states and does not
    evaluate. The
    synchronous design repairs every window member before ranking, so every
    answer is exact over the window and the user waits for up to K' forward
    passes. Here a query repairs at most `budget_m` of its missing members
    inline, ranks whatever has a new vector, and leaves the rest to a
    background repairer.

    Which `budget_m` to repair is not arbitrary. The only evidence available
    before a unit has a new vector is its OLD score, so the window is ordered
    by that and the budget is spent on its head -- the missing units the old
    index itself ranked highest, which are the ones most likely to belong in
    the new top-k.

    `drain` is the background repairer's speed, as the two bounds an operator
    actually sits between. With it, a unit one query queued is available to
    every later query, which is a repairer that keeps up with arrivals. Without
    it, nothing queued ever lands, which is a repairer that never gets a turn.
    The truth is in between and depends on the arrival rate.

    The two endpoints of `budget_m` are the two schemes already in this table:
    at 0 nothing is repaired inline, and at `kprime` or more every miss is, so
    the result is the exact whole-window rerank the `lazy` row scores.

    Returns (served, inline_total, queued_total, inline_per_query).
    """
    have = np.asarray(have0, dtype=bool).copy()
    kk = int(min(kprime, len(V_old)))
    if kk < 1:
        return [[] for _ in Q_new], 0, 0, np.zeros(len(Q_new), dtype=int)
    k = min(k, kk)
    out, inline_q = [], np.zeros(len(Q_new), dtype=int)
    inline_total = queued_total = 0
    for i, (qo, qn) in enumerate(zip(Q_old, Q_new)):
        s_old = V_old @ qo
        win = np.argpartition(-s_old, kk - 1)[:kk]
        win = win[np.argsort(-s_old[win])]          # best old score first
        miss = win[~have[win]]
        take, rest = miss[:max(budget_m, 0)], miss[max(budget_m, 0):]
        if take.size:
            have[take] = True
            inline_total += int(take.size)
            inline_q[i] = int(take.size)
        queued_total += int(rest.size)
        vis = win[have[win]]
        if vis.size == 0:
            out.append([])
        else:
            order = np.argsort(-(V_new[vis] @ qn))[:k]
            out.append([int(vis[j]) for j in order])
        if drain:
            # the background repairer lands them AFTER this query answered,
            # which is the whole point: they help later queries, not this one
            have[rest] = True
    return out, inline_total, queued_total, inline_q


def compare_schemes(V_old, V_new, Q_old_eval, Q_new_eval, Q_old_warm,
                    Q_new_warm=None, k: int = 10, kprime: int = 500,
                    budget: int | None = None, adapter_train: int = 0,
                    seed: int = 0,
                    budgets: tuple[int, ...] | None = (8, 16, 64)) -> list[dict]:
    """Run every scheme at the same budget and return one row each.

    `Q_old_warm` is the traffic that drives lazy migration before evaluation;
    `Q_*_eval` are the queries quality is measured on. They must differ, or
    lazy migration is being scored on the very queries that chose its units.

    `Q_new_warm` is that same warm-up traffic in the NEW model's space, needed
    for the `lazy+adapter` row alone. The adapter index approximates the new
    space, so searching it with old-space query vectors compares incomparable
    things. The hybrid therefore needs warm-up queries encoded by the new
    model -- a real cost of that variant, one new-model pass per distinct
    warm-up query, not a bookkeeping detail. Without it the row is skipped
    rather than computed wrongly.

    **The budget is derived, not imposed.** Lazy migration does not take a
    budget: it spends whatever the traffic demands, and the claim is that the
    total is far below N. So it runs first, unbounded, and what it spends
    becomes the budget every other scheme is given. Capping lazy at an
    arbitrary B instead would measure a system nobody would deploy -- one that
    stops repairing mid-window and serves partial results. Pass `budget` only
    to answer the different question of what each scheme does under a hard cap
    -- and note that lazy may then spend slightly less than the cap, because it
    stops at a window boundary, while the others take the cap exactly.
    """
    N = len(V_new)
    rng = np.random.default_rng(seed)
    truth = topk_chunked(Q_new_eval, V_new, k)
    rows = []

    # Lazy first: its spend sets the budget for everyone else. Lazy repairs
    # every window it serves, the evaluation queries' windows included -- the
    # `lazy` row below scores an exact rerank of the whole evaluation window,
    # which it only gets by encoding the members warm-up never reached. So its
    # spend is the union of warm-up AND evaluation windows. Counting warm-up
    # alone (as an earlier version did) handed lazy those evaluation repairs
    # for free and gave every other scheme a budget too small by the same
    # amount.
    cap = N if budget is None else budget
    have_warm = _touched(Q_old_warm, V_old, kprime, cap)
    have_l = have_warm | _touched(Q_old_eval, V_old, kprime, N)
    if budget is not None and have_l.sum() > budget:
        # a hard cap: keep arrival order, warm-up first, then evaluation
        have_l = _touched(np.vstack([Q_old_warm, Q_old_eval]), V_old, kprime, budget)
    spend = int(have_l.sum())
    budget = spend if budget is None else min(budget, N)

    rows.append(dict(scheme="batch", passes=N, passes_train=0, passes_serve=N,
                     recall_at_k=1.0,
                     note="re-embed everything; the reference"))

    old_top = topk_chunked(Q_old_eval, V_old, k)
    rows.append(dict(scheme="stale", passes=0, passes_train=0, passes_serve=0,
                     recall_at_k=_recall(truth, old_top, k),
                     note="keep the old index; why the migration happens"))

    W = None
    if adapter_train:
        widx = rng.choice(N, min(adapter_train, N - 1), replace=False)
        W = fit_linear_adapter(V_old, V_new, widx)
        V_hat = apply_adapter(V_old, W)
        rows.append(dict(scheme="adapter", passes=int(len(widx)),
                         passes_train=int(len(widx)), passes_serve=0,
                         recall_at_k=_recall(truth, topk_chunked(Q_new_eval, V_hat, k), k),
                         note="learned map; no unit is ever exact"))

    have_r = np.zeros(N, dtype=bool)
    have_r[rng.choice(N, budget, replace=False)] = True
    rows.append(dict(scheme="backfill-random", passes=int(have_r.sum()),
                     passes_train=0, passes_serve=int(have_r.sum()),
                     recall_at_k=_recall(
                         truth, _served_from_subset(Q_new_eval, V_new, have_r, k), k),
                     note="B units at random; the control for the ordering claim"))

    # an ordered backfill: a corpus-side priority, here proximity to the centroid
    centre = _safe_normalize(V_old.mean(axis=0, keepdims=True))
    order = np.argsort(-(V_old @ centre[0]))
    have_p = np.zeros(N, dtype=bool)
    have_p[order[:budget]] = True
    rows.append(dict(scheme="backfill-priority", passes=int(have_p.sum()),
                     passes_train=0, passes_serve=int(have_p.sum()),
                     recall_at_k=_recall(truth, _served_from_subset(Q_new_eval, V_new, have_p, k), k),
                     note="B units by a corpus-side order; serve from the new index"))

    # A traffic-ordered backfill can only know the traffic seen so far: the
    # warm-up windows. It migrates those, tops up at random to the same budget,
    # and serves globally from what it has, with no repair on the query path.
    # (Handing it lazy's full set would let it migrate the evaluation windows
    # before seeing those queries.)
    have_t = have_warm.copy()
    short = budget - int(have_t.sum())
    if short > 0:
        rest = np.flatnonzero(~have_t)
        have_t[rng.choice(rest, min(short, len(rest)), replace=False)] = True
    rows.append(dict(scheme="backfill-traffic", passes=int(have_t.sum()),
                     passes_train=0, passes_serve=int(have_t.sum()),
                     recall_at_k=_recall(
                         truth, _served_from_subset(Q_new_eval, V_new, have_t, k), k),
                     note="warm-up traffic's units, topped up at random to the "
                          "budget; served globally, no repair on the query path"))

    # --- Is calibration an alternative to repairing? ----------------------
    # The premise of the design is that no score may be compared across the two
    # spaces. The cheapest way to attack that premise is to calibrate instead
    # of repairing: keep the traffic-ordered backfill's unit set and rank by a
    # mixture of the two spaces' scores. These two rows give that alternative
    # its best shot -- one scalar fitted on warm-up traffic, and the standard
    # scale-free fusion -- and are charged exactly the traffic backfill's
    # passes, so the comparison is at equal cost and inside the same window.
    # (Keeping the window fixed matters: any loss here is the mixture's, not a
    # narrower candidate set's.)
    lam, lam_fit = 1.0, None
    if Q_new_warm is not None and len(Q_new_warm):
        fit_truth = topk_chunked(Q_new_warm, V_new, k)
        for cand in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
            r = _recall(fit_truth, _calibrated_serve(
                Q_old_warm, Q_new_warm, V_old, V_new, have_t, kprime, k, cand), k)
            if lam_fit is None or r > lam_fit:
                lam, lam_fit = cand, r
    rows.append(dict(scheme="merge-calibrated", passes=int(have_t.sum()),
                     passes_train=0, passes_serve=int(have_t.sum()),
                     recall_at_k=_recall(truth, _calibrated_serve(
                         Q_old_eval, Q_new_eval, V_old, V_new, have_t, kprime, k, lam), k),
                     note=f"source and target scores merged after per-window "
                          f"standardisation, lambda={lam} fitted on warm-up; same "
                          f"unit set and passes as backfill-traffic"))
    rows.append(dict(scheme="merge-rrf", passes=int(have_t.sum()),
                     passes_train=0, passes_serve=int(have_t.sum()),
                     recall_at_k=_recall(truth, _rrf_serve(
                         Q_old_eval, Q_new_eval, V_old, V_new, have_t, kprime, k), k),
                     note="reciprocal-rank fusion of the source window and the "
                          "repaired set; same unit set and passes as "
                          "backfill-traffic"))

    # An oracle backfill: lazy's exact unit set -- warm-up's windows AND the
    # evaluation queries' windows -- served globally from the new index with no
    # repair on the query path. It cannot be deployed, because it migrates the
    # units a query needs before that query arrives, and that is what makes it
    # the right control. The budget correction entangled two mechanisms in the
    # backfill-traffic row above (choosing units by traffic, and repairing them
    # at the moment a query needs them), because that row cannot know the
    # evaluation windows and so tops up at random instead. Holding the unit set
    # fixed to lazy's separates them again: traffic row vs oracle row is the
    # value of the ordering signal, oracle row vs lazy row is the value of
    # repairing on demand.
    rows.append(dict(scheme="backfill-oracle", passes=int(have_l.sum()),
                     passes_train=0, passes_serve=int(have_l.sum()),
                     recall_at_k=_recall(
                         truth, _served_from_subset(Q_new_eval, V_new, have_l, k), k),
                     note="lazy's exact unit set, served globally with no "
                          "query-path repair; not deployable -- it knows the "
                          "evaluation windows in advance"))

    win = topk_chunked(Q_old_eval, V_old, kprime)
    # a window unit traffic has not reached is repaired on this query, so the
    # user sees the exact new-space ranking over the whole window
    served = _rerank(Q_new_eval, V_new, win, k)
    rows.append(dict(scheme="lazy", passes=spend, passes_train=0,
                     passes_serve=spend,
                     recall_at_k=_recall(truth, served, k),
                     note=f"whatever traffic reached, K'={kprime}, exact rerank"))

    # --- What the latency costs, and what buying it back costs in quality ---
    # The rows above all answer a query by ranking something; none of them says
    # what the user waited for. Lazy's answer is exact over the window because
    # it repairs the whole window first, which is up to K' forward passes on the
    # critical path. These rows price the alternative: repair at most m of the
    # window's missing members inline and let a background repairer have the
    # rest.
    #
    # Every row here spends exactly what lazy spends. The background repairer
    # encodes what the inline budget did not, so the set re-embedded is the same
    # union of windows either way -- which is what the paper claims when it says
    # the savings analysis covers both variants, and it is asserted in the
    # test suite rather than argued. Only the latency and the quality move.
    for m, label, drain in ([(0, "lazy-async", True), (0, "lazy-async-cold", False)]
                            + [(int(b), f"lazy-budget-{int(b)}", True)
                               for b in (budgets or ())]):
        served_a, inline, queued, per_q = _async_serve(
            Q_old_eval, Q_new_eval, V_old, V_new, have_warm, kprime, k, m, drain)
        passes = int(have_warm.sum()) + inline + (queued if drain else 0)
        rows.append(dict(
            scheme=label, passes=passes, passes_train=0, passes_serve=passes,
            passes_inline=inline, passes_queued=queued,
            inline_per_query_mean=round(float(per_q.mean()), 2),
            inline_per_query_p95=int(np.percentile(per_q, 95)) if len(per_q) else 0,
            recall_at_k=_recall(truth, served_a, k),
            note=(f"at most {m} inline repairs per query, chosen by old-space "
                  f"rank; the rest "
                  + ("to a background repairer that keeps up"
                     if drain else "never land (a repairer that never runs)"))))

    if W is not None and Q_new_warm is not None:
        V_hat = apply_adapter(V_old, W)
        # the adapter index lives in an approximation of the NEW space, so the
        # traffic that warms it must be new-space queries
        win_h = topk_chunked(Q_new_eval, V_hat, kprime)   # adapter space ~ new space
        # as for lazy: it repairs the evaluation windows too
        have_h = _touched(Q_new_warm, V_hat, kprime, N)
        have_h[np.unique(win_h)] = True
        served_h = _rerank(Q_new_eval, V_new, win_h, k)
        rows.append(dict(scheme="lazy+adapter",
                         passes=int(have_h.sum()) + int(len(widx)),
                         passes_train=int(len(widx)),
                         passes_serve=int(have_h.sum()),
                         recall_at_k=_recall(truth, served_h, k),
                         note="adapter candidates, exact rerank; NOT equal-cost "
                              "-- it pays the adapter sample on top, and its "
                              "window is chosen in a different space, so it "
                              "reaches a different set of units"))
    return rows


def report(rows: list[dict], N: int) -> None:
    print(f"{'scheme':<20}{'train':>9}{'serve':>10}{'% of N':>9}"
          f"{'recall@k':>10}{'inline/q':>10}  note")
    for r in rows:
        # inline repairs per query is blank for every scheme that has no query
        # path to pay for: it is the latency column, not a cost column
        inl = (f"{r['inline_per_query_mean']:>10.1f}"
               if "inline_per_query_mean" in r else f"{'':>10}")
        print(f"{r['scheme']:<20}{r.get('passes_train', 0):>9,}"
              f"{r.get('passes_serve', r['passes']):>10,}"
              f"{100 * r['passes'] / N:>8.1f}%{r['recall_at_k']:>10.3f}{inl}"
              f"  {r['note']}")


def latency_ms(row: dict, unit_ms: float, window_ms: float = 8.3,
               rank_ms: float = 4.3) -> dict:
    """Turn an inline-repair count into the latency an operator would see.

    The counts are what this module measures; milliseconds are a property of a
    device and a window implementation, so they are supplied rather than
    assumed. `window_ms` and `rank_ms` default to the realization's measured
    HNSW window at K' = 500 and its ranking step.
    """
    mean = row.get("inline_per_query_mean", 0.0)
    p95 = row.get("inline_per_query_p95", 0)
    return dict(scheme=row["scheme"],
                mean_ms=round(window_ms + mean * unit_ms + rank_ms, 1),
                p95_ms=round(window_ms + p95 * unit_ms + rank_ms, 1))
