"""A CHAIN of unfinished upgrades: does abandoned work accumulate into an asset?

`gen2_union.py` asks whether ONE unfinished upgrade helps the next. This asks
the question a deployment actually lives in. If a migration need not finish,
then the next model always arrives mid-flight, and after T upgrades the store
holds

    index 0            complete, the oldest model, T generations stale
    index 1 .. T-1     PARTIAL, one per abandoned leg, covering R_1 .. R_{T-1}

Read literally the design nominates from index 0 alone, which ages with every
upgrade: its window is a weaker and weaker proxy for the newest model's top-k.
The invariant permits better. An old score only ever nominates, so every
abandoned index is a candidate source, and none of them costs a forward pass --
its vectors already exist, bought by the migration that left them behind.

Four policies, all ranking exactly in the newest space, all scored against that
space's own top-k:

    original-only     index 0 only: the design read literally
    previous-only     the most recent partial index only
    union-nearest     index 0 + the most recent partial index   (keep one)
    union-all         index 0 + every partial index             (keep all)

`union-nearest` against `union-all` is the system question: a partial index
costs storage and an insert per repair, so it matters whether the gain needs all
of them or only the last.

COST ACCOUNTING, which decides whether any of this is worth it. A unit repaired
in leg s holds a space-s vector, which cannot rank in space t > s. So at the
start of leg t no unit has a usable vector and leg t pays one f_t pass per unit
in the window it builds. A union window is larger, so it costs MORE passes than
`original-only` at the same K'. The control is therefore in the same table: the
`original-only` row at whatever K'' spends the same passes. A previous
generation is worth keeping only if it nominates better per pass than the
original index does.

THE SWEEP THAT MATTERS. A partial index can only nominate inside the set its
leg reached, so the gain is governed by rho = |R_s| / N, not by geometry alone.
`--rho` sets a target reached fraction per leg and the traffic is sized to hit
it, so the table is read along rho rather than at one accidental value. (A leg
that reached 99% leaves an index that is nearly complete, which is the easy
case and not the one to report.)

    python -m semview.runlog --name chain_msmarco --tag final -- \
      python analysis/chain.py \
        --spaces cache/MINILM.npy,cache/BGE_SMALL.npy,cache/E5_BASE.npy,cache/BGE_BASE.f16.npy \
        --queries dump/A/q_old.npy,dump/B/q_old.npy,dump/C/q_old.npy,dump/A/q_new.npy \
        --rho 0.2,0.4,0.6 --kprimes 200,500,1000 --out "{run}/chain.json"

`--spaces` and `--queries` are parallel lists: entry i is the corpus and the
same queries under model i, oldest first, newest last. Produce the query
matrices with `--dump-queries` on `migration`, one run per model. The command
is spelled out here rather than cross-referenced because RUNTESTS.md is not in
the shipped bundle, and a reader who cannot build the inputs cannot reproduce
the one experiment that decides the paper:

    # once per model, oldest first. --old is the model whose query vectors you
    # want; the corpus cache is keyed by hash and reused across runs.
    python -m semview.migration --corpus msmarco --n 500000 \
        --old sentence-transformers/all-MiniLM-L6-v2 \
        --new BAAI/bge-base-en-v1.5 \
        --kprimes 500 --dump-queries dump/leg0 --out /tmp/leg0.json

    # then: --spaces cache/<hash0>.npy,cache/<hash1>.npy,...
    #       --queries dump/leg0/q_old.npy,dump/leg1/q_old.npy,...

`--spaces` entry i must be the corpus under model i and `--queries` entry i the
same queries under model i, in the same order, oldest first. With T+1 spaces the
tool runs T legs.

`--synthetic` needs no files and no models: it builds a calibrated chain
(near-isometry plus a unit-deterministic distortion, alpha tuned so one
generation reproduces the measured MiniLM -> BGE agreement) and is how the test
suite covers this file. Synthetic numbers say nothing about MS MARCO; they only
show the shape to look for.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semview import baselines, migration  # noqa: E402
from semview.results_io import environment, write_results  # noqa: E402

POLICIES = ("original-only", "previous-only", "union-nearest", "union-all")


def _unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


DRIFTS = ("independent", "shared-distortion", "converging", "target-seeking")


def synthetic_chain(N: int, d: int, T: int, alpha: float = 0.5,
                    beta: float = 0.25, n_queries: int = 1200, seed: int = 0,
                    drift: str = "independent"):
    """A chain of spaces and the same queries in each.

    V_{t+1} = normalise(V_t R_t + alpha U_t): R_t orthogonal, so successive
    models largely agree, plus U_t a unit-norm distortion that is a function of
    the unit, as a real model's disagreement is (not per-row noise, which no
    model has).

    TWO ASSUMPTIONS, BOTH DECLARED. `alpha` is calibrated so that ONE
    generation reproduces a measured pair -- and the value that does so depends
    on N and K', not on the pair alone, because recall@K' depends on K'/N; use
    `calibrate_alpha` rather than a remembered constant. `drift` is how the
    legs AFTER the first are generated, which no measurement of one generation
    constrains:

    - `independent` -- a fresh rotation and distortion per leg: a random walk.
    - `shared-distortion` -- one rotation and distortion reused every leg:
      successive models moving consistently in one direction.
    - `converging` -- the step shrinks geometrically: successive models ever
      more alike, which is what sibling models in one family look like.
    - `target-seeking` -- every leg moves a fraction of the way to one fixed
      better space.

    Calibrated to the same one-generation recall, these four put the literal
    design's fourth-leg recall anywhere from 0.90 to 0.13 and the gap the chain
    section argues from anywhere from 0.06 to 0.58 (`analysis/drift_sweep.py`).
    The policy ORDER is stable across all four; the magnitude is not identified
    by anything measured, which is why the simulated table is illustrative and
    why D-08 is the only thing that places it.
    """
    if drift not in DRIFTS:
        raise SystemExit(f"--drift must be one of {', '.join(DRIFTS)}")
    rng = np.random.default_rng(seed)
    cur = _unit(rng.normal(size=(N, d)).astype(np.float32))
    Vs = [cur]
    R0, _ = np.linalg.qr(rng.normal(size=(d, d)))
    S0 = rng.normal(size=(d, d)).astype(np.float32)
    target = _unit(rng.normal(size=(N, d)).astype(np.float32))
    for t in range(T):
        a = alpha * (0.6 ** t) if drift == "converging" else alpha
        if drift == "target-seeking":
            cur = _unit((1 - a) * cur + a * target)
            Vs.append(cur)
            continue
        if drift == "shared-distortion":
            R, S = R0, S0
        else:
            R, _ = np.linalg.qr(rng.normal(size=(d, d)))
            S = rng.normal(size=(d, d)).astype(np.float32)
        cur = _unit(cur @ R.astype(np.float32)
                    + a * _unit(np.tanh(cur @ S.astype(np.float32))))
        Vs.append(cur)
    pick = rng.choice(N, n_queries, replace=False)
    # one query, seen by every model: that unit's vector in each space, nudged
    # by the same perturbation index so the spaces see the same query text
    noise = _unit(rng.normal(size=(n_queries, d)).astype(np.float32))
    Qs = [_unit(V[pick] + beta * noise) for V in Vs]
    return Vs, Qs


def one_generation_recall(N, d, alpha, kprime, k=10, eval_queries=150,
                          seed=0, drift="independent") -> float:
    """Recall@K' of the first leg alone: the quantity alpha is calibrated on."""
    Vs, Qs = synthetic_chain(N, d, 2, alpha, n_queries=max(300, 2 * eval_queries),
                             seed=seed, drift=drift)
    truth = [np.asarray(r, dtype=np.int64) for r in
             migration.topk_chunked(Qs[1][:eval_queries], Vs[1], k)]
    return float(baselines._recall(
        truth, _windows(Qs[0][:eval_queries], Vs[0], kprime), k))


def calibrate_alpha(N, d, kprime, target: float, k=10, eval_queries=150,
                    seed=0, drift="independent", lo=0.02, hi=1.3, steps=14):
    """The alpha whose FIRST leg reproduces `target` at this N and K'.

    Written because the tool shipped alpha=0.5 with a docstring claiming it
    reproduced the measured 0.929, which it did at the N=20,000 it was fitted
    at and not at the N=200,000 the tool runs: there it gives 0.827. Calibrating
    here removes the class of error rather than the instance. Returns
    (alpha, achieved).
    """
    for _ in range(steps):
        mid = (lo + hi) / 2
        if one_generation_recall(N, d, mid, kprime, k, eval_queries, seed,
                                 drift) > target:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    return a, one_generation_recall(N, d, a, kprime, k, eval_queries, seed, drift)


def _windows(Q, V, kprime, rows=None):
    """Exact top-K' per query, optionally within a partial index's rows."""
    kp = int(min(kprime, len(V) if rows is None else len(rows)))
    if kp <= 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(len(Q))]
    if rows is None:
        return [np.asarray(r, dtype=np.int64)
                for r in migration.topk_chunked(Q, V, kp)]
    rows = np.asarray(rows, dtype=np.int64)
    sub = migration.topk_chunked(Q, np.ascontiguousarray(V[rows]), kp)
    return [rows[np.asarray(r, dtype=np.int64)] for r in sub]


def traffic_for_rho(Q, V, kprime: int, rho: float, N: int) -> int:
    """How many queries a leg must serve to reach `rho` of the corpus.

    The reached set is the union of those queries' windows, so it grows with
    the stream and saturates; this walks the stream once and stops at the
    target. Returns the query count, or raises if the stream cannot get there.
    """
    wins = _windows(Q, V, kprime)
    seen = np.zeros(N, dtype=bool)
    # Counted incrementally rather than re-summed: `seen.sum()` is O(N) and was
    # evaluated once per query, to answer a question that only changes by the
    # units each query adds. Measured benefit at the tool's default N=200,000
    # is 1.01x, i.e. none -- after the block fix the loop only walks the queries
    # it actually consumes, which in rho mode is tens. Kept because it is exact
    # (A/B identical sweeps on four configurations) and because an O(N) test
    # inside a per-query loop is a trap for whoever next raises N.
    n_seen = 0
    target = rho * N
    for i, w in enumerate(wins, start=1):
        n_seen += int((~seen[w]).sum())
        seen[w] = True
        if n_seen >= target:
            return i
    raise SystemExit(
        f"the {len(Q)} available queries reach {seen.sum() / N:.1%} of the corpus "
        f"at K'={kprime}, short of the --rho target {rho:.0%}. Dump more queries, "
        f"lower --rho, or raise --kprimes.")


# Which indexes a policy nominates from while it serves a leg. `original`
# says whether index 0 is still a candidate source; `partials` picks the
# abandoned partial indexes, newest last.
_SOURCES = {
    "original-only": (True, lambda t, R: []),
    "union-nearest": (True, lambda t, R: [t - 1] if (t - 1) in R else []),
    "union-all": (True, lambda t, R: sorted(R)),
}
# `previous-only` has no history of its own because it is a diagnostic, not a
# deployment: serving ONLY from the most recent partial index can never repair
# a unit outside what that index already covers, so its reached set shrinks
# monotonically and the policy strangles itself. The row answers "how much of
# the answer lives in the most recent partial index", inside the history
# `union-nearest` would have produced, and every row carries the `history`
# field saying so.
_HISTORY_OF = {"previous-only": "union-nearest"}


def _run_legs(policy, Vs, Qs, slices, N, T, kpr, rho, queries_per_leg):
    """Serve every leg under one policy and return (R, legs_meta).

    R[s] is the coverage of the partial index leg s left behind, under THIS
    policy's nomination rule. A policy that serves from more indexes reaches
    more of the corpus per query, so it leaves richer partial indexes behind
    and, by the same token, spends more forward passes: both halves of that
    trade come from this loop, and scoring every policy against one shared
    history would flatter the narrower ones.
    """
    uses_original, pick = _SOURCES[policy]
    R, legs_meta = {}, []
    for t in range(1, T + 1):
        pool = slices[t - 1]
        parts = pick(t, R)
        srcs = []
        if uses_original or not parts:
            srcs.append((Qs[0][pool], Vs[0], None))
        srcs.extend((Qs[s][pool], Vs[s], R[s]) for s in parts)
        reached = np.zeros(N, dtype=bool)
        n_reached = 0
        n_used = None
        # Two ways to size a leg, and they answer different questions.
        # `queries_per_leg` holds the TRAFFIC constant, which is what a
        # deployment has: a policy serving from more indexes then reaches more
        # of the corpus per query. Holding `rho` constant instead equalises
        # coverage across legs and policies, which isolates candidate quality
        # from coverage -- and makes the traffic each leg needs the variable.
        limit = len(pool) if queries_per_leg is None else min(queries_per_leg,
                                                              len(pool))
        # Windows are computed in blocks and only as far as the stream is
        # actually consumed. Computing them for the whole pool up front was
        # 140x more work than a rho-sized leg uses and 32x more than a
        # traffic-sized one: the cost scaled with the size of the query dump
        # rather than with the traffic being simulated, which is the wrong
        # variable for the one experiment that decides the paper. Same queries,
        # same order, same result -- the discarded ones were never read.
        BLOCK = 64
        done = False
        for lo in range(0, limit, BLOCK):
            hi = min(lo + BLOCK, limit)
            blocks = [_windows(q[lo:hi], V, kpr, rows=r) for q, V, r in srcs]
            for i in range(hi - lo):
                for w in blocks:
                    # same incremental count as `traffic_for_rho`, for the same
                    # reason: this test ran once per query over a length-N array
                    n_reached += int((~reached[w[i]]).sum())
                    reached[w[i]] = True
                if queries_per_leg is None and n_reached >= rho * N:
                    n_used = lo + i + 1
                    done = True
                    break
            if done:
                break
        if queries_per_leg is not None:
            n_used = limit
        if n_used is None:
            raise SystemExit(
                f"{policy}, leg {t}: its {len(pool)} queries reach "
                f"{reached.sum() / N:.1%} of the corpus at K'={kpr}, short of "
                f"the --rho target {rho:.0%}. Dump more queries, lower --rho, "
                f"or raise --reach-depth.")
        R[t] = np.flatnonzero(reached)
        legs_meta.append(dict(policy=policy, leg=t, reach_depth=int(kpr),
                              warm_queries=int(n_used),
                              sources_while_serving=len(srcs),
                              reached=int(reached.sum()),
                              reached_frac=round(float(reached.mean()), 4),
                              cumulative_encoded=int(sum(len(R[s])
                                                         for s in sorted(R))),
                              cumulative_encoded_pct=round(
                                  100 * sum(len(R[s]) for s in sorted(R)) / N, 1)))
    return R, legs_meta


def run_chain(Vs, Qs, k: int = 10, kprimes=(200, 500, 1000), rho: float = 0.4,
              eval_queries: int = 400, kprime_reach: int | None = None,
              seed: int = 0, queries_per_leg: int | None = None):
    """T = len(Vs) - 1 legs. Returns (rows, meta).

    Each leg is driven by its own traffic, disjoint from the evaluation stream
    and from the other legs': a leg must not be warmed by the queries that then
    measure it, and two legs sharing a stream would make the partial indexes
    agree for a reason no deployment has.

    Every policy serves the legs itself, so each is scored against the partial
    indexes it would have produced and charged for the forward passes that
    producing them took. `cumulative_encoded` is that charge and is the cost
    axis the comparison turns on; `passes` is the evaluation stream's own
    repair work at the leg measured, which is a snapshot rather than a bill.
    """
    T = len(Vs) - 1
    N = len(Vs[0])
    if T < 2:
        raise SystemExit("a chain needs at least 3 spaces (2 legs)")
    nq = len(Qs[0])
    ev = min(eval_queries, nq // (T + 1))
    if ev < 20:
        raise SystemExit(f"{nq} queries cannot give {T} legs their own traffic "
                         f"plus an evaluation stream; dump more")
    # the evaluation stream is the first `ev`; each leg draws from its own slice
    # of the rest, so no leg sees the evaluation queries or another leg's
    rest = np.arange(ev, nq)
    slices = np.array_split(rest, T)
    # Reach depth -- how deep a leg repairs while it serves -- follows each
    # row's own K' unless `kprime_reach` pins it, because a deployment serving
    # at K' repairs at K'. Pinning it answers the other question: what several
    # serving depths read out of one fixed history.

    # The cost axis is only informative while the evaluation stream does not
    # touch the whole corpus: its load is x = ev * K' / N, the paper's own
    # quantity. At x >> 1 every policy repairs everything and the passes column
    # says nothing -- which is the trap gen2_union fell into at 99.6% reached.
    x_hi = ev * max(kprimes) / N
    if x_hi > 3:
        print(f"  WARNING: {ev} evaluation queries at K'={max(kprimes)} over "
              f"N={N:,} is a load of x={x_hi:.1f}. Above about 3 every policy "
              f"repairs the whole corpus and the passes column is saturated. "
              f"Lower --eval-queries to about {max(20, int(3 * N / max(kprimes)))} "
              f"or use a larger corpus.")
    rows, legs_meta = [], []
    # hist[(policy, reach_depth)][s] = the rows leg s reached while that policy
    # served at that depth. A deployment that serves at K' repairs at K', so
    # unless --reach-depth pins it, each K' in the sweep gets its own history
    # and every row is charged for the passes its own configuration spent.
    # Cached, because a pinned depth makes every K' share one simulation.
    hist: dict[tuple[str, int], dict[int, np.ndarray]] = {}

    def history(policy, depth):
        key = (policy, depth)
        if key not in hist:
            R, lm = _run_legs(policy, Vs, Qs, slices, N, T, depth, rho,
                              queries_per_leg)
            hist[key] = R
            legs_meta.extend(lm)
        return hist[key]

    q_ev = {s: Qs[s][:ev] for s in range(T + 1)}
    w0_cache: dict[int, list] = {}
    for t in range(1, T + 1):
        truth = [np.asarray(r, dtype=np.int64) for r in
                 migration.topk_chunked(q_ev[t], Vs[t], min(k, N))]
        for kp in kprimes:
            depth = int(kprime_reach) if kprime_reach else int(kp)
            # index 0's evaluation window does not depend on the leg, so it
            # was being recomputed T times per depth -- 12 full top-k passes
            # for T=4 and three depths, 9 of them identical to one already
            # done. Cached by depth.
            if kp not in w0_cache:
                w0_cache[kp] = _windows(q_ev[0], Vs[0], kp)
            w0 = w0_cache[kp]

            def add(policy, wins, note, n_sources, store):
                wins = [np.asarray(w, dtype=np.int64) for w in wins]
                touched = np.zeros(N, dtype=bool)
                for w in wins:
                    touched[w] = True
                served = baselines._rerank(q_ev[t], Vs[t], wins, k)
                R_ = history(_HISTORY_OF.get(policy, policy), depth)
                enc = sum(len(R_[s]) for s in range(1, t + 1))
                # `passes` is every unit the evaluation windows touch, which
                # includes the ones THIS leg's own warm-up already repaired --
                # those need no second forward pass, and they are already
                # charged inside `cumulative_encoded`, so reporting both as a
                # cost double-counts them. The marginal figure is what the
                # evaluation stream adds on top of the warm-up; `total_passes`
                # is the two together and is the only one that is a bill.
                # Marginal alone would flatter the wider policies, whose
                # warm-up already paid for most of what they touch: the cost
                # moves, it does not vanish.
                warm = np.zeros(N, dtype=bool)
                warm[R_[t]] = True
                marginal = int((touched & ~warm).sum())
                rows.append(dict(
                    policy=policy, leg=int(t), kprime=int(kp),
                    passes=int(touched.sum()),
                    passes_pct=round(100 * float(touched.mean()), 2),
                    eval_passes_marginal=marginal,
                    eval_passes_marginal_pct=round(100 * marginal / N, 2),
                    cumulative_encoded=int(enc),
                    cumulative_encoded_pct=round(100 * enc / N, 1),
                    total_passes=int(enc + marginal),
                    total_passes_pct=round(100 * (enc + marginal) / N, 1),
                    window_mean=round(float(np.mean([len(w) for w in wins])), 1),
                    recall_candidates=round(baselines._recall(truth, wins, k), 4),
                    recall_at_k=round(baselines._recall(truth, served, k), 4),
                    sources=int(n_sources),
                    partial_rows_kept=int(store),
                    history=_HISTORY_OF.get(policy, policy),
                    reach_depth=int(depth),
                    note=note))

            add("original-only", w0,
                "index 0 only: the design read literally", 1, 0)
            if t >= 2:
                for pol in ("previous-only", "union-nearest", "union-all"):
                    R = history(_HISTORY_OF.get(pol, pol), depth)
                    prev = _windows(q_ev[t - 1], Vs[t - 1], kp, rows=R[t - 1])
                    if pol == "previous-only":
                        add(pol, prev,
                            f"partial index {t - 1} only, inside the "
                            f"{len(R[t - 1]):,} rows it reached; a diagnostic, "
                            "not a deployment (see _SOURCES)", 1,
                            len(R[t - 1]))
                    elif pol == "union-nearest":
                        add(pol, [np.union1d(a, b) for a, b in zip(w0, prev)],
                            "index 0 plus the most recent partial index", 2,
                            len(R[t - 1]))
                    else:
                        allw = w0
                        kept = 0
                        for s in range(1, t):
                            allw = [np.union1d(a, b) for a, b in
                                    zip(allw, _windows(q_ev[s], Vs[s], kp,
                                                       rows=R[s]))]
                            kept += len(R[s])
                        add(pol, allw,
                            "index 0 plus every partial index left behind",
                            t, kept)
    return rows, dict(N=int(N), T=int(T), k=int(k), rho_target=float(rho),
                      queries_per_leg=(int(queries_per_leg)
                                       if queries_per_leg else None),
                      sizing=("constant traffic" if queries_per_leg
                              else "constant reached fraction"),
                      kprimes=[int(x) for x in kprimes],
                      kprime_reach=(int(kprime_reach) if kprime_reach else None),
                      reach_follows_kprime=kprime_reach is None,
                      eval_queries=int(ev),
                      eval_load_x=round(float(x_hi), 3), legs=legs_meta)



def report(rows, meta) -> None:
    N, T = meta["N"], meta["T"]
    sizing = (f"{meta['queries_per_leg']} queries per leg"
              if meta.get("queries_per_leg")
              else f"rho target {meta['rho_target']:.0%} per leg")
    print(f"\n  N={N:,}  {T} legs  k={meta['k']}  {sizing}  "
          f"{meta['eval_queries']} evaluation queries "
          f"(load x={meta['eval_load_x']})")
    depth = (f"K'={meta['kprime_reach']}" if meta.get("kprime_reach")
             else "each row's own K'")
    print(f"    legs served at reach depth {depth}; each policy serves them "
          f"itself and is charged for the passes that took")
    for pol in POLICIES:
        ls = [L for L in meta["legs"] if L["policy"] == pol]
        if not ls:
            continue
        print(f"    {pol:<16} reached " +
              " ".join(f"{L['reached_frac']:>6.1%}" for L in ls) +
              f"   cumulative encode {ls[-1]['cumulative_encoded_pct']:>6.1f}% of N"
              + ("   (" + " ".join(f"{L['warm_queries']}q" for L in ls) + ")"
                 if meta.get("queries_per_leg") is None else ""))
    for kp in meta["kprimes"]:
        print(f"\n  K'={kp}:  recall@k served, by leg")
        hdr = f"  {'policy':<16}" + "".join(f"{'leg ' + str(t):>12}"
                                            for t in range(1, T + 1))
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for pol in POLICIES:
            cells = []
            for t in range(1, T + 1):
                m = [r for r in rows if r["policy"] == pol and r["leg"] == t
                     and r["kprime"] == kp]
                cells.append(f"{m[0]['recall_at_k']:.3f}" if m else "-")
            print(f"  {pol:<16}" + "".join(f"{c:>12}" for c in cells))
        print(f"  {'passes, % of N':<16}" + "".join(
            f"{next((r['passes_pct'] for r in rows if r['policy'] == 'union-all' and r['leg'] == t and r['kprime'] == kp), 0):>11.1f}%"
            for t in range(1, T + 1)) + "   (union-all)")

    # The comparison that decides it, on the axis that is a bill: how many
    # forward passes the policy spent over the whole chain, against what it
    # serves at the end. `passes` is the evaluation stream's own repair work,
    # which is a snapshot of one leg, not the cost of getting there.
    last = [r for r in rows if r["leg"] == T]
    print(f"\n  at the last leg, cheapest first, charged for the whole chain: "
          f"total = warm-up over {T} legs + what the evaluation adds")
    print(f"  {'policy':<16}{'K′':>6}{'warm-up':>10}{'+eval':>9}"
          f"{'total':>10}{'% of N':>9}{'recall':>9}")
    print("  " + "-" * 70)
    for r in sorted(last, key=lambda r: (r["total_passes"], r["passes"])):
        print(f"  {r['policy']:<16}{r['kprime']:>6}"
              f"{r['cumulative_encoded']:>10,}{r['eval_passes_marginal']:>9,}"
              f"{r['total_passes']:>10,}{r['total_passes_pct']:>8.1f}%"
              f"{r['recall_at_k']:>9.3f}")
    lits = [r for r in last if r["policy"] == "original-only"]
    best_lit = max((r["recall_at_k"] for r in lits), default=0.0)
    # The control, and it takes no threshold: a union row counts only if some
    # `original-only` row spends AT LEAST as much in total and still serves
    # less. If the depth ladder holds no such rung the comparison is not
    # passed, it is UNTESTED, and saying so is the difference between a control
    # and a formality.
    wins, untested = [], []
    for r in last:
        if not r["policy"].startswith("union"):
            continue
        # The control pool is every literal row that spends AT LEAST as much.
        # This has twice been proposed as a bug ("not matched budget") with a
        # fix to a +-5% window. It is deliberate, and the window would WEAKEN
        # it. Literal recall and literal cost are both monotone in K' (checked
        # on every chain run in the bundle), so this pool is the deepest-K'
        # tail and `max(recall)` over it is the deepest literal row measured.
        # The test is therefore "union beats the best literal that is no
        # cheaper", which is strictly harder than "union beats a literal of
        # about the same price": a +-5% window admits controls spending LESS
        # than the union row, and on the rho run it turns an UNTESTED row
        # (union-nearest at K'=1000, 198.1%) into a win against a control at
        # 192.3%. Section 7's separate "within 1.05x encode" clause is
        # redundant for the same reason -- the control already spends at least
        # as much, by construction.
        ctrl = [c for c in lits if c["total_passes"] >= r["total_passes"]]
        if not ctrl:
            untested.append(r)
            continue
        # The control the win is DECIDED against is the STRONGEST row spending
        # at least as much -- beating a weaker one that merely costs more is
        # not a win. An earlier version of this function carried the cheapest
        # qualifying control alongside the row and printed that instead, and a
        # reader of the printout took the printed control for the decisive one:
        # at rho=0.6 that made union-all at K'=2000 look like +0.0287 against
        # literal K'=4000 (0.9040) when the deciding comparison is against
        # literal K'=8000 (0.9307), i.e. +0.0020 -- fourteen times smaller, on
        # the row a branch decision rests on. So the decisive control is what
        # is carried and what is printed.
        best_c = max(ctrl, key=lambda c: c["recall_at_k"])
        if r["recall_at_k"] > best_c["recall_at_k"]:
            wins.append((r, best_c))
    if wins:
        r, c = min(wins, key=lambda p: p[0]["total_passes"])
        print(f"\n  matched on total passes: {r['policy']} at K'={r['kprime']} "
              f"serves {r['recall_at_k']:.3f} for {r['total_passes_pct']:.1f}% "
              f"of N,\n  against the STRONGEST original-only row spending at "
              f"least as much -- K'={c['kprime']}, serving {c['recall_at_k']:.3f} "
              f"for {c['total_passes_pct']:.1f}% -- a margin of "
              f"{r['recall_at_k'] - c['recall_at_k']:+.4f}.")
    elif untested:
        print(f"\n  UNTESTED: no original-only row spends as much as "
              f"{', '.join(sorted({r['policy'] for r in untested}))}, so nothing "
              f"here controls for cost.\n  Add a deeper K' to --kprimes; the "
              f"literal design's best at this depth is {best_lit:.3f}.")
    else:
        print(f"\n  No union policy beat an original-only row that spent at least "
              f"as much; the literal design's best here is {best_lit:.3f}.")


def main() -> None:  # pragma: no cover - exercised through the CLI
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spaces", help="comma-separated .npy corpus matrices, "
                                     "oldest first, newest last")
    ap.add_argument("--queries", help="the same queries under each model, in "
                                      "the same order as --spaces")
    ap.add_argument("--synthetic", action="store_true",
                    help="a calibrated synthetic chain; no files, no models")
    ap.add_argument("--n", type=int, default=200_000,
                    help="synthetic: corpus size. Keep eval_queries * max(K') / n "
                         "below about 3 or the passes column saturates")
    ap.add_argument("--dim", type=int, default=64, help="synthetic: dimension")
    ap.add_argument("--legs", type=int, default=4, help="synthetic: number of legs")
    ap.add_argument("--alpha", type=float, default=None,
                    help="synthetic: per-generation drift. Omit it and alpha is "
                         "CALIBRATED at this run's own N and K' so one "
                         "generation reproduces --calibrate-to; a fixed value "
                         "is only correct at the N it was fitted at, which is "
                         "how 0.5 came to be shipped for an N where it gives "
                         "0.827 rather than the 0.929 it claimed")
    ap.add_argument("--calibrate-to", type=float, default=0.929,
                    help="synthetic: the measured one-generation recall@K' that "
                         "alpha is fitted to (default: the MiniLM->BGE pair)")
    ap.add_argument("--drift", default="independent", choices=DRIFTS,
                    help="synthetic: how the legs after the first are "
                         "generated. NOTHING in a one-generation measurement "
                         "constrains this, and calibrated to the same first "
                         "leg these put the chain gap anywhere from 0.06 to "
                         "0.58, so a single choice is illustrative and the "
                         "sweep is the honest report (analysis/drift_sweep.py)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kprimes", default="200,500,1000")
    ap.add_argument("--rho", default="0.4",
                    help="target reached fraction per leg; a comma-separated "
                         "list sweeps it. Holding this constant isolates chain "
                         "length from the compounding --queries-per-leg shows")
    ap.add_argument("--queries-per-leg", type=int, default=None,
                    help="instead of --rho: give every leg the same traffic, "
                         "which is what a deployment has. Later legs then reach "
                         "more per query, because they nominate from more "
                         "indexes, and leave richer partial indexes behind")
    ap.add_argument("--reach-depth", type=int, default=None,
                    help="the depth a leg repairs at while it serves. A "
                         "deployment serving at K' repairs at K', so a run "
                         "that sweeps --kprimes shares one reach depth and "
                         "only the matching row is self-consistent. Left "
                         "unset, each row repairs at ITS OWN K' -- not at "
                         "max(--kprimes), which this help used to claim -- so "
                         "every row is self-consistent and the rows do not "
                         "share one history. Set it to compare depths against "
                         "a single reach depth instead.")
    ap.add_argument("--eval-queries", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kprimes = tuple(int(x) for x in a.kprimes.split(","))

    alpha_was_none = a.alpha is None
    achieved = None
    if a.synthetic:
        kp_cal = int(min(kprimes))
        alpha = a.alpha
        if alpha is None:
            alpha, achieved = calibrate_alpha(
                a.n, a.dim, kp_cal, a.calibrate_to, a.k, a.eval_queries,
                a.seed, a.drift)
            print(f"  calibrated alpha={alpha:.4f} at N={a.n:,}, K'={kp_cal}, "
                  f"drift={a.drift}: one generation gives {achieved:.4f} "
                  f"against the target {a.calibrate_to}")
        else:
            achieved = one_generation_recall(a.n, a.dim, alpha, kp_cal, a.k,
                                             a.eval_queries, a.seed, a.drift)
            print(f"  alpha={alpha} was GIVEN, not calibrated: at N={a.n:,}, "
                  f"K'={kp_cal} it puts one generation at {achieved:.4f}. "
                  f"Omit --alpha to calibrate it here.")
        a.alpha = alpha
        Vs, Qs = synthetic_chain(a.n, a.dim, a.legs, alpha,
                                 n_queries=max(1200, 4 * a.eval_queries * a.legs),
                                 seed=a.seed, drift=a.drift)
        names = [f"synthetic-{a.drift}-{i}" for i in range(len(Vs))]
    else:
        if not (a.spaces and a.queries):
            raise SystemExit("--spaces and --queries, or --synthetic")
        sp = a.spaces.split(",")
        qp = a.queries.split(",")
        if len(sp) != len(qp):
            raise SystemExit(f"--spaces has {len(sp)} entries, --queries {len(qp)}; "
                             "entry i must be the corpus and the queries under model i")
        Vs = [np.load(p, mmap_mode="r") for p in sp]
        Qs = [np.asarray(np.load(p), dtype=np.float32) for p in qp]
        if len({len(V) for V in Vs}) != 1:
            raise SystemExit("every corpus matrix must share a row order and length")
        if len({len(Q) for Q in Qs}) != 1:
            raise SystemExit("every query matrix must hold the same queries")
        names = sp

    sweeps = []
    for rho in (float(x) for x in a.rho.split(",")):
        rows, meta = run_chain(Vs, Qs, a.k, kprimes, rho, a.eval_queries,
                               kprime_reach=a.reach_depth, seed=a.seed,
                               queries_per_leg=a.queries_per_leg)
        report(rows, meta)
        sweeps.append(dict(meta=meta, rows=rows))

    obj = {"tool": "chain", "k": a.k, "kprimes": list(kprimes),
           "synthetic": bool(a.synthetic),
           # Carried in the file itself so a reader of the JSON cannot mistake
           # its standing, and so the paper's label can be checked against it.
           # alpha is fitted on leg 1 and the table is read at leg 4: fixing the
           # calibration does not make that an extrapolation anyone measured.
           "status": ("illustrative" if a.synthetic else "measured"),
           "inputs": {"spaces": names, "queries": (a.queries.split(",")
                                                   if a.queries else None),
                      # the two assumptions, recorded with the numbers they
                      # produced, so a result can never again be read without
                      # them
                      "alpha": a.alpha if a.synthetic else None,
                      "alpha_calibrated": bool(a.synthetic and alpha_was_none),
                      "alpha_target": a.calibrate_to if a.synthetic else None,
                      "one_generation_recall": achieved if a.synthetic else None,
                      "drift": a.drift if a.synthetic else None},
           "environment": environment(dtype=str(np.asarray(Vs[-1][:1]).dtype)),
           "fingerprint": {f"space_{i}": migration._fingerprint_strided(V)
                           for i, V in enumerate(Vs)},
           "sweeps": sweeps}
    write_results(a.out, obj)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
