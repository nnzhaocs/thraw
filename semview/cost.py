"""Lazy migration against a batch re-embed: the whole cost, both sides.

The obvious objection to lazy migration is that re-embedding a corpus is a
routine overnight batch job, so why trade it for anything. The answer has to
be arithmetic.

Batch migration pays, once:

    N unit passes with the new model.

Lazy migration pays, spread over traffic:

    reached * N unit passes with the new model   (only what queries touch)
  + S         query passes with the OLD model    (candidate generation)

where S is the number of distinct query STRINGS: old-model query embeddings are
cached by exact text. S is not D, the number of distinct intents that governs
coverage -- paraphrases of one intent reach the same units but miss the cache,
so S >= D. Uncached, the query term is Q, the total query volume.

The query term is small per pass (a query is short text) but permanent: it is
paid for as long as any unit stays in the old space. Whether it matters is
decided by one condition:

    Q * query_ms  >  (1 - reached) * N * unit_ms

i.e. the scheme inverts past Q = (1 - reached) * N * (unit_ms / query_ms)
queries if nothing is cached. How large that is depends entirely on the shape.
At the MEASURED MS MARCO point (N = 500,000, reached = 0.57 at D = 1,000,
passages ~4x the cost of a query) it is about 860,000 queries -- well inside a
normal traffic month, so intent-level query caching is load-bearing there. On a
larger corpus with a smaller reached fraction the same arithmetic gives tens of
millions. Compute it for your shape rather than carrying a rule of thumb.

Three things this model does not put in the GPU ledger, and should be reported
beside it:

  * First-touch latency. A query whose whole window is stale re-embeds up to K'
    passages on the query path. That is the cost users see; `report` prints it.
  * Batch-size asymmetry. Measure `repair_unit_ms` at the batch sizes the query
    path actually produces (K' at the start of a migration, a handful once most
    units are fresh), not at the batch job's. Leaving it unset assumes the query
    path is as efficient as an overnight job, which flatters lazy migration.
  * Keeping the old encoder resident, and running two model versions.
  * Drift. `reached` is measured for one intent population. If intents keep
    arriving, D grows with time and savings decay as (1 - K'/N)^D(t); "never
    paid" means never paid for units no intent reaches, not a fixed fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CostParams:
    N: int                      # corpus units
    D: int                      # distinct intents over the migration window
    Q: int                      # total query volume over that window
    reached: float              # measured by semview.estimate, at this D
    unit_ms: float = 10.4       # one unit pass in the BATCH job's large batches
    repair_unit_ms: float | None = None   # one unit pass on the QUERY path
    query_ms: float = 2.0       # one query forward pass (short text)
    cache_hit_rate: float = 1.0 # share of repeat query strings served from cache
    S: int | None = None        # distinct query strings (>= D); defaults to D
    usd_per_1k_units: float = 0.0
    usd_per_1k_queries: float = 0.0


def reached_at(est_row: dict, D: int) -> float:
    """`reached` for D intents, read from an estimate row's coverage curve.

    A row's `reached_observed` holds only for the D it was measured at. Pairing
    it with a different D -- reached measured over 50,000 intents, priced as if
    there were 10,000 -- misstates the saving. The curve is interpolated in
    log D; values beyond the replayed log are refused rather than extrapolated.
    """
    import numpy as np

    if D == est_row.get("D"):
        return float(est_row["reached_observed"])
    ds, rc = est_row.get("curve_d"), est_row.get("curve_reached")
    if not ds or not rc:
        raise ValueError(f"row measured at D={est_row.get('D')} has no coverage "
                         f"curve, so it cannot price D={D}; rerun semview.estimate")
    if len(ds) < 2:
        raise ValueError(f"coverage curve has {len(ds)} point(s); reached_at "
                         "needs at least two to interpolate between")
    if D > max(ds):
        raise ValueError(f"D={D} is beyond the replayed log (max {max(ds)}); "
                         "replay more queries rather than extrapolating")
    if D < min(ds):
        raise ValueError(f"D={D} is below the curve's first point ({min(ds)})")
    return float(np.interp(math.log(D), [math.log(x) for x in ds], rc))


def from_estimate(est_row: dict, Q: int, D: int | None = None,
                  N: int | None = None, **kw) -> CostParams:
    """Build CostParams from a `semview.estimate` row.

    Preferred over constructing CostParams by hand: `reached` is the input that
    decides the answer. D defaults to the row's own D; a different D is read
    off the row's coverage curve, never paired with a `reached` measured
    elsewhere.
    """
    if "reached_observed" not in est_row:
        raise ValueError("expected a row from semview.estimate.estimate(); "
                         f"got keys {sorted(est_row)[:8]}")
    n = int(est_row["N"] if N is None else N)
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    d = int(est_row["D"] if D is None else D)
    reached = reached_at(est_row, d)
    if not 0.0 <= reached <= 1.0:
        raise ValueError(f"reached must lie in [0, 1], got {reached}")
    return CostParams(N=n, D=d, Q=Q, reached=reached, **kw)


def compare(p: CostParams) -> dict:
    S = p.D if p.S is None else p.S
    if S < p.D:
        raise ValueError(f"S (distinct query strings, {S}) cannot be below D "
                         f"(distinct intents, {p.D})")
    batch_units = p.N
    lazy_units = p.reached * p.N
    # A batch job encodes in batches of thousands; the query path encodes the
    # stale units inside one window, which starts at K' and shrinks toward a
    # handful as the migration progresses. Small batches are far less efficient
    # per unit, so using one unit cost for both sides understates lazy cost.
    repair_ms = p.unit_ms if p.repair_unit_ms is None else p.repair_unit_ms

    # Each distinct string costs one old-model pass; repeats of a string cost
    # one more pass each unless served from the cache.
    old_passes = S + (1.0 - p.cache_hit_rate) * max(p.Q - S, 0)

    batch_ms = batch_units * p.unit_ms
    lazy_ms = lazy_units * repair_ms + old_passes * p.query_ms

    saved_units = batch_units - lazy_units
    net_ms = batch_ms - lazy_ms

    return dict(
        batch_unit_passes=int(batch_units),
        lazy_unit_passes=int(round(lazy_units)),
        extra_old_model_query_passes=int(round(old_passes)),
        unit_passes_saved=int(round(saved_units)),
        unit_savings_pct=round(100 * saved_units / max(batch_units, 1), 2),
        batch_gpu_seconds=round(batch_ms / 1000, 1),
        lazy_gpu_seconds=round(lazy_ms / 1000, 1),
        net_gpu_seconds_saved=round(net_ms / 1000, 1),
        net_savings_pct=round(100 * net_ms / max(batch_ms, 1e-9), 2),
        repair_unit_ms_used=repair_ms,
        repair_penalty=round(repair_ms / max(p.unit_ms, 1e-9), 3),
        query_term_share_of_lazy=round(
            (old_passes * p.query_ms) / max(lazy_ms, 1e-9), 4),
        inverts=bool(net_ms <= 0),
        usd_batch=round(batch_units / 1000 * p.usd_per_1k_units, 4),
        usd_lazy=round(lazy_units / 1000 * p.usd_per_1k_units
                       + old_passes / 1000 * p.usd_per_1k_queries, 4),
    )


def predicted_savings(D: int, kprime: int, N: int) -> float:
    """savings ~= (1 - K'/N)^D.

    The exact expectation if each distinct intent's window were an independent
    draw of K' of N units. Its K'/N << 1 limit is exp(-D*K'/N), the form to
    write down; the two agree to 1e-6 at 21M units but not on small corpora --
    at K'/N = 0.4 the exponential overpredicts savings by ~6 points.

    On the measured data this form is conservative: on the MiniLM -> BGE run,
    every point in the informative range x = D*K'/N in [0.05, 3] had measured
    savings at or above it, and on CIPBench, where intents cluster heavily, it
    underestimates by ~22 points. Prefer a measured `reached` whenever you
    have one.
    """
    return (1.0 - min(kprime, N) / max(N, 1)) ** D


def predicted_savings_exp(D: int, kprime: int, N: int) -> float:
    """exp(-D*K'/N): the small-K'/N limit of predicted_savings."""
    return math.exp(-D * kprime / max(N, 1))


def breakeven_D(N: int, kprime: int) -> int:
    """D* = N / K', the characteristic load.

    Not a cliff: savings at D* are about 1/e ~= 37%, not zero. D* is the scale
    at which savings have fallen by a factor of e, which is the honest way to
    state the applicability boundary.
    """
    return max(1, N // max(kprime, 1))


def repair_batch_profile(curve_d, curve_reached, N: int, saturation_batch: float = 32.0,
                         batch_job_batch: float = 2048.0) -> dict:
    """How large are the query-path repair batches, and what do they cost?

    Read straight off an estimate row's coverage curve: between two points of
    the curve, `delta_reached * N` units are re-embedded while `delta_d` new
    intents arrive, so units-per-new-intent is the batch a repair sees. Early
    in a migration that is most of a window; late it is a handful -- but by
    then few units are left, so the WORK is concentrated in large batches.
    Weighting by units rather than by batch count is the whole point: on MS
    MARCO at K'=500 the work-weighted mean batch is ~213 and only 9% of repair
    work happens in batches below 32.

    Throughput is modelled as ms(b) = ms_sat * (1 + saturation_batch / b), i.e.
    a fixed per-batch overhead amortised over the batch. `saturation_batch` is
    where the encoder stops gaining from bigger batches; measure it (§0 of
    RUN_EXPERIMENTS) rather than trusting the default.

    Returns the work-weighted penalty to pass as `repair_unit_ms / unit_ms`.
    """
    import numpy as np

    ds = np.asarray(curve_d, dtype=float)
    rc = np.asarray(curve_reached, dtype=float)
    work = np.diff(rc) * N
    batch = np.divide(work, np.diff(ds), out=np.ones(len(work)), where=np.diff(ds) > 0)
    keep = work > 0
    work, batch = work[keep], np.maximum(batch[keep], 1.0)
    if work.sum() <= 0:
        return dict(weighted_batch=float("nan"), penalty=1.0, work_below_32=0.0)
    pen = ((1 + saturation_batch / batch) / (1 + saturation_batch / batch_job_batch))
    return dict(
        weighted_batch=float(np.average(batch, weights=work)),
        penalty=float(np.average(pen, weights=work)),
        work_below_32=float(work[batch < 32].sum() / work.sum()),
        saturation_batch=saturation_batch,
    )


def chunked_repair_penalty(curve: dict, curve_d, curve_reached, N: int,
                           chunk: int) -> dict:
    """Repair penalty read off the MEASURED throughput curve, for a query path
    that issues each window's repair in chunks of `chunk` (as `serve` does).
    Choose `chunk` as the measured size that minimises the result -- normally
    the best batch, not the saturation batch.

    `curve` maps batch size -> {"total_ms": ..., "ms_per_item": ...}, as
    `analysis/measure_throughput.py` records it. A repair of b units costs
    floor(b / chunk) full chunks at the measured rate for `chunk`, plus one
    remainder batch whose total time is interpolated linearly between the
    measured batch sizes at or below `chunk` (and 0 ms at batch 0). The batch
    job is priced at the best measured rate, the one a backfill would use.

    Unlike `repair_batch_profile`, this assumes nothing about the curve's
    shape. That matters on memory-tight devices, where per-item cost RISES
    past the best batch: the parametric model then prices the batch job at a
    batch size nobody would run.
    """
    import numpy as np

    c = {int(k): v for k, v in curve.items()}
    if chunk not in c:
        raise ValueError(f"chunk {chunk} is not a measured batch size {sorted(c)}")
    best = min(v["ms_per_item"] for v in c.values())
    small = sorted(b for b in c if b <= chunk)
    xs = [0] + small
    ys = [0.0] + [float(c[b]["total_ms"]) for b in small]
    chunk_ms = float(c[chunk]["total_ms"])

    ds = np.asarray(curve_d, dtype=float)
    rc = np.asarray(curve_reached, dtype=float)
    work = np.diff(rc) * N
    dd = np.diff(ds)
    batch = np.divide(work, dd, out=np.ones(len(work)), where=dd > 0)
    keep = work > 0
    work, batch = work[keep], np.maximum(batch[keep], 1.0)
    if work.sum() <= 0:
        return dict(method="chunked", chunk=chunk, weighted_batch=float("nan"),
                    penalty=1.0, work_below_chunk=0.0, interpolated_sizes=[])

    def per_unit(b: float) -> float:
        n = max(1, int(round(b)))
        q, r = divmod(n, chunk)
        rem = float(np.interp(r, xs, ys)) if r else 0.0
        return (q * chunk_ms + rem) / n

    cost = np.array([per_unit(b) for b in batch])
    missing = sorted({int(round(b)) % chunk for b in batch} - set(xs) - {0})
    return dict(
        method="chunked", chunk=int(chunk),
        weighted_batch=float(np.average(batch, weights=work)),
        penalty=float(np.average(cost, weights=work) / best),
        work_below_chunk=float(work[batch < chunk].sum() / work.sum()),
        interpolated_sizes=missing,
    )


def breakeven_reached(unit_ms: float, repair_unit_ms: float | None = None) -> float:
    """The fraction of the corpus that may be re-embedded before lazy loses.

    Ignoring the query term, lazy costs `reached * N * repair_unit_ms` against
    the batch job's `N * unit_ms`, so it pays exactly while

        reached  <  unit_ms / repair_unit_ms  =  1 / penalty.

    This is the useful form of the boundary: it depends on the encoder's batch
    behaviour alone, not on N, D or K', so it transfers across corpora -- but
    it is only as good as the penalty, which must be measured on the device
    (`measure_throughput` for a curve, `thaw_pg backfill-rate` for the matched
    sustained ratio). A break-even expressed as a D is just this point read off
    one corpus's coverage curve.
    """
    r = unit_ms if repair_unit_ms is None else repair_unit_ms
    return unit_ms / max(r, 1e-12)


def first_touch_ms(kprime: int, unit_ms: float, chunk: int | None = None,
                   curve: dict | None = None) -> float:
    """Worst-case work on the query path: a query whose whole window is stale.

    The obvious implementation hands all K' units to the encoder in one call,
    and on a memory-tight device that is the worst thing to do. On a measured
    RTX 4060 curve, 500 units in a single batch runs at 42.96 ms each (21.5 s
    in total) while the same 500 chunked into batches of 8 run at 3.16 ms each
    (1.6 s) -- a 13x difference decided entirely by how the repair is issued.

    Pass `curve` (batch size -> ms per item, as measured) and `chunk` to price
    a specific implementation; with neither, this assumes repairs are issued at
    whatever batch `unit_ms` was measured at, which is the optimistic case and
    should be stated as such.
    """
    if curve:
        # both shapes occur: {batch: ms} and measure_throughput's
        # {batch: {"ms_per_item": ms, ...}}, with int or str keys
        def ms(v):
            return float(v["ms_per_item"] if hasattr(v, "get") else v)
        per = {int(b): ms(v) for b, v in curve.items()}
        b = int(chunk) if chunk is not None and int(chunk) in per \
            else min(per, key=per.get)
        return kprime * per[b]
    return kprime * unit_ms


def report(p: CostParams, kprime: int | None = None) -> dict:
    r = compare(p)
    S = p.D if p.S is None else p.S
    print(f"N={p.N:,}  D={p.D:,}  S={S:,}  Q={p.Q:,}  reached={p.reached:.3f}  "
          f"cache_hit={p.cache_hit_rate:.2f}")
    print(f"  batch : {r['batch_unit_passes']:>12,} unit passes"
          f"   {r['batch_gpu_seconds']:>10,.1f} GPU-s")
    print(f"  lazy  : {r['lazy_unit_passes']:>12,} unit passes"
          f" + {r['extra_old_model_query_passes']:,} old-model query passes"
          f"   {r['lazy_gpu_seconds']:>10,.1f} GPU-s")
    if p.repair_unit_ms is None:
        print("  ! repair_unit_ms unset: query-path passes priced at the batch "
              "job's per-unit cost, which flatters lazy migration")
    elif r["repair_penalty"] != 1.0:
        print(f"  repair passes cost {r['repair_penalty']:.2f}x a batch pass "
              f"({r['repair_unit_ms_used']:.1f} vs {p.unit_ms:.1f} ms)")
    print(f"  net   : {r['net_savings_pct']:>6.1f}% GPU saved"
          f"   (unit passes alone: {r['unit_savings_pct']:.1f}%)"
          + ("   *** INVERTS ***" if r["inverts"] else ""))
    print(f"  the query term is {100*r['query_term_share_of_lazy']:.1f}% "
          "of lazy cost")
    if kprime:
        print(f"  break-even D* = N/K' = {breakeven_D(p.N, kprime):,} intents")
        print(f"  first touch: up to {kprime} re-embeds on the query path, "
              f"~{first_touch_ms(kprime, p.unit_ms):,.0f} ms of GPU at the batched "
              "rate -- the latency users see, not in the GPU totals above")
    return r


if __name__ == "__main__":
    # The MS MARCO shape, as a DEMONSTRATION of the ledger. The recall figures
    # are the measured MiniLM->BGE curve at N=500,000 (BEIR dev), but `reached`
    # below is the closed-form PREDICTION, not a measurement: a real costing
    # reads it from an estimate row with `from_estimate`, which is what
    # RUN_EXPERIMENTS step 7 does.
    N, Q = 8_800_000, 10_000_000
    for kp, recall in ((200, 0.883), (500, 0.929)):
        print(f"=== MS MARCO shape, K'={kp} (measured recall {recall}), "
              f"D* = {breakeven_D(N, kp):,} ===\n")
        for D, cache in ((1_000, 1.0), (5_000, 1.0), (10_000, 1.0),
                         (10_000, 0.0), (breakeven_D(N, kp), 1.0)):
            reached = 1.0 - predicted_savings(D, kp, N)   # conservative
            tag = "cached" if cache == 1.0 else "UNCACHED"
            print(f"-- D={D:,} {tag}")
            report(CostParams(N=N, D=D, Q=Q, reached=reached,
                              cache_hit_rate=cache), kprime=kp)
            print()
