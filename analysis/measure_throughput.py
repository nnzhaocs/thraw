"""Measure the encoder throughput CURVE, and turn it into cost-model inputs.

One number is not enough. The cost model needs three things, and two of them
only exist if you sweep batch size:

    unit_ms         a passage pass in the BATCH job's large batches
    query_ms        a query pass (short text, so measure it separately)
    repair_unit_ms  a passage pass in the QUERY path's small batches

The third is the one that decides whether a headline saving survives: the
query path re-embeds the stale units inside one window, which starts near K'
and shrinks to a handful, and small batches cost far more per unit. This script
finds where throughput stops improving (the saturation batch) and hands that to
`cost.repair_batch_profile`, which weights it by the repair work actually done
at each batch size, read off a measured coverage curve.

    python analysis/measure_throughput.py --model BAAI/bge-base-en-v1.5 \
        --passage-file pool.jsonl --estimate results/estimate_msmarco.json \
        --kprime 500 --out results/throughput_real.json

    # no GPU: recompute every derived field from a curve already measured
    python analysis/measure_throughput.py --from-json results/throughput_real.json \
        --estimate results/estimate_msmarco.json --kprime 500 \
        --out results/throughput_real.json

The repair penalty is computed two ways and both are recorded. "chunked"
(primary) reads the measured curve directly, assuming the query path issues
repairs in chunks at the measured batch size that minimises it (normally the
best batch), as `serve` does with --chunk. "parametric" is
the earlier model, ms(b) = ms_sat (1 + sat/b) against a batch job at the
largest measured batch; it is kept for comparison with old results and is
wrong on devices whose per-item cost rises past the best batch.

Timing uses CUDA events where available, discards warm-up iterations, and
reports the median of several repeats. Tokenisation is included, because a
deployment pays it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import statistics
import time


def load_texts(path: str, n: int, seed: int = 0) -> list[str]:
    """Sample `n` real texts from a JSONL ({"text": ...}) or plain-text file.

    Timing on generated text measures a single length repeated; real corpora
    have a length distribution, and tokenisation and padding both track it.
    Sampling is seeded so a re-run times the same passages.
    """
    import json
    import random
    from pathlib import Path

    lines = [l for l in Path(path).read_text(encoding="utf-8",
                                             errors="replace").splitlines() if l.strip()]
    texts = []
    for l in lines:
        if l.lstrip().startswith("{"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            t = r.get("text") or r.get("passage") or r.get("body") or ""
            if r.get("title"):
                t = f"{r['title']} {t}"
        else:
            t = l
        if t.strip():
            texts.append(t.strip())
    if not texts:
        raise SystemExit(f"no usable texts in {path}")
    rng = random.Random(seed)
    return [texts[rng.randrange(len(texts))] for _ in range(n)]


def _timer(device: str):
    """Returns (start, stop) -> elapsed ms. CUDA events when on GPU, because
    wall-clock around an async kernel launch measures the launch, not the work."""
    if device.startswith("cuda"):
        import torch

        def run(fn):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)
        return run

    def run(fn):
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000.0
    return run


def _env(a) -> dict:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.results_io import environment
    return environment(dtype="float16" if getattr(a, "fp16", False) else "float32",
                       device=getattr(a, "device", None))


def _write(path, result) -> None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.results_io import write_results
    write_results(path, result, required=())


def sweep(model_name: str, texts, batches, repeats: int = 5, warmup: int = 2,
          device: str | None = None, prefix: str = "",
          fp16: bool = False):  # pragma: no cover
    from sentence_transformers import SentenceTransformer

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    m = SentenceTransformer(model_name, device=device)
    if fp16:
        m.half()
    run = _timer(device)
    out = {}
    import random

    rng = random.Random(0)
    for b in batches:
        # A fresh random batch from the pool for every repeat: one fixed batch
        # timed over and over has one padding length and stays warm in every
        # cache, which made the curve faster than the system ever ran.
        draws = [[prefix + t for t in rng.choices(texts, k=b)]
                 for _ in range(warmup + repeats)]
        for batch in draws[:warmup]:
            m.encode(batch, batch_size=b, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        times, real, padded = [], 0, 0
        for batch in draws[warmup:]:
            times.append(run(lambda: m.encode(batch, batch_size=b,
                                              convert_to_numpy=True,
                                              normalize_embeddings=True,
                                              show_progress_bar=False)))
            mask = m.tokenize(batch)["attention_mask"]
            real += int(mask.sum())
            padded += int(mask.numel())
        ms = statistics.median(times)
        out[b] = dict(total_ms=round(ms, 2), ms_per_item=round(ms / b, 4),
                      items_per_s=round(1000.0 * b / ms, 1),
                      padding_factor=round(padded / max(real, 1), 4),
                      tokens_per_item=round(real / (b * repeats), 1))
        print(f"  batch {b:>5}: {ms / b:8.3f} ms/item   {1000.0 * b / ms:9.1f} items/s"
              f"   padding {padded / max(real, 1):.2f}")
    return out, device


def saturation_batch(curve: dict, tol: float = 0.10) -> int:
    """Smallest batch whose per-item cost is within `tol` of the best seen.

    This is the number the cost model needs: past it, bigger batches buy
    nothing, and below it the query path pays a penalty.
    """
    best = min(v["ms_per_item"] for v in curve.values())
    for b in sorted(curve):
        if curve[b]["ms_per_item"] <= best * (1 + tol):
            return b
    return max(curve)


def derive(p_curve: dict, q_curve: dict, estimate: str | None = None,
           kprime: int = 500, method: str = "chunked") -> dict:
    """Every cost-model input that follows from the two measured curves.

    Pure: no model, no GPU. `--from-json` uses it to regenerate a results file
    from a curve measured earlier, so a stale derived field (the bug that left
    unit_ms = 42.96 in throughput_real.json) can be fixed without re-timing.
    """
    p_curve = {int(k): v for k, v in p_curve.items()}
    q_curve = {int(k): v for k, v in q_curve.items()}
    sat = saturation_batch(p_curve)
    # The ACHIEVABLE rate, not the largest batch's: on real passages of mixed
    # length, and on a memory-tight card, the biggest batch can be an order of
    # magnitude WORSE per item. A backfill would run at the best batch.
    best_b = min(p_curve, key=lambda b: p_curve[b]["ms_per_item"])
    unit_ms = p_curve[best_b]["ms_per_item"]
    worst_b = max(p_curve)
    worst = p_curve[worst_b]["ms_per_item"]
    # per-item cost should not rise with batch size; if it does, the device ran
    # short of memory and the parametric penalty model does not apply
    sizes = sorted(p_curve)
    rising = [b for a, b in zip(sizes, sizes[1:])
              if b > best_b and p_curve[b]["ms_per_item"] > 1.5 * unit_ms]
    out = dict(saturation_batch=int(sat), best_batch=int(best_b),
               unit_ms=unit_ms, worst_large_batch=int(worst_b),
               worst_large_batch_ms=round(worst, 4),
               curve_rises_past_best=bool(rising),
               # a query is encoded on its own on the serving path, so it is priced
               # at batch 1 -- pricing it at the batch rate is the mistake this
               # paper criticises in the other direction
               query_ms=q_curve[min(q_curve)]["ms_per_item"],
               query_ms_batched=q_curve[max(q_curve)]["ms_per_item"])
    if estimate:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from semview import cost

        est = json.load(open(estimate, encoding="utf-8"))
        rows = [r for r in est["estimates"] if r["kprime"] == kprime]
        if not rows:
            raise SystemExit(f"no K'={kprime} row in {estimate}; "
                             f"it has {[r['kprime'] for r in est['estimates']]}")
        row = rows[0]
        # The repair chunk is the measured batch size with the lowest
        # penalty -- normally the best batch. Not the saturation batch: that is
        # the smallest batch within 10% of the best, and chunking there pays up
        # to that 10% on every chunk (1.09x against 1.007x on a 4060 Laptop
        # curve whose saturation batch is 4 and best batch 16).
        by_chunk = {b: cost.chunked_repair_penalty(p_curve, row["curve_d"],
                                                   row["curve_reached"], est["N"],
                                                   chunk=b)
                    for b in sorted(p_curve)}
        chunk = min(by_chunk, key=lambda b: by_chunk[b]["penalty"])
        chunked = by_chunk[chunk]
        chunked["penalty_by_chunk"] = {str(b): round(v["penalty"], 4)
                                       for b, v in by_chunk.items()}
        param = cost.repair_batch_profile(row["curve_d"], row["curve_reached"],
                                          est["N"], saturation_batch=sat,
                                          batch_job_batch=max(p_curve))
        param["method"] = "parametric"
        primary = chunked if method == "chunked" else param
        out.update(repair_chunk=int(chunk), repair_profile=primary,
                   repair_profile_alternative=param if method == "chunked" else chunked,
                   penalty_method=method, penalty=primary["penalty"],
                   repair_unit_ms=unit_ms * primary["penalty"],
                   breakeven_reached=1.0 / primary["penalty"],
                   estimate_file=estimate, kprime=kprime, estimate_N=est["N"])
    return out


def report(d: dict) -> None:
    print(f"  best {d['unit_ms']:.3f} ms/item at batch {d['best_batch']}; "
          f"saturation batch ~{d['saturation_batch']} (within 10% of best)")
    if d["curve_rises_past_best"]:
        print(f"  ! per-item cost RISES past batch {d['best_batch']} (batch "
              f"{d['worst_large_batch']}: {d['worst_large_batch_ms']:.2f} ms, "
              f"{d['worst_large_batch_ms'] / d['unit_ms']:.1f}x the best) -- the "
              "device ran short of memory. Issue repairs in chunks of "
              f"{d.get('repair_chunk', d['best_batch'])}; the parametric penalty does not apply.")
    if "penalty" in d:
        a = d["repair_profile_alternative"]
        p = d["repair_profile"]
        print(f"  repair chunk {d['repair_chunk']} (use it as --chunk)")
        print(f"  repair penalty ({d['penalty_method']}): {d['penalty']:.3f}x"
              f"   [{a['method']}: {a['penalty']:.3f}x]")
        print(f"  work-weighted repair batch {p['weighted_batch']:.0f}")
        if p.get("interpolated_sizes"):
            print(f"  note: remainder batches {p['interpolated_sizes']} were "
                  "interpolated; measure --batches 1,2,4,8,... to remove that")
    print("\ncost-model inputs:")
    print(f"  unit_ms        = {d['unit_ms']:.4f}")
    print(f"  query_ms       = {d['query_ms']:.4f}   (one query at a time; "
          f"{d.get('query_ms_batched', float('nan')):.4f} if concurrent queries "
          "are batched)")
    if "repair_unit_ms" in d:
        print(f"  repair_unit_ms = {d['repair_unit_ms']:.4f}")
        print(f"  break-even rho = {d['breakeven_reached']:.3f}")
    else:
        print("  repair_unit_ms = pass --estimate to derive it")


DERIVED = ("repair_chunk", "saturation_batch", "best_batch", "unit_ms", "worst_large_batch",
           "worst_large_batch_ms", "curve_rises_past_best", "query_ms", "query_ms_batched",
           "repair_profile", "repair_profile_alternative", "penalty_method",
           "penalty", "repair_unit_ms", "breakeven_reached", "estimate_file",
           "kprime", "estimate_N")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="measure_throughput")
    ap.add_argument("--model", default=None, help="the NEW model (it does the repairs)")
    ap.add_argument("--old-model", default=None,
                    help="optional: the old model, which encodes every query")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--passage-file", default=None,
                    help="JSONL or plain text to sample real passages from. "
                         "WITHOUT THIS the script times generated text, which is "
                         "fine for the curve's shape but not for absolute ms.")
    ap.add_argument("--query-file", default=None)
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--passage-words", type=int, default=80)
    ap.add_argument("--query-words", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp16", action="store_true",
                    help="half precision; match the encoder step 2 uses")
    ap.add_argument("--estimate", default=None,
                    help="a semview.estimate JSON; its coverage curve gives the "
                         "repair batch-size distribution")
    ap.add_argument("--kprime", type=int, default=500)
    ap.add_argument("--penalty-method", default="chunked",
                    choices=["chunked", "parametric"])
    ap.add_argument("--from-json", default=None,
                    help="recompute derived fields from this file's measured "
                         "curves instead of timing anything (no GPU needed)")
    ap.add_argument("--out", default="throughput.json")
    a = ap.parse_args()

    if a.from_json:
        result = json.load(open(a.from_json, encoding="utf-8"))
        for k in list(result):
            if k in DERIVED:
                del result[k]
        print(f"recomputing from the curves in {a.from_json} (nothing is timed)")
        d = derive(result["passage_curve"], result["query_curve"], a.estimate,
                   a.kprime, a.penalty_method)
        result.update(d)
        report(d)
        result["environment"] = _env(a)
        _write(a.out, result)
        return

    if not a.model:
        raise SystemExit("--model is required unless --from-json is given")
    words = ("retrieval index maintenance embedding migration corpus passage "
             "vector query document").split()
    batches = [int(x) for x in a.batches.split(",")]
    if a.passage_file:
        passages = load_texts(a.passage_file, a.pool)
        lens = sorted(len(t.split()) for t in passages)
        src = (f"{a.passage_file}: {len(passages)} sampled, "
               f"{lens[0]}-{lens[-1]} words, median {lens[len(lens)//2]}")
    else:
        passages = [" ".join(words[i % len(words)] for i in range(a.passage_words))]
        src = (f"GENERATED {a.passage_words}-word text -- the curve's SHAPE is "
               "usable, the absolute ms/passage is not")
        print("WARNING: no --passage-file, so absolute timings are synthetic and "
              "must not be quoted as GPU cost")
    queries = (load_texts(a.query_file, min(a.pool, 512), seed=1) if a.query_file
               else [" ".join(words[i % len(words)] for i in range(a.query_words))])

    print(f"passages through {a.model} ({src}):")
    p_curve, device = sweep(a.model, passages, batches, a.repeats, device=a.device,
                             fp16=a.fp16)
    q_model = a.old_model or a.model
    print(f"\nqueries through {q_model}:")
    q_curve, _ = sweep(q_model, queries, [1, 32, 512], a.repeats, device=device,
                       fp16=a.fp16)

    gpu = None
    if device.startswith("cuda"):
        try:
            import torch
            gpu = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            gpu = None
    result = dict(model=a.model, old_model=a.old_model, device=device, gpu=gpu,
                  passage_source=src, synthetic=not bool(a.passage_file),
                  passage_curve=p_curve, query_curve=q_curve)
    d = derive(p_curve, q_curve, a.estimate, a.kprime, a.penalty_method)
    result.update(d)
    print()
    report(d)
    result["environment"] = _env(a)
    _write(a.out, result)


if __name__ == "__main__":
    main()
