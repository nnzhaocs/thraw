"""Matched backfill-rate: Thaw's repairs and a plain backfill, timed the same way.

The cost model's penalty is a ratio: what one repaired unit costs against what
one unit costs in a full backfill. Both sides have to be measured in the same
process, on the same GPU, with the same encoder, precision and database load,
for the same length of time. Anything else compares two setups, not two
policies. This tool does that:

    warm-up     backfill batches, discarded (the GPU reaches its clocks)
    backfill A  every unit, in random order, batched the way a backfill would be
    repair      Thaw's repair stream: for each query in order, the window members
                not yet repaired, in chunks of --chunk
    backfill B  as A again; A vs B shows drift (heat, clocks) over the run

Each batch is timed in three parts (tokenize on the CPU, forward on the GPU,
copy back) and its real and padded token counts are recorded, so the penalty
splits into

    ms/unit  =  real tokens/unit  x  padded/real  x  forward ms per padded token
                (which units)        (batching)      (device rate)

and each factor is reported as a repair : backfill ratio. Only the device-rate
factor depends on the GPU. With TEI the split and the token counts are not
available, and only ms/unit is reported.

    python -m thaw_pg backfill-rate --enc local:BAAI/bge-base-en-v1.5 \\
        --kprime 500 --chunk 16 --backfill-batch 16 --seconds 300

`analysis/padding.py` computes the batching factor alone, with no GPU.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------- the streams ---
def repair_stream(V_old: np.ndarray, Q_old: np.ndarray, kprime: int,
                  n_queries: int | None = None) -> list[list[int]]:
    """For each query in order, the window members no earlier query repaired:
    exactly the units Thaw encodes for that query (single-threaded)."""
    from semview.migration import topk_chunked

    Q = Q_old if n_queries is None else Q_old[:n_queries]
    W = topk_chunked(np.asarray(Q, dtype=np.float32), V_old, kprime)
    seen = np.zeros(len(V_old), dtype=bool)
    out = []
    for w in W:
        w = np.asarray(w, dtype=np.int64)
        stale = w[~seen[w]]
        seen[stale] = True
        out.append(stale.tolist())
    return out


def repair_batches(stale_sets, chunk: int, lengths, sort: bool = True):
    """Chunks as the coordinator issues them: one query's stale set at a time,
    sorted by length within that set when `sort`, never mixed across queries."""
    for q, ids in enumerate(stale_sets):
        if sort:
            ids = sorted(ids, key=lambda i: lengths[i])
        for lo in range(0, len(ids), chunk):
            yield q, list(ids[lo:lo + chunk])


def _token_lengths(enc_spec: str, texts: list[str],
                   model: str | None = None) -> tuple[list[int], str, str | None]:
    """Token counts for the sort key, from the encoder's own tokenizer.

    Returns `(lengths, key, fallback_reason)` where `key` is what was actually
    measured -- `"token"` or `"char"` -- rather than what was asked for. The
    caller records `key`, never its own flag: a run that asked for a token sort
    and silently got a character sort used to write
    `backfill_sort_key="token"`, which is the same mis-keyed measurement D-02
    exists to correct, moved from the sort into the metadata. A `tei:` spec
    carries a URL and no model name, so that path fell back *every time* unless
    the model is passed in.
    """
    kind, _, arg = enc_spec.partition(":")
    name = arg if kind.startswith("local") else (model or None)
    try:
        if not name:
            raise ValueError(
                "a tei: spec carries a URL, not a model name -- pass --new-model "
                "or --passage-prefix so the tokenizer can be loaded")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        enc = tok(texts, add_special_tokens=True, truncation=True)["input_ids"]
        return [len(ids) for ids in enc], "token", None
    except Exception as e:                                        # noqa: BLE001
        return [len(t) for t in texts], "char", f"{type(e).__name__}: {e}"


def backfill_batches(population, batch: int, lengths, sort_window: int = 0,
                     seed: int = 0):
    """A backfill over `population` in random order. With `sort_window` > 0,
    each run of that many units is sorted by length before batching: what a
    backfill with a global view does to cut padding."""
    order = np.random.default_rng(seed).permutation(np.asarray(population))
    win = sort_window if sort_window and sort_window > 0 else len(order)
    for lo in range(0, len(order), win):
        block = order[lo:lo + win].tolist()
        if sort_window:
            block.sort(key=lambda i: lengths[i])
        for b in range(0, len(block), batch):
            yield None, block[b:b + batch]


def cycled(make, max_rounds: int | None = None):
    """Restart a stream when it ends, so every phase runs for the same wall
    time. A repair stream from a few thousand queries can end well before a
    backfill phase does; comparing 40 s of one with 300 s of the other would
    compare a cool GPU with a warm one. Re-encoding a unit costs the same as
    encoding it the first time, so a restarted stream times the same work."""
    r = 0
    while max_rounds is None or r < max_rounds:
        empty = True
        for item in make(r):
            empty = False
            yield item
        if empty:
            return
        r += 1


def padding_factor(batches, tok_len) -> dict:
    """Padded / real tokens for a sequence of batches, from token lengths
    alone. Needs no GPU."""
    real = padded = units = 0
    for _, ids in batches:
        ls = [tok_len[i] for i in ids]
        real += sum(ls)
        padded += max(ls) * len(ls)
        units += len(ls)
    return dict(units=units, real_tokens=real, padded_tokens=padded,
                tokens_per_unit=round(real / max(units, 1), 2),
                padding_factor=round(padded / max(real, 1), 4))


# --------------------------------------------------------------- telemetry ---
class GPUTelemetry:
    """Samples nvidia-smi once a second in a thread. Absent nvidia-smi, it
    records nothing and says so."""

    FIELDS = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"

    def __init__(self, period: float = 1.0):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.available = True
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.FIELDS}",
                     "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                vals = [float(x) if x.strip() not in ("", "[N/A]") else None
                        for x in out.split(",")]
                if len(vals) == 5:
                    self.samples.append((time.perf_counter(), *vals))
            except Exception:  # noqa: BLE001
                self.available = False
                return
            self._stop.wait(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    def summary(self, t0: float, t1: float) -> dict | None:
        rows = [s for s in self.samples if t0 <= s[0] <= t1]
        if not rows:
            return None

        def col(j):
            v = [r[j] for r in rows if r[j] is not None]
            return v

        sm, temp, pw, util = col(1), col(3), col(4), col(5)
        return dict(samples=len(rows),
                    sm_clock_mean=round(float(np.mean(sm)), 0) if sm else None,
                    sm_clock_min=min(sm) if sm else None,
                    temp_max=max(temp) if temp else None,
                    power_mean=round(float(np.mean(pw)), 1) if pw else None,
                    util_mean=round(float(np.mean(util)), 1) if util else None)


class PGLoad:
    """Optional database load during every phase alike: HNSW window searches
    in a loop on their own connection. Off by default."""

    def __init__(self, dsn: str, Q_old: np.ndarray, kprime: int, threads: int = 1):
        self.dsn, self.Q, self.kprime, self.threads = dsn, Q_old, kprime, threads
        self._stop, self.count, self._ts = threading.Event(), 0, []

    def _run(self, seed):
        from .db import connect

        c = connect(self.dsn)
        c.execute(f"SET hnsw.ef_search = {min(self.kprime, 1000)}")
        c.execute("SET hnsw.iterative_scan = relaxed_order")
        rng = np.random.default_rng(seed)
        while not self._stop.is_set():
            q = self.Q[rng.integers(len(self.Q))]
            c.execute("SELECT id FROM docs ORDER BY emb_old <=> %s LIMIT %s",
                      (q, self.kprime)).fetchall()
            self.count += 1
        c.close()

    def start(self):
        for s in range(self.threads):
            t = threading.Thread(target=self._run, args=(s,), daemon=True)
            t.start()
            self._ts.append(t)
        return self

    def stop(self):
        self._stop.set()
        for t in self._ts:
            t.join(timeout=30)


# ------------------------------------------------------------------ phases ---
def run_phase(name: str, batches, encoder, texts, seconds: float,
              gap_ms: float = 0.0, bin_s: float = 10.0) -> dict:
    """Encode batches until `seconds` of wall time pass or the stream ends.
    Returns per-batch records and the phase's wall-clock bounds."""
    recs = []
    t_start = time.perf_counter()
    for q, ids in batches:
        if time.perf_counter() - t_start >= seconds:
            break
        _, tm = encoder.encode_timed([texts[i] for i in ids])
        tm.update(t=time.perf_counter() - t_start, n=len(ids), q=q)
        recs.append(tm)
        if gap_ms:
            time.sleep(gap_ms / 1e3)
    t_end = time.perf_counter()
    exhausted = (t_end - t_start) < seconds
    return dict(name=name, records=recs, t_start=t_start, t_end=t_end,
                exhausted=exhausted, bin_s=bin_s)


def summarise_phase(ph: dict, burst_s: float = 10.0) -> dict:
    recs = ph["records"]
    if not recs:
        return dict(name=ph["name"], units=0)
    n = sum(r["n"] for r in recs)
    tok = sum(r["tok_ms"] for r in recs)
    fwd = sum(r["fwd_ms"] for r in recs)
    cpy = sum(r["copy_ms"] for r in recs)
    total = tok + fwd + cpy
    out = dict(name=ph["name"], units=n, batches=len(recs),
               wall_s=round(ph["t_end"] - ph["t_start"], 1),
               stream_exhausted=ph["exhausted"],
               mean_batch=round(n / len(recs), 2),
               ms_per_unit=round(total / n, 4),
               tok_ms_per_unit=round(tok / n, 4),
               fwd_ms_per_unit=round(fwd / n, 4),
               copy_ms_per_unit=round(cpy / n, 4))
    if recs[0]["real_tokens"] is not None:
        real = sum(r["real_tokens"] for r in recs)
        padded = sum(r["padded_tokens"] for r in recs)
        out.update(tokens_per_unit=round(real / n, 2),
                   padding_factor=round(padded / real, 4),
                   fwd_ms_per_padded_token=round(fwd / padded, 6))
    # burst = the first burst_s seconds of the phase; sustained = all of it
    first = [r for r in recs if r["t"] <= burst_s] or recs[:1]
    nb = sum(r["n"] for r in first)
    out["burst_ms_per_unit"] = round(
        sum(r["tok_ms"] + r["fwd_ms"] + r["copy_ms"] for r in first) / nb, 4)
    out["sustained_ms_per_unit"] = out["ms_per_unit"]
    series, b = [], ph["bin_s"]
    for lo in np.arange(0, ph["t_end"] - ph["t_start"], b):
        rs = [r for r in recs if lo < r["t"] <= lo + b]
        if rs:
            m = sum(r["n"] for r in rs)
            series.append([round(float(lo + b), 1), round(
                sum(r["tok_ms"] + r["fwd_ms"] + r["copy_ms"] for r in rs) / m, 4)])
    out["ms_per_unit_series"] = series
    return out


def penalty(rep: dict, bfa: dict, bfb: dict) -> dict:
    """repair : backfill, with the backfill as the mean of the two runs around
    the repair phase, and its factors."""
    def mean(k):
        vs = [x[k] for x in (bfa, bfb) if x.get(k) is not None]
        return float(np.mean(vs)) if vs else None

    if not rep.get("units") or mean("ms_per_unit") is None:
        raise SystemExit("a phase encoded nothing; lengthen --seconds")
    walls = [x["wall_s"] for x in (bfa, rep, bfb)]
    out = dict(phase_wall_s=walls,
               phases_equal_length=bool(max(walls) <= 1.1 * min(walls)),penalty=round(rep["ms_per_unit"] / mean("ms_per_unit"), 4),
               penalty_burst=round(rep["burst_ms_per_unit"] / mean("burst_ms_per_unit"), 4),
               backfill_drift_B_over_A=round(bfb["ms_per_unit"] / bfa["ms_per_unit"], 4)
               if bfa.get("units") and bfb.get("units") else None)
    if rep.get("tokens_per_unit") is not None and mean("tokens_per_unit"):
        f_units = rep["tokens_per_unit"] / mean("tokens_per_unit")
        f_pad = rep["padding_factor"] / mean("padding_factor")
        f_dev = rep["fwd_ms_per_padded_token"] / mean("fwd_ms_per_padded_token")
        fwd_share = rep["fwd_ms_per_unit"] / rep["ms_per_unit"]
        out["factors"] = dict(
            which_units_tokens_per_unit=round(f_units, 4),
            batching_padding=round(f_pad, 4),
            device_ms_per_padded_token=round(f_dev, 4),
            forward_penalty=round(f_units * f_pad * f_dev, 4),
            forward_share_of_repair_time=round(fwd_share, 4),
            tokenize_ms_per_unit_repair=rep["tok_ms_per_unit"],
            tokenize_ms_per_unit_backfill=round(mean("tok_ms_per_unit"), 4))
    out["breakeven_reached_fraction"] = round(1 / out["penalty"], 4)
    return out


# --------------------------------------------------------------------- run ---
def run(out_dir: Path, enc_spec: str, kprime: int, n_queries: int | None,
        chunk: int, backfill_batch: int, sort_window: int, repair_sort: bool,
        seconds: float, warmup_s: float, gap_ms: float, dsn: str | None,
        pg_threads: int, device: str | None, out: Path, seed: int = 0,
        new_model_prefix: str | None = None, new_model: str | None = None,
        max_rounds: int | None = None, token_sort: bool = True,
        allow_char_fallback: bool = False) -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semview.migration import resolve_prefixes
    from semview.results_io import environment, write_results

    from . import encoders

    vec = np.load(out_dir / "vectors.npz")
    V_old, Q_old = vec["V_old"], vec["Q_old"]
    with open(out_dir / "texts.jsonl", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    if len(texts) != len(V_old):
        raise SystemExit("texts.jsonl and vectors.npz disagree on N")
    # Sort by TOKEN length, not characters. The sort key decides how much a
    # batch pads, and padding is measured on what the tokenizer produced -- so a
    # character count is a proxy that leaves most of the sort's value on the
    # table. It matters asymmetrically: the backfill sorts within a 10,000-unit
    # window while the repair path can only sort within one query's stale set,
    # so a better key helps the backfill far more. With the old key its sorted
    # batches of 16 still padded at 1.2815, against 1.2570 for essentially
    # unsorted repair chunks -- the sort was buying almost nothing, which is not
    # what a correct length sort does.
    if token_sort:
        lengths, sort_key, fallback = _token_lengths(enc_spec, texts,
                                                     model=new_model)
        if sort_key != "token":
            # Refusing rather than warning. A warning in a log is how the
            # character key survived three rounds of review: the number it
            # produced went into the paper and nothing downstream knew which
            # key it came from. A measurement that cannot use the key it was
            # asked for is not a measurement of that key.
            msg = (f"refusing to measure: a token sort was asked for and the "
                   f"tokenizer is unavailable ({fallback}). Padding is measured "
                   f"on tokens, so a character key understates what a sorted "
                   f"backfill can do and the two are not comparable. Either "
                   f"make the tokenizer loadable (--new-model <hf name> for a "
                   f"tei: encoder), or ask for the character key explicitly "
                   f"with --no-token-sort, or accept the fallback with "
                   f"--allow-char-fallback -- which records "
                   f"backfill_sort_key='char' in the results, not 'token'.")
            if not allow_char_fallback:
                raise SystemExit(msg)
            print(f"  WARNING: {msg}")
    else:
        lengths, sort_key, fallback = [len(t) for t in texts], "char", None

    print(f"repair stream: exact old-space top-{kprime} for "
          f"{n_queries or len(Q_old)} queries over N={len(V_old):,}")
    stale = repair_stream(V_old, Q_old, kprime, n_queries)
    n_rep = sum(len(s) for s in stale)
    print(f"  {n_rep:,} repairs in the stream "
          f"({n_rep / len(V_old):.1%} of the corpus)")

    kind, _, arg = enc_spec.partition(":")
    if new_model_prefix is not None:
        passage_prefix = new_model_prefix
    else:
        # the passage prefix belongs to the MODEL; for TEI the spec holds a URL,
        # so the model has to be named (E5-style models need "passage: ")
        model = new_model or (arg if kind.startswith("local") else None)
        if model is None:
            raise SystemExit("with a tei: encoder pass --new <model name> (or "
                             "--passage-prefix) so the passage prefix is right")
        _, passage_prefix = resolve_prefixes(model)
    enc = encoders.make(enc_spec, passage_prefix, chunk, device)
    dtype = ("float16" if enc_spec.startswith("local-fp16:") else
             "float32" if enc_spec.startswith("local:") else "server")
    tele = GPUTelemetry().start()
    load = PGLoad(dsn, Q_old, kprime, pg_threads).start() if dsn else None

    every = list(range(len(V_old)))
    bf = lambda s: cycled(lambda r: backfill_batches(every, backfill_batch, lengths,
                                                    sort_window, s + 1000 * r),
                          max_rounds)
    print(f"warm-up {warmup_s:.0f}s ...")
    run_phase("warmup", bf(seed + 99), enc, texts, warmup_s)
    phases = []
    for name, it in (("backfill_A", bf(seed)),
                     ("repair", cycled(lambda r: repair_batches(stale, chunk, lengths,
                                                                 repair_sort),
                                       max_rounds)),
                     ("backfill_B", bf(seed + 1))):
        print(f"{name}: up to {seconds:.0f}s ...", flush=True)
        ph = run_phase(name, it, enc, texts, seconds,
                       gap_ms if name == "repair" else 0.0)
        s = summarise_phase(ph)
        s["gpu"] = tele.summary(ph["t_start"], ph["t_end"])
        phases.append(s)
        extra = (f"  tokens/unit {s['tokens_per_unit']}  padding "
                 f"{s['padding_factor']}" if s.get("tokens_per_unit") else "")
        print(f"  {s['units']:,} units  {s['ms_per_unit']:.3f} ms/unit "
              f"(tok {s['tok_ms_per_unit']:.3f}, fwd {s['fwd_ms_per_unit']:.3f}){extra}")
    tele.stop()
    pg = None
    if load:
        load.stop()
        pg = dict(threads=pg_threads, window_searches=load.count)
    bfa, rep, bfb = phases
    pen = penalty(rep, bfa, bfb)
    print("\n" + "=" * 70)
    print(f"penalty (repair : backfill, sustained)  {pen['penalty']:.4f}")
    print(f"penalty (first 10 s)                    {pen['penalty_burst']:.4f}")
    if pen.get("backfill_drift_B_over_A") is not None:
        dr = pen["backfill_drift_B_over_A"]
        print(f"backfill drift B/A                      {dr:.4f}" +
              ("   WARNING: the GPU changed during the run; lengthen --warmup"
               if abs(dr - 1) > 0.03 else ""))
    if "factors" in pen:
        f = pen["factors"]
        print(f"  which units (tokens/unit)             {f['which_units_tokens_per_unit']:.4f}")
        print(f"  batching (padding)                    {f['batching_padding']:.4f}")
        print(f"  device (fwd ms per padded token)      {f['device_ms_per_padded_token']:.4f}")
        print(f"  tokenize ms/unit repair vs backfill   {f['tokenize_ms_per_unit_repair']:.4f}"
              f" vs {f['tokenize_ms_per_unit_backfill']:.4f}")
    print(f"break-even reached fraction             {pen['breakeven_reached_fraction']:.4f}")
    if not pen["phases_equal_length"]:
        print(f"WARNING: phases ran {pen['phase_wall_s']} s; the comparison is "
              "between unequal stretches of time")
    write_results(out, dict(
        tool="backfill-rate", encoder=enc_spec, dtype=dtype, kprime=kprime,
        queries=n_queries or len(Q_old), N=len(V_old), chunk=chunk,
        backfill_batch=backfill_batch, backfill_sort_window=sort_window,
        # what was measured, not what was asked for
        backfill_sort_key=sort_key,
        backfill_sort_key_requested=("token" if token_sort else "char"),
        backfill_sort_key_fallback=fallback,
        repair_sort=repair_sort, seconds_per_phase=seconds, warmup_s=warmup_s,
        repair_gap_ms=gap_ms, repairs_in_stream=n_rep, pg_load=pg,
        telemetry=tele.available and bool(tele.samples),
        phases=phases, result=pen,
        environment=environment(dtype=dtype, device=device)))
    return 0
