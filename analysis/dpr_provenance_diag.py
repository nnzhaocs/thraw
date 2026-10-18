"""Why does the DPR provenance check fail? Six hypotheses, one pass each.

`verify_dpr_provenance` answers one question -- does the shipped `wiki_dpr`
embedding match what the context encoder produces now -- and returns a single
mean cosine. Three runs on 2026-10-17 returned 0.9506 (min 0.8278) and, on a
re-downloaded copy, 0.9312 (min 0.6740), against its 0.98 threshold. A single
number cannot say *why*, and the DPR row in the paper's model-pair table rests
on a run (`migration_dpr_490k_`) whose results file carries no provenance
record at all. So this tool sweeps the candidate causes against one fixed
sample of passages and prints them side by side.

**Read the comparison, not any single number.** Every row uses the SAME
passages and the SAME shipped vectors, so the rows are comparable to each other
even though none of them is a ground truth. The cause is whichever row jumps to
roughly 1.0.

The hypotheses, in the order a review proposed checking them -- code before
data, because a code bug makes a data diagnostic fail too and you would then
misread it as a data problem:

1. **`max_length`.** The check hard-codes 256. A review argued DPR used 512 and
   that truncation of long passages explains a mean of 0.95 with a minimum of
   0.67. Against that: DPR's own README trains the bi-encoder at
   `--sequence_length 256`, and `wiki_dpr` passages are 100 words by
   construction, so they tokenise to roughly 130--150 tokens and truncation at
   256 should bind on almost nothing. The sweep settles it rather than the
   argument: 128, 256, 350 and 512 are all run, and `tokens_at_or_over` reports
   how many passages the limit actually truncates. If that count is zero at 256,
   `max_length` cannot be the cause whatever the cosines say.
2. **Title handling.** `wiki_dpr` is encoded as a (title, text) pair. Encoding
   text alone, or gluing title and text into one string, both give a
   systematically different vector. (`migration.py` normalises a null title to
   `""` at both call sites, so None-handling is already ruled out; this row
   tests the pairing itself, which is a different question.)
3. **The encoder checkpoint.** `psgs_w100.nq` should be `single-nq-base` and
   `psgs_w100.multiset` should be `multiset-base`. A local copy built from the
   other one is the classic `wiki_dpr` mix-up and would show as a broad
   deficit, not a long tail.
4. **Pooling.** DPR's context encoder exposes `pooler_output`; using the CLS
   token of the last hidden state instead is a plausible mis-wiring and gives
   vectors that correlate without matching.
5. **Normalisation and dtype.** The shipped vectors are not unit-norm, and a
   cosine hides that. The report prints both norms so a pure scale difference
   is visible, and checks whether float16 storage could account for the gap
   (it cannot: f16 round-trip costs about 1e-4 of cosine, not 0.05).
6. **Batch and padding effects.** Attention masking should make `pooler_output`
   independent of what else is in the batch. Encoding the sample one passage at
   a time tests that, because if padding does leak the whole comparison is
   unreliable regardless of cause.

If no row reaches ~1.0, the local `wiki_dpr` copy was not produced by either
released context encoder and the paper's DPR row has to be re-measured from
text or dropped. That is the outcome this tool exists to be able to reach.

    python analysis/dpr_provenance_diag.py \\
        --dpr-path /path/to/wiki_dpr --variant psgs_w100.nq.no_index \\
        --n 256 --out dpr_diag.json

Needs `torch` and `transformers`, and the corpus `--dpr-path` points at. No
GPU required: 256 passages on CPU take a couple of minutes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _cos(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Row-wise cosine, float64, with a guard on either side being zero."""
    R = np.asarray(R, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    nr = np.linalg.norm(R, axis=1)
    ns = np.linalg.norm(S, axis=1)
    bad = (nr == 0) | (ns == 0)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} zero-norm vectors; the sample or the "
                         "shipped matrix is not what it should be")
    return np.einsum("ij,ij->i", R, S) / (nr * ns)


def _summary(name: str, cos: np.ndarray, extra: dict | None = None) -> dict:
    row = {
        "case": name,
        "mean_cos": round(float(cos.mean()), 4),
        "min_cos": round(float(cos.min()), 4),
        "p05_cos": round(float(np.percentile(cos, 5)), 4),
        "frac_above_0.99": round(float(np.mean(cos > 0.99)), 4),
        # the shape of the deficit is the diagnosis: a broad shift is a wrong
        # encoder or wrong pooling, a long tail is per-passage (truncation)
        "spread_mean_minus_p05": round(float(cos.mean() - np.percentile(cos, 5)), 4),
    }
    row.update(extra or {})
    return row


def main() -> None:  # pragma: no cover - needs torch, transformers and the corpus
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpr-path", required=True,
                    help="the local wiki_dpr directory, as passed to semview.estimate")
    ap.add_argument("--variant", default="psgs_w100.nq.no_index")
    ap.add_argument("--n", type=int, default=256,
                    help="passages to compare; the default matches "
                         "verify_dpr_provenance's own sample size")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="dpr_diag.json")
    a = ap.parse_args()

    import torch
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

    from semview.migration import dpr_encoders_for, load_dpr_wiki

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev}")

    # ONE fixed sample, reused by every case, so the rows compare to each other
    units, V, pairs = load_dpr_wiki(limit=max(2000, a.n * 4),
                                    variant=a.variant, keep_text=True,
                                    local_dir=a.dpr_path)
    sample = pairs[:a.n]
    idx = np.array([i for i, _, _ in sample])
    titles = [(t or "") for _, t, _ in sample]
    texts = [x for _, _, x in sample]
    S = np.asarray(V[idx])
    print(f"{len(idx)} passages; shipped matrix {V.shape}, dtype {V.dtype}")

    expected = dpr_encoders_for(a.variant)[1]
    rows: list[dict] = []
    cache: dict[str, tuple] = {}

    def model(name: str):
        if name not in cache:
            print(f"  loading {name}")
            tok = DPRContextEncoderTokenizer.from_pretrained(name)
            m = DPRContextEncoder.from_pretrained(name).eval().to(dev)
            cache[name] = (tok, m)
        return cache[name]

    def encode(name, ml, pair=True, join=False, pooling="pooler", bs=32):
        tok, m = model(name)
        out, truncated = [], 0
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                ti, tx = titles[i:i + bs], texts[i:i + bs]
                if join:
                    enc = tok([f"{t} {x}".strip() for t, x in zip(ti, tx)],
                              padding=True, truncation=True, max_length=ml,
                              return_tensors="pt")
                elif pair:
                    enc = tok(ti, tx, padding=True, truncation=True,
                              max_length=ml, return_tensors="pt")
                else:
                    enc = tok(tx, padding=True, truncation=True, max_length=ml,
                              return_tensors="pt")
                # how many passages this limit actually cuts: a full row of
                # attention at the cap means the input hit it
                truncated += int((enc["attention_mask"].sum(1) >= ml).sum())
                enc = {k: v.to(dev) for k, v in enc.items()}
                o = m(**enc)
                if pooling == "pooler":
                    out.append(o.pooler_output.cpu().numpy())
                else:
                    out.append(o.hidden_states[-1][:, 0].cpu().numpy()
                               if getattr(o, "hidden_states", None) is not None
                               else m(**enc, output_hidden_states=True
                                      ).hidden_states[-1][:, 0].cpu().numpy())
        return np.vstack(out), truncated

    # --- 1. max_length -----------------------------------------------------
    print("1. max_length sweep")
    for ml in (128, 256, 350, 512):
        R, trunc = encode(expected, ml)
        rows.append(_summary(f"max_length={ml}", _cos(R, S),
                             {"tokens_at_or_over_limit": trunc,
                              "note": "0 truncated means max_length cannot be the cause"}))
        print(f"   {rows[-1]['case']:<22} mean {rows[-1]['mean_cos']:.4f} "
              f"min {rows[-1]['min_cos']:.4f}  truncated {trunc}")

    # --- 2. title handling -------------------------------------------------
    print("2. title handling")
    for label, kw in (("text only, no title", dict(pair=False)),
                      ("title+text joined", dict(join=True))):
        R, _ = encode(expected, 256, **kw)
        rows.append(_summary(label, _cos(R, S)))
        print(f"   {label:<22} mean {rows[-1]['mean_cos']:.4f}")

    # --- 3. the other released checkpoint ----------------------------------
    print("3. encoder checkpoint")
    other = ("facebook/dpr-ctx_encoder-multiset-base"
             if "single-nq" in expected else
             "facebook/dpr-ctx_encoder-single-nq-base")
    for name in (expected, other):
        try:
            R, _ = encode(name, 256)
        except Exception as e:                               # noqa: BLE001
            # The other released checkpoint is not in every cache, and an
            # offline machine cannot fetch it. Record the skip and move on:
            # the sweep's job is the comparison it CAN make, and one missing
            # row is not a reason to lose the rows already measured.
            # 2026-10-17.
            rows.append({"case": f"encoder={name.split('/')[-1]}",
                         "skipped": f"{type(e).__name__}: {e}"[:200],
                         "is_expected_for_variant": name == expected})
            print(f"   {name.split('/')[-1]:<34} SKIPPED ({type(e).__name__})")
            continue
        rows.append(_summary(f"encoder={name.split('/')[-1]}", _cos(R, S),
                             {"is_expected_for_variant": name == expected}))
        print(f"   {name.split('/')[-1]:<34} mean {rows[-1]['mean_cos']:.4f}")

    # --- 4. pooling --------------------------------------------------------
    print("4. pooling")
    try:
        R, _ = encode(expected, 256, pooling="cls")
        rows.append(_summary("CLS of last hidden state", _cos(R, S)))
        print(f"   CLS instead of pooler  mean {rows[-1]['mean_cos']:.4f}")
    except Exception as e:                                   # noqa: BLE001
        rows.append({"case": "CLS of last hidden state", "error": str(e)[:200]})
        print(f"   CLS variant unavailable: {e}")

    # --- 5. norms and dtype ------------------------------------------------
    print("5. norms and dtype")
    R, _ = encode(expected, 256)
    f16 = R.astype(np.float16).astype(np.float32)
    rows.append({
        "case": "norms and dtype",
        "shipped_norm_mean": round(float(np.linalg.norm(S, axis=1).mean()), 4),
        "reencoded_norm_mean": round(float(np.linalg.norm(R, axis=1).mean()), 4),
        "shipped_dtype": str(V.dtype),
        "f16_roundtrip_mean_cos": round(float(_cos(f16, R).mean()), 6),
        "note": ("if the f16 round-trip cosine is ~1.0 then storage precision "
                 "cannot explain a 0.95; a norm ratio far from 1 is a scale "
                 "difference a cosine would otherwise hide"),
    })
    print(f"   shipped norm {rows[-1]['shipped_norm_mean']}  "
          f"re-encoded {rows[-1]['reencoded_norm_mean']}  "
          f"f16 round-trip {rows[-1]['f16_roundtrip_mean_cos']}")

    # --- 6. batching -------------------------------------------------------
    print("6. batch independence")
    R1, _ = encode(expected, 256, bs=1)
    rows.append(_summary("one passage per batch", _cos(R1, S),
                         {"vs_batched_mean_cos": round(float(_cos(R1, R).mean()), 6),
                          "note": "vs_batched below ~0.999 means padding leaks "
                                  "and every row above is unreliable"}))
    print(f"   unbatched mean {rows[-1]['mean_cos']:.4f}  "
          f"agreement with batched {rows[-1]['vs_batched_mean_cos']}")

    # --- verdict -----------------------------------------------------------
    scored = [r for r in rows if "mean_cos" in r]
    best = max(scored, key=lambda r: r["mean_cos"])
    baseline = next(r for r in scored if r["case"] == "max_length=256")
    if best["mean_cos"] > 0.98:
        verdict = (f"CAUSE FOUND: '{best['case']}' reaches "
                   f"{best['mean_cos']:.4f}. Fix that in "
                   f"verify_dpr_provenance and re-run the four DPR jobs.")
    elif best["mean_cos"] - baseline["mean_cos"] > 0.02:
        verdict = (f"PARTIAL: '{best['case']}' improves on the current "
                   f"configuration by {best['mean_cos']-baseline['mean_cos']:.4f} "
                   f"but still only reaches {best['mean_cos']:.4f}. More than one "
                   f"thing is wrong, or the copy is partly mismatched.")
    else:
        verdict = ("NO CAUSE IN THIS LIST: nothing reaches 0.98 and nothing "
                   "beats the current configuration by more than 0.02, so the "
                   "local wiki_dpr vectors were probably not produced by either "
                   "released context encoder. The paper's DPR row must then be "
                   "re-measured from text or dropped -- do not pass "
                   "--skip-provenance to make the number appear.")
    print(f"\n  {verdict}")

    obj = {"tool": "dpr_provenance_diag", "variant": a.variant,
           "dpr_path": a.dpr_path, "n": int(len(idx)),
           "expected_encoder": expected, "rows": rows, "verdict": verdict}
    try:
        from semview.results_io import environment, write_results
        write_results(a.out, dict(obj, environment=environment()))
    except Exception:                                        # noqa: BLE001
        Path(a.out).write_text(json.dumps(obj, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
