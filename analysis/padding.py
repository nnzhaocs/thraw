"""The batching factor of the penalty, computed from token lengths. No GPU.

Padding is fixed once the passage lengths and the batching policy are fixed:
a batch costs max(length) x size tokens of forward work. This tool computes
padded / real tokens for Thaw's repair stream and for a backfill under several
batch sizes and sorting choices, on the same corpus and the same queries as
step 2. It also reports real tokens per unit for each, since the units
traffic repairs need not be the corpus average.

    python analysis/padding.py --data-dir step1_data --tokenizer BAAI/bge-base-en-v1.5 \\
        --kprime 500 --chunks 8,16,32 --backfill-batches 16,64,256,1024 \\
        --sort-windows 0,10000 --out results/padding_msmarco_500k.json

Read it with `backfill-rate`, which measures the other two factors (which
units, device rate) on a GPU. Because padding depends only on lengths and
policy, this part of the penalty carries across devices; the device-rate part
does not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thaw_pg.backfill import (backfill_batches, padding_factor,  # noqa: E402
                              repair_batches, repair_stream)


def token_lengths(texts, tokenizer: str, max_len: int = 512) -> list[int]:
    if tokenizer == "words":
        return [min(len(t.split()) + 2, max_len) for t in texts]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    out = []
    for lo in range(0, len(texts), 10_000):
        enc = tok(texts[lo:lo + 10_000], truncation=True, max_length=max_len,
                  add_special_tokens=True)["input_ids"]
        out.extend(len(x) for x in enc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="step1_data")
    ap.add_argument("--tokenizer", default="BAAI/bge-base-en-v1.5",
                    help="HF tokenizer name, or 'words' for a rough offline count")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--kprime", type=int, default=500)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--chunks", default="8,16,32")
    ap.add_argument("--backfill-batches", default="16,64,256,1024")
    ap.add_argument("--sort-windows", default="0,10000")
    ap.add_argument("--backfill-sample", type=int, default=200_000,
                    help="units the backfill rows are computed over (random)")
    ap.add_argument("--sort-key", choices=["chars", "tokens"], default="chars",
                    help="what sorting sorts by. 'chars' is what the encoder and "
                         "backfill-rate do, so the factor matches the system that "
                         "runs; 'tokens' shows what sorting by true length would buy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="padding.json")
    a = ap.parse_args()

    d = Path(a.data_dir)
    vec = np.load(d / "vectors.npz")
    V, Q = vec["V_old"], vec["Q_old"]
    with open(d / "texts.jsonl", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    stale = repair_stream(V, Q, a.kprime, a.queries)
    rep_units = sorted({i for s in stale for i in s})
    rng = np.random.default_rng(a.seed)
    bf_units = rng.choice(len(texts), min(a.backfill_sample, len(texts)),
                          replace=False).tolist()
    need = sorted(set(rep_units) | set(bf_units))
    print(f"tokenizing {len(need):,} passages with {a.tokenizer} ...")
    ls = token_lengths([texts[i] for i in need], a.tokenizer, a.max_len)
    tok_len = dict(zip(need, ls))
    # sorting uses one key, padding is counted in true tokens
    key = (tok_len if a.sort_key == "tokens"
           else {i: len(texts[i]) for i in need})

    rows = []
    for c in (int(x) for x in a.chunks.split(",")):
        for srt in (False, True):
            r = padding_factor(repair_batches(stale, c, key, srt), tok_len)
            rows.append(dict(side="repair", batch=c, sort=srt, **r))
    for b in (int(x) for x in a.backfill_batches.split(",")):
        for w in (int(x) for x in a.sort_windows.split(",")):
            r = padding_factor(backfill_batches(bf_units, b, key, w, a.seed), tok_len)
            rows.append(dict(side="backfill", batch=b, sort=w, **r))

    print(f"\n{'side':<9}{'batch':>6}{'sorted':>9}{'tokens/unit':>13}{'padding':>9}")
    for r in rows:
        print(f"{r['side']:<9}{r['batch']:>6}{str(r['sort']):>9}"
              f"{r['tokens_per_unit']:>13.1f}{r['padding_factor']:>9.3f}")
    print("\nbatching factor of the penalty = repair padding / backfill padding, "
          "for the pair of rows you actually run.")

    from semview.migration import _fingerprint
    from semview.results_io import environment, write_results
    write_results(a.out, dict(
        tool="padding", tokenizer=a.tokenizer, sort_key=a.sort_key, max_len=a.max_len, kprime=a.kprime,
        queries=a.queries or len(Q), N=len(texts), repairs=len(rep_units),
        fingerprint=dict(old_head=_fingerprint(V)),
        environment=environment(), rows=rows))


if __name__ == "__main__":
    main()
