"""Lazy encoder migration: the savings-quality frontier.

Under migration every unit is stale, so there is no update stream and no
admission bound. A query is served by searching the OLD index, re-embedding
the top-K' with the NEW model, ranking those in the new space, and persisting
them. Two quantities decide whether that is worth doing, and both are governed
by the same knob:

    quality  =  cross-space candidate recall@K'
                | new_top_k(q) ∩ old_top_K'(q) | / k
                -- if the old space cannot surface what the new model would
                   rank highly, the served results are wrong no matter what is
                   saved. Reranking inside the window is exact, so this recall
                   IS the end-to-end quality.

    savings  =  1 - |union of old_top_K'(q) over distinct intents| / N
                -- measured, not modelled. Each row also reports x = D*K'/N
                   and two closed-form predictions of savings from it:
                   exp(-x), the identity the paper uses, and 1 - x, the
                   earlier linear form, kept only so its error stays visible.

Both move with K', in opposite directions, which makes the operating point a
one-dimensional choice rather than a tuning exercise.

    python -m semview.migration --corpus msmarco --limit 1000000 \\
        --old sentence-transformers/all-MiniLM-L6-v2 \\
        --new BAAI/bge-base-en-v1.5 --out results/migration_msmarco.json

Corpus loaders: `msmarco` and `beir` need `datasets` or a local TSV;
`cipbench` pools the repositories in a benchmark directory; `jsonl` reads
{"id","text"} lines. Queries come from the dataset where it has them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# corpora
# ----------------------------------------------------------------------
def _fingerprint(V) -> str:
    """Hash of the first rows of an embedding matrix, recorded in every results
    file so a later reader can tell which vectors produced it."""
    import hashlib

    head = np.ascontiguousarray(np.asarray(V[:16], dtype=np.float32))
    return hashlib.sha256(head.tobytes()).hexdigest()[:16]


def _fingerprint_strided(V, rows: int = 16) -> str:
    """Hash of rows spread evenly through the matrix. The head alone is the
    same for two subsets that share their first rows (gold_subset puts gold
    first), so it cannot tell a 500k run from an 8.8M one."""
    import hashlib

    n = V.shape[0]
    idx = np.unique(np.linspace(0, n - 1, num=min(rows, n)).astype(np.int64))
    sample = np.ascontiguousarray(np.asarray(V[idx], dtype=np.float32))
    return hashlib.sha256(sample.tobytes() + str(n).encode()).hexdigest()[:16]


def _ids_hash(ids) -> str:
    """Hash of an ordered id list: two results files with the same value used
    the same queries (or units) in the same order."""
    import hashlib

    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def old_label(old_model: str, dpr_variant: str | None, normalize: bool) -> str:
    """The name a results file records for the old side. One function, used by
    both tools, so the same configuration is never labelled two ways."""
    base = f"dpr:{dpr_variant}" if dpr_variant else old_model
    return base + (":normalized" if normalize else "")


def positive_int(text: str) -> int:
    """argparse type: a limit of 0 or less is an error, never 'no limit'.
    `limit or DEFAULT` would silently turn --limit 0 into the full corpus."""
    v = int(text)
    if v <= 0:
        import argparse
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {v}")
    return v


def require_dpr_limit(corpus: str, limit: int | None, tool: str) -> None:
    """Refuse to load all of DPR Wikipedia by default.

    21M x 768 float32 is ~64 GB for one embedding space; `migration` holds two.
    Running out of memory an hour into a load is a bad way to learn that, so
    the full corpus needs an explicit --limit 21015300.
    """
    if corpus == "dpr_wiki" and limit is None:
        need = "~64 GB of vectors" if tool == "estimate" else \
               "~128 GB (old + new vectors) plus ~16 GB of passage text"
        raise SystemExit(
            f"--corpus dpr_wiki without --limit loads all {DPR_WIKI_SIZE:,} "
            f"passages, which needs {need}. Pass --limit 1000000 for a first "
            f"run, or --limit {DPR_WIKI_SIZE} to confirm the full corpus.")


def _reservoir_offer(res: list, k: int, item, n: int, rng) -> None:
    """Offer the n-th item (0-based) of a stream to a size-k reservoir.

    Algorithm R. `rng` is a `random.Random`, whose `randint(a, b)` is INCLUSIVE
    of both ends, so `randint(0, n)` draws uniformly from the n+1 positions a
    uniform sample of the first n+1 items needs. Do not change it to
    `randint(0, n + 1)`: that is numpy's exclusive-upper convention applied to
    an inclusive API, and it over-samples the first k items by ~16% while
    under-sampling the rest. `test_reservoir_is_uniform` fails if it happens.
    """
    if len(res) < k:
        res.append(item)
    else:
        j = rng.randint(0, n)
        if j < k:
            res[j] = item


def _join(title, text) -> str:
    """title + text, tolerating a title that is absent OR present-but-null.
    `r.get("title", "")` only covers the first case; null titles occur."""
    return ((title or "") + " " + (text or "")).strip()


def _sample(units: list, limit: int | None, seed: int = 0) -> list:
    """Seeded random sample of `limit` units, order preserved; all if no limit.
    Never a prefix: every in-memory corpus here lists related units together."""
    if limit is None or limit >= len(units):
        return units
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(units), size=limit, replace=False))
    return [units[i] for i in keep]


def load_cipbench(root: str, limit: int | None = None, seed: int = 0):
    """Pool every CIPBench repository under `root`.

    `limit` caps the CORPUS, by a seeded random sample. A prefix would be
    biased: units are listed repository by repository, so the first N come from
    the first few repositories only -- the same bias that once inflated savings
    on the query side. Queries are capped separately by --max-queries.
    """
    from .cipbench import REPOS, load_repo

    units, queries = [], []
    for repo in REPOS:
        p = Path(root) / repo
        if not (p / "units.jsonl").exists():
            continue
        r = load_repo(p)
        units += [(u["unit_id"], u["text"]) for u in r["units"]]
        queries += [(q["query_id"], q["query"]) for q in r["queries"]]
    return _sample(units, limit, seed), queries


def load_jsonl(path: str, qpath: str | None = None, limit: int | None = None,
               seed: int = 0):
    """{"id", "text"} lines. `limit` samples the corpus at random, not a prefix."""
    units = []
    with open(path, encoding="utf-8") as f:           # streamed, as load_beir
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            units.append((str(r.get("id", len(units))), r["text"]))
    queries = []
    if qpath:
        with open(qpath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    queries.append((str(r.get("id", len(queries))), r["text"]))
    return _sample(units, limit, seed), queries


def load_msmarco(limit: int | None = None):  # pragma: no cover
    """MS MARCO passages (~8.8M) via `BeIR/msmarco`.

    Streamed, so `limit` takes the first N passages rather than a random
    sample -- a random sample would require reading all 8.8M. If this fails to
    load on your `datasets` version, download the BEIR zip instead and use
    `--corpus beir --beir-name msmarco` (see RUN_EXPERIMENTS.md, "Getting the
    data"); that path needs no Hub loader at all.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
        qs = load_dataset("BeIR/msmarco", "queries", split="queries", streaming=True)
    except (OSError, ValueError, RuntimeError, ImportError) as e:
        # Narrow on purpose: a MemoryError or a bug should surface as itself,
        # not be reported as "the Hub failed".
        raise SystemExit(
            f"could not load BeIR/msmarco from the Hub ({type(e).__name__}: {e}). "
            "Download the BEIR msmarco zip and use --corpus beir --beir-name "
            "msmarco instead.") from e
    units = []
    for r in ds:
        units.append((str(r["_id"]), _join(r.get("title"), r["text"])))
        if limit is not None and len(units) >= limit:
            break
    queries = [(str(r["_id"]), r["text"]) for r in qs]
    return units, queries


DPR_WIKI_SIZE = 21_015_300       # passages in psgs_w100
PROVENANCE_SAMPLE = 256


def dpr_parquet_files(variant: str, local_dir: str | None = None) -> str:
    """Where the shipped embeddings live, as a glob for the parquet loader.

    `facebook/wiki_dpr` still carries a Python loading script, and `datasets`
    4.x refuses any repository with one ("Dataset scripts are no longer
    supported"), so `load_dataset("wiki_dpr", ...)` fails on a current install.
    The same repository also hosts the data as parquet shards, which load on
    any version. The embedding set is the second variant segment: the
    exact/compressed/no_index suffix only chooses a FAISS index, and the vectors
    are identical across them.
    """
    key = ".".join(variant.split(".")[:2])
    if key not in DPR_ENCODERS:
        raise ValueError(f"unknown wiki_dpr variant {variant!r}")
    subset = key.split(".")[1]                     # "nq" or "multiset"
    if local_dir:
        return str(Path(local_dir) / "data" / "psgs_w100" / subset / "train-*.parquet")
    return f"hf://datasets/facebook/wiki_dpr/data/psgs_w100/{subset}/train-*.parquet"


DPR_COLUMNS = ("id", "title", "text", "embeddings")


def check_dpr_record(r: dict) -> None:
    missing = [c for c in DPR_COLUMNS if c not in r]
    if missing:
        raise SystemExit(f"wiki_dpr record lacks columns {missing}; found "
                         f"{sorted(r)}. The parquet schema has changed -- fix "
                         "the column names in load_dpr_wiki before running.")


NQ_DEV_URL = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"


def parse_dpr_nq_dev(records: list) -> list[dict]:
    """DPR's NQ dev file -> [{qid, question, gold_ids, gold_keys}].

    The documented schema gives each positive only `title` and `text`; the
    downloadable files carry extra attributes, possibly a `passage_id`. Both
    are recorded: ids are used when present, and exact (title, text) matching
    otherwise. Either is sound, because both sides come from psgs_w100.
    """
    out = []
    for i, r in enumerate(records):
        pos = r.get("positive_ctxs") or []
        ids = {str(p["passage_id"]) for p in pos if p.get("passage_id") is not None}
        keys = {((p.get("title") or ""), p.get("text") or "") for p in pos}
        if ids or keys:
            out.append(dict(qid=str(i), question=r["question"],
                            gold_ids=ids, gold_keys=keys))
    return out


def load_dpr_nq_dev(path_or_url: str = NQ_DEV_URL) -> list[dict]:  # pragma: no cover
    import gzip
    import urllib.request

    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=120) as resp:
            raw = resp.read()
    else:
        raw = Path(path_or_url).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return parse_dpr_nq_dev(json.loads(raw))


def select_gold_and_distractors(rows, gold_ids: set, gold_keys: set, n: int,
                                n_gold_expected: int, seed: int = 0):
    """One pass over a stream of (id, title, text, vec): keep every gold row,
    and a uniform reservoir sample of the others sized to bring the total to n.

    Replaces a contiguous prefix, which (a) is a block of Wikipedia rather than
    a sample of it, and (b) contains the answers to only a small fraction of
    the questions, so most queries would be searching a corpus that cannot
    answer them.
    """
    import random

    cap = max(0, n - n_gold_expected)
    rng = random.Random(seed)
    gold, res, seen = [], [], 0
    for rid, title, text, vec in rows:
        title = title or ""
        if str(rid) in gold_ids or (title, text) in gold_keys:
            gold.append((str(rid), title, text, vec))
            continue
        _reservoir_offer(res, cap, (str(rid), title, text, vec), seen, rng)
        seen += 1
    return gold, res


def dpr_corpus(limit: int | None, variant: str, local_dir: str | None,
               queries_from: str, nq_dev: str, keep_text: bool, seed: int = 0):
    """Corpus and queries for DPR Wikipedia, shared by migration and estimate.

    `queries_from` is "nq-dev" (DPR's NQ dev questions, whose gold passages are
    known), "nq-open-validation" or "nq-open-train". With a --limit below the
    full corpus only "nq-dev" is allowed, and the corpus is every gold passage
    plus uniform distractors -- which needs one full pass over all 21M rows.
    The nq-open sets carry no gold ids, so they require the full corpus.
    Returns (units, V, provenance_pairs, queries).
    """  # pragma: no cover
    full = limit is None or limit >= DPR_WIKI_SIZE
    if queries_from == "nq-dev":
        dev = load_dpr_nq_dev(nq_dev)
    elif not full:
        raise SystemExit(f"--queries {queries_from} has no gold passage ids, so it "
                         "needs the full corpus; use --queries nq-dev with --limit")
    if full:
        units, V, pairs = load_dpr_wiki(limit, variant, local_dir, keep_text)
        if queries_from == "nq-dev":
            queries = [(d["qid"], d["question"]) for d in dev]
        else:
            queries = load_nq_questions(queries_from.split("-")[-1])
        return units, V, pairs, queries

    gold_ids = set().union(*(d["gold_ids"] for d in dev))
    gold_keys = set().union(*(d["gold_keys"] for d in dev))
    n_gold = len(gold_ids) or len(gold_keys)
    if n_gold >= limit:
        raise SystemExit(f"--limit {limit} is below the {n_gold} gold passages")
    print(f"building a gold+distractor subset: {n_gold:,} gold passages + "
          f"{limit - n_gold:,} uniform distractors, from one pass over all "
          f"{DPR_WIKI_SIZE:,} rows")
    gold, res = select_gold_and_distractors(
        _iter_dpr_rows(variant, local_dir), gold_ids, gold_keys, limit, n_gold, seed)
    kept = gold + res
    if len(kept) > limit:
        # n_gold_expected is a lower bound when some questions carry passage
        # ids and others only (title, text), so the reservoir can be sized
        # slightly large. Trim distractors, never gold.
        kept = gold[:limit] + res[:max(0, limit - len(gold))]
    found_ids = {g[0] for g in gold}
    found_keys = {(g[1], g[2]) for g in gold}
    queries = [(d["qid"], d["question"]) for d in dev
               if (d["gold_ids"] & found_ids) or (d["gold_keys"] & found_keys)]
    print(f"  {len(gold):,} gold passages found; {len(queries):,} of {len(dev):,} "
          "questions have an answer in the subset")
    V = np.vstack([k[3] for k in kept]).astype(np.float32)
    units = [(k[0], _join(k[1], k[2]) if keep_text else "") for k in kept]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(kept), size=min(PROVENANCE_SAMPLE, len(kept)), replace=False)
    pairs = [(int(i), kept[i][1], kept[i][2]) for i in pick]
    return units, V, pairs, queries


def _iter_dpr_rows(variant: str, local_dir: str | None):  # pragma: no cover
    """Stream (id, title, text, vector) from the parquet shards."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=dpr_parquet_files(variant, local_dir),
                      split="train", streaming=True)
    dim = None
    for n, r in enumerate(ds):
        if dim is None:
            check_dpr_record(r)
        vec = np.asarray(r["embeddings"], dtype=np.float32)
        if dim is None:
            dim = vec.shape[0]
        elif vec.shape[0] != dim:
            raise SystemExit(f"embedding dimension changed at record {n}: "
                             f"{dim} -> {vec.shape[0]}")
        yield r["id"], r.get("title"), r["text"], vec


def load_dpr_wiki(limit: int | None = None, variant: str = "psgs_w100.nq.no_index",
                  local_dir: str | None = None, keep_text: bool = True):
    """DPR's Wikipedia split: 21,015,300 passages of 100 words, with the DPR
    context embeddings already computed. Returns (units, vectors, pairs), where
    pairs is a uniform sample of (index, title, text) tuples -- the inputs DPR
    actually encoded -- used to verify provenance exactly.

    The shipped vectors ARE the old index, so migrating DPR -> a modern encoder
    costs only the new model's passes. They are NOT normalised: DPR scores by
    raw inner product, and the old side must be searched the way the old
    system actually searched it.

    `keep_text=False` stores ids only. The passage text is ~16 GB at 21M
    including Python object overhead, and only the NEW model needs it:
    `estimate`, which never embeds the corpus, passes False. The provenance
    sample keeps its text either way; it is 256 passages.

    Streamed shard by shard, so `limit` takes the first N passages. Those are a
    contiguous block of Wikipedia articles, not a uniform sample: fine for
    comparing two models on the same corpus, but say so when reporting a
    limited run. The dataset is licensed CC BY-NC 4.0.
    """  # pragma: no cover
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    import random

    from datasets import load_dataset

    files = dpr_parquet_files(variant, local_dir)
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    # Preallocate. Appending 21M small arrays and then np.vstack-ing them holds
    # the list and the stacked copy at once -- ~128 GB peak for 64 GB of
    # vectors -- which is an out-of-memory crash at full scale.
    cap = DPR_WIKI_SIZE if limit is None else limit
    V = None
    units = []
    # Provenance needs a few hundred (title, text) pairs, not 21M of them;
    # reservoir-sample uniformly.
    rng = random.Random(0)
    pairs: list[tuple[int, str, str]] = []
    n = 0
    for r in ds:
        if V is None:
            check_dpr_record(r)
        vec = np.asarray(r["embeddings"], dtype=np.float32)
        if V is None:
            V = np.empty((cap, vec.shape[0]), dtype=np.float32)
        elif vec.shape[0] != V.shape[1]:
            # Fail at the record, not hours later inside an array assignment.
            raise SystemExit(f"embedding dimension changed at record {n}: "
                             f"{V.shape[1]} -> {vec.shape[0]}")
        title, text = r.get("title") or "", r["text"]
        V[n] = vec
        units.append((str(r["id"]), _join(title, text) if keep_text else ""))
        _reservoir_offer(pairs, PROVENANCE_SAMPLE, (n, title, text), n, rng)
        n += 1
        if n >= cap:
            break
    return units, V[:n], pairs


DPR_ENCODERS = {
    # shipped wiki_dpr variant prefix -> (question encoder, context encoder)
    "psgs_w100.nq": ("facebook/dpr-question_encoder-single-nq-base",
                     "facebook/dpr-ctx_encoder-single-nq-base"),
    "psgs_w100.multiset": ("facebook/dpr-question_encoder-multiset-base",
                           "facebook/dpr-ctx_encoder-multiset-base"),
}


def dpr_encoders_for(variant: str) -> tuple[str, str]:
    """Question/context encoder pair that produced a given wiki_dpr variant.

    The pairing is not optional. `psgs_w100.nq.*` vectors came from the NQ
    context encoder and must be queried with the NQ question encoder;
    `psgs_w100.multiset.*` needs the multiset pair. Mixing them runs every
    query through the wrong model and produces plausible-looking numbers.
    """
    # Match on the first two dot-separated segments exactly, not a string
    # prefix: "psgs_w100.nq_extra" must not resolve to the NQ encoders.
    key = ".".join(variant.split(".")[:2])
    if key in DPR_ENCODERS:
        return DPR_ENCODERS[key]
    raise ValueError(f"unknown wiki_dpr variant {variant!r}; "
                     f"expected one of {sorted(DPR_ENCODERS)}")


def encode_dpr_questions(texts, variant: str = "psgs_w100.nq.no_index",
                         batch: int = 128, device: str | None = None):
    """DPR's own question encoder, matched to the shipped context vectors."""  # pragma: no cover
    import torch
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

    name = dpr_encoders_for(variant)[0]
    tok = DPRQuestionEncoderTokenizer.from_pretrained(name)
    m = DPRQuestionEncoder.from_pretrained(name).eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(dev)
            out.append(m(**enc).pooler_output.cpu().numpy())
    return np.vstack(out).astype(np.float32)   # raw inner product, unnormalised


def verify_dpr_provenance(pairs, V_shipped, variant: str, n: int = 256,
                          device: str | None = None) -> dict:
    """Re-encode shipped passages with the matching context encoder.

    Mean cosine ~1.0 means the old system is reproducible from current
    weights and the "old index" really is what DPR produced. Run this before
    anything else; if it fails, every downstream number is about a system
    that never existed.
    """  # pragma: no cover
    import torch
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

    name = dpr_encoders_for(variant)[1]
    tok = DPRContextEncoderTokenizer.from_pretrained(name)
    m = DPRContextEncoder.from_pretrained(name).eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(dev)
    sample = pairs[:n]                       # already a uniform reservoir sample
    idx = np.array([i for i, _, _ in sample])
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 32):
            chunk = sample[i:i + 32]
            # wiki_dpr passages are encoded as ONE string, "title text".
            # Measured on 256 shipped passages: the sentence-pair form
            # (tok(titles, texts), which inserts a [SEP] between them) lands at
            # mean cos 0.944, the conjoined form at 0.985 -- so the pair form
            # reports a data mismatch that does not exist. 2026-10-17.
            titles = [t for _, t, _ in chunk]
            texts = [x for _, _, x in chunk]
            enc = tok([f"{t} {x}" for t, x in zip(titles, texts)],
                      padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(dev)
            out.append(m(**enc).pooler_output.cpu().numpy())
    R = np.vstack(out)
    S = V_shipped[idx]
    cos = np.einsum("ij,ij->i", R, S) / (np.linalg.norm(R, axis=1)
                                         * np.linalg.norm(S, axis=1))
    res = dict(n=len(idx), mean_cos=float(cos.mean()), min_cos=float(cos.min()),
               encoder=name, ok=bool(cos.mean() > 0.98))
    print(f"DPR provenance: mean cos {res['mean_cos']:.4f}, min "
          f"{res['min_cos']:.4f} over {res['n']} passages -> "
          f"{'OK' if res['ok'] else 'MISMATCH: do not trust downstream numbers'}")
    return res


NQ_OPEN = "google-research-datasets/nq_open"
NQ_OPEN_SPLITS = {"train": 87_925, "validation": 3_610}


def load_nq_questions(split: str = "validation"):
    """Natural Questions (open) questions for DPR Wikipedia.

    `nq_open` has only two splits -- train (87,925) and validation (3,610) --
    and no test split, so D can never exceed 3,610 on validation. It is the
    open-domain simplification of Natural Questions that DPR's own evaluation
    used, not the raw NQ format; say so when reporting.

    Use validation for the RECALL curve. DPR's question encoder was trained on
    NQ train and BGE's fine-tuning mix includes NQ, so train questions are
    memorised by both models and inflate cross-space recall. Train is
    acceptable for extending the coverage sweep to larger D, with a small
    residual bias: memorised questions may concentrate their windows slightly.
    """
    if split not in NQ_OPEN_SPLITS:
        raise ValueError(f"nq_open has no {split!r} split; it has "
                         f"{sorted(NQ_OPEN_SPLITS)} only")
    from datasets import load_dataset  # pragma: no cover

    ds = load_dataset(NQ_OPEN, split=split)  # pragma: no cover
    return [(str(i), r["question"]) for i, r in enumerate(ds)]


def orcas_columns(header_or_first: str) -> tuple[int, int]:
    """Which tab-separated fields hold the query text and the document id.

    ORCAS is distributed as a TSV of clicked query-document pairs. The column
    order is not something to assume, so it is detected: a field that looks
    like a MS MARCO document id (D followed by digits) is the doc id, and the
    longest remaining field with a space in it is the query. Override with
    --orcas-query-col / --orcas-doc-col if a release differs.
    """
    fields = header_or_first.rstrip("\n").split("\t")
    if len(fields) < 2:
        raise SystemExit(f"ORCAS line has {len(fields)} tab-separated fields: "
                         f"{header_or_first[:120]!r}")
    doc = next((i for i, f in enumerate(fields)
                if re.fullmatch(r"[Dd]\d+", f.strip())), None)
    cand = [(i, f) for i, f in enumerate(fields) if i != doc and " " in f]
    if not cand:
        # a single-token query has no space, so fall back to the longest field
        # that is not itself shaped like a document id -- otherwise a line of
        # nothing but ids would be accepted with one of them read as the query
        # Never a URL or a bare number (the query id): on a line like
        # "qid <tab> query <tab> D5 <tab> http://..." the URL is the longest
        # field, and reading it as the query is a silent error.
        cand = [(i, f) for i, f in enumerate(fields)
                if i != doc and f.strip()
                and not re.fullmatch(r"[Dd]\d+", f.strip())
                and not re.match(r"(?i)(https?://|www\.)", f.strip())
                and not f.strip().isdigit()]
    qry = max(cand, key=lambda t: len(t[1]))[0] if cand else None
    if doc is None or qry is None:
        raise SystemExit(
            "could not identify the ORCAS query and document columns from "
            f"{fields!r}; pass --orcas-query-col and --orcas-doc-col")
    return qry, doc


def load_orcas(orcas_path: str, docs_path: str, limit: int | None = None,
               max_queries: int | None = None, seed: int = 0,
               query_col: int | None = None,
               doc_col: int | None = None):  # pragma: no cover
    """ORCAS clicked query-document pairs over the MS MARCO document corpus.

    ORCAS is the one public source of a REAL distinct-intent population: about
    10M distinct queries against roughly 1.4M documents, i.e. D far above N.
    That makes it the negative case -- the identity predicts nothing is saved --
    and, subsampled, the only evidence available about what D a real deployment
    produces. It has no timestamps, so it describes the intent population over
    the whole collection period rather than a migration window: a pessimistic
    bound, and it should be reported as one. Research use only.

    `docs_path` is the MS MARCO document corpus TSV (docid, url, title, body).
    Queries are subsampled FIRST, so that only their clicked documents have to
    be kept as gold; distractors then fill the corpus up to `limit`.
    """
    rng = np.random.default_rng(seed)
    by_query: dict[str, tuple[str, set]] = {}
    with open(orcas_path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        qc, dc = ((query_col, doc_col) if query_col is not None and doc_col is not None
                  else orcas_columns(first))
        fh.seek(0)
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(qc, dc):
                continue
            q, d = f[qc].strip(), f[dc].strip()
            if not q or not d or q.lower() in ("query", "query-text"):
                continue
            if q not in by_query:
                by_query[q] = (q, set())
            by_query[q][1].add(d)
    print(f"ORCAS: {len(by_query):,} distinct queries")

    keys = list(by_query)
    if max_queries and len(keys) > max_queries:
        keys = [keys[i] for i in rng.choice(len(keys), max_queries, replace=False)]
    gold = {d for k in keys for d in by_query[k][1]}
    queries = [(str(i), k) for i, k in enumerate(keys)]
    print(f"  sampled {len(queries):,} queries with {len(gold):,} clicked documents")

    # One pass, as for DPR: keep every clicked document, reservoir-sample the
    # rest. Reading all ~3.2M documents into memory and then discarding most of
    # them would cost gigabytes for a subset a fraction of the size.
    import random

    rrng = random.Random(seed)
    gold_units, rest, seen = [], [], 0
    cap = None if limit is None else max(0, limit - len(gold))
    with open(docs_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 2:
                continue
            did = f[0].strip()
            text = _join(f[2] if len(f) > 2 else "", f[3] if len(f) > 3 else f[-1])
            if did in gold:
                gold_units.append((did, text))
            elif cap is None:
                rest.append((did, text))
            else:
                _reservoir_offer(rest, cap, (did, text), seen, rrng)
                seen += 1
    if limit is not None and len(gold_units) > limit:
        # truncating would silently drop some queries' clicked documents
        raise SystemExit(f"--limit {limit} is smaller than the {len(gold_units)} "
                         "clicked documents; raise it or lower --max-queries")
    units = gold_units + rest
    if limit is not None and len(units) > limit:
        units = gold_units + rest[:limit - len(gold_units)]
    print(f"  corpus {len(units):,} documents, {len(gold_units):,} of them clicked")
    return units, queries


def load_beir(name: str, path: str, limit: int | None = None,
              seed: int = 0, qrels_split: str | None = None):
    """Standard BEIR layout: <path>/<name>/corpus.jsonl, queries.jsonl, qrels/.

    With `qrels_split` (e.g. "dev"), queries are restricted to that split and,
    if `limit` is set, the corpus is every gold document for those queries plus
    random distractors up to `limit`. Both matter:

      * a random or prefix corpus of 1M out of 8.8M leaves most queries with no
        relevant passage in it, so recall measures agreement on the nearest
        junk and coverage piles onto hub passages;
      * BEIR's msmarco queries.jsonl includes TRAIN queries, which BGE and
        MiniLM were trained on -- the same contamination guarded against for
        NQ. Use the dev split.

    Without `qrels_split`, every query is kept and `limit` samples at random.
    Each line is parsed once; `title` may be absent or null.
    """
    root = Path(path) / name
    # Streamed line by line: read_text() on MS MARCO's ~3 GB corpus.jsonl
    # held the file, its split lines and the parsed units at once.
    units = []
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                units.append((str(r["_id"]), _join(r.get("title"), r["text"])))
    queries = []
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                queries.append((str(r["_id"]), r["text"]))
    if qrels_split is None:
        return _sample(units, limit, seed), queries
    gold = read_qrels(root, qrels_split)
    queries = [q for q in queries if q[0] in gold]
    return gold_subset(units, gold, limit, seed), queries


def read_qrels(root, split: str) -> dict[str, set[str]]:
    """BEIR qrels/<split>.tsv -> {query_id: {doc_id with positive score}}."""
    f = Path(root) / "qrels" / f"{split}.tsv"
    if not f.exists():
        have = sorted(p.stem for p in (Path(root) / "qrels").glob("*.tsv"))
        raise SystemExit(f"no qrels split {split!r} in {Path(root) / 'qrels'}; "
                         f"available: {have}")
    gold: dict[str, set[str]] = {}
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:                       # header row, or a malformed score
            score = int(parts[2].strip())
        except ValueError:
            continue
        if score > 0:
            gold.setdefault(parts[0], set()).add(parts[1])
    return gold


def gold_subset(units: list, gold: dict, limit: int | None, seed: int = 0) -> list:
    """Every unit that is gold for some query, plus random distractors up to
    `limit`. Without a limit, the whole corpus."""
    if limit is None:
        return units
    want = set().union(*gold.values()) if gold else set()
    g = [u for u in units if u[0] in want]
    if len(g) >= limit:
        raise SystemExit(f"--limit {limit} is smaller than the {len(g)} gold "
                         "documents; raise it so every query keeps its answer")
    rest = [u for u in units if u[0] not in want]
    return g + _sample(rest, limit - len(g), seed)


# ----------------------------------------------------------------------
# encoders
# ----------------------------------------------------------------------
BGE_EN_QUERY = "Represent this sentence for searching relevant passages: "


def resolve_prefixes(model: str, query_prefix: str = "auto",
                     passage_prefix: str = "auto") -> tuple[str, str]:
    """The text a model expects before queries and before passages.

    This matters more than it looks: the NEW model's top-k is the ground truth
    of the whole measurement, so encoding its queries the wrong way measures
    agreement with a model nobody would deploy. BGE's English models take a
    retrieval instruction on the query and nothing on the passage; E5 requires
    "query: " and "passage: " and degrades badly without them; MiniLM and
    lexical stand-ins take nothing. "auto" looks the model up; any explicit
    string (including "") overrides it.
    """
    name = model.lower()
    instruct_family = any(t in name for t in (
        "instruct", "mistral", "qwen", "nomic", "gte-", "stella", "icl", "instructor"))
    if "bge" in name and "-en" in name and "m3" not in name and "icl" not in name:
        auto = (BGE_EN_QUERY, "")
    elif "e5" in name and "bge" not in name and not instruct_family:
        auto = ("query: ", "passage: ")
    else:
        auto = ("", "")
        if instruct_family and "auto" in (query_prefix, passage_prefix):
            import warnings
            warnings.warn(
                f"{model!r} looks instruction-tuned but has no prefix rule here "
                "(e5-mistral, gte-Qwen, nomic and similar each use their own "
                "format); pass --{old,new}-{query,passage}-prefix explicitly, or "
                "its top-k will not be the model as deployed", stacklevel=2)
    q = auto[0] if query_prefix == "auto" else query_prefix
    p = auto[1] if passage_prefix == "auto" else passage_prefix
    return q, p
def corpus_key(model: str, prefix: str, ids) -> str:
    """Cache key for a corpus encoding: the model, its passage prefix, and the
    exact ordered list of unit ids. Anything less risks serving vectors for a
    different corpus -- a silent error of exactly the kind this project has
    hit before."""
    import hashlib

    h = hashlib.sha256(f"{model}|{prefix}|".encode())
    for i in ids:
        h.update(str(i).encode())
        h.update(b"\n")
    return h.hexdigest()[:32]


def encode_all(texts, model: str, batch: int = 256, device: str | None = None,
               prefix: str = "", cache_dir: str | None = None,
               cache_key: str | None = None, dtype: str = "float32"):
    """Real encoder. `lexical:word` / `lexical:char` are offline stand-ins."""
    if model.startswith("lexical:"):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        kind = model.split(":", 1)[1]
        vec = (TfidfVectorizer(analyzer="word", sublinear_tf=True, min_df=2,
                               token_pattern=r"[A-Za-z_][A-Za-z_0-9]*")
               if kind == "word" else
               TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                               sublinear_tf=True, min_df=3))
        # Clamp the SVD rank to the vocabulary. A fixed 192 crashes on any
        # corpus with fewer distinct features -- a small BEIR subset or one
        # small repository is enough to hit it.
        X = vec.fit_transform(texts)
        rank = max(1, min(192, X.shape[1] - 1, X.shape[0] - 1))
        svd = TruncatedSVD(rank, random_state=0)
        # `svd.fit_transform(X)` fits svd once and returns the reduced X; the
        # pipeline below is then a wrapper over the fitted vec and svd, so a
        # later `pipe.transform(new_texts)` reuses both. (The comment here read
        # "vec, svd already fitted" *before* the pipeline was built, which read
        # as though they were fitted twice.)
        Z = Normalizer().fit_transform(svd.fit_transform(X))
        pipe = make_pipeline(vec, svd, Normalizer())
        return Z.astype(np.float32), pipe

    from sentence_transformers import SentenceTransformer  # pragma: no cover

    m = SentenceTransformer(model, device=device)
    path = None
    if cache_dir and cache_key:
        path = Path(cache_dir) / f"{cache_key}{'.f16' if dtype == 'float16' else ''}.npy"
        if path.exists():
            # memory-mapped: the sharded search then reads one shard at a time
            print(f"  corpus embeddings from cache {path.name}")
            return np.load(path, mmap_mode="r"), m
    V = encode_to_array(m, texts, prefix=prefix, batch=batch, dtype=dtype,
                        path=path)
    return V, m


def close_memmap(a) -> None:
    """Release the file handle behind a memory-mapped array, if there is one.

    `encode_to_array` and `encode_corpus` return a `np.memmap` on the cached
    path on purpose: an 8.8M x 768 corpus is read one shard at a time and must
    not be brought into RAM. The cost is that the caller now holds an open
    file handle, and on Windows an open mapping makes the file undeletable --
    a test that encodes into a temporary directory fails to clean it up with
    `WinError 32`, which looks like a bug in the thing under test and is not.
    POSIX unlinks happily, so this only ever shows up on someone else's
    machine.

    Call this when done with such an array. Safe on an ordinary ndarray.
    """
    mm = getattr(a, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except (BufferError, ValueError):
            # still has views on it; dropping the reference is the caller's job
            pass


ENCODE_CHUNK = 100_000


def encode_to_array(m, texts, prefix: str = "", batch: int = 256,
                    dtype: str = "float32", path: Path | None = None,
                    chunk: int = ENCODE_CHUNK):
    """Encode `texts` in chunks, writing each chunk straight into the result.

    The previous version called `m.encode` on the whole corpus. sentence-
    transformers accumulates every batch and then stacks them as float32, so
    at 8.8M x 768 the peak was well over 50 GB before --dtype float16 was
    ever applied. Here the peak is one chunk.

    With a cache `path`, the result is written to `<path>.partial` as a
    memmap, with the rows completed recorded beside it, and renamed to `path`
    only when every row is written. An interrupted run resumes from the last
    completed chunk and a half-written file is never mistaken for a cache.
    float16 halves the corpus on disk and in RAM; every search upcasts one
    shard at a time, so the precision loss is confined to storage.
    """
    np_dtype = np.float16 if dtype == "float16" else np.float32
    n = len(texts)
    done = 0
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".partial")
        prog = path.with_name(path.name + ".progress")
        V = None
        if part.exists() and prog.exists():
            try:
                V = np.load(part, mmap_mode="r+")
                done = int(prog.read_text(encoding="utf-8").strip())
                if V.shape[0] != n or V.dtype != np_dtype or not 0 <= done <= n:
                    V, done = None, 0
                else:
                    print(f"  resuming encode at row {done:,} of {n:,}")
            except (ValueError, OSError):
                V, done = None, 0
    else:
        part = prog = None
        V = None
    for lo in range(done, n, chunk):
        hi = min(lo + chunk, n)
        E = m.encode([prefix + t for t in texts[lo:hi]], batch_size=batch,
                     convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False)
        if V is None:
            shape = (n, E.shape[1])
            V = (np.lib.format.open_memmap(part, mode="w+", dtype=np_dtype,
                                           shape=shape)
                 if part is not None else np.empty(shape, dtype=np_dtype))
        V[lo:hi] = E
        if part is not None:
            V.flush()
            prog.write_text(str(hi), encoding="utf-8")
        print(f"  encoded {hi:,} / {n:,}", flush=True)
    if V is None:                                   # n == 0
        raise ValueError("nothing to encode")
    if part is not None:
        V.flush()
        del V
        part.replace(path)
        prog.unlink(missing_ok=True)
        return np.load(path, mmap_mode="r")
    return V


def encode_with(fitted, texts, model: str, batch: int = 256, prefix: str = ""):
    """Encode queries with an already-loaded model. `prefix` is prepended for
    neural models (see resolve_prefixes); the lexical stand-ins ignore it."""
    if model.startswith("lexical:"):
        return fitted.transform(texts).astype(np.float32)
    return np.ascontiguousarray(  # pragma: no cover
        fitted.encode([prefix + t for t in texts], batch_size=batch, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32)


# ----------------------------------------------------------------------
# the old side
# ----------------------------------------------------------------------
def select_old_side(utext, qtext, old_model: str, V_shipped=None,
                    question_encoder=None, normalize: bool = False,
                    device: str | None = None, query_prefix: str = "",
                    passage_prefix: str = "", cache_dir: str | None = None,
                    cache_key: str | None = None):
    """Return (V_old, Q_old): the old index and the old-model query vectors.

    With shipped vectors (DPR Wikipedia), those vectors ARE the old index. They
    are used unchanged, queries go through the matched question encoder, and
    `encode_all` is never called -- the old side costs no corpus passes, which
    is the whole reason DPR is the headline corpus.

    This lives in one function, called by both `migration` and `estimate`, and
    is covered by a test that fails if the shipped vectors are re-encoded. An
    earlier version inlined it in `main`, and a patch that was meant to add the
    DPR branch silently failed to apply: every "DPR -> BGE" run was in fact
    lexical -> BGE, with the shipped vectors loaded and then overwritten.

    `normalize` is an ablation. DPR ranks by raw inner product, and that ranking
    is the old system's real candidate list, so the default leaves it alone.
    """
    if V_shipped is not None:
        if question_encoder is None:
            raise ValueError("shipped vectors need their matching question "
                             "encoder; refusing to fall back to another model")
        V_old = V_shipped
        Q_old = question_encoder(qtext)
    else:
        V_old, fitted = encode_all(utext, old_model, device=device,
                                   prefix=passage_prefix, cache_dir=cache_dir,
                                   cache_key=cache_key)
        Q_old = encode_with(fitted, qtext, old_model, prefix=query_prefix)
    if normalize:
        V_old = _safe_normalize(V_old)
        Q_old = _safe_normalize(Q_old)
    return V_old, Q_old


def _safe_normalize(M: np.ndarray) -> np.ndarray:
    """Row-normalise, leaving all-zero rows as zeros. Dividing by a zero norm
    yields NaN, and NaN scores make top-k selection undefined; lexical
    stand-ins do produce zero rows for all-out-of-vocabulary queries."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return (M / np.maximum(n, 1e-12)).astype(np.float32)


# ----------------------------------------------------------------------
# the measurement
# ----------------------------------------------------------------------
def topk_faiss(Q: np.ndarray, V: np.ndarray, k: int, gpu: bool = True):
    """Exact inner-product top-k via FAISS. Needed past ~10^6 units.

    Exact (IndexFlatIP), not approximate: this measures cross-space recall and
    coverage, and an ANN index here would add its own recall error to the very
    quantity being measured. Use ANN in the SYSTEM, never in the measurement.

    KNOWN COST, deliberately not optimised: a fresh index is built and pushed to
    the GPU for every shard, so an 8.8M corpus at the 2M shard size pays the
    `index_cpu_to_all_gpus` setup five times. A review put that at 10-30% of
    search time and suggested caching a `StandardGpuResources` and using
    `index_cpu_to_gpu(res, 0, ix)` instead. It is not taken, for two reasons:
    sharding itself is not optional (see `_sharded_topk` -- an IndexFlatIP keeps
    its own copy, and 8.8M x 768 float32 does not fit on one card), and
    `index_cpu_to_gpu(res, 0, ...)` pins the search to device 0 where
    `index_cpu_to_all_gpus` spreads it, which changes throughput and per-device
    memory on any multi-GPU machine. This file cannot be exercised on the
    machine that writes it -- there is no GPU and no faiss here -- so a change
    whose only benefit is speed and whose risk is a different device layout
    would be shipped untested into the one run that produces the paper's 8.8M
    numbers. If search time becomes the binding constraint on machine A, make
    the change there, where it can be timed, and raise `shard` first: it is a
    parameter, and it costs setup calls linearly.
    """
    import faiss

    Q = np.ascontiguousarray(Q, dtype=np.float32)

    def search(Vs: np.ndarray, kk: int):
        ix = faiss.IndexFlatIP(Vs.shape[1])
        if gpu and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
            ix = faiss.index_cpu_to_all_gpus(ix)
        ix.add(np.ascontiguousarray(Vs, dtype=np.float32))
        return ix.search(Q, kk)

    return _sharded_topk(V, k, search)


def _sharded_topk(V: np.ndarray, k: int, search, shard: int = 2_000_000):
    """Exact top-k over V, searched one corpus shard at a time and merged.

    An IndexFlatIP stores its own copy of every vector it holds, so adding all
    of V at once costs a second 64 GB at 21M on top of the array itself. Shards
    bound that copy to one shard. `search(Vs, kk)` returns (scores, local ids)
    for the shard; results are merged into a running exact top-k.
    """
    N = V.shape[0]
    k = min(k, N)
    best_s = best_i = None
    for lo in range(0, N, shard):
        hi = min(lo + shard, N)
        s_, i_ = search(V[lo:hi], min(k, hi - lo))
        i_ = i_.astype(np.int64) + lo
        if best_s is None:
            best_s, best_i = s_, i_
        else:
            cs = np.concatenate([best_s, s_], axis=1)
            ci = np.concatenate([best_i, i_], axis=1)
            if cs.shape[1] <= k:
                # Not yet k candidates in total: keep them all. argpartition
                # would be asked for an element past the end of the row.
                best_s, best_i = cs, ci
            else:
                keep = np.argpartition(-cs, kth=k - 1, axis=1)[:, :k]
                best_s = np.take_along_axis(cs, keep, axis=1)
                best_i = np.take_along_axis(ci, keep, axis=1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_i, order, axis=1).astype(np.int32)


def topk_chunked(Q: np.ndarray, V: np.ndarray, k: int, budget: float = 6e7):
    """Top-k indices per query, memory-bounded. Uses FAISS when installed."""
    try:
        import faiss  # noqa: F401

        return topk_faiss(Q, V, k)
    except ImportError:
        pass
    # Without FAISS, search one corpus shard at a time. The previous version
    # computed Q_chunk @ V.T against the whole corpus: with V a float16 memmap
    # numpy upcasts ALL of V to float32 for every query chunk (27 GB per chunk
    # at 8.8M x 768), so the memmapped float16 path never actually held.
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    shard = max(1, min(V.shape[0], 500_000))

    def search(Vs: np.ndarray, kk: int):
        Vs = np.ascontiguousarray(Vs, dtype=np.float32)   # one shard upcast
        qchunk = max(1, int(budget / max(Vs.shape[0], 1)))
        s_out = np.empty((Q.shape[0], kk), dtype=np.float32)
        i_out = np.empty((Q.shape[0], kk), dtype=np.int64)
        for lo in range(0, Q.shape[0], qchunk):
            S = Q[lo:lo + qchunk] @ Vs.T
            idx = np.argpartition(-S, kth=kk - 1, axis=1)[:, :kk]
            s_out[lo:lo + qchunk] = np.take_along_axis(S, idx, axis=1)
            i_out[lo:lo + qchunk] = idx
            del S, idx
        return s_out, i_out

    return _sharded_topk(V, k, search, shard=shard)


def x_targets_to_fracs(N: int, D: int, kprime: int, targets) -> tuple[list, list]:
    """Choose D so that x = D*K'/N hits each target at the reference K'.

    Fractions of the query set are an error-prone way to specify this -- the
    same fractions give x = 0.0002 on one corpus and x = 3 on another. Returns
    (fractions, unreachable targets); a target needing more intents than the
    query set holds is reported rather than silently clipped.
    """
    fracs, unreachable = [], []
    for x in targets:
        d = int(round(x * N / kprime))
        # Unreachable at either end: more intents than the query set holds, or
        # a load below what a single intent already produces (x = K'/N). The
        # low end used to be clipped to D = 1 without a word, so two targets
        # could silently collapse into one point.
        if d > D or d < 1:
            unreachable.append(x)
            continue
        fracs.append(d / D)
    return sorted(set(fracs)), unreachable


def fit_linear_adapter(V_old: np.ndarray, V_new: np.ndarray, idx: np.ndarray,
                       ridge: float = 1e-2) -> np.ndarray:
    """Least-squares map from the old space to the new one, fitted on `idx`.

    This is the mechanism behind adapter-style upgrades: learn W with
    V_old @ W ~= V_new on a sample, then apply it to every unit so the index
    "moves" to the new space without re-embedding. Fitting costs one new-model
    pass per sampled unit, which is the adapter's real price and is reported
    beside its recall.

    A linear map is the honest floor rather than a straw man: published
    adapters use small MLPs, and if the linear part already closes the gap the
    design question changes.
    """
    idx = np.asarray(idx)
    if idx.size < 2:
        raise ValueError(f"an adapter fitted on {idx.size} unit(s) is singular; "
                         "sample at least a few hundred")
    A = V_old[idx].astype(np.float64)
    B = V_new[idx].astype(np.float64)
    G = A.T @ A + ridge * len(idx) * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ B).astype(np.float32)


RIDGES_DEFAULT = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fit_adapter_selected(V_old, V_new, idx, ridges=RIDGES_DEFAULT, seed: int = 0):
    """Fit at several ridge values and keep the best, chosen on held-out units.

    The adapter is the competing method, so it should be given its best shot
    rather than a constant someone can call cherry-picked. Selection uses
    reconstruction quality on a fifth of the fitting sample that the fit never
    saw -- never the evaluation queries, which would be selecting on the metric
    being reported. The whole sweep is recorded so a reader can see how little
    (or much) the choice mattered.
    """
    idx = np.asarray(idx)
    if idx.size < 10:
        raise ValueError(f"--adapter {idx.size} is too small to fit and select "
                         "on held-out units; use at least a few hundred")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))
    cut = max(1, len(idx) // 5)
    val, fit = idx[perm[:cut]], idx[perm[cut:]]
    sweep, best, bestW = [], -np.inf, None
    for r in ridges:
        W = fit_linear_adapter(V_old, V_new, fit, ridge=r)
        pred = apply_adapter(V_old[val], W)
        score = float(np.mean(np.sum(pred * _safe_normalize(V_new[val]), axis=1)))
        sweep.append({"ridge": r, "heldout_cosine": round(score, 4)})
        if score > best:
            best, bestW = score, W
    chosen = sweep[int(np.argmax([x["heldout_cosine"] for x in sweep]))]["ridge"]
    return bestW, chosen, sweep


def apply_adapter(V_old: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Map the old index into the new space and renormalise."""
    return _safe_normalize((V_old @ W).astype(np.float32))


def adapter_report(V_old, V_new, Q_new, W, k: int, kprimes) -> dict:
    """Two questions an adapter raises, measured on the same corpus.

    1. As a REPLACEMENT: rank with adapter vectors alone and never re-embed.
       Its recall against the new model's true top-k is the quality an
       adapter-only upgrade delivers, permanently, for every unit.
    2. As a BETTER OLD INDEX: generate candidates from adapter vectors and
       still rerank exactly in the new space. The window starts closer to the
       new geometry, so the same K' should surface more of the true top-k --
       and every unit a query touches still ends up exact.

    The second is a strictly better use of the same artifact, and it is the
    hybrid the evaluation should report.
    """
    V_hat = apply_adapter(V_old, W)
    truth = topk_chunked(Q_new, V_new, k)
    out = {"adapter_only_recall_at_k": float(np.mean([
        len(set(t) & set(p)) / k
        for t, p in zip(truth, topk_chunked(Q_new, V_hat, k))]))}
    win_hat = topk_chunked(Q_new, V_hat, max(kprimes))
    out["hybrid"] = [
        {"kprime": kp,
         "hybrid_recall_at_kprime": float(np.mean(
             [len(set(t) & set(w[:kp])) / k for t, w in zip(truth, win_hat)]))}
        for kp in sorted(kprimes)]
    return out


KPRIMES_DEFAULT = (10, 20, 50, 100, 200, 500, 1000, 2000)
"""The candidate depths swept by default. One source of truth: `frontier` takes
it as its default and `main` derives from it the depths handed to the adapter
and baseline paths, so the three cannot drift apart."""


def frontier(V_old, V_new, Q_old, Q_new, k=10, kprimes=KPRIMES_DEFAULT,
             d_fracs=(1.0,), seed=0, repeats=5):
    """Sweep K' and the distinct-intent count D.

    The D sample must be RANDOM. Taking the first d queries biases the result
    whenever the query list is grouped -- pooled CIPBench concatenates
    repositories, so the first few dozen queries all come from one repo and
    their windows cover a fraction of one repo's units, which inflates savings
    dramatically. Sampled without replacement and averaged over `repeats`.
    """
    N, D = V_old.shape[0], Q_old.shape[0]
    # topk_chunked clamps k to N, so the divisor must be clamped too: dividing
    # a 5-unit intersection by k=10 reports 0.5 recall on identical spaces
    k = min(k, N)
    new_top = topk_chunked(Q_new, V_new, k)
    new_sets = [set(r.tolist()) for r in new_top]

    rows = []
    # One search at the deepest K', then prefixes of it. topk_chunked returns
    # each row in descending score order, so its first K' columns ARE the
    # top-K' for any smaller K'. At 21M units this replaces one full search
    # per K' with a single one.
    kps = [kp for kp in kprimes if kp <= N]
    if not kps:
        raise ValueError(f"every K' in {tuple(kprimes)} exceeds the corpus size "
                         f"N={N}; nothing to sweep. Use a smaller K' or a larger "
                         "corpus -- and note that recall@K' is uninformative "
                         "once K' is a large fraction of N.")
    full_win = topk_chunked(Q_old, V_old, max(kps))
    for kp in kps:
        old_win = full_win[:, :kp]
        # recall_at_kprime is over ALL queries, so it is the same on every D
        # row by construction: a query's window and its new-space top-k do not
        # depend on which other queries arrive. It is a property of the two
        # models and the old index, not a finding about D. The D-subset recall
        # below is what the queries in each sample actually got; it differs
        # from the full value only by sampling.
        per_q = np.array([len(new_sets[i] & set(old_win[i].tolist())) / k
                          for i in range(D)])
        rec = float(per_q.mean())
        rng = np.random.default_rng(seed)
        for df in d_fracs:
            # round, not int: --x-targets encodes an intent count as d/D, and
            # int(D * (d/D)) can land a unit low on the float, shifting the
            # achieved load off the target it was chosen to hit
            d = max(1, round(D * df))
            reps = 1 if d >= D else repeats
            vals, rsub = [], []
            for _ in range(reps):
                sel = (np.arange(D) if d >= D
                       else rng.choice(D, size=d, replace=False))
                seen = np.zeros(N, dtype=bool)
                seen[np.unique(old_win[sel])] = True
                vals.append(float(seen.mean()))
                rsub.append(float(per_q[sel].mean()))
            reached = float(np.mean(vals))
            # Recall@K' has a chance floor: two unrelated spaces still share
            # about K'/N of any top-k by accident, and recall -> 1 as K' -> N
            # whatever the spaces agree on. Report it corrected for chance so a
            # small corpus (or a large K') cannot pass for agreement.
            floor = min(1.0, kp / N)
            # At K' = N the window is the corpus, recall is 1 by construction
            # and the correction is undefined -- reporting 0.0 there would plot
            # as a collapse. Report None instead.
            rec_cc = (rec - floor) / (1.0 - floor) if floor < 1.0 else None
            rows.append(dict(
                k=k, kprime=kp, N=N, D=d,
                recall_at_kprime=round(rec, 4),
                recall_scope="all queries",
                recall_on_d_subset=round(float(np.mean(rsub)), 4),
                recall_on_d_subset_sd=round(float(np.std(rsub)), 4),
                recall_chance_floor=round(floor, 4),
                recall_chance_corrected=(None if rec_cc is None else round(rec_cc, 4)),
                reached=round(reached, 4),
                reached_sd=round(float(np.std(vals)), 4),
                savings_pct=round(100 * (1 - reached), 2),
                x_dk_over_n=round(d * kp / N, 4),
                # (1 - K'/N)^D is the exact expectation if each intent's window
                # were an independent draw; exp(-D*K'/N) is its K'/N << 1 limit.
                # They agree to 1e-6 at 21M but not on small corpora: at
                # K'/N = 0.4 the exponential overpredicts savings by ~6 points.
                pred_savings_indep=round(100 * float((1 - kp / N) ** d), 2),
                pred_savings_exp=round(100 * float(np.exp(-d * kp / N)), 2),
                pred_savings_linear=round(100 * max(0.0, 1 - d * kp / N), 2),
                embeddings_lazy=int(round(reached * N)),
                embeddings_full=N,
            ))
    return rows


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(prog="semview.migration")
    ap.add_argument("--corpus", default="cipbench",
                    choices=["cipbench", "jsonl", "msmarco", "beir", "dpr_wiki",
                             "orcas"])
    ap.add_argument("--path", default="bench_all_clean")
    ap.add_argument("--queries-file", default=None,
                    help="--corpus jsonl: a {\"id\",\"text\"} query file")
    ap.add_argument("--beir-name", default="nfcorpus")
    ap.add_argument("--limit", type=positive_int, default=None)
    ap.add_argument("--max-queries", type=positive_int, default=5000)
    ap.add_argument("--old", default="lexical:word")
    ap.add_argument("--new", default="lexical:char")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dpr-variant", default="psgs_w100.nq.no_index")
    ap.add_argument("--queries", default="nq-dev",
                    choices=["nq-dev", "nq-open-validation", "nq-open-train"],
                    help="DPR query set. nq-dev (default) has known gold passages "
                         "and is the only choice with --limit; the nq-open sets "
                         "need the full corpus. nq-open-train is memorised by both "
                         "encoders: coverage only, never recall.")
    ap.add_argument("--nq-dev", default=NQ_DEV_URL,
                    help="path or URL of DPR's biencoder-nq-dev.json(.gz)")
    ap.add_argument("--orcas-docs", default=None,
                    help="--corpus orcas: the MS MARCO document corpus TSV")
    ap.add_argument("--orcas-query-col", type=int, default=None)
    ap.add_argument("--orcas-doc-col", type=int, default=None)
    ap.add_argument("--qrels-split", default=None,
                    help="BEIR: restrict queries to this qrels split (use 'dev' for "
                         "msmarco, whose query file includes train queries) and, "
                         "with --limit, keep every gold document")
    ap.add_argument("--dpr-path", default=None,
                    help="local copy of facebook/wiki_dpr (the directory containing data/); default streams the parquet shards from the Hub")
    ap.add_argument("--skip-provenance", action="store_true")
    ap.add_argument("--adapter", type=int, default=0, metavar="M",
                    help="also measure a learned old->new linear adapter fitted "
                         "on M sampled units (M new-model passes, its real "
                         "cost). Reports what an adapter-only upgrade would "
                         "deliver, and what adapter-generated candidates plus "
                         "exact reranking deliver. 0 disables.")
    ap.add_argument("--baselines", action="store_true",
                    help="compare every migration scheme at equal cost: batch, "
                         "stale, adapter, an ordered backfill, a traffic-ordered "
                         "backfill, lazy, and the adapter hybrid. The budget is "
                         "not set here -- lazy spends what the traffic demands "
                         "and every other scheme is given exactly that. Add "
                         "--adapter for the two adapter rows. Half the queries "
                         "warm the migration, half evaluate it.")
    ap.add_argument("--budgets", default="8,16,64",
                    help="inline repair budgets per query for the latency rows "
                         "of --baselines: at most m of a window's missing units "
                         "are repaired before answering and the rest go to a "
                         "background repairer. 0 and K' are the two schemes "
                         "already in the table, so these fill the curve between "
                         "them. Empty string disables the rows.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="storage dtype for the NEW-side corpus embeddings, in "
                         "RAM and in the cache. float16 halves 8.8M x 768 from "
                         "27 GB to 13.5 GB; searches upcast one shard at a time.")
    ap.add_argument("--reencode-old", action="store_true",
                    help="DPR only: ignore the shipped vectors and encode the old "
                         "side with --old instead. The control experiment for "
                         "'is the window depth the encoder pair or the corpus?': "
                         "it holds the corpus and queries fixed and changes only "
                         "the pair. Costs one pass over the corpus with --old.")
    ap.add_argument("--old-normalize", action="store_true",
                    help="ablation: L2-normalise the old side. The default "
                         "keeps DPR's native raw inner product, which is how "
                         "the deployed system actually ranked candidates.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--x-targets", default=None,
                    help="comma-separated loads x = D*K'/N to hit at --x-kprime, "
                         "e.g. 0.05,0.2,1,3; overrides --d-fracs. Unreachable "
                         "targets are reported, never silently clipped.")
    ap.add_argument("--x-kprime", type=positive_int, default=500)
    ap.add_argument("--cache-dir", default=None,
                    help="cache neural corpus embeddings here, keyed by model, "
                         "prefix and the exact unit ids; ablations and reruns "
                         "then skip the corpus encode")
    ap.add_argument("--d-fracs", default="1.0",
                    help="comma-separated fractions of the query set to treat as "
                         "the distinct-intent population, e.g. 0.01,0.1,1.0. "
                         "Savings are governed by D*K'/N, so this is the axis "
                         "that decides whether deferral pays at all.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the --limit corpus sample (and the D sample "
                         "unless --d-seed is given)")
    ap.add_argument("--d-seed", type=int, default=None,
                    help="seeds ONLY the D sample, leaving the corpus and its "
                         "embeddings untouched. Use this for error bars: with "
                         "--seed the corpus changes too, so every repeat "
                         "re-encodes it; with --d-seed the cache is reused and "
                         "a repeat takes seconds.")
    for side in ("old", "new"):
        for kind in ("query", "passage"):
            ap.add_argument(f"--{side}-{kind}-prefix", default="auto",
                            help=f"text prepended to {side}-model {kind}s; "
                                 "'auto' looks the model up (BGE, E5), '' disables")
    ap.add_argument("--out", default="migration.json")
    ap.add_argument("--dump-queries", default=None, metavar="DIR",
                    help="write q_old.npy and q_new.npy for this run's query "
                         "set into DIR, for analyses that need matrices rather "
                         "than a frontier (analysis/n_ladder.py). Corpus vectors "
                         "are already cached under cache/<key>.npy and the run "
                         "log names them; query vectors are never cached, which "
                         "is what this exists for")
    a = ap.parse_args()

    require_dpr_limit(a.corpus, a.limit, "migration")
    V_old_pre = None
    if a.corpus == "dpr_wiki":
        units, V_old_pre, dpr_pairs, queries = dpr_corpus(
            a.limit, a.dpr_variant, a.dpr_path, a.queries, a.nq_dev,
            keep_text=True, seed=a.seed)
        if a.reencode_old:
            print("skipping the DPR provenance check: --reencode-old means the "
                  "shipped vectors are not used")
        elif not a.skip_provenance:
            prov = verify_dpr_provenance(dpr_pairs, V_old_pre,
                                         a.dpr_variant, device=a.device)
            if not prov["ok"]:
                raise SystemExit("DPR provenance check failed; aborting")
        print(f"DPR Wikipedia: {len(units):,} passages, shipped DPR vectors "
              f"{V_old_pre.shape}; old side costs no encoder passes")
    elif a.corpus == "orcas":
        if not a.orcas_docs:
            raise SystemExit("--corpus orcas needs --orcas-docs (the MS MARCO "
                             "document corpus TSV)")
        units, queries = load_orcas(a.path, a.orcas_docs, a.limit, a.max_queries,
                                    a.seed, a.orcas_query_col, a.orcas_doc_col)
    elif a.corpus == "cipbench":
        units, queries = load_cipbench(a.path, a.limit, a.seed)
    elif a.corpus == "jsonl":
        units, queries = load_jsonl(a.path, a.queries_file, a.limit, a.seed)
    elif a.corpus == "msmarco":
        if a.limit is not None:
            print("WARNING: --corpus msmarco with --limit takes the first N passages "
                  "and keeps all 500k+ queries (train included), so most queries "
                  "have no answer in the corpus and some were trained on. Prefer "
                  "--corpus beir --beir-name msmarco --qrels-split dev.")
        units, queries = load_msmarco(a.limit)
    else:
        units, queries = load_beir(a.beir_name, a.path, a.limit, a.seed,
                                   a.qrels_split)

    if a.max_queries and len(queries) > a.max_queries:
        rng = np.random.default_rng(a.seed)
        queries = [queries[i] for i in
                   rng.choice(len(queries), a.max_queries, replace=False)]

    utext = [t for _, t in units]
    qtext = [t for _, t in queries]
    print(f"corpus={a.corpus}  N={len(utext)}  D={len(qtext)}")
    # --reencode-old drops the shipped vectors so that --old is honoured; the
    # label must follow, or the results file would name a model that was not used.
    shipped = None if a.reencode_old else V_old_pre
    label = old_label(a.old, a.dpr_variant if shipped is not None else None,
                      a.old_normalize)
    if a.reencode_old and V_old_pre is not None:
        print(f"old model: {a.old} (re-encoding the DPR corpus; shipped vectors "
              "ignored by --reencode-old)")
        qenc = None
    elif shipped is not None:
        print("old model: DPR (shipped context vectors + matched question encoder)")
        qenc = (lambda t: encode_dpr_questions(t, a.dpr_variant, device=a.device))
    else:
        # Lexical stand-ins are fitted on units only; see analysis/leak_test.py.
        print(f"old model: {a.old}")
        qenc = None
    oq, op = resolve_prefixes(a.old, a.old_query_prefix, a.old_passage_prefix)
    nq, np_ = resolve_prefixes(a.new, a.new_query_prefix, a.new_passage_prefix)
    print(f"prefixes  old: query={oq!r} passage={op!r}   new: query={nq!r} passage={np_!r}")
    # With shipped vectors the old side is never encoded, so hashing 21M ids
    # for a cache key that is then ignored is pure waste.
    old_key = (None if shipped is not None
               else corpus_key(a.old, op, (i for i, _ in units)))
    V_old, Q_old = select_old_side(utext, qtext, a.old, V_shipped=shipped,
                                   question_encoder=qenc,
                                   normalize=a.old_normalize, device=a.device,
                                   query_prefix=oq, passage_prefix=op,
                                   cache_dir=a.cache_dir, cache_key=old_key)
    if a.old_normalize:
        print("ABLATION: old side L2-normalised (not how DPR actually ranked)")
    print(f"new model: {a.new}")
    V_new, f_new = encode_all(utext, a.new, device=a.device, prefix=np_,
                              cache_dir=a.cache_dir, dtype=a.dtype,
                              cache_key=corpus_key(a.new, np_, (i for i, _ in units)))
    Q_new = encode_with(f_new, qtext, a.new, prefix=nq)

    if a.dump_queries:
        # Query vectors are never cached, so an analysis that wants matrices
        # instead of a frontier has no way to obtain them. Dumping them here --
        # from the same encoder, the same prefix resolution and the same query
        # set the frontier below uses -- is what keeps the two consistent; the
        # corpus side is already on disk under cache/<key>.npy and the log names
        # it. Nested-subset ladders (analysis/n_ladder.py) are the caller.
        d = Path(a.dump_queries)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "q_old.npy", Q_old)
        np.save(d / "q_new.npy", Q_new)
        print(f"dumped query vectors to {d}: q_old {Q_old.shape} "
              f"{Q_old.dtype}, q_new {Q_new.shape} {Q_new.dtype}")

    if a.x_targets:
        targets = [float(x) for x in a.x_targets.split(",")]
        dfr, unreach = x_targets_to_fracs(len(utext), len(qtext), a.x_kprime, targets)
        if unreach:
            lo, hi = a.x_kprime / len(utext), len(qtext) * a.x_kprime / len(utext)
            print(f"WARNING: x targets {unreach} are outside what this corpus and "
                  f"query set can produce at K'={a.x_kprime} (x from {lo:.3g} with "
                  f"one intent to {hi:.3g} with all {len(qtext)}); omitted")
        if not dfr:
            raise SystemExit("no x target is reachable with this query set")
        dfr = tuple(dfr)
    else:
        dfr = tuple(sorted({float(x) for x in a.d_fracs.split(',')}))
    xs = [max(1, round(len(qtext) * f)) * a.x_kprime / len(utext) for f in dfr]
    print(f"load x = D*K'/N at K'={a.x_kprime}: " + ", ".join(f"{x:.3g}" for x in xs)
          + ("" if any(0.05 <= x <= 3 for x in xs)
             else "   (none in the informative range 0.05-3)"))
    # the depths the adapter and baseline paths report at; derived from the
    # same constant frontier() sweeps, and clipped to corpora smaller than it
    kps = tuple(kp for kp in KPRIMES_DEFAULT if kp <= len(units))
    if not kps:                       # a corpus smaller than the shallowest depth
        raise SystemExit(f"corpus has {len(units)} units, fewer than the "
                         f"shallowest depth swept ({KPRIMES_DEFAULT[0]}); "
                         f"there is nothing to measure at this scale")

    rep = None
    if a.adapter:
        if a.adapter < 10 or a.adapter >= len(units):
            raise SystemExit(f"--adapter {a.adapter} must be at least 10 and "
                             f"smaller than the corpus ({len(units)})")
        arng = np.random.default_rng(a.seed)
        widx = arng.choice(len(units), a.adapter, replace=False)
        W, chosen, sweep = fit_adapter_selected(V_old, V_new, widx, seed=a.seed)
        rep = adapter_report(V_old, V_new, Q_new, W, a.k, kps)
        rep["ridge_chosen"] = chosen
        rep["ridge_sweep"] = sweep
        rep["train_units"] = int(a.adapter)
        rep["train_frac_of_corpus"] = round(a.adapter / len(units), 5)
        print(f"adapter fitted on {a.adapter:,} units "
              f"({100 * a.adapter / len(units):.2f}% of the corpus); "
              f"ridge {chosen:g} chosen on held-out units, sweep "
              f"{[x['heldout_cosine'] for x in sweep]}:")
        print(f"  adapter alone, nothing re-embedded: recall@{a.k} = "
              f"{rep['adapter_only_recall_at_k']:.3f}")
        for r in rep["hybrid"]:
            print(f"  adapter candidates + exact rerank: K'={r['kprime']:<5} "
                  f"recall = {r['hybrid_recall_at_kprime']:.3f}")

    base = None
    if a.baselines:
        from .baselines import compare_schemes, report
        half = len(Q_old) // 2
        if half < 2:
            raise SystemExit("--baselines needs at least 4 queries")
        # the traffic that drives lazy migration must not be the traffic it is
        # scored on, or it is being credited for units its own evaluation chose
        budgets = tuple(int(b) for b in a.budgets.split(",") if b.strip())
        base = compare_schemes(V_old, V_new, Q_old[half:], Q_new[half:],
                               Q_old[:half], Q_new_warm=Q_new[:half], k=a.k,
                               kprime=min(kps, key=lambda x: abs(x - 500)),
                               budget=None,
                               adapter_train=a.adapter, seed=a.seed,
                               budgets=budgets)
        spent = [r for r in base if r["scheme"] == "lazy"][0]["passes"]
        print(f"\nschemes at equal cost: lazy spent {spent:,} new-model passes "
              f"({100 * spent / len(units):.1f}% of the corpus), and every other "
              f"scheme is given the same. {half} warm-up queries, "
              f"{len(Q_old) - half} evaluation queries:")
        report(base, len(units))
        print()

    rows = frontier(V_old, V_new, Q_old, Q_new, k=a.k, d_fracs=dfr,
                    seed=a.seed if a.d_seed is None else a.d_seed)

    ds = sorted({r["D"] for r in rows})
    hdr = (f"{'K':>6}{'recall@K':>10}" +
           "".join(f"{('sav%@D=' + str(d)):>14}" for d in ds))
    print("\n" + hdr); print("-" * len(hdr))
    for kp in sorted({r["kprime"] for r in rows}):
        at = {r["D"]: r for r in rows if r["kprime"] == kp}
        print(f"{kp:>6}{at[ds[0]]['recall_at_kprime']:>10.3f}" +
              "".join(f"{at[d]['savings_pct']:>14.1f}" for d in ds))
    print(f"\nfull recomputation would embed {len(utext):,} units")

    from .results_io import environment, write_results
    write_results(a.out,
        dict(corpus=a.corpus, N=len(utext), D=len(qtext), k=a.k,
             old=label, new=a.new,
             limit=a.limit, seed=a.seed, max_queries=a.max_queries,
             qrels_split=getattr(a, "qrels_split", None),
             query_ids=[q[0] for q in queries],
             fingerprint=dict(old_dim=int(V_old.shape[1]), new_dim=int(V_new.shape[1]),
                              old_head=_fingerprint(V_old), new_head=_fingerprint(V_new),
                              old_strided=_fingerprint_strided(V_old),
                              new_strided=_fingerprint_strided(V_new),
                              new_storage_dtype=a.dtype,
                              query_ids_sha256=_ids_hash(q[0] for q in queries),
                              unit_ids_sha256=_ids_hash(u[0] for u in units)),
             environment=environment(dtype=a.dtype, device=a.device),
             prefixes=dict(old_query=oq, old_passage=op, new_query=nq,
                           new_passage=np_),
             adapter=rep,
             baselines=base,
             frontier=rows))


if __name__ == "__main__":
    main()
