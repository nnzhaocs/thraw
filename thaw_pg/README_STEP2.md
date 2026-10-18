# Step 2: the vertical slice

> The full run order is in `RUNTESTS.md`; this file explains one part of it.

> **Run every command whose numbers may reach the paper through `python -m semview.runlog`** (its own directory, log, checks, manifest, a line in `results/RUNS.md`). See `docs/history/README_RESULTS.md`.

Step 2 builds the Thaw query path end to end on PostgreSQL + pgvector and
measures it on the real 500k MS MARCO subset, MiniLM → BGE. It answers two
questions:

- **E1:** with an exact old-space window, does the real system reproduce the
  instrument's recall (0.929 at K′ = 500 over all 5,000 dev queries)?
- **E2:** with the HNSW window a deployment would use, how much recall does
  the index lose, and what latency does a query see?

**Start only after step 1 passes on the real data.** Step 2 relies on its
findings: iterative scan on (`relaxed_order`), claims in sorted order, and the
`m` / `ef_construction` you settled on.

---

## 1. What runs

```
query text ──► old-model query encoder (TEI :8081) ──► q_old
           ──► docs:  SELECT id ORDER BY emb_old <=> q_old LIMIT K'      window
           ──► docs_new: SELECT id WHERE id = ANY(window)                present
           ──► RepairCoordinator (in memory, sorted claims)               stale
                 ├─ docs: fetch text for stale ids
                 ├─ new-model encoder (TEI :8080), chunks of --chunk
                 └─ COPY into docs_new (origin = 'repair')
           ──► new-model query encoder (TEI :8080) ──► q_new
           ──► docs_new: SELECT id WHERE id = ANY(window) ORDER BY emb <=> q_new LIMIT 10
```

| file | role |
|---|---|
| `thaw_pg/retriever.py` | `ThawRetriever` (the query path) and `RepairCoordinator` (each unit encoded at most once per process, claims in sorted order, waits on units another thread is encoding) |
| `thaw_pg/encoders.py` | `TEIEncoder` (HTTP), `LocalEncoder` (sentence-transformers in-process), `TableEncoder` (lookup, for the self-test) |
| `thaw_pg/step2.py` | `prepare2`, `verify-encoder`, `step2`, `selftest2` |
| `thaw_pg/docker-compose.tei.yml` | TEI for both models on the GPU server |

Each query records where its time went: old query encode, window, present,
text fetch, encode, write, wait, new query encode, rank. Step 2 stays single
process. Database claims (for several coordinators), the background filler
and the repair policies come in steps 3–4.

---

## 2. Machines

The final latency numbers use **two machines**. Development can use one.

| machine | runs | spec |
|---|---|---|
| **DB server** | Postgres + pgvector (step 1's `docker-compose.yml`) | 16+ cores, 64–128 GB RAM, NVMe; no GPU |
| **GPU server** | TEI (both models) and the `step2` runner | 1 × 24–48 GB GPU, 16+ cores, 64 GB RAM |

Put them in the same datacenter or LAN. Record the round-trip time
(`ping -c 20 <db-host>`) in your notes, because every query makes four or five
round trips to the database. Splitting the machines keeps Postgres from
competing with the encoder and the runner for CPU, and it matches how
deployments separate the database from inference.

**On one machine,** everything below works unchanged with
`THAW_PG_DSN` pointing at localhost. Say in the paper that latency then
includes CPU contention between the database and the encoder, which makes it
an upper bound.

---

## 3. Set up

### 3a. DB server

This is step 1's setup, already loaded. The `docs` table holds the 500k
vectors and text.

`docs_new` still holds step 1's random vectors. `step2` truncates it before
every run, so nothing needs doing.

Allow the GPU server to connect. With the step-1 compose file, Postgres
listens on port 5432. Open it to the GPU server's address only.

**Fix the memory settings first.** Step 1 ran on Postgres defaults
(`shared_buffers` 256 MB), smaller than the HNSW index alone. Follow
`thaw_pg/README_POSTGRES.md`: set the values for your RAM, restart, confirm
with `python -m thaw_pg settings`, and prewarm the tables. Do the same prewarm
after every Postgres restart, before any run whose latency you will report.

### 3b. GPU server: TEI

```bash
export TEI_IMAGE=ghcr.io/huggingface/text-embeddings-inference:TAG_FOR_YOUR_GPU
docker compose -f thaw_pg/docker-compose.tei.yml up -d
curl -s localhost:8080/embed -H 'Content-Type: application/json' \
     -d '{"inputs":["hello"]}' | head -c 80      # a vector comes back
```

`--dtype float32` is set deliberately. TEI defaults to fp16 on GPU, which
moves vectors slightly away from the fp32 ground truth. Keep fp32 for E1. You
can try fp16 later as a separate measurement.

### 3c. GPU server: Python

```bash
pip install -r requirements.txt -r thaw_pg/requirements.txt
export THAW_PG_DSN="postgresql://postgres:postgres@DBHOST:5432/thaw"
```

Copy `step1_data/` from wherever step 1's `prepare` ran (`vectors.npz`,
`texts.jsonl`).

---

## 4. Run

### 4a. Self-test (synthetic, about 1 minute, no model or GPU)

Run it against a database loaded by step 1's `quick`, **never the one holding
the real 500k load**: `quick` drops and reloads the tables and `selftest2`
overwrites `docs.text`. Both now refuse when they find real data (use another
`--data-dir` and a separate database such as `.../thaw_test`, or `--force`).

```bash
python -m thaw_pg --data-dir step1_quick quick --n 50000      # if not done already
python -m thaw_pg --data-dir step1_quick selftest2
```

It runs the real query path with 4 threads and a lookup encoder. Expect every
gate to PASS (42 at 50k rows: four configurations, each with and without the
union; the HNSW-plan gates are added at 100k rows or more, which takes the count
to 46). Its recall numbers come from synthetic data and mean nothing.

**The count has grown with each gate added** — 36, then 40 when *union produced
candidates outside the window* went in, now 42. The last two are the plan gates
for the exact-window union configurations, added when they were found to be
missing: an exact window with the union was falling through to the HNSW rule and
being failed for correctly avoiding an index. See
`test_exact_window_with_the_union_is_not_required_to_use_hnsw`.

### 4b. Ground truth (once; GPU)

```bash
python -m thaw_pg prepare2 --beir-path beir/ --cache-dir cache/
```

This encodes the 500k corpus with BGE, only to get each query's true top-10.
That cost belongs to the evaluation, not the system. If `cache/` holds the
BGE vectors from the paper's 500k run, nothing is re-encoded. It writes
`step1_data/truth.npz` and `queries.jsonl`.

### 4c. Encoder check (a gate)

```bash
python -m thaw_pg verify-encoder --enc-new tei:http://localhost:8080 \
    --enc-old tei:http://localhost:8081
```

The system's encoders must reproduce the ground truth's vectors: minimum
cosine ≥ 0.999 on 256 passages and 256 queries. If this fails, the cause is
pooling, truncation, prefix or model revision. Fix it first, or every recall
number below is measuring encoder drift.

### 4d. E1 headline: reproduce 0.929 (recall only, about 30–60 min)

```bash
python -m semview.runlog --name step2_e1_5000 --tag final -- \
  python -m thaw_pg step2 --enc-new tei:http://localhost:8080 \
    --precomputed --warm --windows exact --kprimes 500 --queries 5000 \
    --out "{run}/step2.json"
```

All 5,000 dev queries, exact window, stored query vectors. Only the database
path and the repair encoder are under test. `--warm` keeps repairs between
runs, so the latencies here are not a migration's. **Expect `e2e` ≈ 0.9295**,
the paper's number. The gate compares it against the offline number for the
same queries.

### 4e. E1 + E2: the migration replay (about 1–2 h on a 4060-class GPU, less on a datacenter GPU)

```bash
python -m semview.runlog --name step2_e2_500 --tag final -- \
  python -m thaw_pg step2 --enc-new tei:http://localhost:8080 \
    --enc-old tei:http://localhost:8081 \
    --windows exact,hnsw --kprimes 200,500,1000 --queries 500 --chunk 16 \
    --out "{run}/step2.json"
```

Each of the six runs starts from an empty `docs_new` and replays the same 500
queries in order with live query encoding, so latency is what a user sees
during a migration. `--chunk` is the repair batch: use `repair_chunk` from
`analysis/measure_throughput.py` run on **this** GPU (see
`docs/history/README_THROUGHPUT.md`), the measured batch size with the lowest repair
penalty. On the RTX 4060 Laptop GPU it is 16; chunking at 8 there, or at 4,
loses 6 % and 9 % respectively on the curve. Expect the best batch to be larger
on a datacenter card. Note this only fixes the **chunk**; the penalty the paper
quotes comes from the matched `backfill-rate` run, not from this curve — see
`docs/history/README_MEASURE.md`.

Add `--queries 1000` or `--kprimes 2000` once the 500-query run looks right.
K′ = 2000 needs iterative scan, which is on by default.

---

## 5. Gates and outputs

`step2.json` in the run directory (plus `step2.traces.json`, one row per
query; both checked and checksummed) and a summary:

```
  window      K     e2e   exact  ANN loss      p50      p99  reached
  exact     500  0.93xx  0.93xx   +0.0000      ...      ...      ...
  hnsw      500  0.9xxx  0.93xx   +0.0xxx      ...      ...      ...
```

| gate | meaning |
|---|---|
| **persisted once** | rows in `docs_new` = distinct ids = units encoded. Nothing was encoded or written twice |
| **rank is exact within the window** | for ≥ 99% of queries the final top-10 contains exactly the true top-10 members that were in the window. Repair, write-back and SQL ranking are correct |
| **full window** | every query got K′ rows |
| **E1 reproduces offline recall** (exact) | system recall within 0.005 of the offline exact-window recall for the same queries |
| **exact plan has no HNSW / HNSW plan** | each mode actually ran the plan it claims (the HNSW gate applies at N ≥ 100k) |

Columns to report:
- **ANN loss** = offline exact-window recall − HNSW-window recall. This is the
  error the paper's exact-search numbers don't include. Report it for each K′,
  next to step 1's `m` / `ef_construction`.
- **latency**: p50/p95/p99 and first-decile versus last-decile p50 (how fast
  first-touch cost decays), plus the per-component means in the JSON. Encode
  time per unit, compared with step 1's throughput curve, shows how much TEI
  and HTTP add.
- **reached**: the fraction of the corpus encoded after the replay. Compare
  with `estimate` at the same D and K′.

If **E1 fails**, first look for the `score tie at the window boundary`
warning. Ties make the window depend on tie-breaking, which Postgres and numpy
do differently. Duplicate passages tie (220 of 5,000 MS MARCO queries had a boundary tie at
K′ = 500, and E1 still matched to four decimals). A hashed stand-in encoder
ties on nearly every query, and failed E1 at K′ = 500 for that reason alone. Without ties, rerun with `--precomputed`. If it then passes,
the gap is in the query encoders, so go back to `verify-encoder`.

---

## 6. What we verified here

Development and gating ran on PostgreSQL 16 + pgvector 0.8.1. **The shipped
`results/step1/step1_report.json` is from PostgreSQL 18.1 + pgvector 0.8.5**, and
the paper quotes that pair, so the two are stated separately rather than
blended: everything below is the 16/0.8.1 result, and the five step-1 gates the
paper cites are the 18.1/0.8.5 report.

- **`selftest2`** with 4 threads and a lookup encoder: every gate passed at
  50k synthetic rows (42 now that it also gates the union configurations'
  plans), and at 150k, where the HNSW-plan gates apply. In
  exact mode the system's recall equalled the offline recall to four decimals.
  No unit was encoded or written twice under concurrency.
- **The full CLI path** (`prepare` → `load` → `prepare2` → `verify-encoder` →
  `step2`, with a stand-in model on a 5k toy BEIR corpus): everything ran.
  E1 failed only at K′ = 500, where every query had a boundary tie because of
  the stand-in encoder. That failure is what the tie warning above is for.

Neither run says anything about real recall or latency. That comes from 4d
and 4e on your machines.

---

## 7. Exit criterion → step 3

- `verify-encoder` passes.
- 4d gives e2e ≈ 0.9295 and passes its gates.
- 4e passes all gates, and the report gives ANN loss and latency for K′ =
  200/500/1000.

Then step 3: database claims for several coordinators, the background filler,
crash and restart (E6).

## Troubleshooting

| symptom | fix |
|---|---|
| `verify-encoder` min cosine ≈ 0.98 | pooling or prefix mismatch: check that TEI loaded `BAAI/bge-base-en-v1.5` with CLS pooling, and that the passage prefix is empty and the query prefix is BGE's instruction |
| `verify-encoder` ≈ 0.9995 but not 1.0 | fp16 or kernel differences; above 0.999 is fine |
| `HTTP 413` / `422` from TEI | raise `--max-client-batch-size` or lower `--chunk` |
| encode ms/unit far above the throughput curve | HTTP overhead per small request; compare `--chunk 8` with `16`/`32` on this GPU |
| `FAIL full window` on HNSW | iterative scan off or `max_scan_tuples` too low; keep `--iterative relaxed_order` |
| a run is too slow | `--queries 200`; or `--warm` for recall-only runs |
