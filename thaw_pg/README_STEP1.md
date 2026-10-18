# Step 1: pgvector feasibility

> The full run order is in `RUNTESTS.md`; this file explains one part of it.

Step 1 checks, with throwaway scripts and no service, that PostgreSQL + pgvector
behaves the way the Thaw architecture assumes. Each check corresponds to one
assumption. If a check fails, the design changes, so you find out before
building step 2 on top of it.

**Exit criterion:** `thaw_pg check` prints five PASS lines on the real 500k
MS MARCO data, and its `step1_report.json` is saved in a runlog run directory. Then
start step 2.

---

## 1. What is checked, and why

| # | check | the assumption it tests | if it fails |
|---|---|---|---|
| 1 | **index use** | The planner uses the old-side HNSW index for the window, *can* use the partial HNSW index for post-switch rows, never uses it where it can't apply, and uses the primary key for window fetches | Partial index unusable: keep post-switch rows in their own table instead of one table with `origin` |
| 2 | **window depth** | pgvector can return a full window of K′ rows for K′ = 200/500/1000, and it reports which setting achieves that. Overlap with the exact window and latency are also reported: that is a first look at E2 | `ef_search` caps at 1000. If iterative scans can't fill K′ = 2000, the DPR pair becomes optional |
| 3 | **NULL skip** | After `UPDATE docs SET emb_old = NULL` (how a document leaves the old space when it's rewritten), the row is never returned | A correctness bug: filter in SQL (`WHERE emb_old IS NOT NULL`) and recheck |
| 4 | **dead tuples** | Before VACUUM, dead index entries don't shrink a window below K′, at least under one setting | Run with `hnsw.iterative_scan` on, or make VACUUM part of the update path |
| 5 | **claims** | Concurrent queries with overlapping windows never claim (and so encode) the same row twice. A crashed worker's expired claim is reclaimed; live claims are left alone | The claim protocol is wrong; fix it before step 3 builds on it |

The claim protocol under test is a sorted multi-row `INSERT ... ON CONFLICT
DO NOTHING RETURNING id` into `repair_claims`. Each returned id belongs to
exactly one worker. Workers wait for ids that someone else claimed. A lease
reclaims claims older than 5 minutes whose row was never finished.

---

## 2. Machines and configuration

### How many machines

| stage | machines | GPU |
|---|---|---|
| **Step 1 (this)** | **1**, CPU only. A laptop is fine for `quick` | not needed. Only `prepare --source msmarco` encodes queries and passages with MiniLM, and that is free if you have `cache/` from the 500k run |
| Steps 2–4 (system, E1–E4) | **1 GPU server** running Postgres, TEI and the Thaw service, plus **1 small CPU machine** for the load generator in E4, so the replayer doesn't take CPU from Postgres | 1 GPU, 24 GB or more |
| E9 (8.8M in pgvector) | the same GPU server, with more RAM | — |

Use one server for the system, not separate database and GPU boxes. The paper
then reports a single-node system, as the scope sentence says, and network
hops between the service and Postgres don't distort the latency numbers.

### Recommended specs

| | step 1 | steps 2–4 (500k) | E9 (8.8M) |
|---|---|---|---|
| CPU | 8+ cores | 16–32 cores | 32 cores |
| RAM | 32 GB | 64–128 GB | 128 GB (the HNSW build for 8.8M × 384 wants tens of GB of `maintenance_work_mem`) |
| Disk | 100 GB NVMe | 500 GB NVMe | 1 TB NVMe |
| GPU | none | 1 × 24–48 GB (L4, A10, 4090, A100, …) | same |
| OS | Linux with Docker | Linux, Docker, NVIDIA driver + container toolkit | same |

Use NVMe, not network disks. Index scans on cold pages are random reads, and
a network disk shows up directly in p99.

### Postgres settings

`docker-compose.yml` is set for a 32–64 GB machine:

| setting | value | why |
|---|---|---|
| `shared_buffers` | 8GB (¼ of RAM, max ~32GB) | keeps the HNSW graph hot |
| `effective_cache_size` | 24GB (¾ of RAM) | planner hint |
| `maintenance_work_mem` | 8GB | HNSW builds much faster when the graph fits; 500k × 384 is ~1 GB, 8.8M needs much more |
| `max_parallel_maintenance_workers` | 7 (cores − 1) | parallel HNSW build |
| `max_connections` | 200 | the claim check opens 16 connections; the service will open more |
| `shm_size` (Docker) | 8g | Docker's 64 MB default breaks parallel index builds |

On a 128 GB machine, double `shared_buffers`, `effective_cache_size` and
`maintenance_work_mem`.

---

## 3. Install

From the bundle root (the directory with `semview/`; put `thaw_pg/` beside it):

```bash
# Postgres 17 + pgvector 0.8.1
docker compose -f thaw_pg/docker-compose.yml up -d
docker exec thaw-pg psql -U postgres -d thaw -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector';"
#   expect 0.8.1. If that image tag is missing, use pgvector/pgvector:pg17 and confirm the version is 0.8.0 or later.

pip install -r thaw_pg/requirements.txt
export THAW_PG_DSN="postgresql://postgres:postgres@localhost:5432/thaw"
```

**Without Docker:** install PostgreSQL 16 or 17 and pgvector 0.8.x from your
distribution (`postgresql-17-pgvector` on Debian/Ubuntu with the PGDG
repository), apply the settings above in `postgresql.conf`, create a database
`thaw`, and point `THAW_PG_DSN` at it.

pgvector must be **0.8.0 or later**. Checks 2 and 4 depend on iterative index
scans, which arrived in 0.8.0.

---

## 4. Run

### 4a. Smoke test on synthetic data (about 1 minute)

```bash
python -m thaw_pg --data-dir step1_quick quick --n 50000
```

This generates 50k clustered vectors, loads them, and runs all five checks
with small sizes. It tests the tooling, not the design. Expect five PASS lines.
Synthetic overlap with the exact window is low, around 0.5–0.8. That comes
from the synthetic distribution, so don't read anything into it.

### 4b. The real run: 500k MS MARCO, MiniLM (about 20–40 minutes)

```bash
# 1. vectors: same 500k subset and 5,000 dev queries as the paper's run
python -m thaw_pg prepare --source msmarco --beir-path beir/ --cache-dir cache/
#    reuses cache/ from the 500k run; without it, MiniLM encodes 500k passages
#    (minutes on a GPU, longer on CPU). Writes step1_data/vectors.npz + texts.jsonl

# 2. tables and indexes
python -m thaw_pg load --m 16 --ef-construction 64 --workers 7 --maint-mem 8GB

# 3. the checks
python -m semview.runlog --name step1_check --tag final -- \
  python -m thaw_pg check --kprimes 200,500,1000,2000 --queries 200 \
    --claim-workers 16 --claim-windows 200 --out "{run}/step1_report.json"
```

`check` exits with status 0 only if all five gates pass. The full report is
`step1_report.json` in the run directory runlog printed, and the load timings
and index sizes are in `step1_data/load_report.json` (copy it into the same run
directory; it describes the database the checks ran against).

If check 2 warns that overlap with the exact window is below 0.9, rebuild with
a denser graph and run the checks again:

```bash
python -m thaw_pg load --m 32 --ef-construction 128
python -m semview.runlog --name step1_check_m32 -- \
  python -m thaw_pg check --out "{run}/step1_report.json"
```

Record which `m` / `ef_construction` you settle on. Step 2 measures
cross-space recall on top of this index, so the paper must state it.

---

## 5. Reading the output

```
[2/5] window depth (and a preview of E2)
  K'=500   plain    ef=500   rows min/mean 500/500.0   overlap 0.675  p50 5.5 ms  p99 7.1 ms
  ...
[3-4/5] NULLed old vectors and dead index entries (K'=500)
  before_vacuum  plain    rows min/mean 353/380.0  nulled returned 0  ...
  before_vacuum  relaxed  rows min/mean 500/500.0  nulled returned 0  ...
...
  PASS  1 index_use
  PASS  2 depth (K' in [200, 500, 1000])
  PASS  3 null_skip
  PASS  4 dead_tuples
  PASS  5 claims
```

- **rows min/mean:** how many rows the window query returned. Below K′ means
  the index ran out of candidates.
- **overlap:** the fraction of the exact top-K′ that the HNSW window contains.
  This is ANN error on the old side, which the paper's exact-search numbers
  don't include. Report it in step 2.
- **plain / relaxed / strict:** `hnsw.iterative_scan` off, `relaxed_order`,
  `strict_order`. `ef_search` is set to min(K′, 1000).

---

## 6. What our test run found (synthetic, 50k rows)

We ran this code end to end against PostgreSQL 16 with pgvector 0.8.1. Four
findings already change the design. Confirm them on the real data. **The
verifying run on the real 500k load, whose report ships as
`results/step1/step1_report.json`, was on PostgreSQL 18.1 with pgvector
0.8.5** — cite that pair, and read `report.json`'s `server` block rather than
this line when a version matters:

1. **`hnsw.ef_search` caps at 1000.** With iterative scans off, a K′ = 2000
   query returns only 1000 rows. With `hnsw.iterative_scan = relaxed_order`
   it returns all 2000. **The DPR pair is feasible, but only with iterative
   scans on.**
2. **Dead index entries shrink windows.** After NULLing 30% of the rows in
   the sampled windows, plain scans returned 353–500 rows (mean 380) out of
   500 until VACUUM ran. With `relaxed_order` every window stayed full.
   **The service should run with `hnsw.iterative_scan = relaxed_order`.**
   Repairs write to `docs_new`, not `docs`, so they create no dead entries in
   the old index. Only document updates do.
3. **Unsorted claim inserts deadlock.** The first version of check 5 inserted
   claims in window order. Two workers with overlapping windows then waited on
   each other's keys, and Postgres aborted one with `deadlock detected`.
   Inserting in sorted id order fixed it: 0 double claims and 0 errors across
   16 workers. **The real coordinator must claim in sorted order.**
4. **The planner chose a sequential scan over the partial index** at 2,500
   post-switch rows. It can use the index (it does with `enable_seqscan` off),
   and the gate tests that capability. At small sizes a seq scan is the
   cheaper plan. Check what the planner chooses at the real size.

The NULL check passed: no NULLed row was ever returned.

---

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `could not resize shared memory segment` during `load` | raise `shm_size` in the compose file, or lower `--workers` |
| HNSW build is slow and warns that the graph no longer fits in `maintenance_work_mem` | raise `--maint-mem` (the server's limit also applies) |
| `unrecognized configuration parameter "hnsw.iterative_scan"` | pgvector is older than 0.8.0; upgrade |
| `prepare --source msmarco` re-encodes even though you have `cache/` | the cache key is model + prefix + the exact ordered ids; you need the same `--limit` (500000), seed and model as the 500k run |
| check 5 reports errors | the error text is in `step1_report.json` → `claims.errors` |
| want a clean slate | `load` drops and recreates every table it owns; `docker compose down -v` removes the whole database |
