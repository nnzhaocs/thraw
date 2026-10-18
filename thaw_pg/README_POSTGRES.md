# Restarting Postgres with the right settings

> The full run order is in `RUNTESTS.md`; this file explains one part of it.

Step 1 ran with Postgres defaults:

```
shared_buffers 256MB   maintenance_work_mem 512MB   max_parallel_maintenance_workers 4
```

The old-side HNSW index alone is 977 MB and the `docs` heap another 900 MB, so
most window searches read pages from the OS cache instead of Postgres's own
buffer pool. Recall and correctness don't depend on this, so the step-1 gates
and overlap numbers stand. **Latency does**, so set the values below before
any latency measurement (step 1's latency table, all of step 2).

Run every command on the machine where Postgres runs.

---

## 1. Check what you have now

From the bundle root, with `THAW_PG_DSN` set as for step 1:

```bash
python -m thaw_pg settings
```

```
Postgres 18.1, pgvector 0.8.5
  shared_buffers                          256 MB   (configuration file)
  maintenance_work_mem                    512 MB   (configuration file)
  ...
```

The column in brackets says where each value comes from. If it says
**`command line`**, Postgres was started with `-c` flags (Docker `command:`
lines do this) and those flags override everything in section 3. Change them
where they're set instead (section 4a).

---

## 2. Pick values for your RAM

Size these for the machine's **total RAM**, minus what else runs on it. With
TEI and the step-2 runner on the same laptop, leave them 4–6 GB.

| setting | 16 GB machine | 32 GB | 64 GB | why |
|---|---|---|---|---|
| `shared_buffers` | 4GB | 8GB | 16GB | holds the HNSW index (977 MB), the heap (900 MB) and `docs_new` (~1.5 GB once repaired) in Postgres's own cache |
| `effective_cache_size` | 10GB | 22GB | 44GB | planner hint only (no memory is allocated): about ¾ of RAM |
| `maintenance_work_mem` | 2GB | 4GB | 8GB | HNSW builds. Only matters if you rebuild the index (e.g. `--m 32`) |
| `max_parallel_maintenance_workers` | cores − 1, at most 7 | same | same | parallel HNSW build |
| `work_mem` | 32MB | 64MB | 64MB | per sort; the exact-window mode sorts 500k distances |
| `jit` | off | off | off | JIT compilation can add tens of ms to the exact window's sequential scan, which would distort exact-vs-HNSW latency |
| `max_connections` | 200 | 200 | 200 | the claim check opens 16, step 2 opens 2 per thread |

Don't change `random_page_cost` or other planner costs. Step 1 confirmed the
planner already picks the HNSW index at 500k, and the paper should report
default planner behaviour.

---

## 3. Set them (any installation)

`ALTER SYSTEM` writes `postgresql.auto.conf` in the data directory, so it
works the same whether Postgres runs natively or in Docker. Example for a
**16 GB laptop** (substitute your row from section 2):

```bash
psql "$THAW_PG_DSN" <<'SQL'
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '10GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET max_parallel_maintenance_workers = 7;
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET jit = off;
ALTER SYSTEM SET max_connections = 200;
SELECT pg_reload_conf();
SQL
python -m thaw_pg settings     # shared_buffers / max_connections show RESTART PENDING
```

`shared_buffers` and `max_connections` only take effect after a **restart**.
The others apply on reload.

(No `psql` installed? Use `docker exec -it <container> psql -U postgres -d thaw`
and paste the same statements.)

---

## 4. Restart

Use whichever matches how Postgres runs. To find out: `docker ps` lists a
`pgvector/pgvector` or `postgres` container if it's Docker. Otherwise it's a
native install.

### 4a. Docker

```bash
docker ps --format '{{.Names}}  {{.Image}}'      # find the container name
docker restart CONTAINER
```

- **If you started it with `thaw_pg/docker-compose.yml`:** its `command:`
  section passes `-c` flags, which override `ALTER SYSTEM`. Edit the values
  there instead (section 2's table), then
  `docker compose -f thaw_pg/docker-compose.yml up -d`, which recreates the
  container. The data volume is kept.
- **If you started it with plain `docker run`:** there is no `command:` to
  edit, so `ALTER SYSTEM` + `docker restart` is all you need.
- **Docker Desktop (Windows / macOS):** the Linux VM has a memory cap
  (Settings → Resources, or `.wslconfig` on WSL 2). `shared_buffers` must fit
  inside it with room to spare, or Postgres fails to start.
- **Shared memory:** a container needs `--shm-size=4g` or more (or
  `shm_size: 8g` in compose) for parallel index builds and parallel queries.
  `docker inspect -f '{{.HostConfig.ShmSize}}' <container>` shows the current
  value in bytes; the 64 MB default is too small. It can only be set when the
  container is created.

### 4b. Linux, native (apt / dnf)

```bash
sudo systemctl restart postgresql            # Debian/Ubuntu also: postgresql@18-main
systemctl status postgresql --no-pager | head -5
```

### 4c. macOS, Homebrew

```bash
brew services restart postgresql@18
```

### 4d. Windows, native (EDB installer)

In an **Administrator** PowerShell:

```powershell
Get-Service *postgres*                      # find the name, e.g. postgresql-x64-18
Restart-Service postgresql-x64-18
```

Or Services (`services.msc`) → postgresql-x64-18 → Restart.

### 4e. Anything else

```bash
pg_ctl restart -D "$PGDATA" -w
# the data directory: psql "$THAW_PG_DSN" -c "SHOW data_directory"
```

---

## 5. Verify

```bash
python -m thaw_pg settings
```

Every value should match section 2, with no `RESTART PENDING`.

**If Postgres doesn't come back up**, `shared_buffers` is too large for the
machine or container. Check the server log, then:

```bash
# Docker:  docker logs <container> | tail -20
# Linux:   journalctl -u postgresql -n 30
# Windows: <data directory>\log\  (newest file)
```

To undo, stop Postgres, delete the bad line from
`<data directory>/postgresql.auto.conf`, and start it again. Or set a smaller
value with `ALTER SYSTEM` once it's up.

---

## 6. Warm the cache before measuring

After a restart, Postgres's buffer pool is empty, so the first queries read
from disk and inflate p99. Load the tables once:

```bash
psql "$THAW_PG_DSN" <<'SQL'
CREATE EXTENSION IF NOT EXISTS pg_prewarm;
SELECT pg_prewarm('docs_emb_old_hnsw'), pg_prewarm('docs'),
       pg_prewarm('docs_pkey'), pg_prewarm('docs_new_pkey');
SQL
```

The paper should say latencies were measured with a warm cache.

On a **laptop**, also plug it in, set the power mode to best performance, and
close other heavy applications. Battery power-saving changes CPU clocks
between runs and shows up directly in p99.

---

## 7. Then

1. Re-run step 1's checks to get a clean latency table. Data and indexes are
   already loaded, so this takes a few minutes and needs no `load`:
   ```bash
   python -m semview.runlog --name step1_check_tuned --tag final -- \
     python -m thaw_pg check --kprimes 200,500,1000,2000 --queries 200 \
     --out "{run}/step1_report.json"
   ```
   The overlap numbers should be identical to the first run (they don't depend
   on memory). The p50/p99 columns are the ones to keep.
2. Start step 2 (`README_STEP2.md`), with `--chunk 16` on the RTX 4060 Laptop
   GPU.

We tested the procedure here on PostgreSQL 16 with pgvector 0.8.1:
`ALTER SYSTEM`, reload, `settings` showing the pending restart, restart, the
new values taking effect, and `pg_prewarm` on the HNSW index and heap.

The shipped `results/step1/step1_report.json` was produced on **PostgreSQL 18.1
with pgvector 0.8.5**, which is the pair the paper quotes, and the step-2 runs
were made on the same server. The settings procedure above is the same on both;
the version differs because the development box and the reporting box were not
the same machine. Where a number depends on the version — the
`hnsw.ef_search` cap, iterative-scan behaviour — the report's `server` block is
the authority, not any prose here.
