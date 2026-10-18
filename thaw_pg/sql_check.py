"""The claim protocol's SQL, checked against a real PostgreSQL.

`D-07` wants the online path run against a live database with a real encoder.
This is the part of that which needs no encoder, no GPU and no Python driver:
the statements `PgClaimSet`, `online_io` and the schema rely on, executed by a
real server under real concurrent transactions.

It matters because those statements have only ever been exercised by a fake.
`semview/tests.py` models `INSERT ... ON CONFLICT (id) DO NOTHING RETURNING id`
as a dictionary insert under a Python lock, which is a *claim* that the SQL
behaves that way, not evidence of it. Atomicity across concurrent transactions,
what `RETURNING` gives each of several writers racing on one key, whether an
`ORDER BY` inside the insert actually prevents the deadlock it is there to
prevent, and whether `ON DELETE CASCADE` reaches `docs_new` -- a database
answers all four, and nothing else does.

It drives `psql` rather than psycopg on purpose: the driver is not installed
everywhere this needs to run, and the point here is the server's behaviour, not
the client's.

    python -m thaw_pg.sql_check --dsn "host=/var/run/postgresql port=5432 user=postgres"
    python -m thaw_pg.sql_check --dsn ... --out results/sql_check.json

**What this does NOT cover**, and `integration.py` does: the encoder, the
repair loop, latency, recall, process kill and restart, and -- when pgvector is
absent -- the vector type and its indexes. Each check says which it is.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

N_UNITS = 200


# --------------------------------------------------------------------- psql --
#: Every statement runs inside this schema, which the run creates and drops.
#: It is not an optimisation. The first version of this file ran
#: `DROP TABLE IF EXISTS claim_done, repair_claims, docs_new, docs CASCADE`
#: against whatever `--dsn` pointed at, unconditionally, while the README told
#: people to point `THAW_SQLCHECK_DSN` at "any scratch PostgreSQL" -- so one
#: wrong environment variable destroyed a real corpus and everything
#: referencing it. Nothing in `public` is touched now: the checks cannot name
#: a table outside their own schema, and the schema is refused if it already
#: exists, so a collision never drops someone else's work.
_SCHEMA = {"name": None}


def _search_path() -> str:
    """The run's schema, plus wherever pgvector's types live.

    `vector(8)` is an unqualified type name, so it resolves through
    `search_path` -- and setting the path to the run's own schema alone put
    the extension out of reach. Every check here happened to run on a server
    without pgvector, where the column type falls back to
    `double precision[]`, so the one configuration this would break under is
    the one that was never exercised: a server that HAS pgvector, which is
    every real deployment.
    """
    parts = [f'"{_SCHEMA["name"]}"']
    if _EMB.get("schema"):
        parts.append(f'"{_EMB["schema"]}"')
    return ", ".join(parts)


def psql(dsn: str, sql: str, timeout: float = 60.0) -> str:
    """Run one SQL string in the run's own schema; return stdout."""
    if _SCHEMA["name"]:
        sql = f'SET search_path TO {_search_path()}; {sql}'
    p = subprocess.run(
        ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-X", "-q", "-t", "-A", "-c", sql],
        capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"psql failed: {p.stderr.strip()[:500]}\nSQL: {sql[:200]}")
    return p.stdout


def psql_raw(dsn: str, sql: str, timeout: float = 60.0) -> str:
    """Run a statement OUTSIDE the run's schema: creating and dropping it."""
    was, _SCHEMA["name"] = _SCHEMA["name"], None
    try:
        return psql(dsn, sql, timeout)
    finally:
        _SCHEMA["name"] = was


#: The shape of a name this tool generates for itself. The sweep removes only
#: these, and `--schema` refuses to take one, so "ours" and "yours" cannot be
#: confused by a name that merely looks the part.
_AUTO_NAME = re.compile(r"^thaw_sqlcheck_\d+_\d+(_ext)?$")


def _sweep_age(name: str) -> int | None:
    r"""The unix time in one of this tool's generated schema names, or None.

    Factored out and returning rather than `continue`-ing so it can be tested
    without a database, because the `_ext` case has now been got wrong twice.

    One check builds a second schema named `<run>_ext` to prove an unqualified
    type resolves across schemas, and it creates the ownership marker there, so
    the sweep's query does return it. Two separate things then dropped it:

      1. `_AUTO_NAME` was anchored `...\d+$`, which `_ext` cannot match. Fixed
         by the optional group above -- and fixing only that is what the
         sixteenth-round changelog wrongly described as fixing the bug.
      2. The age parse was `int(name.rsplit("_", 1)[1])`, which on
         `..._4567890_ext` is `int("ext")`: ValueError, skipped, same outcome
         as before. The suffix has to come off before the timestamp is read.

    So a hard kill (SIGKILL, a container going away) still left a schema in
    someone's database with nothing looking for it, under a comment at the
    naming site saying it was covered.
    """
    base = name[:-4] if name.endswith("_ext") else name
    try:
        return int(base.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None

#: A schema name this tool will accept. Anything else is refused rather than
#: escaped: the name is interpolated into DDL, which cannot take a parameter,
#: so the only safe position is a whitelist. `--schema 'x"; DROP SCHEMA public
#: CASCADE; --'` was accepted before this.
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,48}$")

#: Dropped into every schema this tool creates, so the sweep can tell its own
#: litter from a schema that merely matches the naming pattern.
_MARKER = "_thaw_sqlcheck_owner_v1"


def _check_ident(name: str, is_custom: bool = False) -> str:
    if is_custom and _AUTO_NAME.match(name or ""):
        raise SystemExit(
            f"refusing the schema name {name!r}: it matches the pattern this "
            f"tool generates for its own schemas, which the stale sweep "
            f"removes after an hour. A name you pass must not be mistakable "
            f"for one this tool owns.")
    if not _IDENT.match(name or ""):
        raise SystemExit(
            f"refusing the schema name {name!r}: it must match "
            f"{_IDENT.pattern}. A schema name goes into DDL, which takes no "
            f"parameters, so this is a whitelist rather than an escape.")
    return name


def make_schema(dsn: str, name: str) -> None:
    """Create the run's own schema, refusing to reuse an existing one."""
    _check_ident(name)
    exists = rows(psql_raw(
        dsn, "SELECT 1 FROM information_schema.schemata "
             f"WHERE schema_name = '{name}'"))
    if exists:
        raise SystemExit(
            f"schema {name!r} already exists in this database. This run would "
            f"drop it at the end, and it is not this run's to drop. Pass a "
            f"different --schema.")
    psql_raw(dsn, f'CREATE SCHEMA "{name}"')
    _SCHEMA["name"] = name
    # the ownership marker: a schema without it is not ours to drop
    psql_raw(dsn, f'CREATE TABLE "{name}".{_MARKER} '
                  f'(created_at timestamptz NOT NULL DEFAULT now())')


def sweep_old_schemas(dsn: str, older_than_s: int = 3600) -> list[str]:
    """Drop this tool's own leftover schemas from runs that died.

    A `finally` covers the ordinary failures, and it does not cover every one:
    a run piped into `head` can take a BrokenPipeError at a moment where the
    cleanup does not get to finish, and then its schema sits in someone's
    database forever. The name prefix is unambiguous -- nothing but this file
    creates `thaw_sqlcheck_<pid>_<unix time>` -- so an old one is litter and
    safe to remove. Recent ones are left alone: another run may be using them.
    """
    cutoff = int(time.time()) - int(older_than_s)
    # Only schemas that carry this tool's marker table AND match the name it
    # generates. The name pattern alone is not ownership: a `--schema` can be
    # anything, and another tool or person on a shared database could use the
    # same convention. Two independent signals, and a `DROP ... CASCADE` only
    # where both hold.
    names = [n.strip() for n in rows(psql_raw(
        dsn, "SELECT n.nspname FROM pg_namespace n "
             f"JOIN pg_class c ON c.relnamespace = n.oid "
             f"WHERE c.relname = '{_MARKER}' AND c.relkind = 'r' "
             "AND n.nspname LIKE 'thaw\\_sqlcheck\\_%'"))]
    gone = []
    for n in names:
        if not _AUTO_NAME.match(n):
            continue                       # a custom --schema: not ours to judge
        when = _sweep_age(n)
        if when is None:
            continue
        if when < cutoff:
            psql_raw(dsn, f'DROP SCHEMA IF EXISTS "{n}" CASCADE')
            gone.append(n)
    return gone


def drop_schema(dsn: str) -> None:
    name = _SCHEMA["name"]
    if not name:
        return
    _SCHEMA["name"] = None
    psql_raw(dsn, f'DROP SCHEMA IF EXISTS "{name}" CASCADE')


def rows(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.strip()]


# -------------------------------------------------------------------- schema --
def has_pgvector(dsn: str) -> bool:
    """Is pgvector installed, and in which schema? This does not install it.

    It used to run `CREATE EXTENSION IF NOT EXISTS vector`, which changes
    database-wide state that the run does not own and cannot undo -- the one
    thing left reaching outside the isolated schema after that isolation was
    added. A check should not install anything. If pgvector is absent the
    columns fall back to `double precision[]` and the result says so.
    """
    got = rows(psql_raw(
        dsn, "SELECT n.nspname FROM pg_extension e "
             "JOIN pg_namespace n ON n.oid = e.extnamespace "
             "WHERE e.extname = 'vector'"))
    if got:
        _EMB["schema"] = got[0].strip()
        return True
    return False


def schema(dsn: str, vector_ok: bool, d: int = 8) -> None:
    """The real schema, with one substitution when pgvector is unavailable.

    `emb` becomes `double precision[]` rather than `vector(d)`. Nothing checked
    here touches the vector type -- the claim protocol, the lease and the
    cascade are all about rows and keys -- but the substitution is recorded in
    the result so that nobody reads this as covering the index.

    These tables are created inside the run's own schema, which did not exist
    a moment ago, so there is nothing here to drop and nothing of anyone
    else's to lose.
    """
    emb = f"vector({d})" if vector_ok else "double precision[]"
    psql(dsn, f"""
        CREATE TABLE docs (
            id bigint PRIMARY KEY, doc_id text NOT NULL, text text,
            emb_old {emb}, updated_at timestamptz NOT NULL DEFAULT now());
        CREATE TABLE docs_new (
            id bigint PRIMARY KEY REFERENCES docs (id) ON DELETE CASCADE,
            emb {emb} NOT NULL,
            origin text NOT NULL CHECK (origin IN ('repair', 'write')),
            repaired_at timestamptz NOT NULL DEFAULT now());
        CREATE TABLE repair_claims (
            id bigint PRIMARY KEY, owner text NOT NULL,
            claimed_at timestamptz NOT NULL DEFAULT now());
        CREATE TABLE claim_done (id bigint PRIMARY KEY, owner text NOT NULL);
        INSERT INTO docs (id, doc_id, text)
            SELECT g, 'd' || g, 'unit ' || g FROM generate_series(1, {N_UNITS}) g;
    """)


# -------------------------------------------------------------------- checks --
def check_claim_is_exclusive(dsn: str, workers: int = 8, units: int = 2_500,
                             batch: int = 16) -> dict:
    """Many sessions claiming the same units in small batches, as workers do.

    This is the guarantee every cost number in the paper rests on, and the
    fake models it with a dict under a lock -- an assertion that the SQL
    behaves this way rather than evidence of it.

    Getting real contention took two tries. Offering one big set per session
    does not contend at all: the first `INSERT ... ON CONFLICT` commits before
    the others have finished starting, so they find every row present, win
    nothing, and exclusivity holds vacuously. 40,000 ids did not help -- the
    statement is atomic, so a bigger one is still one commit. What contends is
    the real access pattern: each session claims in batches of 16, over and
    over, for as long as there is work. Then they are inside the server
    together and the race is real.

    Each session logs what `RETURNING` gave it into `claim_log`, in the same
    statement, so the evidence is the database's rather than the client's:
    no id may appear twice in that log however many sessions ran.
    """
    psql(dsn, """
        TRUNCATE repair_claims;
        DROP TABLE IF EXISTS claim_log;
        CREATE TABLE claim_log (seq bigserial PRIMARY KEY, id bigint NOT NULL,
                                owner text NOT NULL);
    """)
    tmpdir = tempfile.TemporaryDirectory()   # removed on exit; `mkdtemp` left
    tmp = Path(tmpdir.name)                  # eight .sql files behind each run
    scripts = []
    for w in range(workers):
        # Each worker gets a different shuffle, so the BATCHES differ: worker 0's
        # third batch is a different random subset of ids from worker 1's third.
        # That is what makes them contend. It is not that they take locks from
        # opposite ends and meet in the middle -- the `ORDER BY x` inside each
        # INSERT sorts the batch, so every statement takes its own locks in
        # ascending id order, which is the deadlock-free discipline the claim
        # protocol requires and this check is partly here to confirm.
        order = list(range(1, units + 1))
        rng = __import__("random").Random(w)
        rng.shuffle(order)
        stmts = []
        for lo in range(0, units, batch):
            arr = ",".join(str(i) for i in order[lo:lo + batch])
            stmts.append(
                f"WITH ins AS (INSERT INTO repair_claims (id, owner) "
                f"SELECT x, 'w{w}' FROM unnest(ARRAY[{arr}]::bigint[]) AS x "
                f"ORDER BY x ON CONFLICT (id) DO NOTHING RETURNING id) "
                f"INSERT INTO claim_log (id, owner) SELECT id, 'w{w}' FROM ins;")
        f = tmp / f"w{w}.sql"
        # a script runs through `psql -f`, which does not go through psql()
        # above, so it carries the search_path itself
        head = f'SET search_path TO "{_SCHEMA["name"]}";\n' if _SCHEMA["name"] else ""
        f.write_text(head + "\n".join(stmts) + "\n", encoding="utf-8")
        scripts.append(f)

    def run(f):
        p = subprocess.run(["psql", dsn, "-X", "-q", "-v", "ON_ERROR_STOP=1",
                            "-f", str(f)], capture_output=True, text=True,
                           timeout=300)
        return p.returncode, p.stderr.strip()[:200]

    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(run, scripts))
    finally:
        tmpdir.cleanup()
    wall = time.perf_counter() - t0

    errs = [e for rc, e in res if rc != 0]
    logged = int(rows(psql(dsn, "SELECT count(*) FROM claim_log"))[0])
    distinct = int(rows(psql(dsn, "SELECT count(DISTINCT id) FROM claim_log"))[0])
    held = int(rows(psql(dsn, "SELECT count(*) FROM repair_claims"))[0])
    by_owner = {r.split("|")[0]: int(r.split("|")[1]) for r in rows(psql(
        dsn, "SELECT owner, count(*) FROM claim_log GROUP BY owner ORDER BY owner"))}
    winners = sum(1 for v in by_owner.values() if v)
    return {
        "check": "claim is exclusive across concurrent sessions",
        "workers": workers, "units": units, "batch": batch,
        "wall_s": round(wall, 2), "errors": errs[:2],
        "logged": logged, "distinct_ids": distinct, "rows_held": held,
        "won_by_owner": by_owner, "winners": winners,
        "contended": winners > 1,
        "pass": (not errs and logged == distinct == units == held
                 and winners > 1),
        "covers": "real transactions, real concurrency, real row locks",
        "note": ("every unit was returned to exactly one session and "
                 f"{winners} of {workers} sessions won some, so the race was "
                 "real" if winners > 1 else
                 "NOT contended: the sessions did not overlap"),
    }


def check_no_deadlock_on_opposite_orders(dsn: str, rounds: int = 20) -> dict:
    """Two sessions claiming overlapping sets, offered in opposite orders.

    `PgClaimSet.claim` sorts its ids and the SQL carries `ORDER BY x` for this
    reason: two writers taking the same keys in opposite orders deadlock, and
    PostgreSQL resolves that by killing one with a 40P01. The fake cannot show
    the ordering matters because Python dict insertion never deadlocks. If the
    `ORDER BY` were dropped this check is what would notice.
    """
    psql(dsn, "TRUNCATE repair_claims")
    lo = ",".join(str(i) for i in range(1, 60))
    hi = ",".join(str(i) for i in range(59, 0, -1))
    errors = []

    def run(which):
        arr = lo if which == 0 else hi
        for r in range(rounds):
            try:
                psql(dsn, f"BEGIN; INSERT INTO repair_claims (id, owner) "
                          f"SELECT x, 'w{which}' FROM unnest(ARRAY[{arr}]::bigint[]) "
                          f"AS x ORDER BY x ON CONFLICT (id) DO NOTHING; "
                          f"DELETE FROM repair_claims WHERE owner = 'w{which}'; "
                          f"COMMIT;")
            except RuntimeError as e:
                errors.append(str(e)[:200])
                return

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, range(2)))
    deadlocks = [e for e in errors if "deadlock" in e.lower() or "40P01" in e]
    return {
        "check": "claiming in key order does not deadlock",
        "rounds": rounds, "errors": errors[:3], "deadlocks": len(deadlocks),
        "pass": not errors,
        "covers": "real lock waits and deadlock detection",
    }


def check_lease_reclaims_only_the_dead(dsn: str, lease_s: int = 300) -> dict:
    """The sweep must take a dead owner's claims and leave everything else.

    Three cases in one table: a claim older than the lease whose unit has no
    vector (reclaim), one older than the lease whose unit was repaired after
    all (leave -- it is done, not lost), and a fresh claim (leave).
    """
    psql(dsn, "TRUNCATE repair_claims; DELETE FROM docs_new;")
    psql(dsn, f"""
        INSERT INTO repair_claims (id, owner, claimed_at)
            SELECT g, 'crashed', now() - interval '1 hour'
            FROM generate_series(1, 10) g;
        INSERT INTO repair_claims (id, owner, claimed_at)
            SELECT g, 'finished-late', now() - interval '1 hour'
            FROM generate_series(11, 15) g;
        INSERT INTO repair_claims (id, owner)
            SELECT g, 'alive' FROM generate_series(16, 25) g;
        INSERT INTO docs_new (id, emb, origin)
            SELECT g, {_zero_emb(dsn)}, 'repair' FROM generate_series(11, 15) g;
    """)
    got = rows(psql(dsn, f"""
        DELETE FROM repair_claims c
        WHERE c.claimed_at < now() - make_interval(secs => {lease_s})
          AND NOT EXISTS (SELECT 1 FROM docs_new d WHERE d.id = c.id)
        RETURNING id"""))
    left = {r.split("|")[0]: r for r in rows(psql(
        dsn, "SELECT owner, count(*) FROM repair_claims GROUP BY owner"))}
    reclaimed = sorted(int(i) for i in got)
    return {
        "check": "the lease reclaims a dead owner's claims and nothing else",
        "reclaimed": reclaimed,
        "remaining_by_owner": left,
        "pass": (reclaimed == list(range(1, 11))
                 and left.get("alive", "").endswith("|10")
                 and left.get("finished-late", "").endswith("|5")
                 and "crashed" not in left),
        "covers": "a real sweep over a real table",
    }


def check_staged_write_is_idempotent(dsn: str) -> dict:
    """`online_io`'s writer: stage, then insert ON CONFLICT DO NOTHING.

    Two writers may legitimately hold the same unit after a reclaimed lease.
    A bare `COPY` into `docs_new` raises a duplicate-key error inside a serving
    request; this must not.
    """
    psql(dsn, "DELETE FROM docs_new")
    emb = _zero_emb(dsn)
    stage = f"""
        CREATE TEMP TABLE IF NOT EXISTS stage_new
            (id bigint PRIMARY KEY, emb {_emb_type(dsn)}, origin text);
        TRUNCATE stage_new;
        INSERT INTO stage_new SELECT g, {emb}, 'repair'
            FROM generate_series(30, 40) g;
        INSERT INTO docs_new (id, emb, origin)
            SELECT id, emb, origin FROM stage_new
            ON CONFLICT (id) DO NOTHING;
    """
    err = None
    try:
        for _ in range(3):                       # the same batch, three times
            psql(dsn, stage)
    except RuntimeError as e:
        err = str(e)[:300]
    n = int(rows(psql(dsn, "SELECT count(*) FROM docs_new "
                           "WHERE id BETWEEN 30 AND 40"))[0])
    return {
        "check": "a repeated staged write is a no-op, not a duplicate-key error",
        "rows": n, "error": err, "pass": err is None and n == 11,
        "covers": "the real primary key and the real conflict clause",
    }


def check_delete_cascades(dsn: str) -> dict:
    """A document deleted from `docs` must leave `docs_new` too, or the online
    path would keep answering with a vector whose document is gone."""
    psql(dsn, "DELETE FROM docs_new")
    psql(dsn, f"INSERT INTO docs_new (id, emb, origin) "
              f"SELECT g, {_zero_emb(dsn)}, 'repair' FROM generate_series(50, 55) g")
    before = int(rows(psql(dsn, "SELECT count(*) FROM docs_new"))[0])
    psql(dsn, "DELETE FROM docs WHERE id = 52")
    after = int(rows(psql(dsn, "SELECT count(*) FROM docs_new"))[0])
    gone = rows(psql(dsn, "SELECT count(*) FROM docs_new WHERE id = 52"))[0]
    # Does a leftover claim row for a deleted unit block anything? Asserted
    # rather than asserted-in-a-comment: claim the unit, delete the document,
    # and check the orphan claim neither disappears by cascade (there is no
    # foreign key, so it must not) nor stops a fresh claim from being taken.
    # This line used to be a bare TRUNCATE under a comment that promised the
    # check, which is the same shape as a test that cannot fail.
    psql(dsn, "INSERT INTO repair_claims (id, owner) VALUES (53, 'w0') "
              "ON CONFLICT (id) DO NOTHING")
    psql(dsn, "DELETE FROM docs WHERE id = 53")
    orphan = int(rows(psql(dsn,
        "SELECT count(*) FROM repair_claims WHERE id = 53"))[0])
    reclaimed = rows(psql(dsn,
        "WITH ins AS (INSERT INTO repair_claims (id, owner) VALUES (53, 'w1') "
        "ON CONFLICT (id) DO NOTHING RETURNING id) "
        "SELECT count(*) FROM ins"))[0]
    psql(dsn, "TRUNCATE repair_claims")
    return {
        "check": "deleting a document removes its new vector by cascade",
        "before": before, "after": after, "row_52": int(gone),
        # the orphan claim must survive (no foreign key) and must not block a
        # fresh claim on the same id
        "orphan_claim_rows": orphan,
        "reclaimed_after_delete": int(reclaimed),
        "pass": (before == 6 and after == 5 and int(gone) == 0
                 and orphan == 1 and int(reclaimed) == 0),
        "covers": "the real foreign key, and a claim row outliving its document",
    }


def check_claim_survives_a_killed_backend(dsn: str) -> dict:
    """A backend killed mid-transaction must not leave a half-claim.

    This is the database half of "a process died holding claims": the claim
    row is either there (committed) or not (rolled back), never partial, and
    the lease is what recovers the committed case. `pg_terminate_backend` is
    the closest a SQL-level check gets to `kill -9`; the process-level case is
    `integration.py`.
    """
    psql(dsn, "TRUNCATE repair_claims")
    ids = ",".join(str(i) for i in range(100, 120))
    sp = f'SET search_path TO "{_SCHEMA["name"]}"; ' if _SCHEMA["name"] else ""
    out = subprocess.Popen(
        ["psql", dsn, "-X", "-q", "-t", "-A", "-c",
         sp + f"BEGIN; INSERT INTO repair_claims (id, owner) "
         f"SELECT x, 'doomed' FROM unnest(ARRAY[{ids}]::bigint[]) AS x "
         f"ORDER BY x ON CONFLICT (id) DO NOTHING; "
         f"SELECT pg_sleep(10); COMMIT;"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # poll for the backend rather than sleeping a guessed interval: on a
    # loaded machine it may not be in pg_stat_activity within any fixed time,
    # and the check would then fail for a reason that is not the subject
    deadline = time.time() + 20.0
    killed = []
    while time.time() < deadline:
        killed = rows(psql(
            dsn, "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                 "WHERE query LIKE '%doomed%' AND pid <> pg_backend_pid()"))
        if killed:
            break
        time.sleep(0.1)
    if not killed:
        out.kill()                      # do not wait out the sleep on failure
    out.wait(timeout=20)
    held = int(rows(psql(dsn, "SELECT count(*) FROM repair_claims "
                              "WHERE owner = 'doomed'"))[0])
    # the uncommitted claim is gone, so another worker can take those units
    retaken = len(rows(psql(
        dsn, f"INSERT INTO repair_claims (id, owner) "
             f"SELECT x, 'next' FROM unnest(ARRAY[{ids}]::bigint[]) AS x "
             f"ORDER BY x ON CONFLICT (id) DO NOTHING RETURNING id")))
    return {
        "check": "a killed backend leaves no half-claim",
        "terminated": len(killed), "rows_from_doomed": held, "retaken": retaken,
        "pass": held == 0 and retaken == 20,
        "covers": "a real backend dying inside a real transaction",
    }


# ------------------------------------------------------------------ helpers --
_EMB = {}


def _emb_type(dsn: str) -> str:
    return "vector(8)" if _EMB.get("vector") else "double precision[]"


def _zero_emb(dsn: str) -> str:
    return ("'[0,0,0,0,0,0,0,0]'::vector(8)" if _EMB.get("vector")
            else "ARRAY[0,0,0,0,0,0,0,0]::double precision[]")


def check_an_unqualified_type_from_another_schema_resolves(dsn: str) -> dict:
    """The mechanism the pgvector column depends on.

    `vector(8)` is unqualified, so it resolves through `search_path` -- and
    setting the path to the run's own schema alone put the extension's types
    out of reach. The bug could not show itself here because every run so far
    was on a server WITHOUT pgvector, where the type falls back to
    `double precision[]`: the only configuration that breaks is the one that
    was never exercised, which is every real deployment.

    pgvector cannot be built in this container, so this proves the mechanism
    rather than the extension: a type defined in a DIFFERENT schema, used
    unqualified from the run's schema. If `_search_path` stops including the
    extension's schema, the create fails here exactly as `vector(8)` would.
    """
    # named so the stale sweep can collect it too: `<run>_ext` alone matches
    # neither the generated-name pattern nor the marker, so a failed cleanup
    # would leave it in the database with nothing looking for it
    ext = f'{_SCHEMA["name"]}_ext'
    psql_raw(dsn, f'DROP SCHEMA IF EXISTS "{ext}" CASCADE')
    psql_raw(dsn, f'CREATE SCHEMA "{ext}"')
    psql_raw(dsn, f'CREATE DOMAIN "{ext}".standin AS double precision[]')
    psql_raw(dsn, f'CREATE TABLE "{ext}".{_MARKER} (created_at timestamptz)')
    # and prove the domain is really there, so a failure below means "the
    # unqualified name did not resolve" and not "the schema was never made"
    made_domain = rows(psql_raw(
        dsn, "SELECT 1 FROM information_schema.domains "
             f"WHERE domain_schema = '{ext}' AND domain_name = 'standin'"))
    before = _EMB.get("schema")
    try:
        # without the extension's schema on the path, as the first version did
        _EMB["schema"] = None
        failed_without = False
        try:
            psql(dsn, "CREATE TABLE probe_a (id bigint, v standin)")
        except RuntimeError:
            failed_without = True
        # and with it, as the fix does
        _EMB["schema"] = ext
        psql(dsn, "CREATE TABLE probe_b (id bigint, v standin)")
        made = int(rows(psql(
            dsn, "SELECT count(*) FROM information_schema.tables "
                 f"WHERE table_schema = '{_SCHEMA['name']}' "
                 "AND table_name = 'probe_b'"))[0])
    finally:
        _EMB["schema"] = before
        psql_raw(dsn, f'DROP SCHEMA IF EXISTS "{ext}" CASCADE')
    return {
        "check": "an unqualified type from another schema resolves",
        "failed_without_the_extension_schema": failed_without,
        "the_type_really_existed": bool(made_domain),
        "created_with_it": made,
        "pass": bool(made_domain) and failed_without and made == 1,
        "covers": "how pgvector's type is found; NOT pgvector itself",
        "note": ("the search path reaches the extension's schema; without it "
                 "the create fails, which is what a pgvector server would "
                 "have hit" if failed_without and made == 1 else
                 "the probe did not behave as expected -- read it before "
                 "trusting the pgvector branch"),
    }


CHECKS = (check_an_unqualified_type_from_another_schema_resolves,
          check_claim_is_exclusive, check_no_deadlock_on_opposite_orders,
          check_lease_reclaims_only_the_dead, check_staged_write_is_idempotent,
          check_delete_cascades, check_claim_survives_a_killed_backend)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="thaw_pg.sql_check", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", required=True,
                    help="anything psql accepts, e.g. "
                         "\"host=/var/run/postgresql port=5432 user=postgres\"")
    ap.add_argument("--out", default=None, help="write the results here as JSON")
    ap.add_argument("--schema", default=None,
                    help="the schema to create and drop for this run "
                         "(default: thaw_sqlcheck_<pid>_<time>). Lowercase "
                         "letters, digits and underscores only, at most 49 "
                         "characters; it must not already exist, and may not "
                         "look like a name this tool generates. Nothing "
                         "outside it is touched.")
    a = ap.parse_args(argv)

    server = psql_raw(a.dsn, "SELECT version()").strip()
    stale = sweep_old_schemas(a.dsn)
    if stale:
        print(f"removed {len(stale)} leftover schema(s) from runs that died: "
              f"{', '.join(stale[:3])}")
    name = (_check_ident(a.schema, is_custom=True) if a.schema
            else _check_ident(f"thaw_sqlcheck_{os.getpid()}_{int(time.time())}"))
    make_schema(a.dsn, name)
    print(f"schema: {name} (created for this run, dropped at the end; "
          f"nothing outside it is touched)")
    _EMB["vector"] = has_pgvector(a.dsn)   # detects; never installs
    print(f"{server}")
    where = ("present" if _EMB["vector"]
             else "ABSENT -- emb columns are double precision[]")
    print(f"pgvector: {where}")
    out, bad = [], 0
    try:
        schema(a.dsn, _EMB["vector"])
        for fn in CHECKS:
            t0 = time.perf_counter()
            try:
                r = fn(a.dsn)
            except Exception as e:                                 # noqa: BLE001
                r = {"check": fn.__name__, "pass": False,
                     "error": f"{type(e).__name__}: {e}"[:400]}
            r["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            out.append(r)
            bad += not r["pass"]
            print(f"  {'ok  ' if r['pass'] else 'FAIL'}  {r['check']}")
            for k, v in r.items():
                if k not in ("check", "pass", "ms", "covers"):
                    print(f"          {k}: {v}")
    finally:
        # Always, including on a crash, a KeyboardInterrupt or a broken pipe
        # -- a run piped through `head` died before this and left its schema
        # behind, which is litter in someone else's database and exactly the
        # kind of mess the schema was introduced to avoid.
        try:
            drop_schema(a.dsn)
        except Exception as e:                                     # noqa: BLE001
            print(f"WARNING: could not drop {name!r}: {e}", file=sys.stderr)
    print(f"\n{len(out) - bad} of {len(out)} checks passed against a real server")
    if not _EMB["vector"]:
        print("pgvector was absent: the vector type and its indexes are NOT "
              "covered here. Nothing above depends on them -- claims, leases, "
              "the cascade and the staged write are about rows and keys -- but "
              "the index behaviour is still only covered by `checks.py`.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"tool": "sql_check", "server": server,
             "pgvector": _EMB["vector"], "checks": out,
             "passed": len(out) - bad, "total": len(out)},
            indent=1) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
