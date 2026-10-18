"""Connection handling. One place for the DSN and the session settings."""
from __future__ import annotations

import os

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/thaw"


def dsn(arg: str | None = None) -> str:
    return arg or os.environ.get("THAW_PG_DSN", DEFAULT_DSN)


def connect(d: str | None = None, autocommit: bool = True,
            ensure_extension: bool = False):
    """Open a connection and register the vector type.

    `ensure_extension` runs `CREATE EXTENSION IF NOT EXISTS vector`, and it is
    off by default. It used to run on EVERY connect, which put a DDL statement
    on the hot path of a service -- the heartbeat opens a connection per batch
    -- and meant every component needed CREATE privileges at run time for
    something that belongs to deployment. `load` turns it on, because that is
    where the schema is created; nothing else needs to.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn(d), autocommit=autocommit)
    if ensure_extension:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def server_info(conn) -> dict:
    row = conn.execute(
        "SELECT current_setting('server_version'), "
        "(SELECT extversion FROM pg_extension WHERE extname = 'vector'), "
        "current_setting('shared_buffers'), current_setting('maintenance_work_mem'), "
        "current_setting('max_parallel_maintenance_workers')").fetchone()
    return dict(postgres=row[0], pgvector=row[1], shared_buffers=row[2],
                maintenance_work_mem=row[3], max_parallel_maintenance_workers=row[4])


SETTINGS = ("shared_buffers", "effective_cache_size", "maintenance_work_mem",
            "work_mem", "max_parallel_maintenance_workers", "max_parallel_workers",
            "max_worker_processes", "max_connections", "jit")


def show_settings(conn) -> list[tuple]:
    """Current value, where it came from, and whether a restart is pending."""
    rows = conn.execute(
        "SELECT name, setting, unit, source, pending_restart FROM pg_settings "
        "WHERE name = ANY(%s) ORDER BY name", (list(SETTINGS),)).fetchall()
    return rows
