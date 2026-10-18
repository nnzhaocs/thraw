"""python -m thaw_pg SUBCOMMAND -- run from the bundle root; --help lists them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="thaw_pg")
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN (default $THAW_PG_DSN or "
                         "postgresql://postgres:postgres@localhost:5432/thaw)")
    ap.add_argument("--data-dir", default="step1_data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="write vectors.npz (and texts.jsonl)")
    p.add_argument("--source", choices=["msmarco", "synthetic"], required=True)
    p.add_argument("--beir-path", default="beir/")
    p.add_argument("--limit", type=int, default=500_000)
    p.add_argument("--max-queries", type=int, default=5000)
    p.add_argument("--old", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--cache-dir", default="cache/")
    p.add_argument("--device", default=None)
    p.add_argument("--n", type=int, default=100_000, help="synthetic: corpus size")
    p.add_argument("--dim", type=int, default=384, help="synthetic: dimension")
    p.add_argument("--n-queries", type=int, default=1000, help="synthetic")

    l = sub.add_parser("load", help="create tables, COPY, build indexes")
    l.add_argument("--m", type=int, default=16)
    l.add_argument("--ef-construction", type=int, default=64)
    l.add_argument("--workers", type=int, default=7,
                   help="max_parallel_maintenance_workers for the HNSW build")
    l.add_argument("--maint-mem", default="8GB",
                   help="maintenance_work_mem for the HNSW build")

    c = sub.add_parser("check", help="run the five checks")
    c.add_argument("--kprimes", default="200,500,1000,2000")
    c.add_argument("--queries", type=int, default=200)
    c.add_argument("--max-scan-tuples", type=int, default=100_000)
    c.add_argument("--update-frac", type=float, default=0.3)
    c.add_argument("--claim-windows", type=int, default=200)
    c.add_argument("--claim-workers", type=int, default=16)
    c.add_argument("--encode-ms", type=float, default=0.5,
                   help="simulated encode cost per unit in the claim check")
    c.add_argument("--out", default=None)

    q = sub.add_parser("quick", help="synthetic prepare + load + check, small")
    q.add_argument("--n", type=int, default=50_000)
    q.add_argument("--force", action="store_true",
                   help="allow it to replace real data (it drops the tables)")

    # ---- step 2
    p2 = sub.add_parser("prepare2", help="step 2: ground truth (new-model top-10)")
    p2.add_argument("--beir-path", default="beir/")
    p2.add_argument("--limit", type=int, default=500_000)
    p2.add_argument("--max-queries", type=int, default=5000)
    p2.add_argument("--old", default="sentence-transformers/all-MiniLM-L6-v2")
    p2.add_argument("--new", default="BAAI/bge-base-en-v1.5")
    p2.add_argument("--cache-dir", default="cache/")
    p2.add_argument("--device", default=None)

    v2 = sub.add_parser("verify-encoder",
                        help="step 2: the system encoder must match the ground truth")
    v2.add_argument("--enc-new", required=True,
                    help="tei:http://gpu-host:8080 or local:BAAI/bge-base-en-v1.5")
    v2.add_argument("--enc-old", default=None,
                    help="tei:http://gpu-host:8081 or local:sentence-transformers/all-MiniLM-L6-v2")
    v2.add_argument("--old", default="sentence-transformers/all-MiniLM-L6-v2")
    v2.add_argument("--chunk", type=int, default=16)
    v2.add_argument("--device", default=None)

    s2 = sub.add_parser("step2", help="step 2: replay queries through the query path")
    s2.add_argument("--enc-new", default=None,
                    help="new-model encoder: tei:<url> or local:<model>")
    s2.add_argument("--enc-old", default=None,
                    help="old-model QUERY encoder (only without --precomputed)")
    s2.add_argument("--old", default="sentence-transformers/all-MiniLM-L6-v2")
    s2.add_argument("--kprimes", default="200,500,1000")
    s2.add_argument("--windows", default="exact,hnsw")
    s2.add_argument("--queries", type=int, default=500)
    s2.add_argument("--chunk", type=int, default=16,
                    help="repair chunk = repair_chunk from measure_throughput on this GPU")
    s2.add_argument("--threads", type=int, default=1)
    s2.add_argument("--iterative", default="relaxed_order",
                    choices=["off", "relaxed_order", "strict_order"])
    s2.add_argument("--max-scan-tuples", type=int, default=100_000)
    s2.add_argument("--warm", action="store_true",
                    help="keep docs_new between runs (recall only; latency is "
                         "then not a migration's)")
    s2.add_argument("--precomputed", action="store_true",
                    help="use stored query vectors: takes the query encoders out "
                         "of E1, so a mismatch is the database path's alone")
    s2.add_argument("--union", default="0",
                    help="comma-separated K_n values; 0 = base design. e.g. 0,100 "
                         "runs each configuration without and with the union")
    s2.add_argument("--device", default=None)
    s2.add_argument("--out", default=None)

    st = sub.add_parser("selftest2", help="step 2 machinery on synthetic data, no model")
    st.add_argument("--force", action="store_true",
                    help="allow it to run against real data (it overwrites docs.text)")

    b = sub.add_parser("backfill-rate",
                       help="repair vs backfill encode cost, matched (the penalty)")
    b.add_argument("--enc", required=True,
                   help="local:<model>, local-fp16:<model> or tei:<url> -- the "
                        "same encoder step 2 used")
    b.add_argument("--kprime", type=int, default=500)
    b.add_argument("--queries", type=int, default=None,
                   help="queries in the repair stream (default: all in step1_data)")
    b.add_argument("--chunk", type=int, default=16, help="repair chunk, as in step 2")
    b.add_argument("--backfill-batch", type=int, default=16,
                   help="backfill batch; rerun with 32/64/... to find the "
                        "backfill's own best, and compare against that")
    b.add_argument("--sort-window", type=int, default=10_000,
                   help="backfill sorts each run of this many units by length "
                        "before batching (0 = no sorting)")
    b.add_argument("--no-token-sort", action="store_true",
                   help="sort the backfill by character count instead of token "
                        "count. The old behaviour, kept only to reproduce the "
                        "pre-fix measurement: padding is measured on tokenizer "
                        "lengths, so characters are a proxy that understates "
                        "what a sorted backfill can do.")
    b.add_argument("--allow-char-fallback", action="store_true",
                   help="proceed with character lengths if the tokenizer cannot "
                        "be loaded, instead of refusing. The results then record "
                        "backfill_sort_key='char' and why -- never 'token'.")
    b.add_argument("--no-repair-sort", action="store_true",
                   help="don't sort each query's stale set by length")
    b.add_argument("--seconds", type=float, default=300, help="per phase")
    b.add_argument("--warmup", type=float, default=60)
    b.add_argument("--repair-gap-ms", type=float, default=0.0,
                   help="idle time after each repair chunk (the window search "
                        "and rank a real query spends off the GPU)")
    b.add_argument("--pg-load", type=int, default=0,
                   help="threads running HNSW window searches during every phase "
                        "(0 = database idle; needs --dsn or $THAW_PG_DSN)")
    b.add_argument("--new", default=None,
                   help="model name, for the passage prefix (required with tei:)")
    b.add_argument("--passage-prefix", default=None,
                   help="override the passage prefix outright")
    b.add_argument("--device", default=None)
    b.add_argument("--out", default=None)
    sub.add_parser("settings", help="show the Postgres settings that matter here")
    a = ap.parse_args()

    from . import checks, data, db, load
    out_dir = Path(a.data_dir)

    def guard(what: str) -> None:
        """quick drops and reloads the tables; selftest2 overwrites docs.text.
        Either one run against a database holding the real 500k load destroys
        it, so both refuse unless the data in play is synthetic or --force."""
        if getattr(a, "force", False):
            return
        f = out_dir / "vectors.npz"
        src = None
        if f.exists():
            import numpy as _np
            z = _np.load(f)
            src = str(z["source"]) if "source" in z.files else None
        real_rows = 0
        db_error = None
        try:
            c = db.connect(a.dsn)
            if c.execute("SELECT to_regclass('docs')").fetchone()[0]:
                real_rows = c.execute(
                    "SELECT count(*) FROM docs WHERE text IS NOT NULL "
                    "AND text NOT LIKE 'k:%'").fetchone()[0]
            c.close()
        except Exception as e:  # noqa: BLE001
            db_error = f"{type(e).__name__}: {e}"
        if db_error is not None:
            # Fail CLOSED. This used to warn and continue, "assuming no real
            # data is at risk" -- which is the one assumption the guard exists
            # to avoid making. A guard that cannot see the database does not
            # know whether `docs` holds a real corpus, and the command it is
            # guarding will connect again moments later: a transient error, a
            # DSN pointing at a database that is briefly down, or a pooler
            # hiccup would let a DROP through on live data. Not knowing is a
            # refusal, and --force is the way to say you know better.
            raise SystemExit(
                f"refusing: {what} destroys data, and the guard could not "
                f"inspect the database to find out whose ({db_error}). It does "
                f"not assume the database is empty. Fix the DSN or start the "
                f"server, point --dsn at a scratch database "
                f"(e.g. .../thaw_test), or pass --force if you know the target "
                f"holds nothing you want.")
        if (src not in (None, "synthetic")) or real_rows:
            raise SystemExit(
                f"refusing: {what} would destroy real data (vectors.npz source="
                f"{src!r}, {real_rows:,} rows of real text in docs). Use another "
                "--data-dir and database (e.g. --dsn .../thaw_test), or --force.")

    if a.cmd == "prepare":
        if a.source == "synthetic":
            d = data.synthetic(a.n, a.dim, a.n_queries)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            d = data.msmarco(a.beir_path, a.limit, a.max_queries, a.old,
                             a.cache_dir, a.device, out_dir)
        data.save(d, out_dir)
        return 0

    if a.cmd == "load":
        # `load.run` begins with `DROP TABLE IF EXISTS claim_done,
        # repair_claims, docs_new, docs CASCADE`, and this command stood
        # unguarded from the first round through eight rounds of review -- two
        # of which I spent calling exactly this shape "the worst thing I have
        # shipped" about a different file. `load` is also the command most
        # likely to be aimed at a database that already holds a corpus.
        guard("load")
        conn = db.connect(a.dsn, ensure_extension=True)   # the one place it belongs
        print(db.server_info(conn))
        load.run(conn, data.load(out_dir), out_dir, a.m, a.ef_construction,
                 a.workers, a.maint_mem)
        return 0

    if a.cmd == "check":
        # check 5 truncates repair_claims and claim_done, and check 2 drops a
        # table it creates; all of it runs against whatever --dsn names
        guard("check")
        kps = [int(x) for x in a.kprimes.split(",")]
        out = Path(a.out) if a.out else out_dir / "step1_report.json"
        return checks.run(db.dsn(a.dsn), data.load(out_dir), out, kps, a.queries,
                          a.max_scan_tuples, a.update_frac, a.claim_windows,
                          a.claim_workers, a.encode_ms)

    if a.cmd in ("prepare2", "verify-encoder", "step2", "selftest2"):
        from . import step2
        if a.cmd == "prepare2":
            step2.prepare2(out_dir, a.beir_path, a.limit, a.max_queries, a.old,
                           a.new, a.cache_dir, a.device)
            return 0
        if a.cmd == "verify-encoder":
            return step2.verify_encoder(out_dir, a.enc_new, a.enc_old, a.chunk,
                                        a.device, a.old)
        if a.cmd == "step2":
            guard("step2")                 # it truncates docs_new
            if not a.enc_new:
                raise SystemExit("--enc-new is required")
            if not a.precomputed and not a.enc_old:
                raise SystemExit("--enc-old is required unless --precomputed")
            out = Path(a.out) if a.out else out_dir / "step2_report.json"
            return step2.run(db.dsn(a.dsn), out_dir,
                             [int(x) for x in a.kprimes.split(",")],
                             a.windows.split(","), a.queries, a.enc_new, a.enc_old,
                             a.chunk, a.threads, a.iterative, a.max_scan_tuples,
                             not a.warm, a.precomputed, a.device, out,
                             old_model=a.old,
                             union_ks=tuple(int(x) for x in a.union.split(",")))
        guard("selftest2")
        return step2.selftest2(db.dsn(a.dsn), out_dir)

    if a.cmd == "backfill-rate":
        from . import backfill
        out = Path(a.out) if a.out else out_dir / "backfill_rate.json"
        return backfill.run(out_dir, a.enc, a.kprime, a.queries, a.chunk,
                            a.backfill_batch, a.sort_window, not a.no_repair_sort,
                            a.seconds, a.warmup, a.repair_gap_ms,
                            db.dsn(a.dsn) if a.pg_load else None, a.pg_load,
                            a.device, out, new_model_prefix=a.passage_prefix,
                            new_model=a.new, token_sort=not a.no_token_sort,
                            allow_char_fallback=a.allow_char_fallback)

    if a.cmd == "settings":
        conn = db.connect(a.dsn)
        info = db.server_info(conn)
        print(f"Postgres {info['postgres']}, pgvector {info['pgvector']}")
        pending = False
        for name, setting, unit, source, restart in db.show_settings(conn):
            val = setting
            if unit == "8kB":
                val = f"{int(setting) * 8 // 1024} MB"
            elif unit == "kB":
                val = f"{int(setting) // 1024} MB"
            flag = "   <- RESTART PENDING" if restart else ""
            pending |= bool(restart)
            print(f"  {name:<34}{val:>12}   ({source}){flag}")
        if pending:
            print("a changed setting needs a restart before it takes effect")
        return 0

    if a.cmd == "quick":
        guard("quick")
        d = data.synthetic(a.n, 384, 500)
        data.save(d, out_dir)
        conn = db.connect(a.dsn)
        load.run(conn, d, out_dir, 16, 64, 2, "1GB")
        conn.close()
        return checks.run(db.dsn(a.dsn), d, out_dir / "step1_report.json",
                          [200, 500, 1000], 50, 100_000, 0.3, 50, 8, 0.2)
    return 2


if __name__ == "__main__":
    sys.exit(main())
