"""Step 1 of the Thaw system build: pgvector feasibility checks.

    python -m thaw_pg prepare ...   # vectors to step1_data/
    python -m thaw_pg load ...      # tables and indexes in Postgres
    python -m thaw_pg check ...     # the five gated checks, plus a preview of E2
"""
