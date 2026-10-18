"""Run analysis/dpr_provenance_diag.py with the Windows DLL guard applied.

The tool imports torch BEFORE anything imports semview (whose __init__ carries
the guard), and it calls load_dpr_wiki -> datasets.load_dataset("parquet"),
which pulls pyarrow.dataset. On this machine that ordering access-violates the
process, so the guard is applied here first: USE_TF/TRANSFORMERS_NO_TF before
transformers loads, pyarrow before torch. All argv is passed through, so the
effective command names the real paths used for the 2026-10-17 diagnostic:

    python _local/_run_dpr_diag.py --dpr-path D:/SemView/semview_v14/wiki_dpr_clean
        --variant psgs_w100.nq.no_index --n 256 --out dpr_diag.json
"""
import os
import runpy
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pyarrow  # noqa: F401  # must precede torch (Windows DLL conflict guard)

sys.argv = ["analysis/dpr_provenance_diag.py"] + sys.argv[1:]
runpy.run_path("analysis/dpr_provenance_diag.py", run_name="__main__")
