"""Generate ASSETS.md: the mapping from release assets to local paths, the
split-part reassembly commands, and the restore procedure."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "upl", str(Path(__file__).resolve().parent / "upload_release_assets.py"))
upl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upl)

plan = upl.plan()
by_src = {}
for a in plan:
    by_src.setdefault(a["src"], []).append(a)

lines = [
    "# Release assets: THAW measured data (data-v1)",
    "",
    "The repository holds the code, the 77 run directories and the query dumps.",
    "The 38 GB of measured data — encoded caches, corpora, model weights and the",
    "database dump — is attached to the **data-v1** release because GitHub caps a",
    "repository file at 100 MB. Assets are named after the local path with `/`",
    "replaced by `__`, so `cache/foo.npy` is the asset `cache__foo.npy`.",
    "",
    "Files larger than 1.9 GB are split into `.partNN` pieces. Reassemble with:",
    "",
    "```bash",
    "# bash / Git Bash, run in the folder that will hold the restored tree",
    "cat cache__3844646357568e4071b35b3700ad7871.npy.part* > cache/3844646357568e4071b35b3700ad7871.npy",
    "```",
    "",
    "Then check each restored file against the size in the table (and, for the",
    "npz/npy files, load the header: the first dimension is the corpus size).",
    "",
    f"## Assets ({len(plan)} files, "
    f"{sum(a['length'] for a in plan) / 1e9:.1f} GB)",
    "",
    "| asset | local path | size |",
    "|---|---|---|",
]
for a in plan:
    rel = str(Path(a["src"]).relative_to(upl.ROOT)).replace("\\", "/")
    lines.append(f"| `{a['asset']}` | `{rel}` | {a['length'] / 1e6:.0f} MB |")

lines += [
    "",
    "## Restoring the database",
    "",
    "```bash",
    "pg_restore -d thaw external/db/thaw.dump",
    "```",
    "",
    "## What each piece is",
    "",
    "- `cache/` — the encoded matrices keyed by content hash: the 500k spaces for",
    "  MiniLM, bge-small, bge-base, bge-m3 and Qwen3-Embedding-0.6B, the 8.8M",
    "  MiniLM space, and the f16 variants. This is the expensive part (hours of",
    "  GPU) and the only data here that had to exist on a particular machine.",
    "- `step1_data/` — the step1 vectors and the queries/texts they were built from.",
    "- `external/db/thaw.dump` — the loaded PostgreSQL database: 500k documents",
    "  with both models' vectors, the table `thaw_pg` reads.",
    "- `beir/` — the MS MARCO corpus in BEIR layout (public data, included for",
    "  convenience; `--corpus beir --path beir/` is the form the runs use).",
    "- `external/wiki_dpr_clean/` — the wiki_dpr parquet shards the DPR estimates",
    "  read (public data from facebook/wiki_dpr).",
    "- `external/models/` — the encoder weights (public, from huggingface and",
    "  modelscope): MiniLM, bge-small/base/m3, Qwen3-Embedding-0.6B, the DPR",
    "  question and context encoders, e5 and mpnet.",
    "",
    "One matrix is missing and cannot be restored: the 8.8M bge-base f16 file",
    "existed only on a cloud instance that has since been released. It gates local",
    "re-runs of the scaling-mechanism analysis; the numbers it produced are in",
    "`results/runs/`.",
]
out = Path(__file__).resolve().parent.parent / "ASSETS.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(plan)} assets)")
