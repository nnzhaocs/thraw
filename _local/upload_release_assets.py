"""Upload the THAW data to the data-v1 GitHub release, splitting >1.9 GB files.

GitHub caps a release asset at 2 GiB, so anything larger is cut into
1,900,000,000-byte parts and uploaded as separate assets named
`<path with / -> __>.partNN`. Progress is also appended to
D:\\SemView\\_thaw_upload.log so it can be watched without a pipe (a pipe that
closes kills the process on SIGPIPE). Restarting is safe: assets already on the
release with the right size are skipped, and an asset with the right name but
the wrong size (a killed upload) is deleted before the retry.

Usage:  python _local/upload_release_assets.py [--list-only]
Token:  read from the machine's git credential helper (never written to disk).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
# The release is addressed by the owner/repo environment variables so that no
# account name is baked into the tree.
OWNER = os.environ.get("THAW_GH_OWNER", "OWNER")
REPO = os.environ.get("THAW_GH_REPO", "thraw")
RELEASE_ID = os.environ.get("THAW_RELEASE_ID", "0")
PART = 1_900_000_000
TMP = Path(r"D:\SemView\_thaw_upload_tmp")
LOG = Path(r"D:\SemView\_thaw_upload.log")
API = "https://api.github.com"
UP = f"https://uploads.github.com/repos/{OWNER}/{REPO}/releases/{RELEASE_ID}/assets"

GROUPS = [
    ("cache", "cache"),
    ("step1_data", "step1_data"),
    ("external/db", "external/db"),
    ("beir", "beir"),
    ("external/wiki_dpr_clean", "external/wiki_dpr_clean"),
    ("external/models", "external/models"),
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def token() -> str:
    r = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    sys.exit("no github credential found")


def plan() -> list[dict]:
    out = []
    for dirname, rel in GROUPS:
        base = ROOT / rel
        for f in sorted(base.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            size = f.stat().st_size
            # 0-byte files cannot be release assets (GitHub rejects
            # Content-Length: 0), and the HF cache's `.no_exist` markers and
            # CACHEDIR.TAG are not data -- they record what upstream lacks.
            if size == 0 or ".no_exist" in f.parts or f.name == "CACHEDIR.TAG":
                continue
            name = str(f.relative_to(ROOT)).replace("\\", "/").replace("/", "__")
            if size <= PART:
                out.append({"src": str(f), "asset": name, "size": size,
                            "offset": 0, "length": size})
            else:
                n = (size + PART - 1) // PART
                for i in range(n):
                    off = i * PART
                    ln = min(PART, size - off)
                    out.append({"src": str(f), "asset": f"{name}.part{i:02d}",
                                "size": ln, "offset": off, "length": ln})
    return out


def main() -> int:
    auth = {"Authorization": f"token {token()}",
            "Accept": "application/vnd.github+json"}
    existing = {}
    page = 1
    while True:
        r = requests.get(f"{API}/repos/{OWNER}/{REPO}/releases/{RELEASE_ID}/assets",
                         headers=auth, params={"per_page": 100, "page": page}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for a in batch:
            existing[a["name"]] = (a["id"], a["size"], a["state"])
        page += 1
    todo = plan()
    total = sum(a["length"] for a in todo)
    def done(a):
        cur = existing.get(a["asset"])
        return bool(cur) and cur[2] == "uploaded" and cur[1] == a["size"]

    done_gb = sum(a["length"] for a in todo if done(a)) / 1e9
    log(f"plan: {len(todo)} assets, {total / 1e9:.2f} GB — {done_gb:.2f} GB already up")
    if "--list-only" in sys.argv:
        for a in todo:
            mark = "OK " if existing.get(a["asset"], (0, -1))[1] == a["size"] else "   "
            print(f"  {mark} {a['asset']}  {a['length'] / 1e6:.0f} MB")
        return 0

    TMP.mkdir(exist_ok=True)
    sent = 0
    t_all = time.time()
    for i, a in enumerate(todo, 1):
        cur = existing.get(a["asset"])
        if cur and cur[2] == "uploaded" and cur[1] == a["size"]:
            continue
        if cur:  # killed upload (state 'starter') or wrong size: delete first
            log(f"  deleting stale asset {a['asset']} "
                f"(state {cur[2]}, size {cur[1]} vs {a['size']})")
            requests.delete(f"{API}/repos/{OWNER}/{REPO}/releases/assets/{cur[0]}",
                            headers=auth, timeout=60)
        src = Path(a["src"])
        part_path = None
        if a["length"] != src.stat().st_size:
            part_path = TMP / a["asset"]
            if not part_path.exists() or part_path.stat().st_size != a["length"]:
                with open(src, "rb") as fi, open(part_path, "wb") as fo:
                    fi.seek(a["offset"])
                    left = a["length"]
                    while left:
                        chunk = fi.read(min(64 * 1024 * 1024, left))
                        if not chunk:
                            break
                        fo.write(chunk)
                        left -= len(chunk)
        upload = part_path or src
        t0 = time.time()
        ok = False
        for attempt in range(5):
            try:
                with open(upload, "rb") as fh:
                    r = requests.post(
                        UP, headers={**auth, "Content-Type": "application/octet-stream"},
                        params={"name": a["asset"]}, data=fh, timeout=7200)
                if r.status_code in (200, 201):
                    ok = True
                    break
                log(f"  [{attempt + 1}] HTTP {r.status_code}: {r.text[:120]}")
                if r.status_code == 422:  # name exists: delete and retry once
                    q = requests.get(f"{API}/repos/{OWNER}/{REPO}/releases/{RELEASE_ID}/assets",
                                     headers=auth, params={"per_page": 100}, timeout=60)
                    for x in q.json():
                        if x["name"] == a["asset"]:
                            requests.delete(f"{API}/repos/{OWNER}/{REPO}/releases/assets/{x['id']}",
                                            headers=auth, timeout=60)
                time.sleep(5 * (attempt + 1))
            except Exception as e:                              # noqa: BLE001
                log(f"  [{attempt + 1}] {type(e).__name__}: {e}")
                time.sleep(5 * (attempt + 1))
        if not ok:
            log(f"FAILED {a['asset']} after 5 attempts")
            return 1
        el = time.time() - t0
        sent += a["length"]
        log(f"[{i}/{len(todo)}] {a['asset']}  {a['length'] / 1e6:.0f} MB "
            f"in {el:.0f}s ({a['length'] / max(el, 1e-9) / 1e6:.1f} MB/s)  "
            f"session {sent / 1e9:.2f} GB")
        if part_path and part_path.exists():
            part_path.unlink()
    log(f"DONE {sent / 1e9:.2f} GB in {(time.time() - t_all) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
