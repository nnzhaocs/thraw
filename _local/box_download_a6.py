"""Download the A6 inputs (and box-side backup) from the cloud box over SFTP.

Files (relative to the SFTP cwd /root):
  thaw-r18/cache/ad3f5957fc3400a29de3e268ada99b08.npy   2.05 GB  Qwen3 500k
  thaw-r18/cache/c60e09474748d76ad5f3eafb61126bce.npy   2.05 GB  bge-m3 500k
  thaw-r18/dump/leg{0,1,2,3}/q_{old,new}.npy                   ~100 MB  query dumps

Each file downloads to `<dst>.partial` and is renamed only after the size
matches the remote stat. Also asks the box for sha256 of its 8.8M bge-base f16
matrix so the local .partial copy can be verified and promoted.
"""
import hashlib
import os
import sys
import time

import paramiko

import os

# The box is gone and its address is not part of the record. Set THAW_BOX_HOST /
# THAW_BOX_PORT / THAW_BOX_USER / THAW_BOX_PW to point these scripts at any
# SSH host; the defaults are deliberately inert.
HOST = os.environ.get("THAW_BOX_HOST", "box.invalid")
PORT = int(os.environ.get("THAW_BOX_PORT", "22"))
USER = os.environ.get("THAW_BOX_USER", "root")
PW = os.environ.get("THAW_BOX_PW", "")
ROOT = r"D:\SemView\thaw-1017"

FILES = [
    ("thaw-r18/cache/ad3f5957fc3400a29de3e268ada99b08.npy",
     os.path.join(ROOT, "cache", "ad3f5957fc3400a29de3e268ada99b08.npy")),
    ("thaw-r18/cache/c60e09474748d76ad5f3eafb61126bce.npy",
     os.path.join(ROOT, "cache", "c60e09474748d76ad5f3eafb61126bce.npy")),
]
for leg in ("leg0", "leg1", "leg2", "leg3"):
    for q in ("q_old", "q_new"):
        FILES.append((
            f"thaw-r18/dump/{leg}/{q}.npy",
            os.path.join(ROOT, "dump", leg, f"{q}.npy"),
        ))

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PW)
sftp = paramiko.SFTPClient.from_transport(t)

# box-side sha256 of the 8.8M f16 matrix, for verifying the local .partial
ch = t.open_session()
ch.exec_command("cd /root && sha256sum thaw-r18/cache/555c0c6032336a208fbb259eb16cbb59.f16.npy")
box_hash = b""
while True:
    if ch.recv_ready():
        box_hash += ch.recv(65536)
    elif ch.exit_status_ready():
        break
print("BOX sha256 555c0c60.f16.npy:", box_hash.decode().strip()[:64], flush=True)
ch.close()

for src, dst in FILES:
    st = sftp.stat(src)
    need = st.st_size
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == need:
        print(f"SKIP {os.path.basename(dst)}: already {need} bytes", flush=True)
        continue
    part = dst + ".partial"
    t0 = time.time()
    got = [0]

    def _cb(transferred, total, _t0=t0, _got=got, _name=os.path.basename(dst)):
        _got[0] = transferred
        if int(transferred / (32 * 1024 * 1024)) != int((transferred - 65536) / (32 * 1024 * 1024)):
            el = time.time() - _t0
            print(f"  {_name}: {transferred / 1e6:.0f}/{total / 1e6:.0f} MB "
                  f"({transferred / max(el, 1e-9) / 1e6:.1f} MB/s, {el:.0f}s)",
                  flush=True)

    sftp.get(src, part, callback=_cb)
    if os.path.getsize(part) != need:
        print(f"FAIL {src}: size mismatch", flush=True)
        sys.exit(1)
    os.replace(part, dst)
    print(f"OK   {os.path.basename(dst)}: {need} bytes in {time.time() - t0:.0f}s",
          flush=True)

sftp.close()
t.close()
print("ALL DOWNLOADS DONE", flush=True)
