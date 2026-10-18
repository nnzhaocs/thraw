"""Back up the 8.8M bge-base f16 matrix from the box (13.58 GB).

The local copy in thaw-8m is a .partial whose size is exactly right but whose
sha256 does not match the box's (08026bc9...), so it cannot be promoted. This
downloads a fresh copy to the same cache path, verifies the hash, and renames.
~2 MB/s means about two hours; it runs in the background while A6 and the DPR
rerun finish.
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
REMOTE = "thaw-r18/cache/555c0c6032336a208fbb259eb16cbb59.f16.npy"
LOCAL = r"D:\SemView\thaw-1017\cache\555c0c6032336a208fbb259eb16cbb59.f16.npy"
WANT = "08026bc905b8bcc96b4f9bd0b0137926d80d3c08ecc10919b1c77b98f9088768"

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PW)
sftp = paramiko.SFTPClient.from_transport(t)
need = sftp.stat(REMOTE).st_size
print(f"remote size {need}", flush=True)

part = LOCAL + ".partial"
if os.path.exists(part) and os.path.getsize(part) > need:
    os.remove(part)
t0 = time.time()
done = [0]

def cb(n, total, _t0=t0, _d=done):
    _d[0] = n
    if int(n / (256 * 1024 * 1024)) != int((n - 65536) / (256 * 1024 * 1024)):
        el = time.time() - _t0
        print(f"  {n / 1e9:.2f}/{total / 1e9:.2f} GB "
              f"({n / max(el, 1e-9) / 1e6:.1f} MB/s, {el / 60:.0f} min)", flush=True)

sftp.get(REMOTE, part, callback=cb)
sftp.close()
t.close()
if os.path.getsize(part) != need:
    sys.exit(f"size mismatch: {os.path.getsize(part)} != {need}")

h = hashlib.sha256()
with open(part, "rb") as f:
    while True:
        b = f.read(64 * 1024 * 1024)
        if not b:
            break
        h.update(b)
got = h.hexdigest()
print(f"sha256 {got}", flush=True)
if got != WANT:
    sys.exit(f"HASH MISMATCH: {got} != {WANT}")
os.replace(part, LOCAL)
print(f"OK {LOCAL} verified and renamed", flush=True)
