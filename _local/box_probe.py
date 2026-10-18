"""Probe the cloud box: connect, inventory /root, check for the two matrices
A6 needs (ad3f5957 = Qwen3 500k, c60e0947 = bge-m3 500k) and the leg dumps."""
import paramiko
import sys

import os

# The box is gone and its address is not part of the record. Set THAW_BOX_HOST /
# THAW_BOX_PORT / THAW_BOX_USER / THAW_BOX_PW to point these scripts at any
# SSH host; the defaults are deliberately inert.
HOST = os.environ.get("THAW_BOX_HOST", "box.invalid")
PORT = int(os.environ.get("THAW_BOX_PORT", "22"))
USER = os.environ.get("THAW_BOX_USER", "root")
PW = os.environ.get("THAW_BOX_PW", "")

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PW)
ch = t.open_session()
ch.exec_command(
    "echo '=== /root ==='; ls /root; "
    "echo '=== thaw trees ==='; ls -d /root/thaw* 2>/dev/null; "
    "echo '=== target matrices ==='; "
    "for h in ad3f5957fc3400a29de3e268ada99b08 c60e09474748d76ad5f3eafb61126bce; do "
    "  for tree in /root/thaw-r18 /root/thaw-1017 /root/thaw; do "
    "    f=$tree/cache/$h.npy; "
    "    if [ -f \"$f\" ]; then echo \"$f $(stat -c %s $f) bytes\"; fi; "
    "  done; done; "
    "echo '=== dump legs ==='; "
    "for tree in /root/thaw-r18 /root/thaw-1017 /root/thaw; do "
    "  if [ -d \"$tree/dump\" ]; then echo \"$tree/dump:\"; ls -la $tree/dump/; "
    "    for l in leg0 leg1 leg2 leg3; do ls -la $tree/dump/$l/ 2>/dev/null | grep q_; done; "
    "  fi; done; "
    "echo '=== df ==='; df -h /root | tail -1; "
    "echo PROBE_DONE")
out = b""
while True:
    if ch.recv_ready():
        out += ch.recv(65536)
    elif ch.exit_status_ready():
        break
print(out.decode("utf-8", "replace"))
ch.close()
t.close()
