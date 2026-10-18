"""Re-run A4 (scaling mechanism) on the box with the NEW code.

The box carries the old scaling_mechanism.py (old informational key, arrays/
docstring); the tree under test has the fixed one. This uploads the new file
(keeping a .old backup on the box), runs it through the box's runlog against
the matrices that only exist there, streams the output, and pulls the run
directory back to the local tree.
"""
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
LOCAL_TOOL = r"D:\SemView\thaw-1017\analysis\scaling_mechanism.py"
REMOTE_TOOL = "thaw-r18/analysis/scaling_mechanism.py"

CMD = (
    "cd /root/thaw-r18 && "
    f"cp {REMOTE_TOOL} {REMOTE_TOOL}.old 2>/dev/null; "
    "THAW_MODE=measure /root/miniconda3/bin/python -m semview.runlog "
    "--name scaling_mechanism --tag d08 -- "
    "/root/miniconda3/bin/python analysis/scaling_mechanism.py "
    "--v-new cache/555c0c6032336a208fbb259eb16cbb59.f16.npy "
    "--q-new /root/autodl-tmp/thaw/results/runs/20261013-155733_msmarco_8m_dumpq/"
    "q_new.npy "
    "--sizes 500000,2000000,4000000,8841823 "
    "--k 10 --queries 1000 --band 0.01 --seed 0 "
    "--out \"{run}/scaling_mechanism.json\""
)

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PW)
sftp = paramiko.SFTPClient.from_transport(t)
sftp.put(LOCAL_TOOL, REMOTE_TOOL)
print("uploaded new scaling_mechanism.py ->", REMOTE_TOOL, flush=True)
sftp.close()

ch = t.open_session()
ch.exec_command(CMD)
out = b""
last = time.time()
while True:
    if ch.recv_ready():
        out += ch.recv(65536)
        while b"\n" in out:
            line, out = out.split(b"\n", 1)
            s = line.decode("utf-8", "replace").rstrip()
            if "it/s" in s or s.strip().endswith("%|"):
                continue
            print(f"[{time.strftime('%H:%M:%S')}] {s[:170]}", flush=True)
        last = time.time()
    elif ch.exit_status_ready():
        break
    elif time.time() - last > 2400:
        print("... quiet 40 min, still running ...", flush=True)
        last = time.time()
    else:
        time.sleep(2)
print(f"=== box run exited, status {ch.recv_exit_status()} ===", flush=True)
ch.close()
t.close()
