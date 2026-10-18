"""Pull the box's A4 rerun directory back into the local tree and check the
new code's output key is present."""
import os
import shutil

import paramiko

import os

# The box is gone and its address is not part of the record. Set THAW_BOX_HOST /
# THAW_BOX_PORT / THAW_BOX_USER / THAW_BOX_PW to point these scripts at any
# SSH host; the defaults are deliberately inert.
HOST = os.environ.get("THAW_BOX_HOST", "box.invalid")
PORT = int(os.environ.get("THAW_BOX_PORT", "22"))
USER = os.environ.get("THAW_BOX_USER", "root")
PW = os.environ.get("THAW_BOX_PW", "")
REMOTE = "thaw-r18/results/runs/20261017-225704_scaling_mechanism"
LOCAL = r"D:\SemView\thaw-1017\results\runs\20261017-225704_scaling_mechanism"

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PW)
sftp = paramiko.SFTPClient.from_transport(t)
os.makedirs(LOCAL, exist_ok=True)
for name in sftp.listdir(REMOTE):
    sftp.get(f"{REMOTE}/{name}", os.path.join(LOCAL, name))
    print("pulled", name)
sftp.close()
t.close()

import json
d = json.load(open(os.path.join(LOCAL, "scaling_mechanism.json"), encoding="utf-8"))
new_key = "band_frac_fell_by_division_not_evidence"
print("new-code key present:", new_key in d, "| value:", d.get(new_key))
print("old key absent:", "band_frac_thins_INFORMATIONAL" not in d)
print("verdict:", d["verdict"][:120])
print("code_hash:", d.get("environment", {}).get("code_hash"))
