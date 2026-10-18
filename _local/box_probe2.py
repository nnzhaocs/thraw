"""Check which version of analysis/scaling_mechanism.py the box's tree carries,
and whether the box can reach huggingface (for the multiset checkpoint)."""
import paramiko

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PW)
ch = t.open_session()
ch.exec_command(
    "cd /root/thaw-r18 && "
    "echo '=== scaling_mechanism key ==='; "
    "grep -c 'band_frac_fell_by_division_not_evidence' analysis/scaling_mechanism.py || true; "
    "grep -c 'band_frac_thins_INFORMATIONAL' analysis/scaling_mechanism.py || true; "
    "echo '=== arrays/ in docstring ==='; "
    "grep -c 'arrays/q_new' analysis/scaling_mechanism.py || true; "
    "echo '=== HF cache: multiset ctx encoder? ==='; "
    "ls ~/.cache/huggingface/hub/ 2>/dev/null | grep -i dpr || echo 'no dpr in HF cache'; "
    "echo '=== network to huggingface ==='; "
    "timeout 8 curl -sI https://huggingface.co 2>&1 | head -2 || echo 'HF unreachable'; "
    "echo '=== autodl-tmp thaw tree (q_new source) ==='; "
    "ls /root/autodl-tmp/thaw/results/runs/20261013-155733_msmarco_8m_dumpq/ 2>/dev/null; "
    "echo PROBE2_DONE")
out = b""
while True:
    if ch.recv_ready():
        out += ch.recv(65536)
    elif ch.exit_status_ready():
        break
print(out.decode("utf-8", "replace"))
ch.close()
t.close()
