import os

p = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--dpr-question_encoder-single-nq-base/refs/main")
with open(p, "rb") as f:
    raw = f.read()
print("before:", len(raw), "bytes, head:", raw[:6])
clean = raw.lstrip(b"\xef\xbb\xbf").strip()
with open(p, "wb") as f:
    f.write(clean)
with open(p, "rb") as f:
    after = f.read()
print("after:", len(after), "bytes, head:", after[:6])
assert len(after) == 40 and after.isalnum() or after.decode().isalnum(), "unexpected content"
print("content:", after.decode())
