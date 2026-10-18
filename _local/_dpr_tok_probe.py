import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pyarrow  # noqa: F401  # must precede torch (Windows DLL conflict guard)

from transformers import DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

Q = "facebook/dpr-question_encoder-single-nq-base"
C = "facebook/dpr-ctx_encoder-single-nq-base"

qt = DPRQuestionEncoderTokenizer.from_pretrained(Q)
ct = DPRContextEncoderTokenizer.from_pretrained(C)
print("question tok OK:", type(qt).__name__, "| vocab size:", qt.vocab_size)
print("ctx     tok OK:", type(ct).__name__, "| vocab size:", ct.vocab_size)

qs = ["what is the capital of france", "who wrote hamlet"]
enc = qt(qs, padding=True, truncation=True, max_length=64, return_tensors="pt")
print("encoded input_ids shape:", tuple(enc["input_ids"].shape))
print("first seq ids:", enc["input_ids"][0].tolist()[:12])
print("first seq decoded:", qt.decode(enc["input_ids"][0][:12]))

# sanity: question and ctx tokenizers must share the same vocab
assert qt.vocab == ct.vocab, "vocab mismatch between question and ctx tokenizers"
print("vocab identical between question and ctx tokenizers: OK")
