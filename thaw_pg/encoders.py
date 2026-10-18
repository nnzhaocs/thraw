"""Encoders behind one interface: encode(texts) -> (n, d) float32, unit length.

  TEIEncoder    Hugging Face Text Embeddings Inference over HTTP. The system
                path: a separate GPU server, as a deployment would run it.
  LocalEncoder  sentence-transformers in-process. For a first run without TEI,
                and for the old model's query encoder if you do not want a
                second TEI container.
  TableEncoder  looks vectors up instead of computing them. Used by the
                self-test, so the whole path can be exercised without a model.

Every encoder sends at most `batch` texts per call. For the new model that is
the repair chunk: `repair_chunk` from analysis/measure_throughput.py on this
GPU, the measured batch size with the lowest repair penalty (16 on an RTX 4060
Laptop GPU, 8 in the coarser original curve).
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np


def _normalize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-12)


class _Base:
    def __init__(self, prefix: str = "", batch: int = 8):
        self.prefix, self.batch = prefix, int(batch)
        self.sort_by_length = True
        self.calls = 0
        self.items = 0
        self.ms = 0.0

    def _encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_timed(self, texts: list[str]):
        """Fallback for encoders that can't split the time (TEI): the whole
        call is counted as forward, and token counts are unknown."""
        t0 = time.perf_counter()
        V = self._encode([self.prefix + t for t in texts])
        ms = (time.perf_counter() - t0) * 1e3
        return _normalize(V), dict(tok_ms=0.0, fwd_ms=ms, copy_ms=0.0,
                                   real_tokens=None, padded_tokens=None, max_len=None)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        t0 = time.perf_counter()
        # Chunk in order of length, so each chunk pads to a similar length
        # rather than to its one longest passage; rows go back in input order.
        # The vectors are identical either way: only the padding changes.
        order = (sorted(range(len(texts)), key=lambda i: len(texts[i]))
                 if self.sort_by_length else list(range(len(texts))))
        out = []
        for lo in range(0, len(order), self.batch):
            chunk = [self.prefix + texts[i] for i in order[lo:lo + self.batch]]
            out.append(self._encode(chunk))
            self.calls += 1
        self.items += len(texts)
        M = np.vstack(out)
        res = np.empty_like(M)
        res[np.asarray(order)] = M
        self.ms += (time.perf_counter() - t0) * 1e3
        return _normalize(res)


class TEIEncoder(_Base):
    """POST {url}/embed. Start TEI with --max-client-batch-size >= batch."""

    def __init__(self, url: str, prefix: str = "", batch: int = 8,
                 timeout: float = 120.0):
        super().__init__(prefix, batch)
        self.url, self.timeout = url.rstrip("/"), timeout
        self.dim = self._encode(["probe"]).shape[1]

    def _encode(self, texts):
        body = json.dumps({"inputs": texts, "normalize": True,
                           "truncate": True}).encode()
        req = urllib.request.Request(self.url + "/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return np.asarray(json.loads(r.read()), dtype=np.float32)


class LocalEncoder(_Base):
    def __init__(self, model: str, prefix: str = "", batch: int = 8,
                 device: str | None = None, fp16: bool = False):
        super().__init__(prefix, batch)
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model, device=device)
        if fp16:
            # about 2x on tensor-core GPUs; vectors change slightly, so run
            # verify-encoder and keep the ground truth's precision in mind
            self.m.half()
        self.fp16 = fp16
        self.dim = self.m.get_sentence_embedding_dimension()

    def _encode(self, texts):
        return self.m.encode(texts, batch_size=len(texts), convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)

    def encode_timed(self, texts: list[str]):
        """One batch, as `encode` would run it, with the time split into
        tokenize / forward / copy-back and the token counts recorded. The prefix
        is applied here. Returns (normalised vectors, timing dict)."""
        import torch

        texts = [self.prefix + t for t in texts]
        dev = self.m.device
        sync = (torch.cuda.synchronize if dev.type == "cuda" else (lambda: None))
        t0 = time.perf_counter()
        feats = self.m.tokenize(texts)
        mask = feats["attention_mask"]
        real, padded = int(mask.sum()), int(mask.numel())
        feats = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                 for k, v in feats.items()}
        sync()
        t1 = time.perf_counter()
        with torch.inference_mode():
            emb = self.m(feats)["sentence_embedding"]
        sync()
        t2 = time.perf_counter()
        V = emb.float().cpu().numpy()
        t3 = time.perf_counter()
        return _normalize(V), dict(tok_ms=(t1 - t0) * 1e3, fwd_ms=(t2 - t1) * 1e3,
                                   copy_ms=(t3 - t2) * 1e3, real_tokens=real,
                                   padded_tokens=padded, max_len=int(mask.shape[1]))


class TableEncoder(_Base):
    """texts are keys 'k:<row>' into `table`; optional `delay_ms` per item
    stands in for GPU time."""

    def __init__(self, table: np.ndarray, batch: int = 8, delay_ms: float = 0.0):
        super().__init__("", batch)
        self.table, self.delay_ms = table, delay_ms
        self.dim = table.shape[1]

    def _encode(self, texts):
        if self.delay_ms:
            time.sleep(self.delay_ms * len(texts) / 1e3)
        return self.table[[int(t.split(":", 1)[1]) for t in texts]]


def make(spec: str, prefix: str, batch: int, device: str | None = None):
    """'tei:http://host:8080' | 'local:<model name>' | 'local-fp16:<model name>'"""
    kind, _, arg = spec.partition(":")
    if kind == "tei":
        return TEIEncoder(arg, prefix, batch)
    if kind == "local":
        return LocalEncoder(arg, prefix, batch, device)
    if kind == "local-fp16":
        return LocalEncoder(arg, prefix, batch, device, fp16=True)
    raise SystemExit("encoder spec must be tei:<url>, local:<model> or "
                     f"local-fp16:<model>, got {spec!r}")
