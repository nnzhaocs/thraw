"""Windows guard: keep TF/jax out and preload pyarrow (see the r18 tree)."""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
try:
    import pyarrow  # noqa: F401
except Exception:  # noqa: BLE001
    pass
