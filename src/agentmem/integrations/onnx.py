"""Local semantic embeddings: all-MiniLM-L6-v2 on ONNX Runtime (CPU, free, offline after
first download). The de-facto default open embedding model (sentence-transformers,
Chroma, LangChain). Requires ``pip install agentmem[onnx]``.

    from agentmem import AgentMemory
    from agentmem.integrations.onnx import MiniLMEmbedder

    mem = AgentMemory("memory.db", embedder=MiniLMEmbedder())   # 384-dim, ~90 MB, auto-downloaded

The archive is fetched from the same public mirror Chroma uses and verified by SHA-256.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, Union

MODEL_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
DEFAULT_CACHE = Path(os.environ.get("AGENTMEM_MODEL_DIR", Path.home() / ".cache" / "agentmem" / "models"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(cache_dir: Union[str, Path] = DEFAULT_CACHE) -> Path:
    """Download + verify + extract the model once; returns the model directory."""
    root = Path(cache_dir) / "all-MiniLM-L6-v2"
    model_dir = root / "onnx"
    if (model_dir / "model.onnx").exists() and (model_dir / "tokenizer.json").exists():
        return model_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "onnx.tar.gz"
    if not archive.exists() or _sha256(archive) != MODEL_SHA256:
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if _sha256(tmp) != MODEL_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("all-MiniLM-L6-v2 download failed checksum verification")
        tmp.replace(archive)
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(root, filter="data")          # Python >= 3.12 / patched 3.8+
        except TypeError:  # pragma: no cover - very old Pythons
            for m in tf.getmembers():
                if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts:
                    raise RuntimeError(f"unsafe path in archive: {m.name}")
            tf.extractall(root)
    return model_dir


class MiniLMEmbedder:
    dim = 384

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, max_length: int = 256,
                 batch_size: int = 32, threads: Optional[int] = None) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("MiniLMEmbedder needs: pip install 'agentmem[onnx]'") from e
        self._np = np
        d = Path(model_dir) if model_dir else ensure_model()
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(d / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size
        self._cache: dict = {}

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        np = self._np
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
            enc = self.tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]                    # (b, seq, 384)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out += pooled.astype(np.float32).tolist()
        return out

    def embed(self, text: str) -> List[float]:
        v = self._cache.get(text)
        if v is None:
            v = self.embed_many([text])[0]
            if len(self._cache) < 10_000:
                self._cache[text] = v
        return v
