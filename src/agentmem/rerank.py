"""Stage-3 reranking: re-score the top candidates with a cross-encoder.

Bi-encoders (embeddings) score query and memory independently; a cross-encoder
reads them *together*, which is far more precise but too slow to run on every
memory - hence: index -> hybrid ranker -> cross-encoder on the top ``rerank_depth``.

    from agentmem import AgentMemory
    from agentmem.rerank import CrossEncoderReranker

    mem = AgentMemory("memory.db", embedder="minilm", reranker=CrossEncoderReranker())

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0), ONNX export by
Xenova, run with ONNX Runtime on CPU (``pip install agentmem[onnx]``).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, Union

HF_BASE = "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/"
FILES = {  # path -> sha256 (pinned; a changed upstream file fails loudly instead of loading)
    "onnx/model.onnx": "c623d0bcb99f4622beb413eaef00cfbe5db20df9f1dd982da4b4f26022881870",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}
DEFAULT_CACHE = Path(os.environ.get("AGENTMEM_MODEL_DIR", Path.home() / ".cache" / "agentmem" / "models"))


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> List[float]: ...


class CallableReranker:
    """Wrap ``fn(query, texts) -> scores`` (e.g. a hosted rerank API or an LLM judge)."""

    def __init__(self, fn: Callable[[str, Sequence[str]], Sequence[float]]) -> None:
        self.fn = fn

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        return [float(x) for x in self.fn(query, texts)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_reranker_model(cache_dir: Union[str, Path] = DEFAULT_CACHE) -> Path:
    root = Path(cache_dir) / "ms-marco-MiniLM-L-6-v2"
    for rel, sha in FILES.items():
        dest = root / rel
        if dest.exists() and (sha is None or _sha256(dest) == sha):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(HF_BASE + rel, tmp)
        if sha is not None and _sha256(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{rel} failed checksum verification")
        tmp.replace(dest)
    return root


class CrossEncoderReranker:
    def __init__(self, model_dir: Optional[Union[str, Path]] = None, max_length: int = 256,
                 batch_size: int = 32) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("CrossEncoderReranker needs: pip install 'agentmem[onnx]'") from e
        self._np = np
        d = Path(model_dir) if model_dir else ensure_reranker_model()
        if _sha256(d / "onnx" / "model.onnx") != FILES["onnx/model.onnx"]:
            raise RuntimeError(f"unexpected reranker weights in {d} (checksum mismatch)")
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(d / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
        self.batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        np = self._np
        out: List[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch([(query, t or " ") for t in texts[i:i + self.batch_size]])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                     "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64)}
            out += self.session.run(None, feeds)[0].ravel().tolist()
        return out
