"""agentmem - three-layer long-term memory for AI agents.

Working memory (bounded, current session) -> Episodic memory (time-stamped
experiences) -> Semantic memory (consolidated facts), with importance scoring,
decay/forgetting and cross-session persistence.
"""
from .consolidation import (CallableExtractor, ConsolidationReport, Fact, FactExtractor,
                            RuleBasedExtractor)
from .embeddings import CallableEmbedder, Embedder, HashingEmbedder, cosine
from .episodic import EpisodicMemory
from .memory import AgentMemory
from .scoring import (CallableImportanceScorer, HeuristicImportanceScorer, ImportanceScorer,
                      RetrievalWeights)
from .selfembed import SelfEmbedder
from .semantic import SemanticMemory
from .storage import InMemoryStorage, SQLiteStorage, StorageBackend
from .types import Layer, MemoryItem, ScoredMemory, Session
from .working import WorkingMemory

__version__ = "0.1.0"
__all__ = [
    "AgentMemory", "WorkingMemory", "EpisodicMemory", "SemanticMemory",
    "MemoryItem", "ScoredMemory", "Session", "Layer",
    "StorageBackend", "InMemoryStorage", "SQLiteStorage",
    "Embedder", "HashingEmbedder", "CallableEmbedder", "SelfEmbedder", "cosine",
    "ImportanceScorer", "HeuristicImportanceScorer", "CallableImportanceScorer", "RetrievalWeights",
    "FactExtractor", "RuleBasedExtractor", "CallableExtractor", "Fact", "ConsolidationReport",
]
