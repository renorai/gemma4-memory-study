"""`swegemma eval` with the scorer's settings the bare CLI leaves unset.

Same arguments as `swegemma eval`. It only wraps swegemma.cli.main and fills four EvalConfig
fields the CLI never passes:
    graph_dir, embeddings_dir      from SWEGEMMA_GRAPH_DIR / SWEGEMMA_EMBEDDINGS_DIR (the CLI
                                   default 'data/graphs' is relative and misses the competition data)
    context_cache_config,          the values in the official Getting Started notebook, so context
    events_compaction_config       compaction starts at about 14k tokens as in scoring
"""
import os
import sys

import swegemma.cli as cli
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig

_GRAPH = os.environ.get("SWEGEMMA_GRAPH_DIR")
_EMB = os.environ.get("SWEGEMMA_EMBEDDINGS_DIR")


def _config(**kw):
    if _GRAPH:
        kw.setdefault("graph_dir", _GRAPH)
    if _EMB:
        kw.setdefault("embeddings_dir", _EMB)
    kw.setdefault("context_cache_config", ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10))
    kw.setdefault("events_compaction_config", EventsCompactionConfig(
        compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5))
    return EvalConfig(**kw)


cli.EvalConfig = _config

if __name__ == "__main__":
    sys.exit(cli.main(sys.argv[1:]))
