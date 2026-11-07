---
name: graph_memory
description: Recall lessons the agent wrote after its own earlier attempts, anchored to code-graph nodes. Use before editing or answering about a function you have located.
---

# graph_memory

`recall_lessons(node_id=None, query=None)` returns up to 5 short lessons.

- With `node_id` (preferred): lessons anchored to that node, to its call-graph
  neighbours within 2 hops, and to nodes whose precomputed embedding is close to it.
- With `query`: the same, seeded from `search_similar_code(query)`.

Lessons were distilled by this same model from its own successful and failed
attempts on the four dev repositories. They can be wrong. Confirm a lesson with
a code tool before acting on it. Which retrieval runs (none / flat / graph) is set by
`memory_config.json`; the tool has the same signature in the flat and graph arms and is absent in the no-memory arm.
