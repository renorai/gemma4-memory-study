# Related work

**How this list was checked (7 Nov 2026).** Every arXiv entry below was opened at
`https://arxiv.org/abs/<id>` with WebFetch, and its ID, title, authors and abstract were checked
against the page. EA-Graph (2608.04278) and Repository Memory (2510.01003) were re-opened a
second time because they are the closest prior work.

**What this list does not do.** The summaries paraphrase abstracts. They make no claim about a
paper's internals beyond its abstract. Venues are given only where the arXiv page states them.

**Not cited, because they were not fetched:** Voyager, Generative Agents, Synapse, RepoCoder,
SWE-Bench-CL, SWE-ContextBench, MemGovern and ExpeRepair. Add them only after opening their
pages.

## 1. Experience and lesson memory for agents

| arXiv | Work | What it does (from the abstract) | Difference from ours |
| --- | --- | --- | --- |
| 2303.11366 | **Reflexion**, Shinn et al., 2023 | Agents reflect in words on task feedback and keep the reflections in an episodic buffer for later trials. No weight updates. | Reflections serve retries of the *same* task and are free text. Ours are reused across tasks and repositories, anchored to code-graph nodes. |
| 2308.10144 | **ExpeL**, Zhao et al., 2023 (AAAI-24 per the arXiv comments) | Gathers experience on training tasks, extracts natural-language insights, and at inference retrieves insights and similar past trajectories. | This is our arm b in spirit: experience retrieved by text or task similarity. Arm c changes only the retrieval key, to the code graph. |
| 2409.07429 | **Agent Workflow Memory**, Wang et al., 2024 | Induces reusable workflows from past experience, offline or online, and gives them selectively to web agents (Mind2Web, WebArena). | Procedural routines keyed to tasks, for web navigation. Ours are local lessons keyed to code entities. |
| 2509.25140 | **ReasoningBank**, Ouyang et al., 2025 (ICLR 2026 per the comments) | Distils strategies from the agent's own self-judged successes *and failures*, retrieves them at test time and adds new ones back; also adds memory-aware test-time scaling. Evaluated on web and SWE benchmarks. | **Closest in lesson source:** it learns from its own successes and failures. Retrieval is by text similarity, with no code structure. |
| 2507.23361 | **SWE-Exp**, Chen et al., 2025 | Distils a multi-faceted experience bank from earlier repair trajectories, successful and failed, for SWE-bench issue resolution. | **Closest in domain.** Experience is retrieved by issue similarity, with a frontier model as the agent. Ours anchors to call-graph nodes and studies small open models. |
| 2602.21611 | **Structurally Aligned Subtask-Level Memory for SWE agents**, Shen et al., 2026 | Aligns memory storage, retrieval and update with the agent's subtask decomposition instead of whole episodes. Reports gains on SWE-bench Verified. | Memory is keyed to *subtasks*, not code entities. |
| 2603.13258 | **MemCoder** ("Your Code Agent Can Grow Alongside You with Structured Memory"), Deng et al., 2026 | Distils intent-to-code mappings from past commits, adds self-refinement from verification, and keeps human-validated solutions. | Memory comes from human commits and validated solutions; ours comes only from the model's own attempts. |
| 2504.07952 | **Dynamic Cheatsheet**, Suzgun et al., 2025 | A persistent, self-curated memory of strategies and snippets at test time, with no labels and no weight updates. | A flat global cheatsheet, which is again arm b's family. |
| 2507.06229 | **Agent KB**, Tang et al., 2025 | A knowledge base shared across agent frameworks and domains, with two-stage retrieval of workflows and fixes. | Pooled cross-agent experience. Ours is self-distilled and model-specific by construction. |

## 2. Memory anchored to code artifacts (closest prior work)

| arXiv | Work | What it does | Difference from ours |
| --- | --- | --- | --- |
| 2608.04278 | **EA-Graph: Artifact-Anchored Verification Memory for Coding Agents under Upstream Drift**, Hsu, Chi and Everett, Aug 2026 | Stores *verification claims* anchored to the artifacts they were checked against, at sub-path granularity with aliases resolved, so claims can be re-checked after upstream drift. On generated repos it reports that anchored memory beat prose notes and no memory for the smaller model (Haiku) in all test worlds; the larger model hit a ceiling. | **The closest idea, and the fairest framing of novelty: "anchored beats prose, especially for smaller models" has precedent here.** Differences: it anchors claims for staleness detection on synthetic repos, not lessons for solving tasks; it uses no call-graph neighbourhood or node-embedding retrieval; there is no cross-repository transfer; it uses Claude models, where ours uses open Gemma models with self-distilled lessons. |
| 2510.01003 | **Improving Code Localization with Repository Memory**, Wang et al., 2025 (ICLR 2026) | Gives the LocAgent localization agent tools over the repo's commit history, linked issues and summaries of actively changing code. | The memory is *human* repository history about the same repo. Ours is the agent's own experience and must transfer to unseen repos. |

## 3. Graph-structured code retrieval for agents (no learned memory)

| arXiv | Work | What it does | Difference from ours |
| --- | --- | --- | --- |
| 2410.14684 | **RepoGraph**, Ouyang et al., 2024 (ICLR 2025) | A repository-level code graph module that improves several SWE-bench frameworks. | The graph supplies *code* context. Nothing learned is stored on its nodes. We store experience on nodes and use the graph as the index. |
| 2408.03910 | **CodexGraph**, Liu et al., 2024 | The LLM writes queries against a code graph database to navigate and retrieve. | Code retrieval only. |
| 2503.09089 | **LocAgent**, Chen et al., 2025 | A heterogeneous code graph of files, classes and functions, with import, call and inheritance edges, used for multi-hop localization. | Code retrieval only; it is the base system for 2510.01003. |
| 2606.22417 | **Code Isn't Memory**, Bhola et al., 2026 | A structural codebase index (ASTs, call-graph lookup) inside a fixed coding harness; reports localization and resolve gains at lower cost. | The "graph index without memory" condition. Our harness already provides that, so our contrast isolates the memory. |
| 2404.05427 | **AutoCodeRover**, Zhang et al., 2024 (ISSTA 2024) | Issue resolution with AST-aware code search over classes and methods, plus optional fault localization. | No memory between tasks. |

## 4. Coding-agent scaffolds and benchmarks

| arXiv | Work | Relevance |
| --- | --- | --- |
| 2405.15793 | **SWE-agent**, Yang et al., 2024 | Agent-computer interfaces for repository work. It is the kind of base agent memory sits on. |
| 2407.01489 | **Agentless**, Xia et al., 2024 | A simple non-agentic pipeline (localize, repair, validate), and a reminder that simple baselines are strong. |
| 2310.06770 | **SWE-bench**, Jimenez et al., 2023 (ICLR 2024) | The Python-repository issue benchmark family the competition's tasks resemble. |
| 2609.23570 | **VibeMemBench**, Fan et al., Sep 2026 | Evaluates memory systems for coding agents. In its setting, 11 of 12 existing memory-system × solver pairings fail to beat no memory, while injecting verified experience directly helps most solvers. This is the strongest motivation for testing memory *against a no-memory arm and a flat-memory arm*, not only against nothing. |

## 5. General long-term memory systems and benchmarks

| arXiv | Work | What it does | Difference from ours |
| --- | --- | --- | --- |
| 2310.08560 | **MemGPT**, Packer et al., 2023 | Virtual context management with OS-like memory tiers. | Manages the context window; not experience lessons. |
| 2504.19413 | **Mem0**, Chhikara et al., 2025 | Extracts, consolidates and retrieves facts from multi-session dialogue; also has a graph-memory variant. Evaluated on LoCoMo. | The graph is over conversational entities, not program structure. |
| 2501.13956 | **Zep** (Graphiti), Rasmussen et al., 2025 | A temporal knowledge graph over conversation and business data. Evaluated on DMR and LongMemEval. | The knowledge graph is built from dialogue and enterprise data, not an AST call graph. |
| 2512.12818 | **Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects**, Latimer et al., Dec 2025 | Four memory networks (world facts, agent experiences, entity summaries, evolving beliefs) with retain, recall and reflect operations. Evaluated on LongMemEval and LoCoMo. Code: github.com/vectorize-io/hindsight. | Conversational and entity-centred, with no code structure. An ACL 2026 demo version appeared in a search result but was **not verified**, so cite the arXiv version. |
| 2410.10813 | **LongMemEval**, Wu et al., 2024 (ICLR 2025) | Benchmarks long-term memory of chat assistants: extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention. | Chat memory. It shows that the standard memory benchmarks are conversational, not about code. |

## What is new, stated honestly

**Not new:**
- Experience or lesson memory for agents: Reflexion, ExpeL, AWM, ReasoningBank.
- Lessons from one's own failures as well as successes: ReasoningBank, SWE-Exp.
- Code graphs for agent retrieval: RepoGraph, LocAgent, CodexGraph.
- The idea that *anchored* memory beats prose notes for a smaller model: EA-Graph, on verification
  claims over synthetic repos.

**New, to the best of our search** (three targeted searches plus the list above; not
exhaustive, and the 2026 preprints should be re-checked before submission):
1. Using the code graph's **nodes as the keys of experience memory**. Retrieval runs through the
   call-graph neighbourhood and **precomputed node embeddings**, so a lesson written on one
   repository can reach structurally similar code in an **unseen** repository.
2. A **controlled comparison** that holds the lesson pool, top-k and context budget fixed and
   changes only the retrieval key (graph vs text). The comparison runs leave-one-repo-out and
   across **three sizes of one open model family**.
3. **Strict self-distillation:** each Gemma model learns only from its own attempts, so any gain
   cannot come from a stronger model's outputs.
