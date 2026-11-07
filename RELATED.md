# Related work

**How this list was checked (7 Nov 2026).** Every arXiv entry below was opened at
`https://arxiv.org/abs/<id>` with WebFetch, and its ID, title, authors and abstract were checked
against the page. EA-Graph (2608.04278) and Repository Memory (2510.01003) were re-opened a
second time because they are the closest prior work. In a second pass the same day, every work
named in the auditor's brief was opened as well. That pass covered SWE-MeM, CGM, SWE-Gym,
SWE-smith, Prometheus, ExpHarness and DeepSWE, and re-checked VibeMemBench, ReasoningBank,
Repository Memory, LocAgent, SWE-Exp, subtask-level memory, AWM, Agent KB and RepoGraph. Where
the brief said more than the page, the row below says what the page says.

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
| 2509.25140 | **ReasoningBank**, Ouyang et al., 2025 (ICLR 2026 per the comments; code at github.com/google-research/reasoning-bank) | Distils generalizable reasoning strategies from the agent's own self-judged successes *and failures*, retrieves them at test time and adds new ones back; memory-aware test-time scaling (MaTTS) amplifies the gains. The abstract names only "web browsing and software engineering" and gives no numbers. | **Closest in lesson source, and our `text` arm's model.** The abstract does not state the title/description/content item format or embedding retrieval; we take those from the paper body (not verified here), so the Writeup says "ReasoningBank-style". Retrieval is by text, with no code structure. |
| 2507.23361 | **SWE-Exp**, Chen et al., 2025 | Distils a multi-faceted experience bank from earlier repair trajectories, successful and failed, for SWE-bench issue resolution. The abstract reports 73.0% Pass@1 on SWE-bench Verified with Claude 4 Sonnet. | **Closest in domain.** Experience is retrieved by issue similarity, with a frontier model as the agent. Ours anchors to call-graph nodes and studies small open models. |
| 2602.21611 | **Structurally Aligned Subtask-Level Memory for SWE agents**, Shen et al., 2026 | Aligns memory storage, retrieval and update with the agent's subtask decomposition instead of whole episodes. The abstract reports +4.7 pp mean Pass@1 on SWE-bench Verified, up to +6.8 pp with Gemini 2.5 Pro. | Memory is keyed to *subtasks*, not code entities. |
| 2603.13258 | **MemCoder** ("Your Code Agent Can Grow Alongside You with Structured Memory"), Deng et al., 2026 | Distils intent-to-code mappings from past commits, adds self-refinement from verification, and keeps human-validated solutions. | Memory comes from human commits and validated solutions; ours comes only from the model's own attempts. |
| 2504.07952 | **Dynamic Cheatsheet**, Suzgun et al., 2025 | A persistent, self-curated memory of strategies and snippets at test time, with no labels and no weight updates. | A flat global cheatsheet, which is again arm b's family. |
| 2507.06229 | **Agent KB**, Tang et al., 2025 | A knowledge base shared across agent frameworks and domains, with two-stage retrieval of workflows and fixes. The abstract reports OpenHands +4.0 pp on SWE-bench pass@1 (24.3% to 28.3%). | Pooled cross-agent experience. Ours is self-distilled and model-specific by construction. |

## 2. Memory anchored to code artifacts (closest prior work)

| arXiv | Work | What it does | Difference from ours |
| --- | --- | --- | --- |
| 2608.04278 | **EA-Graph: Artifact-Anchored Verification Memory for Coding Agents under Upstream Drift**, Hsu, Chi and Everett, Aug 2026 | Stores *verification claims* anchored to the artifacts they were checked against, at sub-path granularity with aliases resolved, so claims can be re-checked after upstream drift. On generated repos it reports that anchored memory beat prose notes and no memory for the smaller model (Haiku) in all test worlds; the larger model hit a ceiling. | **The closest idea, and the fairest framing of novelty: "anchored beats prose, especially for smaller models" has precedent here.** Differences: it anchors claims for staleness detection on synthetic repos, not lessons for solving tasks; it uses no call-graph neighbourhood or node-embedding retrieval; there is no cross-repository transfer; it uses Claude models, where ours uses open Gemma models with self-distilled lessons. |
| 2510.01003 | **Improving Code Localization with Repository Memory**, Wang et al., 2025 (ICLR 2026) | Gives the LocAgent localization agent long-term memory built from commit history: recent commits with their linked issues, plus summaries of frequently changed code. It "can significantly improve LocAgent" on SWE-bench Verified and SWE-bench Live. The abstract names no model and no number, and does not use the name "RepoMem". | The memory is *human* repository history about the same repo. Ours is the agent's own experience and must transfer to unseen repos. |

## 3. Graph-structured code retrieval for agents (no learned memory)

| arXiv | Work | What it does | Difference from ours |
| --- | --- | --- | --- |
| 2410.14684 | **RepoGraph**, Ouyang et al., 2024 (ICLR 2025) | A repository-level code graph module that improves several SWE-bench frameworks. | The graph supplies *code* context. Nothing learned is stored on its nodes. We store experience on nodes and use the graph as the index. |
| 2408.03910 | **CodexGraph**, Liu et al., 2024 | The LLM writes queries against a code graph database to navigate and retrieve. | Code retrieval only. |
| 2503.09089 | **LocAgent**, Chen et al., 2025 (ACL 2025 Long, 2025.acl-long.426, per the ACL Anthology) | A heterogeneous code graph of files, classes and functions, with import, call and inheritance edges, used for multi-hop localization. The abstract reports 92.7% file-level localization accuracy and a fine-tuned Qwen-2.5-Coder-Instruct-32B close to proprietary models. | Code retrieval only; it is the base system for 2510.01003. |
| 2606.22417 | **Code Isn't Memory**, Bhola et al., 2026 | A structural codebase index (ASTs, call-graph lookup) inside a fixed coding harness; reports localization and resolve gains at lower cost. | The "graph index without memory" condition. Our harness already provides that, so our contrast isolates the memory. |
| 2505.16901 | **Code Graph Model (CGM)**, Tao et al., 2025 | Puts the repository's code graph into the LLM's attention, inside an agentless graph-RAG framework. The abstract reports 43.00% on SWE-bench Lite with Qwen2.5-72B. | The graph is *trained into* the model. Ours is a retrieval key for lessons, with no training. |
| 2507.19942 | **Prometheus**, Pan et al., 2025. v1 is titled "Unified Knowledge Graphs for Issue Resolution in Multilingual Codebases"; v2 (Feb 2026) is titled "Towards Long-Horizon Codebase Navigation for Repository-Level Problem Solving" | v1 builds a knowledge graph of the codebase *within* the task. The v2 abstract reports 74.4% on SWE-bench Verified and 33.8% on SWE-PolyBench Verified "powered by GPT-5". | In-task graph, with no memory across tasks. We cite v2's arXiv page and say which version. |
| 2404.05427 | **AutoCodeRover**, Zhang et al., 2024 (ISSTA 2024) | Issue resolution with AST-aware code search over classes and methods, plus optional fault localization. | No memory between tasks. |

## 4. Coding-agent scaffolds and benchmarks

| arXiv | Work | Relevance |
| --- | --- | --- |
| 2405.15793 | **SWE-agent**, Yang et al., 2024 | Agent-computer interfaces for repository work. It is the kind of base agent memory sits on. |
| 2407.01489 | **Agentless**, Xia et al., 2024 | A simple non-agentic pipeline (localize, repair, validate), and a reminder that simple baselines are strong. |
| 2310.06770 | **SWE-bench**, Jimenez et al., 2023 (ICLR 2024) | The Python-repository issue benchmark family the competition's tasks resemble. |
| 2609.23570 | **VibeMemBench**, Fan et al., Sep 2026 | Evaluates memory systems for coding agents on 111 targets from 90 SWE-rebench V2 repositories, with 3,634 history trajectories. "Eleven of twelve solver and system pairings fail to exceed the matched memory-off baseline" (four existing systems). Direct injection of verified experience raises resolution for **four of the five** held-out solvers, by 1.1 to 4.5 pp, and cuts steps for all five. **This is our framing:** the question is *when* memory helps, not whether some memory helps. |
| 2606.28434 | **SWE-MeM**, Gao et al., Jun 2026 | RL-trained memory management (Memory-aware GRPO): the agent learns when, what and how to compress. The abstract reports 43.4% and 60.2% on SWE-bench Verified "with 4B and 30B models", without naming the base models. | Memory *within* a long episode, learned by RL. Ours is across tasks and repositories, with no training. |
| 2412.21139 | **SWE-Gym**, Pan et al., 2024 (ICML 2025) | A training environment of 2,438 real Python tasks. Agents plus verifiers reach 32.0% on SWE-bench Verified. | Training data and an environment. It is relevant to the LoRA stretch, not to memory. |
| 2504.21798 | **SWE-smith**, Yang et al., 2025 | 50k task instances from 128 repositories. SWE-agent-LM-32B reaches 40.2% Pass@1 on SWE-bench Verified. | As above. |
| blog | **DeepSWE**, Luo et al., Agentica and Together AI, 2 Jul 2025, together.ai/blog/deepswe | RL-only training of Qwen3-32B. It reports 42.2% Pass@1 on SWE-bench Verified, and 59% only with hybrid test-time scaling. | Not on arXiv, so it is cited as a URL. The 59% must keep its TTS qualifier. |

## 4b. Experience graphs for agents

| arXiv | Work | What it does | Difference from ours |
| --- | --- | --- | --- |
| 2605.30712 | **ExpHarness: Model-Agnostic Experience Learning through a Trainable Harness**, Feng et al., May 2026 (v2 Oct 2026; some indexes show an earlier title "ExpGraph") | A learnable experience harness for frozen LLM executors: an experience graph plus a retrieval copilot. The abstract reports +12.1% and 4.5% on static tasks, and 21.4% and 12.7% on agentic tasks, for the smaller and larger executors. | The graph is a graph *of experiences*, not of code, and the harness is trained. Ours keys untrained lessons to code-graph properties. |

## 4c. Small open coding models (context only; all self-reported by the vendor, each primary source opened)

| Model | Number as stated | Source | Caveat |
| --- | --- | --- | --- |
| Qwen3.6-27B | SWE-bench Verified 77.2 | huggingface.co/Qwen/Qwen3.6-27B | Qwen's internal scaffold; 200K context |
| Devstral Small 2 (24B) | 68.0% SWE-bench Verified | mistral.ai/news/devstral-2-vibe-cli (9 Dec 2025) | vendor harness |
| FrogNano-4B | 61.5% SWE-bench Verified, **Avg@3** | huggingface.co/microsoft/FrogNano-4B-2609 | Microsoft's Leaf harness; no independent reproduction found |

Gemma 4 has no official SWE-bench number that we found. None of these numbers is comparable to
the competition's harness, and the Writeup does not compare against them.

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
- Memory that rarely beats memory-off for coding agents: VibeMemBench. Our framing starts from it.
- Experience organised as a graph: ExpHarness (a graph of experiences, not of code).
- Experience or lesson memory for agents: Reflexion, ExpeL, AWM, ReasoningBank.
- Lessons from one's own failures as well as successes: ReasoningBank, SWE-Exp.
- Code graphs for agent retrieval: RepoGraph, LocAgent, CodexGraph.
- The idea that *anchored* memory beats prose notes for a smaller model: EA-Graph, on verification
  claims over synthetic repos.

**New, to the best of our search** (targeted searches plus the list above; not exhaustive; the
2026 preprints should be re-checked before submission). No paper we found keys lessons from past
runs to code-graph nodes for models of 31B or smaller. The nearest are Repository Memory
(commit-history memory beside LocAgent tools), Prometheus (an in-task knowledge graph) and
ExpHarness (a graph of experiences, not of code).
1. **Transferable anchor keys.** A lesson is written about one call-graph node, but it is keyed by
   properties of that node that recur in an *unseen* repository: the external APIs it calls, its
   leaf name, its external decorators and bases, its raised exceptions and its fan-in/out shape.
   It is retrieved at the node the agent stands on, or within k hops, inside a declarative
   submission.
2. **A controlled, preregistered "when does it help" study.**
   - Arms: no memory, ReasoningBank-style text keys, node keys, node keys plus k-hop, and a
     shuffled-key control. Every memory arm has the same pool, budget and instructions.
   - Evaluation: leave-one-repo-out, 3 runs on the headline pair, with the flip rate reported.
   - Models: Gemma 4 31B and 12B.
3. **Strict self-distillation.** Each Gemma model learns only from its own attempts.

**Overlap with other competitors (from the brief; not verified by us).**
- One team releases 152 regenerated PR graphs. That is data, not memory.
- Another has preregistered an "anchor distance" study of small-agent failure. If it is published
  first, we cite it. Our moderator (b) and the key-recurrence probe are related but distinct.
