"""A tiny, deterministic, offline stand-in for the Kaggle harness.

It exists so that every arm, the distiller, the runner and the statistics can be
exercised in unit tests without the 22 GB dataset or a model. It proves the
MECHANICS only. Nothing measured on it is a result, and nothing from it goes into
the paper.

Fixture: three toy repos (alpha, beta, gamma) with the same five roles. Node
embeddings are 8-d: one basis vector per role plus a small repo-specific tilt, so
the same role in two repos is close (cos ~0.99) and different roles are far apart.

    client.send -> client.build_request -> auth.sign
    client.send -> utils.parse_headers  -> models.Response
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from memory.adapter import Task
from memory.schema import allow_test_author

allow_test_author("fake-policy")

REPOS = ("alpha", "beta", "gamma")
ROLES = ("client.send", "client.build_request", "auth.sign", "utils.parse_headers", "models.Response")
EDGES = (("client.send", "client.build_request"), ("client.build_request", "auth.sign"),
         ("client.send", "utils.parse_headers"), ("utils.parse_headers", "models.Response"))
DOCS = {
    "client.send": "def send(request): dispatch an http request and return the response",
    "client.build_request": "def build_request(method, url): assemble method url headers",
    "auth.sign": "def sign(request): add an hmac signature header",
    "utils.parse_headers": "def parse_headers(raw): split raw header lines into a dict",
    "models.Response": "class Response: status code body and headers",
}
DIM = 8


def node(repo: str, role: str) -> str:
    return f"{repo}:{role}"


def role_of(node_id: str) -> str:
    return node_id.split(":", 1)[1]


def _vec(repo: str, role: str) -> list[float]:
    v = [0.0] * DIM
    v[ROLES.index(role)] = 1.0
    v[5 + REPOS.index(repo)] = 0.15  # repo tilt
    return v


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


class FakeHarness:
    """Implements memory.adapter.HarnessAdapter."""

    def __init__(self):
        self._nodes = {node(r, ro): (r, ro) for r in REPOS for ro in ROLES}
        self._adj: dict[str, list[str]] = {n: [] for n in self._nodes}
        for r in REPOS:
            for a, b in EDGES:
                self._adj[node(r, a)].append(node(r, b))
                self._adj[node(r, b)].append(node(r, a))
        for n in self._adj:
            self._adj[n].sort()
        self._tasks = []
        for r in REPOS:
            # easy: the answer is one hop from the seed
            self._tasks.append(Task(f"{r}-easy", r,
                                    "Starting from client.send, which function builds the request?",
                                    (node(r, "client.send"),), {"gold": node(r, "client.build_request")}))
            # hard: the answer is two hops away; the fake agent only explores one hop
            self._tasks.append(Task(f"{r}-hard", r,
                                    "Which function adds the signature to requests made by client.send?",
                                    (node(r, "client.send"),), {"gold": node(r, "auth.sign")}))

    # --- HarnessAdapter ---
    def tasks(self):
        return list(self._tasks)

    def repo_of(self, node_id):
        return self._nodes[node_id][0]

    def embedding(self, node_id):
        if node_id not in self._nodes:
            return None
        return _vec(*self._nodes[node_id])

    def neighbors(self, node_id):
        return list(self._adj.get(node_id, ()))

    def similar_nodes(self, *, node_id=None, query=None, k=10):
        if node_id is not None:
            q = self.embedding(node_id)
            nq = math.sqrt(sum(x * x for x in q))
            scored = []
            for n in self._nodes:
                if n == node_id:
                    continue
                v = self.embedding(n)
                scored.append((n, sum(a * b for a, b in zip(q, v)) / (nq * math.sqrt(sum(x * x for x in v)))))
        else:
            qt = _tokens(query or "")
            scored = []
            for n, (_r, ro) in self._nodes.items():
                dt = _tokens(DOCS[ro] + " " + ro)
                scored.append((n, len(qt & dt) / len(qt | dt) if qt | dt else 0.0))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def subgraph(self, node_id, hops=1):
        seen, frontier, edges = {node_id}, [node_id], []
        for _ in range(hops):
            nxt = []
            for n in frontier:
                for m in self._adj[n]:
                    edges.append([n, m])
                    if m not in seen:
                        seen.add(m)
                        nxt.append(m)
            frontier = nxt
        return {"nodes": sorted(seen), "edges": sorted(edges)}


@dataclass
class FakeAgent:
    """Scripted policy standing in for Gemma. Deterministic.

    It reads the seed's one-hop neighbourhood. It answers correctly when the gold
    node is in that neighbourhood, or when a rendered lesson names the gold ROLE
    (it then maps the role onto its own repo). Otherwise it answers the first
    neighbour, which is wrong for the hard task.
    """

    harness: FakeHarness
    model: str = "fake-policy"

    def __call__(self, task: Task, arm, run_id: str = "fake") -> dict:
        steps = []
        lessons = arm.for_task(task)
        rendered = "\n".join(f"{r.lesson.node_id} {r.lesson.text}" for r in lessons)
        seed = task.seed_nodes[0]
        nbrs = self.harness.neighbors(seed)
        steps.append({"tool": "get_code_neighbors", "args": {"node_id": seed}, "nodes": nbrs})
        gold = task.meta["gold"]
        answer = None
        target_role = role_of(gold)
        if gold in nbrs:
            answer = gold
        elif target_role in rendered:
            answer = node(task.repo, target_role)
            steps.append({"tool": "get_code_neighbors", "args": {"node_id": answer},
                          "nodes": self.harness.neighbors(answer)})
        else:
            answer = nbrs[0]
        prompt_chars = len(task.prompt) + len(rendered)
        return {
            "run_id": run_id, "task_id": task.task_id, "repo": task.repo, "model": self.model,
            "arm": arm.arm, "prompt": task.prompt, "steps": steps, "final_answer": answer,
            "outcome": "success" if answer == gold else "failure", "error": None,
            "tokens_in": prompt_chars // 4 + 50 * len(steps), "tokens_out": 20 * len(steps),
            "wall_s": 1.0 * len(steps), "lesson_ids": [r.lesson.lesson_id for r in lessons],
        }


def fake_generate(prompt: str) -> str:
    """Stands in for the attempting Gemma model during distillation. Deterministic:
    anchors one lesson on the first listed anchor and mentions every node it touched."""
    anchors = re.findall(r"^- (\S+)$", prompt.split("you may ONLY anchor", 1)[-1], re.M)
    if not anchors:
        return "[]"
    outcome = re.search(r"Outcome of the attempt \(from the harness scorer\): (\w+)", prompt).group(1)
    roles = ", ".join(role_of(a) for a in anchors)
    verb = "worked" if outcome == "success" else "was not enough"
    return ('[{"node_id": "%s", "lesson": "Walking %s %s; the request path also passes auth.sign '
            'two hops down via build_request."}]' % (anchors[0], roles, verb))
