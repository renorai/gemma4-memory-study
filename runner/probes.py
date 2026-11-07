"""Pre-run checks whose outcome can change the protocol (DESIGN.md section 8, threat 3)."""
from __future__ import annotations

import math
import random
from typing import Sequence

from memory.adapter import HarnessAdapter


def _cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def cross_repo_embedding_probe(adapter: HarnessAdapter, matched_pairs: Sequence[tuple[str, str]],
                               all_nodes: Sequence[str], n_sets: int = 1000, seed: int = 0) -> dict:
    """Is the mean cosine of same-role cross-repo pairs higher than chance?

    `matched_pairs` are chosen by name/role BEFORE looking at any cosine (e.g. the
    `send` methods of the requests and httpx clients). Null: the mean cosine of
    len(matched_pairs) random cross-repo pairs, drawn n_sets times. Pass = the matched
    mean exceeds the null's 95th percentile (one-sided, alpha 0.05).
    """
    matched = [_cos(adapter.embedding(a), adapter.embedding(b)) for a, b in matched_pairs]
    if not matched:
        raise ValueError("no matched pairs")
    by_repo: dict[str, list[str]] = {}
    for n in all_nodes:
        by_repo.setdefault(adapter.repo_of(n), []).append(n)
    repos = sorted(by_repo)
    if len(repos) < 2:
        raise ValueError("need nodes from at least two repos")
    rng = random.Random(seed)
    k = len(matched)
    means = []
    for _ in range(n_sets):
        tot = 0.0
        for _ in range(k):
            ra, rb = rng.sample(repos, 2)
            tot += _cos(adapter.embedding(rng.choice(by_repo[ra])), adapter.embedding(rng.choice(by_repo[rb])))
        means.append(tot / k)
    means.sort()
    p95 = means[int(0.95 * (len(means) - 1))]
    m = sum(matched) / k
    return {"matched_mean": m, "null_p95": p95, "n_matched": k, "n_sets": n_sets,
            "p_value": (1 + sum(x >= m for x in means)) / (1 + n_sets), "pass": m > p95}


def flip_rate(rows_a: Sequence[dict], rows_b: Sequence[dict]) -> dict:
    """Run-to-run noise: share of tasks whose success differs between two identical runs."""
    a = {(r["model"], r["task_id"]): r["success"] for r in rows_a}
    b = {(r["model"], r["task_id"]): r["success"] for r in rows_b}
    keys = sorted(set(a) & set(b))
    flips = sum(a[k] != b[k] for k in keys)
    return {"n": len(keys), "flips": flips, "rate": flips / len(keys) if keys else float("nan")}
