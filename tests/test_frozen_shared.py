"""The code both arms share is frozen while the `none` baseline runs: this test fails when any of it
differs from the frozen baseline commit.

Memory-arm work (memory/, submission/skills/graph_memory/, the lesson store format, the
memory-only prompt part runner/prompt_parts/memory.md) may change; nothing below may. The
baseline commit is 65f28b4 (9 Nov 2026), the last commit that changed shared code: the
memory-arm-only notes sub-agent and explorer output_key in runner/build_submission.py and
runner/validate.py (automatic recall, DESIGN.md section 10); tests/test_none_build_identity.py proves the `none` build is
byte-identical to the earlier baselines (824dc84, ff9bf75d); set
FROZEN_BASELINE_COMMIT to compare against another one. Skipped, never passed, when git or that
commit is not available (a shallow clone, a Kaggle dataset copy).
"""
import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = os.environ.get("FROZEN_BASELINE_COMMIT", "65f28b4012a587857a622ec5ae21464a841c110b")

# Shared by both arms: runner, harness adapter, kernels/common.py, eval_runner.py, the prompts and
# agent config both arms ship, the budget (eval_config.yaml, runner/budget.py), the scoring
# (harness outcome readers) and the vLLM flags (kernels/common.py).
FROZEN = ("harness", "runner", "kernels/common.py", "kernels/eval_runner.py", "submission/agent.yaml",
          "submission/sub_agents", "submission/prompts", "submission/eval_config.yaml",
          # the test environment, task validation, arm runner and analysis (DESIGN.md section 10)
          "kernels/scorer_sandbox.py", "kernels/envcheck", "kernels/arm_run/arm_run.py", "stats",
          "pipeline/experiment.py")
MEMORY_ONLY = ("runner/prompt_parts/memory.md",)


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


class FrozenSharedPaths(unittest.TestCase):
    def test_shared_paths_match_the_frozen_baseline(self):
        if shutil.which("git") is None or _git("rev-parse", "--show-toplevel").returncode != 0:
            self.skipTest("not a git checkout")
        if _git("cat-file", "-e", BASELINE + "^{commit}").returncode != 0:
            self.skipTest(f"baseline commit {BASELINE} is not in this clone")
        top = _git("rev-parse", "--show-toplevel").stdout.strip()
        prefix = _git("rev-parse", "--show-prefix").stdout.strip()
        paths = [prefix + p for p in FROZEN]
        # the working tree (committed or not) and untracked files against the baseline, all paths
        # relative to the repository root
        changed = _git("diff", "--name-only", BASELINE, "--", *paths, cwd=top).stdout.split()
        changed += _git("ls-files", "--others", "--exclude-standard", "--", *paths, cwd=top).stdout.split()
        changed = sorted({c for c in changed if c[len(prefix):] not in MEMORY_ONLY
                          and not c.endswith((".pyc",)) and "__pycache__" not in c})
        self.assertEqual(changed, [], f"shared code changed since the frozen baseline {BASELINE[:12]}")


if __name__ == "__main__":
    unittest.main()
