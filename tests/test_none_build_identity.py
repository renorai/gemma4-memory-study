"""The automatic-recall change (DESIGN.md section 10, 2026-11-09) touched two frozen files,
runner/build_submission.py and runner/validate.py, for the memory arms only. This test proves the
no-memory arm did not change: the `none` submission built by this tree is byte-identical to the one
built by the frozen baseline 824dc84 and by ff9bf75d (the pin of the Modal no-memory runs).

Each old tree is taken with `git archive` and builds in its own Python process. Skipped, never
passed, when git or a commit is not available (a shallow clone, a Kaggle dataset copy)."""
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from runner.build_submission import build

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct")
COMMITS = {"frozen baseline 824dc84": "824dc84b024c56f3482fcec6a60e61716fb3e5ed",
           "Modal no-memory pin ff9bf75d": "ff9bf75d7397e70377deafff0815df345c6405de"}


def tree_hash(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


class NoneBuildIsUnchanged(unittest.TestCase):
    def test_none_submission_is_byte_identical_to_the_frozen_builds(self):
        if shutil.which("git") is None or _git("rev-parse", "--show-toplevel").returncode != 0:
            self.skipTest("not a git checkout")
        prefix = _git("rev-parse", "--show-prefix").stdout.decode().strip().rstrip("/")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for label, commit in COMMITS.items():
                if _git("cat-file", "-e", commit + "^{commit}").returncode != 0:
                    self.skipTest(f"{label} is not in this clone")
                arch = _git("archive", "--format=tar", commit, ":/" + prefix)
                self.assertEqual(arch.returncode, 0, arch.stderr)
                src = d / commit[:8]
                src.mkdir()
                tar = src / "t.tar"
                tar.write_bytes(arch.stdout)
                with tarfile.open(tar) as t:
                    t.extractall(src, filter="data")
                # git archive from a subdirectory may write paths relative to it; find the tree root
                old_root = next(src.rglob("runner/build_submission.py")).parents[1]
                for alias in ALIASES:
                    old_out, new_out = d / f"old-{commit[:8]}-{alias}", d / f"new-{alias}"
                    r = subprocess.run(
                        [sys.executable, "-c",
                         "import sys; sys.path.insert(0, sys.argv[1]); "
                         "from runner.build_submission import build; build('none', None, sys.argv[2], sys.argv[3])",
                         str(old_root), alias, str(old_out)],
                        cwd=str(old_root), capture_output=True, text=True, timeout=120)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    build("none", None, alias, new_out)
                    self.assertGreaterEqual(len(tree_hash(new_out)), 4)        # agent, explorer, eval config, prompt
                    self.assertEqual(tree_hash(new_out), tree_hash(old_out), (label, alias))


if __name__ == "__main__":
    unittest.main()
