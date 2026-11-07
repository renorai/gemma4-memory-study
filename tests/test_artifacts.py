"""Checks on the non-code deliverables: agent config, submission zip, notebook, paper."""
import json
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yaml_instruction(text: str) -> str:
    lines = text.split("\n")
    i = lines.index("instruction: |") + 1
    out = []
    while i < len(lines) and (lines[i].startswith("  ") or lines[i] == ""):
        out.append(lines[i][2:])
        i += 1
    return "\n".join(out).rstrip("\n")


class AgentConfigTests(unittest.TestCase):
    def test_instruction_matches_system_prompt(self):
        y = (ROOT / "agent" / "agent.yaml").read_text()
        sysp = (ROOT / "agent" / "prompts" / "system.txt").read_text().rstrip("\n")
        self.assertEqual(yaml_instruction(y), sysp.replace("{{LESSONS}}", "{memory_lessons?}"))

    def test_memory_config_matches_defaults(self):
        from memory import RetrievalConfig
        cfg = json.loads((ROOT / "agent" / "memory_config.json").read_text())
        rc = RetrievalConfig(**cfg["retrieval"])
        self.assertEqual(rc, RetrievalConfig(**{k: getattr(rc, k) for k in cfg["retrieval"]}))
        defaults = RetrievalConfig()
        for k, v in cfg["retrieval"].items():
            self.assertEqual(getattr(defaults, k), v, k)   # shipped config == preregistered defaults


class SubmissionTests(unittest.TestCase):
    def test_zip_per_arm_and_tool_imports_from_zip(self):
        from runner.build_submission import build
        with tempfile.TemporaryDirectory() as d:
            lessons = Path(d) / "l.jsonl"
            lessons.write_text('{"lesson_id": "", "node_id": "alpha:auth.sign", "outcome": "success", '
                               '"repo": "alpha", "source_model": "gemma-4-e4b", "source_run": "", '
                               '"source_task": "t", "text": "x"}\n')
            g = build("graph", lessons, "gemma-4-e4b-it-qat-w4a16", Path(d) / "g.zip")
            n = build("none", None, "gemma-4-e4b-it-qat-w4a16", Path(d) / "n.zip")
            self.assertIn("skills/graph_memory/lessons.jsonl", g)
            self.assertNotIn("skills/graph_memory/lessons.jsonl", n)
            with zipfile.ZipFile(Path(d) / "n.zip") as z:
                y = z.read("agent.yaml").decode()
                self.assertNotIn("tool.recall_lessons", y)
                self.assertIn("model: gemma-4-e4b-it-qat-w4a16", y)
                self.assertEqual(json.loads(z.read("memory_config.json"))["arm"], "none")
            with zipfile.ZipFile(Path(d) / "g.zip") as z:
                z.extractall(Path(d) / "unz")
            code = ("import sys; sys.path.insert(0, 'skills/graph_memory'); import tool, memory; "
                    "assert memory.__file__.startswith(%r)" % str((Path(d) / "unz").resolve()))
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=Path(d) / "unz",
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with self.assertRaises(ValueError):
                build("graph", None, "gemma-4-e4b", Path(d) / "x.zip")
            with self.assertRaises(ValueError):
                build("graph", lessons, "llama-3", Path(d) / "x.zip")


class NotebookTests(unittest.TestCase):
    NB = ROOT / "notebook" / "gemma4_graph_memory.ipynb"

    def test_valid_and_offline(self):
        nb = json.loads(self.NB.read_text())
        self.assertEqual(nb["nbformat"], 4)
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                self.assertEqual(c.get("outputs"), [])        # shipped without stale outputs
        for banned in ("pip install", "requests.get", "urllib.request", "http://", "https://",
                       "generativelanguage", "ollama"):
            self.assertNotIn(banned, src, banned)
        compile(src.replace("%", "#"), "nb", "exec")


class PaperTests(unittest.TestCase):
    def test_writeup_word_limit_and_sections(self):
        text = (ROOT / "paper" / "WRITEUP.md").read_text()
        body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        words = len(re.findall(r"\S+", body))
        self.assertLessEqual(words, 3000, words)
        for h in ("## Abstract", "## 1. Introduction", "## 2. Method", "## 3. Experiments", "## 4. Related work"):
            self.assertIn(h, body)

    def test_no_invented_results(self):
        """Every cell of a results table is TODO until the notebook fills it."""
        text = (ROOT / "paper" / "WRITEUP.md").read_text()
        for table in re.findall(r"<!-- RESULTS-TABLE -->\n(.*?)\n\n", text, flags=re.S):
            for line in table.splitlines()[2:]:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                self.assertGreater(len(cells), 1, line)
                self.assertTrue(all(c == "TODO" for c in cells[1:]), line)
        tex = (ROOT / "paper" / "arxiv" / "main.tex").read_text()
        self.assertIn("\\TODO", tex)

    def test_bib_entries_have_arxiv_ids(self):
        bib = (ROOT / "paper" / "arxiv" / "refs.bib").read_text()
        entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S)
        self.assertGreaterEqual(len(entries), 9)
        for key, body in entries:
            self.assertRegex(body, r"eprint\s*=\s*\{\d{4}\.\d{4,5}\}", key)
        tex = (ROOT / "paper" / "arxiv" / "main.tex").read_text()
        keys = {k for k, _ in entries}
        for cited in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
            for k in cited.split(","):
                self.assertIn(k.strip(), keys)


if __name__ == "__main__":
    unittest.main()


class NotebookRunsOnFakeHarness(unittest.TestCase):
    def test_executes_end_to_end_offline(self):
        nb = json.loads((ROOT / "notebook" / "gemma4_graph_memory.ipynb").read_text())
        src = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "nb.py"
            script.write_text(src)
            env = {"NB_OUT": str(Path(d) / "out"), "PATH": "/usr/bin:/bin"}
            r = subprocess.run([sys.executable, str(script)], cwd=ROOT / "notebook", env=env,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertIn("H1 graph-flat", r.stdout)
            for f in ("analysis.json", "rows-all.jsonl", "tables.md", "budget_plan.json",
                      "probe_embeddings.json", "noise.json", "lessons-fake-policy.jsonl"):
                self.assertTrue((Path(d) / "out" / f).exists(), f)
