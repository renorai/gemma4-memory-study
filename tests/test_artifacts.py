"""Checks on the non-code deliverables: preregistration freeze, kernels, notebook, paper."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PreregTests(unittest.TestCase):
    def test_prereg_is_frozen(self):
        digest = hashlib.sha256((ROOT / "PREREG.md").read_bytes()).hexdigest()
        self.assertEqual(digest, (ROOT / "PREREG.sha256").read_text().strip(),
                         "PREREG.md changed: log a deviation in DESIGN.md section 10 instead of editing it")

    def test_prereg_parameters_match_code(self):
        from memory import RetrievalConfig
        from memory.keys import STRONG, WEIGHTS
        from runner.build_submission import SHUFFLE_SEED
        from stats.analyze import N_BOOT, SEED
        text = (ROOT / "PREREG.md").read_text()
        c = RetrievalConfig()
        for k in ("k", "per_task_cap", "n_seeds", "hops", "hop_decay", "max_region",
                  "bm25_k1", "bm25_b"):
            self.assertIn(f"{k} = {getattr(c, k)}", text, k)
        # char_budget = 1600 was replaced by Renor's token budget before any memory-arm result:
        # the change is a logged deviation (DESIGN.md section 10), never a silent edit.
        self.assertIn("char_budget = 1600", text)
        self.assertTrue(f"char_budget = 1600 is replaced by token_budget = {c.token_budget}"
                        in (ROOT / "DESIGN.md").read_text(), "DESIGN.md section 10 must log the budget change")
        for t, w in WEIGHTS.items():
            self.assertIn(f"{t} {w}", text, t)
        self.assertIn("{" + ", ".join(sorted(STRONG, key=["api", "base", "deco", "exc", "name"].index)) + "}", text)
        self.assertEqual((SEED, SHUFFLE_SEED, N_BOOT), (20261107, 20261107, 10_000))
        ev = (ROOT / "submission" / "eval_config.yaml").read_text()
        self.assertIn("max_tool_calls: 100", ev)
        self.assertIn("max_time_minutes: 60", ev)
        self.assertIn("thinking_budget: 4096", (ROOT / "submission" / "agent.yaml").read_text())


class KernelTests(unittest.TestCase):
    def test_metadata_and_offline(self):
        for k in ("throughput", "arm_run"):
            meta = json.loads((ROOT / "kernels" / k / "kernel-metadata.json").read_text())
            self.assertTrue((ROOT / "kernels" / k / meta["code_file"]).is_file())
            self.assertFalse(meta["enable_internet"])
            self.assertTrue(meta["enable_gpu"])
            self.assertIn("gemma-4-developer-agent", meta["competition_sources"])
            self.assertEqual(meta["kernel_type"], "script")
            compile((ROOT / "kernels" / k / meta["code_file"]).read_text(), k, "exec")
        for p in (ROOT / "kernels").glob("**/*.py"):
            for url in re.findall(r"https?://[^\s\"']+", p.read_text()):
                self.assertRegex(url, r"^http://(127\.0\.0\.1|localhost)", (p.name, url))

    def test_throughput_kernel_reports_what_was_asked(self):
        src = (ROOT / "kernels" / "throughput" / "throughput.py").read_text()
        for needle in ("gemma-4-31b-it-qat-w4a16-ct", "CONCURRENCY = (8, 16)", "N_TASKS = 16", "tasks_per_hour",
                       "agent_minutes_median", "model_calls_median", "cpu_info"):
            self.assertIn(needle, src)
        self.assertIn('sandbox="subprocess"', (ROOT / "kernels" / "common.py").read_text())

    def test_configure_fills_placeholders(self):
        from kernels.configure import configure
        with tempfile.TemporaryDirectory() as d:
            out = configure("someone", "google/gemma-4/x/y/1", "someone/wheels", Path(d) / "k")
            for k in ("throughput", "arm_run"):
                meta = json.loads((out / k / "kernel-metadata.json").read_text())
                self.assertTrue(meta["id"].startswith("someone/"))
                self.assertIn("someone/wheels", meta["dataset_sources"])
                self.assertEqual(meta["model_sources"], ["google/gemma-4/x/y/1"])
            self.assertTrue((out / "code" / "memory" / "keys.py").is_file())
            self.assertTrue((out / "code" / "kernels" / "common.py").is_file())
            self.assertFalse(list(out.rglob("__pycache__")))


class NotebookTests(unittest.TestCase):
    NB = ROOT / "notebook" / "gemma4_graph_memory.ipynb"

    def code(self):
        nb = json.loads(self.NB.read_text())
        self.assertEqual(nb["nbformat"], 4)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                self.assertEqual(c.get("outputs"), [])
        return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    def test_valid_offline_and_current(self):
        src = self.code()
        for banned in ("pip install", "requests.get", "urllib.request", "http://", "https://",
                       "generativelanguage", "ollama"):
            self.assertNotIn(banned, src, banned)
        r = subprocess.run([sys.executable, str(ROOT / "notebook" / "make_notebook.py")], capture_output=True)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(self.code(), src)    # the committed notebook is the generator's output

    def test_executes_end_to_end_on_the_fake_harness(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "nb.py"
            script.write_text(self.code())
            r = subprocess.run([sys.executable, str(script)], cwd=ROOT / "notebook",
                               env={"NB_OUT": str(Path(d) / "out"), "PATH": "/usr/bin:/bin"},
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertIn("fake harness (NOT results)", r.stdout)
            self.assertIn("node-none", r.stdout)
            for f in ("analysis.json", "rows-all.jsonl", "tables.md", "moderators.json"):
                self.assertTrue((Path(d) / "out" / f).exists(), f)


class PaperTests(unittest.TestCase):
    def body(self):
        return re.sub(r"<!--.*?-->", "", (ROOT / "paper" / "WRITEUP.md").read_text(), flags=re.S)

    def test_writeup_word_limit_and_sections(self):
        body = self.body()
        self.assertLessEqual(len(re.findall(r"\S+", body)), 3000)   # tables and captions count
        for h in ("## Abstract", "## 1. Introduction", "## 2. Method", "## 3. Experiments", "## 4. Related work"):
            self.assertIn(h, body)
        self.assertNotIn("state of the art", body.lower())
        self.assertIn("VibeMemBench", body)

    def test_no_invented_results(self):
        text = (ROOT / "paper" / "WRITEUP.md").read_text()
        tables = re.findall(r"<!-- RESULTS-TABLE -->\n(.*?)\n\n", text, flags=re.S)
        self.assertGreaterEqual(len(tables), 5)
        for table in tables:
            for line in table.splitlines()[2:]:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                self.assertTrue(all(c == "TODO" for c in cells[1:]), line)
        self.assertIn("\\TODO", (ROOT / "paper" / "arxiv" / "main.tex").read_text())

    def test_bib_entries_are_verifiable_and_cited_keys_exist(self):
        bib = (ROOT / "paper" / "arxiv" / "refs.bib").read_text()
        entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, flags=re.S)
        self.assertGreaterEqual(len(entries), 25)
        for kind, key, body in entries:
            if kind == "misc":
                self.assertRegex(body, r"url\s*=\s*\{https://", key)
            else:
                self.assertRegex(body, r"eprint\s*=\s*\{\d{4}\.\d{4,5}\}", key)
        keys = {k for _, k, _ in entries}
        tex = (ROOT / "paper" / "arxiv" / "main.tex").read_text()
        for cited in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
            for k in cited.split(","):
                self.assertIn(k.strip(), keys)
        related = (ROOT / "RELATED.md").read_text()
        for _kind, _key, body in entries:
            m = re.search(r"eprint\s*=\s*\{([\d.]+)\}", body)
            if m:
                self.assertIn(m.group(1), related, f"{m.group(1)} cited but not in RELATED.md")


if __name__ == "__main__":
    unittest.main()
