"""memory/plumbing.py on synthetic ATIF traces: which call a shared observation.content belongs to
(_owner, read_steps), and the `--page` exclusion that keeps the notes sub-agent's own page from
being counted twice. These feed `plumbing_failed`, which defines the per-protocol set."""
import json
import os
import tempfile
import unittest

from memory import plumbing as P

NOTES_FILE = "/tmp/t1/renor-notes/notes.jsonl"
STATUS = f"working notes: on (notes {NOTES_FILE}; recap every 6)"
PAGE = "where was I (3 notes):\n- [N:1] read models.py"


def call(name, args, cid):
    return {"function_name": name, "arguments": args, "tool_call_id": cid}


def skill(path, *argv, cid="c"):
    return call("run_skill_script", {"file_path": path, "args": list(argv)}, cid)


def step(author, calls=(), content=None, results=None, pt=0, message=""):
    s = {"source": "agent", "extra": {"author": author}, "tool_calls": list(calls),
         "metrics": {"prompt_tokens": pt}, "message": message}
    obs = {}
    if content is not None:
        obs["content"] = content
    if results is not None:
        obs["results"] = results
    if obs:
        s["observation"] = obs
    return s


def out(stdout):
    return json.dumps({"stdout": stdout, "stderr": ""})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def trace(self, steps):
        path = os.path.join(self.tmp.name, "trace.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"steps": steps}, f)
        return path


class Owner(Base):
    def test_shared_content_naming_a_file_path_goes_to_that_skill_call(self):
        calls = [call("notes", {"request": "x"}, "a"),
                 skill("graph-memory/scripts/workmem.py", "note", "x", cid="b")]
        content = json.dumps({"file_path": "graph-memory/scripts/workmem.py", "stdout": "noted #1"})
        self.assertIs(P._owner(calls, content), calls[1])
        steps, _ = P.read_steps(self.trace([step("coder", calls, content=content)]))
        self.assertEqual([(n, o) for _, n, _, o in steps], [("notes", ""), ("run_skill_script", content)])

    def test_file_path_in_string_arguments_and_quotes_still_match(self):
        calls = [call("explorer", {}, "a"),
                 call("run_skill_script", json.dumps({"file_path": '"s/recall.py"'}), "b")]
        self.assertIs(P._owner(calls, {"file_path": "s/recall.py"}), calls[1])

    def test_unmatched_or_unparseable_content_falls_back_to_the_first_call(self):
        calls = [call("notes", {}, "a"), skill("s/workmem.py", cid="b")]
        self.assertIs(P._owner(calls, "not json {"), calls[0])
        self.assertIs(P._owner(calls, json.dumps({"file_path": "s/other.py"})), calls[0])
        self.assertIs(P._owner(calls, json.dumps(["list"])), calls[0])
        self.assertIsNone(P._owner([], "x"))
        steps, _ = P.read_steps(self.trace([step("coder", calls, content="not json {")]))
        self.assertEqual([o for *_, o in steps], ["not json {", ""])

    def test_per_call_results_win_over_shared_content(self):
        calls = [call("notes", {}, "a"), skill("s/workmem.py", cid="b")]
        steps, _ = P.read_steps(self.trace([step(
            "coder", calls, content="shared", results=[{"source_call_id": "b", "content": "own"}])]))
        self.assertEqual([o for *_, o in steps], ["shared", "own"])


class PageExclusion(Base):
    def run_trace(self, wm_args, notes_answer=False):
        steps = [step("coder", [skill("s/recall.py", "--issue", "x", cid="r")], out(STATUS), pt=1000),
                 step("coder", [skill("s/workmem.py", *wm_args, cid="w")], out(PAGE), pt=2000)]
        if notes_answer:
            steps.append(step("notes", message=PAGE))
        return P.plumbing(self.trace(steps), "recap")

    def test_a_page_the_coder_asked_for_is_counted(self):
        row = self.run_trace(["recap"])
        self.assertEqual(row["skill_status"], "on")
        self.assertEqual(row["recaps"], 1)

    def test_a_page_note_is_not_counted_from_the_tool_output(self):
        self.assertEqual(self.run_trace(["note", "x", "--page"])["recaps"], 0)

    def test_the_notes_sub_agent_answer_is_counted_once(self):
        self.assertEqual(self.run_trace(["note", "x", "--page"], notes_answer=True)["recaps"], 1)


class Failed(Base):
    def test_compaction_with_nothing_injected_fails_and_an_earlier_page_passes(self):
        recall = step("coder", [skill("s/recall.py", cid="r")], out(STATUS), pt=9000)
        page = step("coder", [skill("s/workmem.py", "recap", cid="w")], out(PAGE), pt=9500)
        after = step("coder", pt=3000)                       # 9500 -> 3000: one compaction
        bare = P.plumbing(self.trace([recall, step("coder", pt=9500), after]), "recap")
        self.assertIs(bare["recall_runs_after_compaction"], False)
        self.assertTrue(bare["plumbing_failed"])
        ok = P.plumbing(self.trace([recall, page, after]), "recap")
        self.assertIs(ok["recall_runs_after_compaction"], True)
        self.assertFalse(ok["plumbing_failed"])
        self.assertIsNone(P.plumbing(self.trace([recall, page, after]), "none")["plumbing_failed"])

    def test_recall_never_run_fails(self):
        self.assertTrue(P.plumbing(self.trace([step("coder", pt=100)]), "recap")["plumbing_failed"])


if __name__ == "__main__":
    unittest.main()
