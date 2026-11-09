"""Old vs fixed read_trace on real traces: args, repeat metrics, outputs. argv: code_root old_module traces..."""
import importlib.util, json, sys
code, old_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, code)
import harness.swegemma as new
from stats.repeats import repeat_metrics
spec = importlib.util.spec_from_file_location("old_swegemma", old_path); old = importlib.util.module_from_spec(spec); sys.modules["old_swegemma"] = old; spec.loader.exec_module(old)
same_args = same_rep = 0; out_old = out_new = calls = 0
for f in files:
    a, b = old.read_trace(f), new.read_trace(f)
    ka = [(s["tool"], json.dumps(s["args"], sort_keys=True), s["turn"]) for s in a["steps"]]
    kb = [(s["tool"], json.dumps(s["args"], sort_keys=True), s["turn"]) for s in b["steps"]]
    same_args += ka == kb
    same_rep += repeat_metrics(a) == repeat_metrics(b)
    calls += len(b["steps"]); out_old += sum(1 for s in a["steps"] if s["output"]); out_new += sum(1 for s in b["steps"] if s["output"])
print(f"traces {len(files)}: identical tool/args/turn sequences {same_args}, identical repeat metrics {same_rep}; "
      f"tool calls {calls}, non-empty outputs old {out_old} -> fixed {out_new}")
