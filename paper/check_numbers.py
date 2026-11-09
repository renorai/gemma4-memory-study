"""Check every number in paper/WRITEUP.md against its sources, and count the words.

    python paper/check_numbers.py [--verbose]

Sources, in order:
  1. results/analysis.json (every value, at the roundings the text uses: as is, x100 for points and
     percent, 0-3 decimals, half away from zero);
  2. results/run_rows.jsonl through notebook/paper_figures.py:delivery (the delivery counts only);
  3. DESIGN.md (every number written in it, exactly as written).
A number that matches none of them fails. Exempt: arXiv ids, section/table/figure numbers and the
year in "Gemma 4". Small integers (<= 10) match almost anything, so they are listed separately as
weak matches; the rest are strong. Exit 1 on any failure or on more than 3,000 words.
"""
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebook"))
from paper_figures import delivery  # noqa: E402

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")


def body():
    return re.sub(r"<!--.*?-->", "", (ROOT / "paper" / "WRITEUP.md").read_text(), flags=re.S)


def norm(tok):
    t = tok.replace(",", "").lstrip("+")
    return t[1:] if t.startswith("-") else t


def renderings(v):
    out = set()
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return out
    if isinstance(v, int):
        return {str(abs(v))}
    for x in (v, 100 * v):
        for q in ("1", "0.1", "0.01", "0.001"):
            out.add(str(abs(Decimal(repr(x)).quantize(Decimal(q), rounding=ROUND_HALF_UP))))
        # also the shortest form without trailing zeros (0.50 -> 0.5)
        for q in ("0.1", "0.01", "0.001"):
            s = str(abs(Decimal(repr(x)).quantize(Decimal(q), rounding=ROUND_HALF_UP)))
            out.add(s.rstrip("0").rstrip(".") if "." in s else s)
    return out


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from walk(v)
    else:
        yield o


def sources():
    analysis = json.loads((ROOT / "results" / "analysis.json").read_text())
    a = set()
    for v in walk(analysis):
        a |= renderings(v)
    rows = [json.loads(x) for x in (ROOT / "results" / "run_rows.jsonl").read_text().splitlines() if x.strip()]
    d = set()
    for v in walk(delivery(rows)):
        d |= renderings(v)
    design = {norm(t) for t in NUM.findall((ROOT / "DESIGN.md").read_text())}
    design |= {t.rstrip("0").rstrip(".") for t in design if "." in t}
    return [("analysis.json", a), ("run_rows.jsonl", d), ("DESIGN.md", design)]


def exempt(text, m):
    s = text[max(0, m.start() - 12):m.end() + 3]
    tok = m.group(0)
    if re.fullmatch(r"\d{4}\.\d{4,5}", tok):                              # arXiv id
        return True
    if re.search(r"(Gemma |Section |Table |Figure |section )" + re.escape(tok), s):
        return True
    if re.search(re.escape(tok) + r"\.\d ", text[m.start():m.end() + 4]):  # "2.1 Setting"
        return True
    line_start = text.rfind("\n", 0, m.start()) + 1
    prefix = text[line_start:m.start()]
    return bool(re.fullmatch(r"#+ |#+ \d+\.|\s*", prefix) and re.match(r"\d+\.", text[m.start():]))


def selftest(srcs):
    """The check must be able to fail: planted wrong numbers match no source."""
    for planted in ("17.9", "63.2", "3,967", "+11.4", "0.044"):
        hit = [name for name, s in srcs if norm(planted) in s]
        assert not hit, f"self-test: planted {planted} matched {hit}"


def main(verbose=False):
    text = body()
    words = len(re.findall(r"\S+", text))
    srcs = sources()
    selftest(srcs)
    strong, weak, bad = [], [], []
    for m in NUM.finditer(text):
        if exempt(text, m):
            continue
        n = norm(m.group(0))
        hit = next((name for name, s in srcs if n in s), None)
        line = text.count("\n", 0, m.start()) + 1
        if hit is None:
            bad.append((line, m.group(0)))
        elif "." not in n and float(n) <= 10:
            weak.append((line, m.group(0), hit))
        else:
            strong.append((line, m.group(0), hit))
    print(f"words: {words} (limit 3000, tables and captions included, HTML comments excluded)")
    print(f"numbers checked: {len(strong) + len(weak) + len(bad)}; matched {len(strong)} strong "
          f"+ {len(weak)} small integers; unmatched {len(bad)}")
    by = {}
    for _, _, h in strong + weak:
        by[h] = by.get(h, 0) + 1
    print("matched by source:", ", ".join(f"{k} {v}" for k, v in by.items()))
    if verbose:
        for line, tok, h in strong:
            print(f"  ok  line {line}: {tok}  <- {h}")
    for line, tok in bad:
        print(f"  UNMATCHED line {line}: {tok}")
    return 0 if not bad and words <= 3000 else 1


if __name__ == "__main__":
    sys.exit(main("--verbose" in sys.argv))
