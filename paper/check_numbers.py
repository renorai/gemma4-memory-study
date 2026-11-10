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
from paper_figures import delivery, extension_context  # noqa: E402

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
    for v in walk([delivery(rows), extension_context(rows)]):
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


# Headline claims bound to the specific analysis.json key they report. The loose check above only asks
# that a number exists somewhere in the sources, so a stale value that also appears elsewhere passes
# (that is how -2.1 survived as the sensitivity bound: it is also the primary interval's lower bound).
# Each claim: (anchor text that must appear in WRITEUP.md, [(key path, scale, decimals, signed)]).
# Every rendered value must appear inside the anchor's line; the anchor is the line containing it.
P = "primary_resolve_run1"
E = "extension_primary_resolve_run1"
C = "combined_secondary_resolve_run1"
S = "sensitivity_resolve_all_available_runs"


def _set(prefix, scaled=True):
    return [(prefix + ["x_resolved"], 1, 0, False), (prefix + ["y_resolved"], 1, 0, False),
            (prefix + ["x_only"], 1, 0, False), (prefix + ["y_only"], 1, 0, False),
            (prefix + ["diff"], 100, 1, True), (prefix + ["newcombe95", 0], 100, 1, True),
            (prefix + ["newcombe95", 1], 100, 1, True), (prefix + ["mcnemar_p"], 1, 3, False)]


CLAIMS = [
    ("| Pooled, 48 (primary)", _set([P, "pooled"])),
    ("| Modal H100, 16", _set([P, "modal"])),
    ("| Kaggle 4x L4, 32", _set([P, "kaggle"])),
    ("| AWS L40S, 50 (extension primary)", _set([E])),
    ("| Combined, 98 (secondary)", _set([C])),
    ("per-task mean over all available runs", [([S, "pooled", "mean_diff"], 100, 1, True),
                                                ([S, "pooled", "bootstrap95", 0], 100, 1, True),
                                                ([S, "pooled", "bootstrap95", 1], 100, 1, True)]),
    ("backend-stratified bootstrap gives", [([P, "pooled", "bootstrap95", 0], 100, 1, False),
                                               ([P, "pooled", "bootstrap95", 1], 100, 1, True),
                                               ([E, "bootstrap95", 0], 100, 1, True),
                                               ([E, "bootstrap95", 1], 100, 1, False),
                                               ([C, "bootstrap95", 0], 100, 1, True),
                                               ([C, "bootstrap95", 1], 100, 1, True)]),
]


def _rend(v, scale, dec, signed):
    x = Decimal(repr(v)) * scale
    s = str(x.quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))
    if signed and not s.startswith("-"):
        s = "+" + s
    return s


def check_claims():
    """Return a list of failures; every claim's values must be rendered, from its own key, in its line."""
    analysis = json.loads((ROOT / "results" / "analysis.json").read_text())
    lines = body().splitlines()
    bad = []
    for anchor, items in CLAIMS:
        found = [i for i, l in enumerate(lines) if anchor in l]
        if not found:
            bad.append(f"claim anchor not found: {anchor!r}")
            continue
        # a sentence can wrap, so the anchor's paragraph (up to a blank line) is the searched text
        i = found[0]
        j = i
        while j + 1 < len(lines) and lines[j + 1].strip() and not lines[j + 1].startswith("|"):
            j += 1
        k = i
        while k > 0 and lines[k - 1].strip() and not lines[k - 1].startswith("|"):
            k -= 1
        para = " ".join(lines[k:j + 1]) if not lines[i].startswith("|") else lines[i]
        flat = para.replace(",", "")
        for path, scale, dec, signed in items:
            v = analysis
            for key in path:
                v = v[key]
            want = _rend(v, scale, dec, signed)
            cand = {want, want.replace("+", "") if signed else want}
            if dec == 3:    # p-values are written at 2 or 3 decimals
                cand |= {want.rstrip("0").rstrip("."), _rend(v, scale, 2, False)}
                cand |= {c.rstrip("0").rstrip(".") if "." in c else c for c in list(cand)}
            if not any(c in flat for c in cand):
                bad.append(f"{anchor!r}: {'.'.join(map(str, path))} = {want} not in its line")
    return bad


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
    claims = check_claims()
    print(f"headline claims bound to analysis.json keys: {len(CLAIMS)} checked; mismatches {len(claims)}")
    for c in claims:
        print("  CLAIM MISMATCH:", c)
    return 0 if not bad and not claims and words <= 3000 else 1


if __name__ == "__main__":
    sys.exit(main("--verbose" in sys.argv))
