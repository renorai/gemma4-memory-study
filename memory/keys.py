"""Transferable anchor keys: what a node-anchored lesson is keyed by (DESIGN.md section 4).

A raw node id such as `httpx._client.Client.send` will not exist in a private repo. These keys
are things that *recur across repositories*, derived from the source of one symbol:

| type   | example                     | why it can recur in an unseen repo                          |
|--------|-----------------------------|-------------------------------------------------------------|
| api    | api:ssl.create_default_context | the same third-party or stdlib API is called             |
| base   | base:enum.Enum              | the same external base class                                |
| deco   | deco:pytest.fixture         | the same external decorator (internal ones by leaf name)    |
| exc    | exc:ValueError              | the same exception is raised                                |
| name   | name:send                   | the same leaf name (`send`, `__init__`, `parse_header`)     |
| tok    | tok:header                  | a shared identifier token of the leaf or its class          |
| motif  | motif:method/in1/out2-4     | the same kind and call-graph fan-in/fan-out bucket          |
| kind   | kind:method                 | the symbol kind                                             |

Strong keys (api, base, deco, exc, name) can make a lesson eligible; weak keys (tok, motif,
kind) only add to the score of a lesson that is already eligible. This stops every method in
the pool from matching every method in the task.

Python >= 3.8, standard library only: this file ships inside the submission's skill.
"""
from __future__ import annotations

import re

WEIGHTS = {"api": 1.0, "base": 0.8, "deco": 0.8, "exc": 0.6, "name": 1.0,
           "tok": 0.4, "motif": 0.3, "kind": 0.1}
STRONG = frozenset({"api", "base", "deco", "exc", "name"})
MAX_KEYS = 40

# Too common to identify anything on their own.
STOP_API = frozenset({"builtins.isinstance", "builtins.len", "builtins.str", "builtins.int",
                      "builtins.list", "builtins.dict", "builtins.super", "builtins.getattr",
                      "builtins.setattr", "builtins.hasattr", "builtins.print", "builtins.tuple",
                      "builtins.set", "builtins.bool", "builtins.range", "builtins.sorted",
                      "builtins.enumerate", "builtins.zip", "builtins.type", "builtins.repr",
                      "builtins.iter", "builtins.next", "builtins.min", "builtins.max",
                      "builtins.any", "builtins.all", "builtins.map", "builtins.filter",
                      "builtins.object", "builtins.property"})
STOP_TOK = frozenset({"get", "set", "self", "cls", "init", "the", "and", "for", "new", "obj",
                      "value", "data", "func", "test", "args", "kwargs", "none", "call"})


def key_type(key):
    return key.split(":", 1)[0]


def split_tokens(name):
    out = []
    for part in re.split(r"_+", name):
        out += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)
    return [t.lower() for t in out if len(t) >= 3 and t.lower() not in STOP_TOK]


def _bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2-4" if n <= 4 else "5+"


def _ext_key(dotted_name):
    """Normalise an external dotted name: drop builtins. prefix for exceptions; keep up to 3 parts."""
    parts = dotted_name.split(".")
    return ".".join(parts[:3])


def keys_for(index, qualname):
    """Sorted transferable keys of one symbol in a RepoIndex. Empty if unknown."""
    sym = index.symbols.get(qualname)
    if sym is None:
        return []
    keys = set()
    leaf = sym.leaf
    keys.add("name:" + leaf.lower())
    keys.add("kind:" + sym.kind)
    for t in split_tokens(leaf) + (split_tokens(sym.cls) if sym.cls else []):
        keys.add("tok:" + t)
    for ref in sym.calls:
        r = index.resolve_ref(sym, ref)
        if r and r[0] == "ext":
            k = _ext_key(r[1])
            if k not in STOP_API and not (k.startswith("builtins.") and k[9:10].isupper()):
                keys.add("api:" + k)
    for d in sym.decorators:
        r = index.resolve_ref(sym, d)
        if r and r[0] == "ext":
            keys.add("deco:" + _ext_key(r[1]).replace("builtins.", ""))
        else:
            keys.add("deco:" + d.rsplit(".", 1)[-1])
    for b in sym.bases:
        r = index.resolve_ref(sym, b)
        if r and r[0] == "ext":
            keys.add("base:" + _ext_key(r[1]).replace("builtins.", ""))
    for e in sym.raises:
        if not e.rsplit(".", 1)[-1][:1].isupper():
            continue  # `raise exc` re-raises a variable: not a recurring property
        r = index.resolve_ref(sym, e)
        if r and r[0] == "ext":
            keys.add("exc:" + r[1].rsplit(".", 1)[-1])
        elif r is None:
            keys.add("exc:" + e.rsplit(".", 1)[-1])
    fan_in, fan_out = len(index.callers(qualname)), len(index.callees(qualname))
    keys.add(f"motif:{sym.kind}/in{_bucket(fan_in)}/out{_bucket(fan_out)}")
    if sym.has_yield:
        keys.add("motif:generator")
    ordered = sorted(keys, key=lambda k: (-WEIGHTS[key_type(k)], k))
    return sorted(ordered[:MAX_KEYS])


def has_strong(keys):
    return any(key_type(k) in STRONG for k in keys)


_IDENT = re.compile(r"`([^`\n]{2,120})`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b"
                    r"|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"
                    r"|\b([A-Za-z_][A-Za-z0-9_]*)\(\)")


def mentioned_symbols(index, text, limit=5):
    """Repo symbols named in free text (backticks, dotted names, CamelCase, snake_case), in order
    of first mention; a bare word counts only when written as a call, `name()`. Only exact and suffix matches count, so prose words do not seed anything."""
    out = []
    for m in _IDENT.finditer(text or ""):
        cand = next(g for g in m.groups() if g)
        cand = cand.strip().rstrip("()").split("(")[0]
        hits = []
        if cand in index.symbols:
            hits = [cand]
        else:
            suf = sorted(q for q in index.symbols if q.endswith("." + cand))
            if 0 < len(suf) <= 3:
                hits = suf
        for h in hits:
            if h not in out:
                out.append(h)
        if len(out) >= limit:
            break
    return out[:limit]
