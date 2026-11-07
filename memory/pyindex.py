"""A small static index of a Python repository: symbols, imports and a name-resolved call graph.

Why it exists: the hidden test repos are private, so a lesson cannot be keyed by a node id
learned on fastapi, rich, requests or httpx. The skill script must derive *transferable* keys
(memory/keys.py) for the code the agent is looking at, inside the task sandbox, with no model
call and no harness tool (a skill script cannot call get_code_neighbors). This index gives it
the symbols and k-hop neighbours from the repository source with the standard library only.

It is deliberately simple. Calls are resolved by name: `self.m()` to a method of the same
class, `f()` to a definition in the same module or one imported from the repo, and otherwise
to a repo definition with that leaf name when there is exactly one. That is coarser than the
harness's AST call graph, and DESIGN.md section 5 says so.

Symbol lookup follows the harness's order: exact, then suffix, then case-insensitive, then
substring.

Python >= 3.8, standard library only: this file ships inside the submission's skill.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict

SKIP_DIRS = frozenset({".git", ".hg", ".tox", ".nox", ".venv", "venv", "env", "node_modules",
                       "build", "dist", "__pycache__", "site-packages", ".mypy_cache", ".eggs"})
MAX_FILES = 4000
MAX_FILE_BYTES = 1_000_000


def dotted(node):
    """`a.b.c` for a Name/Attribute chain, else None. Calls in the chain are skipped: `a().b` -> None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class Symbol:
    __slots__ = ("qualname", "module", "path", "lineno", "end_lineno", "kind", "cls", "decorators",
                 "bases", "calls", "raises", "has_yield", "n_params", "is_test")

    def __init__(self, qualname, module, path, lineno, end_lineno, kind, cls=None):
        self.qualname, self.module, self.path = qualname, module, path
        self.lineno, self.end_lineno, self.kind, self.cls = lineno, end_lineno, kind, cls
        self.decorators, self.bases, self.calls, self.raises = [], [], [], []
        self.has_yield, self.n_params, self.is_test = False, 0, False

    @property
    def leaf(self):
        return self.qualname.rsplit(".", 1)[-1]


def module_name(rel_path):
    p = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    parts = [x for x in p.replace(os.sep, "/").split("/") if x]
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Collector(ast.NodeVisitor):
    def __init__(self, index, module, rel_path, is_pkg):
        self.ix, self.module, self.path, self.is_pkg = index, module, rel_path, is_pkg
        self.imports = {}            # local alias -> dotted target
        self.stack = []              # enclosing class names
        self.current = None

    # imports -----------------------------------------------------------
    def visit_Import(self, node):
        for a in node.names:
            if a.asname:
                self.imports[a.asname] = a.name
            else:
                self.imports[a.name.split(".")[0]] = a.name.split(".")[0]

    def visit_ImportFrom(self, node):
        base = node.module or ""
        if node.level:
            pkg = self.module.split(".") if self.module else []
            if not self.is_pkg:
                pkg = pkg[:-1]
            up = node.level - 1
            pkg = pkg[: len(pkg) - up] if up else pkg
            base = ".".join([p for p in pkg + ([base] if base else []) if p])
        for a in node.names:
            if a.name == "*":
                continue
            self.imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name

    # definitions -------------------------------------------------------
    def _def(self, node, kind):
        if self.current is not None and self.current.kind != "class":
            return  # function-local definitions are not indexed
        parent = ".".join([self.module] + self.stack) if self.module else ".".join(self.stack)
        qual = f"{parent}.{node.name}" if parent else node.name
        cls = self.stack[-1] if self.stack else None
        if kind == "function" and cls is not None:
            kind = "method"
        sym = Symbol(qual, self.module, self.path, node.lineno, getattr(node, "end_lineno", node.lineno),
                     kind, cls)
        sym.decorators = [d for d in (dotted(x.func if isinstance(x, ast.Call) else x)
                                      for x in node.decorator_list) if d]
        sym.is_test = node.name.startswith("test") or "/tests/" in "/" + self.path or \
            os.path.basename(self.path).startswith("test_")
        if isinstance(node, ast.ClassDef):
            sym.bases = [b for b in (dotted(x) for x in node.bases) if b]
        else:
            a = node.args
            sym.n_params = len(a.args) + len(a.kwonlyargs) + len(getattr(a, "posonlyargs", []))
        self.ix._add(sym, self)
        prev, self.current = self.current, sym
        if isinstance(node, ast.ClassDef):
            self.stack.append(node.name)
            for child in node.body:
                self.visit(child)
            self.stack.pop()
        else:
            for child in node.body:
                self._scan_body(child, sym)
        self.current = prev

    def _scan_body(self, node, sym):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                d = dotted(n.func)
                if d:
                    sym.calls.append(d)
            elif isinstance(n, ast.Raise) and n.exc is not None:
                d = dotted(n.exc.func if isinstance(n.exc, ast.Call) else n.exc)
                if d:
                    sym.raises.append(d)
            elif isinstance(n, (ast.Yield, ast.YieldFrom)):
                sym.has_yield = True

    def visit_FunctionDef(self, node):
        self._def(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self._def(node, "async")

    def visit_ClassDef(self, node):
        self._def(node, "class")


class RepoIndex:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.symbols = {}                       # qualname -> Symbol
        self.imports_of = {}                    # module -> {alias: target}
        self.modules = set()
        self.by_leaf = defaultdict(list)
        self._callees = None
        self._callers = None
        self.n_files = 0
        self.truncated = False

    # build ---------------------------------------------------------------
    @classmethod
    def build(cls, root, max_files=MAX_FILES):
        ix = cls(root)
        files = []
        for dirpath, dirnames, filenames in os.walk(ix.root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
            for f in sorted(filenames):
                if f.endswith(".py"):
                    files.append(os.path.join(dirpath, f))
        if len(files) > max_files:
            files, ix.truncated = files[:max_files], True
        for full in files:
            rel = os.path.relpath(full, ix.root).replace(os.sep, "/")
            ix.modules.add(module_name(rel))
        for full in files:
            ix._parse(full)
        return ix

    def _parse(self, full):
        try:
            if os.path.getsize(full) > MAX_FILE_BYTES:
                return
            with open(full, "rb") as f:
                tree = ast.parse(f.read(), filename=full)
        except (SyntaxError, ValueError, OSError):
            return
        rel = os.path.relpath(full, self.root).replace(os.sep, "/")
        mod = module_name(rel)
        col = _Collector(self, mod, rel, rel.endswith("__init__.py"))
        for node in tree.body:
            col.visit(node)
        self.imports_of[mod] = col.imports
        self.n_files += 1

    def _add(self, sym, _col):
        self.symbols[sym.qualname] = sym
        self.by_leaf[sym.leaf].append(sym.qualname)

    # name resolution -------------------------------------------------------
    def is_internal(self, dotted_name):
        parts = dotted_name.split(".")
        return any(".".join(parts[:i]) in self.modules for i in range(len(parts), 0, -1))

    def resolve_ref(self, sym, ref):
        """A call/decorator/base reference inside `sym` -> ("int", qualname) | ("ext", dotted) | None."""
        imports = self.imports_of.get(sym.module, {})
        head, _, rest = ref.partition(".")
        if head in ("self", "cls") and sym.cls and rest and "." not in rest:
            q = f"{sym.module}.{sym.cls}.{rest}" if sym.module else f"{sym.cls}.{rest}"
            if q in self.symbols:
                return ("int", q)
            hits = self.by_leaf.get(rest, [])
            return ("int", hits[0]) if len(hits) == 1 else None
        if head in imports:
            target = imports[head] + ("." + rest if rest else "")
            if self.is_internal(target):
                if target in self.symbols:
                    return ("int", target)
                hits = self.by_leaf.get(target.rsplit(".", 1)[-1], [])
                return ("int", hits[0]) if len(hits) == 1 else None
            return ("ext", target)
        if not rest:
            q = f"{sym.module}.{head}" if sym.module else head
            if q in self.symbols:
                return ("int", q)
            if head in BUILTIN_NAMES:
                return ("ext", "builtins." + head)
        leaf = ref.rsplit(".", 1)[-1]
        hits = self.by_leaf.get(leaf, [])
        if len(hits) == 1 and hits[0] != sym.qualname:
            return ("int", hits[0])
        return None

    def _graph(self):
        if self._callees is None:
            callees, callers = defaultdict(set), defaultdict(set)
            for q, s in self.symbols.items():
                for ref in s.calls:
                    r = self.resolve_ref(s, ref)
                    if r and r[0] == "int" and r[1] != q:
                        callees[q].add(r[1])
                        callers[r[1]].add(q)
                if s.cls:  # a method is adjacent to its class (contains edge)
                    owner = q.rsplit(".", 1)[0]
                    if owner in self.symbols:
                        callees[owner].add(q)
                        callers[q].add(owner)
            self._callees, self._callers = callees, callers
        return self._callees, self._callers

    def callees(self, q):
        return sorted(self._graph()[0].get(q, ()))

    def callers(self, q):
        return sorted(self._graph()[1].get(q, ()))

    def neighbors(self, q):
        return sorted(set(self.callees(q)) | set(self.callers(q)))

    def khop(self, q, k):
        """{qualname: hops} within k hops (callers and callees), the start node at 0."""
        seen, frontier = {q: 0}, [q]
        for h in range(1, k + 1):
            nxt = []
            for n in frontier:
                for m in self.neighbors(n):
                    if m not in seen:
                        seen[m] = h
                        nxt.append(m)
            frontier = sorted(nxt)
        return seen

    def lookup(self, name, limit=5):
        """Harness order: exact, then suffix, then case-insensitive, then substring."""
        name = (name or "").strip().strip("`'\"()")
        if not name:
            return []
        name = name.replace("::", ".").replace("/", ".")
        if name.endswith(".py"):
            name = name[:-3]
        if name in self.symbols:
            return [name]
        suf = sorted(q for q in self.symbols if q.endswith("." + name))
        if suf:
            return suf[:limit]
        low = name.lower()
        ci = sorted(q for q in self.symbols if q.lower() == low or q.lower().endswith("." + low))
        if ci:
            return ci[:limit]
        if len(name) < 4:
            return []
        sub = sorted(q for q in self.symbols if low in q.lower())
        return sub[:limit]

    def symbols_at(self, rel_path, lines):
        """Innermost symbols enclosing any of `lines` in file `rel_path` (for patch hunks)."""
        out = []
        cands = [s for s in self.symbols.values() if s.path == rel_path]
        for ln in lines:
            enc = [s for s in cands if s.lineno <= ln <= s.end_lineno]
            if enc:
                best = max(enc, key=lambda s: (s.lineno, -s.end_lineno))
                if best.qualname not in out:
                    out.append(best.qualname)
        return out


BUILTIN_NAMES = frozenset("""
abs all any bool bytes callable chr dict dir divmod enumerate filter float format frozenset getattr
hasattr hash hex id int isinstance issubclass iter len list map max min next object open ord pow
print property range repr reversed round set setattr slice sorted str sum super tuple type vars zip
ValueError TypeError KeyError IndexError AttributeError RuntimeError NotImplementedError
StopIteration Exception BaseException OSError IOError UnicodeDecodeError UnicodeEncodeError
AssertionError ImportError LookupError ArithmeticError ZeroDivisionError TimeoutError
ConnectionError PermissionError FileNotFoundError
""".split())
