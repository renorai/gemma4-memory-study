"""A scorer-like `--sandbox subprocess` for Kaggle, where there is no Docker daemon.

What the scorer does (competition files docker/Dockerfile.sandbox and sandbox/setup.py, and
swegemma.harness.container_setup):
  - every task runs in a python:3.13-slim container with `pip install pytest pytest-timeout==2.1.0
    typer pdm-backend setuptools wheel poetry-core hatchling flit-core editables` and the
    imp.py / telnetlib.py shims in site-packages;
  - swegemma's Docker path then unpacks the competition wheels (highest version per package,
    py3.13-compatible, pytest/pluggy/pip/setuptools/wheel excluded) into that site-packages,
    pip-installs the repo in editable mode and runs sandbox/setup.py --fast-path.

What swegemma's SubprocessManager does instead: a venv of the HOST Python (3.12 on Kaggle) that
sees every Kaggle package, and no test-dependency setup at all (install_test_dependencies returns
early for it). Scores from that are not the scorer's.

DockerSandbox below is a SubprocessManager whose venv is a clean CPython 3.13 venv (no system
site-packages) holding its own copy of the Dockerfile's pip layer, and which offers the hooks
swegemma's Docker path looks for, so the harness's OWN setup code runs unchanged:
  extract_archive_to_container / _stream_tar_via_exec_socket  -> wheels unpacked into this venv
  '/usr/local/lib/python3.13/site-packages' in a command       -> this venv's site-packages
It is named DockerSandbox because swegemma picks its setup path and its strict JUnit check by type
name: 'SubprocessManager' skips both, 'DockerSandbox' is treated as a real sandbox.

Gaps filled (DESIGN.md section 10, 2026-11-08): the scorer reads a private wheel set; the public
competition wheels lack some test dependencies (typing_inspection for pydantic, dirty_equals, attrs,
the httpbin fixture, ...). After the harness's own setup, post_setup() then:
  1. adds a per-repo overlay of test dependencies resolved from PyPI (Python 3.13, released before
     IMAGE_DATE), from every snapshot's pyproject.toml and requirements*.txt (read with the
     competition's sandbox/setup.py functions) plus EXTRA_TEST_DEPS, ONLY for distribution names
     that have no competition wheel;
  2. removes from site-packages any top-level package the task's own workspace provides (and its
     dist-info), so tests import the repo under test, never an installed wheel of it;
  3. for rich, replaces Pygments by the newest release published on or before the snapshot's HEAD
     commit date (rich's styling tests encode the Pygments of their day).

build_env() makes these inputs once, with internet on (kernels/envcheck). load_env() unpacks
them from an attached input. patch_harness() makes swegemma's Evaluator use DockerSandbox and
runs post_setup() after every install_test_dependencies call.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

from swegemma.sandbox.subprocess import SubprocessManager

SP = "/usr/local/lib/python3.13/site-packages"
IMAGE_DATE = "2026-10-13T00:00:00Z"   # the competition's sandbox files are dated 2026-10-13
IMAGE_PKGS = ("pip", "pytest", "pytest-timeout==2.1.0", "typer", "pdm-backend", "setuptools", "wheel",
              "poetry-core", "hatchling", "flit-core", "editables")
ENV_ROOT = Path(os.environ.get("SWEGEMMA_ENV_ROOT", "/kaggle/tmp/scorer_env"))
EXTRA_TEST_DEPS = ("typing_inspection", "dirty_equals", "attrs", "pytest-httpbin", "httpbin")
REPO_KEYS = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests",
             "encode/httpx": "httpx"}
PYGMENTS_PIN_REPOS = ("rich",)


def _norm(name):
    import re
    return re.sub(r"[-_.]+", "_", name).lower()


def _remove_dist(sp, dist_info):
    """Remove one installed distribution (files listed in its RECORD, then the dist-info)."""
    rec = dist_info / "RECORD"
    if rec.exists():
        for line in rec.read_text(errors="ignore").splitlines():
            rel = line.split(",", 1)[0]
            f = (sp / rel).resolve()
            if rel and str(f).startswith(str(sp.resolve())) and f.is_file():
                f.unlink(missing_ok=True)
    shutil.rmtree(dist_info, ignore_errors=True)


def _workspace_packages(ws):
    out = set()
    for base in (ws, ws / "src"):
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and (d / "__init__.py").exists():
                    out.add(d.name)
    return out


def _venv_sp(venv_dir):
    return next(Path(venv_dir).glob("lib/python3.*/site-packages"))


class DockerSandbox(SubprocessManager):
    """SubprocessManager with a clean CPython 3.13 venv laid out like the scorer's container."""

    def __init__(self, timeout_seconds: int = 300, base_dir: Path | None = None,
                 system_site_packages: bool = False):
        super().__init__(timeout_seconds=timeout_seconds, base_dir=base_dir, system_site_packages=False)
        self.py313 = os.environ["SWEGEMMA_PY313"]
        self.image_sp = os.environ["SWEGEMMA_IMAGE_SP"]

    def start(self) -> str:
        sid = uuid.uuid4().hex[:12]
        root = Path(tempfile.mkdtemp(prefix=f"swegemma_sandbox_{sid}_",
                                     dir=str(self.base_dir) if self.base_dir else None))
        paths = {"root": root, "workspace": root / "workspace", "tmp": root / "tmp",
                 "wheels": root / "wheels", "venv": root / "venv"}
        for k in ("workspace", "tmp", "wheels"):
            paths[k].mkdir(parents=True, exist_ok=True)
        subprocess.run([self.py313, "-m", "venv", "--without-pip", str(paths["venv"])],
                       check=True, capture_output=True)
        shutil.copytree(self.image_sp, _venv_sp(paths["venv"]), dirs_exist_ok=True)
        bindir = paths["venv"] / "bin"
        for name, mod in (("pytest", "pytest"), ("py.test", "pytest"), ("pip", "pip"), ("pip3", "pip")):
            p = bindir / name
            p.write_text(f'#!/bin/sh\nexec "{bindir}/python3" -m {mod} "$@"\n')
            p.chmod(0o755)
        with self._lock:
            self._sandboxes[sid] = paths
            if self._default_sandbox_id is None:
                self._default_sandbox_id = sid
        return sid

    def _sp_of(self, sandbox_id):
        with self._lock:
            if sandbox_id not in self._sandboxes:
                raise ValueError(f"Unknown sandbox_id: {sandbox_id}")
            return _venv_sp(self._sandboxes[sandbox_id]["venv"])

    def exec(self, sandbox_id, command, *, timeout=None):
        return super().exec(sandbox_id, command.replace(SP, str(self._sp_of(sandbox_id))), timeout=timeout)

    def _host_dir(self, sandbox_id, dest_dir):
        d = str(dest_dir).rstrip("/")
        if d == SP:
            return self._sp_of(sandbox_id)
        with self._lock:
            paths = self._sandboxes[sandbox_id]
        if d == "/workspace":
            return paths["workspace"]
        if d == "/tmp":
            return paths["tmp"]
        raise ValueError(f"no sandbox mapping for {dest_dir}")

    def _stream_tar_via_exec_socket(self, container_id, fileobj, dest_dir):
        """swegemma streams the unpacked competition wheels (an uncompressed tar) here."""
        with tarfile.open(fileobj=fileobj, mode="r|*") as t:
            t.extractall(self._host_dir(container_id, dest_dir), filter="tar")
        return True

    def post_setup(self, sandbox_id, repo):
        """Run after each install_test_dependencies (idempotent); see the module docstring."""
        sp = self._sp_of(sandbox_id)
        with self._lock:
            ws = self._sandboxes[sandbox_id]["workspace"]
        key = REPO_KEYS.get(repo, repo.rsplit("/", 1)[-1].lower())
        overlay = Path(os.environ.get("SWEGEMMA_OVERLAY_DIR", "/nonexistent")) / key
        if overlay.is_dir() and not (sp / ".renor_overlay").exists():
            shutil.copytree(overlay, sp, dirs_exist_ok=True)
            (sp / ".renor_overlay").write_text(key + "\n")
        own = _workspace_packages(ws)
        for name in own:
            shutil.rmtree(sp / name, ignore_errors=True)
            (sp / f"{name}.py").unlink(missing_ok=True)
            for di in sp.glob("*.dist-info"):
                if _norm(di.name[:-len(".dist-info")].rsplit("-", 1)[0]) == _norm(name):
                    _remove_dist(sp, di)
        pyg_root = Path(os.environ.get("SWEGEMMA_OVERLAY_DIR", "/nonexistent")) / "_pygments"
        if key in PYGMENTS_PIN_REPOS and pyg_root.is_dir() and not (sp / ".renor_pygments").exists():
            when = subprocess.run(["git", "-C", str(ws), "log", "-1", "--format=%cI"],
                                  capture_output=True, text=True).stdout.strip()
            dates = json.loads((pyg_root / "dates.json").read_text())   # version -> ISO upload date
            from packaging.version import Version
            ok = [v for v, d in dates.items() if when and d[:10] <= when[:10] and (pyg_root / v).is_dir()]
            if ok:
                v = max(ok, key=Version)
                for di in sp.glob("[Pp]ygments-*.dist-info"):
                    _remove_dist(sp, di)
                shutil.rmtree(sp / "pygments", ignore_errors=True)
                shutil.copytree(pyg_root / v, sp, dirs_exist_ok=True)
                (sp / ".renor_pygments").write_text(f"{v} (HEAD {when})\n")

    def extract_archive_to_container(self, container_id, tar_path, dest_dir=SP):
        """swegemma extracts task snapshots (.tgz) into /workspace here; same tar flags as its
        copy-and-extract fallback. Returns False on failure so the harness falls back to that."""
        dest = self._host_dir(container_id, dest_dir)
        r = subprocess.run(["tar", "--no-same-owner", "-xzf", str(tar_path), "-C", str(dest)],
                           capture_output=True, text=True)
        return r.returncode == 0


def build_env(dest, comp):
    """With internet on: a relocatable CPython 3.13 (uv / python-build-standalone) and the
    Dockerfile's pip layer for it, pinned to packages released before IMAGE_DATE. Writes
    dest/py313.tar.gz, dest/image_sp.tar.gz and dest/manifest.json."""
    dest, comp = Path(dest), Path(comp)
    dest.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="scorer_env_build_"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(work / "py"))
    subprocess.run([sys.executable, "-m", "uv", "python", "install", "3.13"], check=True, env=env)
    py = sorted(glob.glob(str(work / "py" / "cpython-3.13*" / "bin" / "python3.13")))[0]
    sp = work / "image_sp"
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", py, "--target", str(sp),
                    "--exclude-newer", IMAGE_DATE, *IMAGE_PKGS], check=True, env=env)
    for shim in ("imp.py", "telnetlib.py"):
        shutil.copy(comp / "docker" / shim, sp / shim)
    shutil.rmtree(sp / "bin", ignore_errors=True)
    overlays = build_overlays(work / "overlays", comp, py, env)
    with tarfile.open(dest / "overlays.tar.gz", "w:gz") as t:
        t.add(work / "overlays", arcname="overlays")
    version = subprocess.run([py, "-c", "import sys; print(sys.version.split()[0])"],
                             capture_output=True, text=True, check=True).stdout.strip()
    dists = sorted(p.name[:-len(".dist-info")] for p in sp.glob("*.dist-info"))
    with tarfile.open(dest / "py313.tar.gz", "w:gz") as t:
        t.add(work / "py", arcname="py")
    with tarfile.open(dest / "image_sp.tar.gz", "w:gz") as t:
        t.add(sp, arcname="image_sp")
    (dest / "manifest.json").write_text(json.dumps(
        {"python": version, "python_rel": str(Path(py).relative_to(work)), "image_date": IMAGE_DATE,
         "image_pkgs": list(IMAGE_PKGS), "image_dists": dists, "overlays": overlays}, indent=2))
    return dest


def _comp_wheel_names(comp):
    return {_norm(w.name.split("-", 1)[0]) for w in (Path(comp) / "wheels").glob("*.whl")}


def _snapshot_dep_names(comp):
    """{repo key: set of dependency names} from every snapshot's pyproject.toml and
    requirements*.txt, parsed by the competition's own sandbox/setup.py functions."""
    import importlib.util
    from concurrent.futures import ThreadPoolExecutor
    spec = importlib.util.spec_from_file_location("comp_setup", Path(comp) / "sandbox" / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    snaps = sorted((Path(comp) / "snapshots").glob("*.tgz"))

    def one(snap):
        tmp = Path(tempfile.mkdtemp(prefix="deps_"))
        subprocess.run(["tar", "-xzf", str(snap), "-C", str(tmp), "--wildcards", "--no-anchored",
                        "pyproject.toml", "*requirements*.txt"], capture_output=True)
        names = set(setup.collect_from_pyproject(tmp / "pyproject.toml"))
        names |= set(setup.collect_from_requirements_files(tmp))
        shutil.rmtree(tmp, ignore_errors=True)
        return snap.name.split("_", 1)[0], names

    out = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, names in pool.map(one, snaps):
            out.setdefault(key, set()).update(names)
    return out


def build_overlays(dest, comp, py, env):
    """Per-repo site-packages overlays of test dependencies missing from the competition wheels,
    plus every Pygments release from 2.13 on (for the rich pin). Returns a manifest."""
    dest = Path(dest)
    comp_names = _comp_wheel_names(comp)
    deps = _snapshot_dep_names(comp)
    uv = [sys.executable, "-m", "uv", "pip", "install", "--python", py, "--exclude-newer", IMAGE_DATE,
          "--only-binary", ":all:"]
    manifest = {}
    for key, names in sorted(deps.items()):
        want = sorted({_norm(n) for n in names | set(EXTRA_TEST_DEPS)} - {_norm(key)})
        tgt = dest / key
        r = subprocess.run(uv + ["--target", str(tgt), *want], env=env, capture_output=True, text=True)
        failed = []
        if r.returncode != 0:                      # joint resolution failed: one name at a time
            for n in want:
                if subprocess.run(uv + ["--target", str(tgt), n], env=env, capture_output=True).returncode:
                    failed.append(n)
        shutil.rmtree(tgt / "bin", ignore_errors=True)
        kept, dropped = [], []
        for di in sorted(tgt.glob("*.dist-info")):
            name = _norm(di.name[:-len(".dist-info")].rsplit("-", 1)[0])
            if name in comp_names or name == _norm(key):
                _remove_dist(tgt, di)
                dropped.append(di.name)
            else:
                kept.append(di.name[:-len(".dist-info")])
        manifest[key] = {"requested": want, "unresolvable": failed, "added": kept,
                         "not_added_competition_has_wheel": dropped}
    import urllib.request
    from packaging.version import Version
    info = json.loads(urllib.request.urlopen("https://pypi.org/pypi/pygments/json", timeout=60).read())
    dates = {}
    for v, files in info["releases"].items():
        try:
            ok = Version(v) >= Version("2.13") and not Version(v).is_prerelease
        except Exception:
            ok = False
        up = min((f["upload_time_iso_8601"] for f in files), default="")
        if ok and up and up[:10] <= IMAGE_DATE[:10]:
            r = subprocess.run(uv + ["--no-deps", "--target", str(dest / "_pygments" / v), f"pygments=={v}"],
                               env=env, capture_output=True)
            if r.returncode == 0:
                shutil.rmtree(dest / "_pygments" / v / "bin", ignore_errors=True)
                dates[v] = up
    (dest / "_pygments" / "dates.json").write_text(json.dumps(dates, indent=2, sort_keys=True))
    manifest["_pygments"] = dates
    return manifest


def load_env(src=None, root=ENV_ROOT):
    """Unpack the build_env() output (found under /kaggle/input, or `src`) and point
    SWEGEMMA_PY313 / SWEGEMMA_IMAGE_SP at it. Idempotent."""
    if src is None:
        hits = sorted(glob.glob("/kaggle/input/**/py313.tar.gz", recursive=True))
        if not hits:
            raise SystemExit("scorer env not attached: no py313.tar.gz under /kaggle/input "
                             "(attach the envcheck kernel's output)")
        src = Path(hits[0]).parent
    src, root = Path(src), Path(root)
    manifest = json.loads((src / "manifest.json").read_text())
    if not (root / ".ok").exists():
        root.mkdir(parents=True, exist_ok=True)
        for name in ("py313.tar.gz", "image_sp.tar.gz", "overlays.tar.gz"):
            with tarfile.open(src / name) as t:
                t.extractall(root, filter="tar")
        (root / ".ok").write_text("ok\n")
    os.environ["SWEGEMMA_PY313"] = str(root / manifest["python_rel"])
    os.environ["SWEGEMMA_IMAGE_SP"] = str(root / "image_sp")
    os.environ["SWEGEMMA_OVERLAY_DIR"] = str(root / "overlays")
    return manifest


def patch_harness():
    """Make swegemma's Evaluator build a DockerSandbox for --sandbox subprocess, and run
    DockerSandbox.post_setup after every install_test_dependencies the harness makes."""
    import swegemma.evaluate as ev
    import swegemma.harness.agent_runner as ar
    import swegemma.harness.container_setup as cs
    import swegemma.harness.verification as ver
    ev.SubprocessManager = DockerSandbox
    orig = cs.install_test_dependencies
    if getattr(orig, "_renor_wrapped", False):
        return

    def install_test_dependencies(docker, container_id, repo="", *a, **kw):
        orig(docker, container_id, repo, *a, **kw)
        if hasattr(docker, "post_setup"):
            docker.post_setup(container_id, repo)
    install_test_dependencies._renor_wrapped = True
    for mod in (cs, ar, ver, ev):
        if hasattr(mod, "install_test_dependencies"):
            mod.install_test_dependencies = install_test_dependencies
