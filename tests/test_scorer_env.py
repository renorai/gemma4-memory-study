"""kernels/scorer_sandbox.py pieces that run without the harness installed."""
import json
import os
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _import():
    if "swegemma.sandbox.subprocess" not in sys.modules:          # stub the harness import
        for name in ("swegemma", "swegemma.sandbox", "swegemma.sandbox.subprocess"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["swegemma.sandbox.subprocess"].SubprocessManager = object
    sys.path.insert(0, str(ROOT / "kernels"))
    import scorer_sandbox
    return scorer_sandbox


class ScorerEnv(unittest.TestCase):
    def test_load_env_skips_uv_symlinked_python_dir(self):
        sb = _import()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            build = d / "build" / "py"
            real = build / "cpython-3.13.18-linux-x86_64-gnu" / "bin"
            real.mkdir(parents=True)
            (real / "python3.13").write_text("")
            (build / "cpython-3.13-linux-x86_64-gnu").symlink_to(real.parent.resolve())   # absolute, like uv
            src = d / "src"
            src.mkdir()
            with tarfile.open(src / "py313.tar.gz", "w:gz") as t:
                t.add(build, arcname="py")
            for name in ("image_sp", "overlays"):
                (d / name).mkdir()
                with tarfile.open(src / f"{name}.tar.gz", "w:gz") as t:
                    t.add(d / name, arcname=name)
            (src / "manifest.json").write_text(json.dumps(
                {"python": "3.13.18", "python_rel": "py/cpython-3.13-linux-x86_64-gnu/bin/python3.13"}))
            import shutil
            shutil.rmtree(d / "build")                               # the symlink target is gone now
            sb.load_env(src, root=d / "root")
            self.assertEqual(os.environ["SWEGEMMA_PY313"],
                             str(d / "root" / "py" / "cpython-3.13.18-linux-x86_64-gnu" / "bin" / "python3.13"))

    def test_poetry_and_pep621_constraints(self):
        sb = _import()
        self.assertEqual(sb._poetry_spec("^0.40.0"), ">=0.40.0,<0.41")
        self.assertEqual(sb._poetry_spec("^1.2"), ">=1.2,<2")
        self.assertEqual(sb._poetry_spec({"version": "~1.2"}), ">=1.2,<1.3")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pyproject.toml").write_text(
                '[project]\ndependencies = ["starlette>=0.40.0,<0.49.0", "x>=1; python_version < \'3.10\'"]\n'
                '[tool.poetry.dependencies]\npython = "^3.8"\npygments = "^2.13.0"\n')
            c = sb.snapshot_constraints(d)
        self.assertEqual(set(c), {"starlette", "pygments"})
        self.assertTrue(c["starlette"].contains("0.48.0") and not c["starlette"].contains("1.6.0"))


if __name__ == "__main__":
    unittest.main()
