"""Task snapshots ship as .tgz archives (first real Kaggle run, 8 Nov), but lesson writing reads
folders: snapshot_for extracts an archive once, Python sources only, safely, and the static
index and patch paths line up with the extracted checkout."""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import fake_harness as fh
from harness.swegemma import Task, patch_files, snapshot_for
from pipeline.experiment import IndexCache


def task(task_id="alpha-header", repo="alpha"):
    return Task(task_id, repo, "issue", "", "--- a/alpha/auth.py\n+++ b/alpha/auth.py\n", {})


class Snapshots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        fh.write_repos(self.d / "repos")
        self.snaps = self.d / "snapshots"
        self.snaps.mkdir()
        self.cache = self.d / "extracted"

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, name, prefix="alpha-1.0/", hostile=True):
        path = self.snaps / name
        with tarfile.open(path, "w:gz") as tf:
            for f in sorted((self.d / "repos" / "alpha").rglob("*")):
                if f.is_file():
                    tf.add(f, prefix + f.relative_to(self.d / "repos" / "alpha").as_posix())
            readme = b"not python"
            info = tarfile.TarInfo(prefix + "README.md")
            info.size = len(readme)
            tf.addfile(info, io.BytesIO(readme))
            if hostile:
                for evil in ("../evil.py", "/abs_evil.py"):
                    info = tarfile.TarInfo(evil)
                    info.size = 1
                    tf.addfile(info, io.BytesIO(b"x"))
                link = tarfile.TarInfo(prefix + "alpha/link.py")
                link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tf.addfile(link)
        return path

    def test_tgz_snapshot_is_extracted_and_indexed(self):
        self.archive("alpha-header.tgz")
        root = snapshot_for(self.snaps, task(), extract_to=self.cache)
        self.assertIsNotNone(root)
        self.assertEqual(root.name, "alpha-1.0")     # the single top-level folder is the checkout
        for f in patch_files(task().gold_patch):
            self.assertTrue((root / f).is_file(), f)  # patch paths line up with the checkout
        ix = IndexCache(lambda t: snapshot_for(self.snaps, t, extract_to=self.cache))(task())
        self.assertIn("alpha.client.Client.send", ix.symbols)
        self.assertIn("alpha.auth.sign", ix.symbols)

    def test_extraction_is_safe_python_only_and_done_once(self):
        self.archive("alpha-header.tgz")
        root = snapshot_for(self.snaps, task(), extract_to=self.cache)
        files = sorted(p.relative_to(self.cache).as_posix() for p in self.cache.rglob("*") if p.is_file())
        self.assertTrue(all(f.endswith(".py") or f.endswith(".renor-extracted") for f in files), files)
        self.assertFalse((root / "alpha" / "link.py").exists())
        self.assertFalse((self.cache / "evil.py").exists() or (self.d / "evil.py").exists())
        self.assertFalse(Path("/abs_evil.py").exists())
        before = (root / "alpha" / "client.py").stat().st_mtime_ns
        self.assertEqual(snapshot_for(self.snaps, task(), extract_to=self.cache), root)
        self.assertEqual((root / "alpha" / "client.py").stat().st_mtime_ns, before)

    def test_archive_names_and_flat_archives(self):
        self.archive("alpha-v2.tar.gz", prefix="", hostile=False)   # found by the repo prefix
        root = snapshot_for(self.snaps, task(), extract_to=self.cache)
        self.assertEqual(root, self.cache / "alpha-v2")             # no single top folder
        self.assertTrue((root / "alpha" / "auth.py").is_file())

    def test_folders_still_win_and_missing_is_none(self):
        self.assertIsNone(snapshot_for(self.snaps, task(), extract_to=self.cache))
        self.archive("alpha-header.tgz")
        folder = self.snaps / "alpha-header"
        folder.mkdir()
        self.assertEqual(snapshot_for(self.snaps, task(), extract_to=self.cache), folder)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
