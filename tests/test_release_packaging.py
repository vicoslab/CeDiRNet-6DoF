"""Offline release checks: python -m unittest discover -s tests -v.

Downloads exercise the real shell helper and curl/wget against a loopback HTTP
server. Only the expensive Python training process is replaced with a recorder.
"""

import fnmatch
import http.server
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("ConvNext-B-RGB.pth", "localization_checkpoint.pth")
PAYLOAD = b"complete test checkpoint\n"


def _glob_matches(pattern, parts):
    """Docker's slash-separated globs: * stays in a segment; ** crosses them."""
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return any(_glob_matches(pattern[1:], parts[i:]) for i in range(len(parts) + 1))
    return bool(parts and fnmatch.fnmatchcase(parts[0], pattern[0])
                and _glob_matches(pattern[1:], parts[1:]))


def _docker_ignored(path):
    # Evaluate the glob/negation subset used by this .dockerignore, including
    # ancestor matches and last-rule-wins. This is not a Docker build smoke test.
    parts = PurePosixPath(path).parts
    ignored = False
    for line in (ROOT / ".dockerignore").read_text().splitlines():
        rule = line.strip()
        if not rule or rule.startswith("#"):
            continue
        negated = rule.startswith("!")
        pattern = rule.lstrip("!").strip("/").split("/")
        if any(_glob_matches(pattern, parts[:i]) for i in range(1, len(parts) + 1)):
            ignored = not negated
    return ignored


class ReleaseManifestTests(unittest.TestCase):
    def test_docker_copy_model_metadata_is_included(self):
        configs = sorted((ROOT / "models/configs").glob("*.py"))
        self.assertTrue(configs, "Docker needs the released model configs")
        paths = [ROOT / "models/model_definitions.json"] + configs
        for path in paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                relative = path.relative_to(ROOT).as_posix()
                self.assertFalse(_docker_ignored(relative), relative)

    def test_docker_excludes_weights_bundles_and_non_config_model_files(self):
        paths = [
            "models/private.json", "models/private.py", "models/release_upload/README.md",
            "models/release_upload/configs/private.py", "models/configs/private.json",
            "models/configs/unreleased.py",
            "models/configs/__pycache__/config.cpython-310.pyc",
        ]
        for folder in ("", "models/", "models/configs/", "models/release_upload/", "nested/"):
            paths.extend(folder + "artifact" + suffix for suffix in
                         (".pth", ".pt", ".ckpt", ".onnx", ".safetensors", ".zip", ".tar", ".tar.gz", ".tgz"))
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(_docker_ignored(path), path)

    def test_both_install_manifests_include_compatible_cvxpy(self):
        # 1.5.4 supports Python 3.10 / NumPy 1.x and still installs ECOS,
        # explicitly selected by NashMTL; 1.6 removes that default dependency.
        for name in ("requirements.txt", "environment.yml"):
            with self.subTest(manifest=name):
                text = (ROOT / name).read_text()
                dependencies = re.findall(r"^\s*(?:-\s*)?(cvxpy[^\s#]*)", text, re.MULTILINE)
                self.assertEqual(dependencies, ["cvxpy>=1.5.4,<1.6"])

    def test_demo_helper_is_readable_and_executable_by_all_users(self):
        mode = stat.S_IMODE((ROOT / "scripts/run_demo_inference.sh").stat().st_mode)
        self.assertEqual(mode, 0o755)


class TrainingDownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="release packaging ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = []
        self.response = "success"
        self.transfer_started = threading.Event()
        self.finish_transfer = threading.Event()
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                test.requests.append(self.path)
                if test.response == "http_error":
                    self.send_error(404, "checkpoint missing")
                    return
                response = test.response
                body = b"" if response == "empty" else PAYLOAD
                if response == "slow":
                    body = PAYLOAD * 1024
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if response == "partial":
                    body = body[:5]  # Deliberately shorter than Content-Length.
                if response == "slow":
                    split = len(body) // 2
                    self.wfile.write(body[:split])
                    self.wfile.flush()
                    test.transfer_started.set()
                    test.finish_transfer.wait(timeout=5)
                    body = body[split:]
                self.wfile.write(body)
                self.close_connection = True

            def log_message(self, format, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)
        self.url = "http://127.0.0.1:%s" % self.server.server_port

    def prepare(self, downloader, custom=False, scenario="default"):
        executable = shutil.which(downloader)
        if executable is None:
            self.skipTest("%s is required for its download integration test" % downloader)
        scenario_root = self.root / scenario / downloader
        repo = scenario_root / "repository with spaces"
        (repo / "scripts").mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/run_training_and_evalution.sh", repo / "scripts")
        bin_dir = repo / "bin"
        bin_dir.mkdir()
        for name in ("dirname", "mkdir", "mktemp", "mv", "rm"):
            binary = shutil.which(name)
            self.assertIsNotNone(binary, name)
            (bin_dir / name).symlink_to(binary)
        # PATH exposes exactly one real downloader, with bounded local retries.
        options = "--tries=1 --timeout=3" if downloader == "wget" else "--max-time 3"
        wrapper = bin_dir / downloader
        wrapper.write_text("#!/bin/sh\nexec %s %s \"$@\"\n" % (shlex.quote(executable), options))
        wrapper.chmod(0o755)
        trainer = bin_dir / "python"
        trainer.write_text('#!/bin/sh\nprintf "%s\\n" "$PRETRAINED_MODEL_DIR" "$@" >> "$TRAIN_LOG"\n')
        trainer.chmod(0o755)
        env = {
            "PATH": str(bin_dir), "HOME": str(self.root), "LC_ALL": "C",
            "NO_PROXY": "*", "no_proxy": "*", "VARIANTS": "crop_rgb",
            "TRAIN_LOG": str(repo / "training.log"),
            "CONVNEXT_RGB_URL": self.url + "/" + MODEL_NAMES[0],
            "LOCALIZATION_CHECKPOINT_URL": self.url + "/" + MODEL_NAMES[1],
        }
        target = repo / "models"
        if custom:
            target = scenario_root / "external pretrained models"
            env["PRETRAINED_MODEL_DIR"] = str(target)
        return repo, target, env

    def run_helper(self, repo, env, mode="train"):
        bash = shutil.which("bash")
        assert bash is not None, "bash is required for the training helper"
        return subprocess.run(
            [bash, str(repo / "scripts/run_training_and_evalution.sh"), mode],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=15,
        )

    def assert_downloaded(self, target):
        self.assertEqual(sorted(p.name for p in target.iterdir()), sorted(MODEL_NAMES))
        for name in MODEL_NAMES:
            self.assertEqual((target / name).read_bytes(), PAYLOAD)

    def test_custom_pretrained_directory_is_used(self):
        for downloader in ("wget", "curl"):
            with self.subTest(downloader=downloader):
                repo, target, env = self.prepare(downloader, custom=True)
                result = self.run_helper(repo, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(target.is_dir(), "PRETRAINED_MODEL_DIR was ignored")
                self.assert_downloaded(target)
                self.assertFalse((repo / "models").exists())
                self.assertIn(str(target), (repo / "training.log").read_text())

    def test_default_directory_and_cache_are_preserved(self):
        for downloader in ("wget", "curl"):
            with self.subTest(downloader=downloader):
                repo, target, env = self.prepare(downloader)
                result = self.run_helper(repo, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_downloaded(target)
                count = len(self.requests)
                env["CONVNEXT_RGB_URL"] = env["LOCALIZATION_CHECKPOINT_URL"] = "TODO"
                result = self.run_helper(repo, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(self.requests), count, "cached models should not be downloaded")

    def test_custom_cached_checkpoints_need_no_download_url(self):
        for downloader in ("wget", "curl"):
            with self.subTest(downloader=downloader):
                repo, target, env = self.prepare(downloader, custom=True)
                target.mkdir(parents=True)
                for name in MODEL_NAMES:
                    (target / name).write_bytes(PAYLOAD)
                env["CONVNEXT_RGB_URL"] = env["LOCALIZATION_CHECKPOINT_URL"] = "TODO"
                result = self.run_helper(repo, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.requests, [])

    def test_failed_or_empty_downloads_leave_no_cached_checkpoint_and_can_retry(self):
        for response in ("partial", "http_error", "empty"):
            for downloader in ("wget", "curl"):
                with self.subTest(response=response, downloader=downloader):
                    repo, target, env = self.prepare(downloader, scenario=response)
                    self.response = response
                    result = self.run_helper(repo, env)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(list(target.iterdir()), [], "partial/temporary file was cached")
                    self.assertFalse((repo / "training.log").exists())
                    self.response = "success"
                    result = self.run_helper(repo, env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assert_downloaded(target)

    def test_zero_byte_cached_files_are_downloaded_again(self):
        for downloader in ("wget", "curl"):
            with self.subTest(downloader=downloader):
                repo, target, env = self.prepare(downloader)
                target.mkdir()
                for name in MODEL_NAMES:
                    (target / name).touch()
                result = self.run_helper(repo, env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_downloaded(target)

    def test_final_checkpoint_is_not_visible_until_download_completes(self):
        bash = shutil.which("bash")
        assert bash is not None
        for downloader in ("wget", "curl"):
            with self.subTest(downloader=downloader):
                repo, target, env = self.prepare(downloader)
                self.response = "slow"
                self.transfer_started.clear()
                self.finish_transfer.clear()
                process = subprocess.Popen(
                    [bash, str(repo / "scripts/run_training_and_evalution.sh"), "train"],
                    cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    self.assertTrue(self.transfer_started.wait(timeout=3))
                    deadline = time.monotonic() + 1
                    while not any(p.stat().st_size for p in target.iterdir()):
                        self.assertLess(time.monotonic(), deadline, "download never wrote bytes")
                        time.sleep(0.01)
                    self.assertFalse((target / MODEL_NAMES[0]).exists(), "partial model became visible")
                finally:
                    self.response = "success"
                    self.finish_transfer.set()
                    stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 0, stdout + stderr)
                self.assertEqual((target / MODEL_NAMES[0]).read_bytes(), PAYLOAD * 1024)
                self.assertEqual((target / MODEL_NAMES[1]).read_bytes(), PAYLOAD)
                self.assertEqual(sorted(p.name for p in target.iterdir()), sorted(MODEL_NAMES))

    def test_dry_run_does_not_download(self):
        repo, target, env = self.prepare("curl", custom=True)
        result = self.run_helper(repo, env, "dry_run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.requests, [])
        self.assertFalse(target.exists())
        self.assertFalse((repo / "models").exists())


if __name__ == "__main__":
    unittest.main()
