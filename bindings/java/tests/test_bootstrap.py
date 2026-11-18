"""Portable bootstrap contracts, using only local archives and fake tools."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import os
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location("jvm_bootstrap", Path(__file__).parents[1] / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="JVM setup with spaces ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"

    def archive(self, name, files, zipped=False):
        path = self.root / (name + (".zip" if zipped else ".tar.gz"))
        if zipped:
            with zipfile.ZipFile(path, "w") as bundle:
                for filename, data in files.items():
                    entry = zipfile.ZipInfo(name + "/" + filename)
                    entry.external_attr = 0o100755 << 16
                    bundle.writestr(entry, data)
        else:
            with tarfile.open(path, "w:gz") as bundle:
                for filename, data in files.items():
                    entry = tarfile.TarInfo(name + "/" + filename)
                    entry.size, entry.mode = len(data), 0o755
                    bundle.addfile(entry, io.BytesIO(data))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_local_jdk_archive_and_cached_offline_reuse(self):
        source, digest = self.archive("jdk11", {"bin/javac": b"compiler", "bin/java": b"runtime"})
        home = bootstrap.install("jdk", str(source), digest, self.cache, offline=True)
        self.assertEqual((home / "bin/java").read_bytes(), b"runtime")
        source.unlink()
        self.assertEqual(home, bootstrap.install("jdk", "https://unreachable.invalid/never", digest, self.cache, True))

    def test_zip_preserves_executable_permissions(self):
        source, digest = self.archive("gradle", {"bin/gradle": b"#!/bin/sh\nexit 0\n"}, zipped=True)
        home = bootstrap.install("gradle", source.as_uri(), digest, self.cache, True)
        if os.name != "nt":
            self.assertTrue(os.access(home / "bin/gradle", os.X_OK))

    def test_selected_jdk_is_available_to_direct_subprocesses_without_host_java(self):
        source, digest = self.archive("jdk", {"bin/javac": b"compiler", "bin/java": b"runtime"})
        original = {
            "TDIGEST_JDK_URL": str(source),
            "TDIGEST_JDK_SHA256": digest,
            "TDIGEST_TOOLCHAIN_CACHE": str(self.cache),
            "TDIGEST_OFFLINE": "1",
            "PATH": "/no-host-java",
        }
        selected = bootstrap.java_environment(original)
        self.assertNotIn("JAVA_HOME", original)
        self.assertEqual(Path(selected["PATH"].split(os.pathsep)[0]), Path(selected["JAVA_HOME"]) / "bin")
        self.assertTrue((Path(selected["JAVA_HOME"]) / "bin/java").is_file())

    def test_checksum_failure_never_installs(self):
        source, _ = self.archive("jdk", {"bin/javac": b"compiler"})
        with self.assertRaisesRegex(bootstrap.BootstrapError, "checksum mismatch"):
            bootstrap.install("jdk", str(source), "0" * 64, self.cache, False)
        self.assertFalse(list(self.cache.glob("*/complete.json")))

    def test_offline_cold_http_source_never_opens_network(self):
        with patch.object(bootstrap.urllib.request, "build_opener") as opener:
            with self.assertRaisesRegex(bootstrap.BootstrapError, "Offline mode"):
                bootstrap.install("jdk", "https://unreachable.invalid/jdk", "0" * 64, self.cache, True)
            opener.assert_not_called()

    def test_archive_cannot_escape_destination(self):
        for zipped in [False, True]:
            source, _ = self.archive("bad", {"../../escape": b"bad"}, zipped=zipped)
            target = self.root / ("unpack-zip" if zipped else "unpack-tar")
            target.mkdir()
            with self.assertRaisesRegex(bootstrap.BootstrapError, "outside"):
                bootstrap.extract(source, target)
        self.assertFalse((self.root.parent / "escape").exists())

    def test_concurrent_cache_preparation_is_atomic(self):
        source, digest = self.archive("jdk", {"bin/javac": b"compiler", "bin/java": b"runtime"})
        with ThreadPoolExecutor(max_workers=2) as workers:
            homes = list(
                workers.map(lambda _: bootstrap.install("jdk", str(source), digest, self.cache, True), range(2))
            )
        self.assertEqual(homes[0], homes[1])
        self.assertTrue((homes[0] / "bin/java").is_file())

    def test_internal_http_basic_auth_and_no_fallback(self):
        payload = b"internal archive"
        expected = "Basic " + base64.b64encode(b"build:secret").decode()
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                requests.append(self.path)
                if self.headers.get("Authorization") != expected:
                    self.send_response(401)
                    self.send_header("WWW-Authenticate", 'Basic realm="mirror"')
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/internal"
            with patch.dict(os.environ, TDIGEST_DOWNLOAD_USERNAME="build", TDIGEST_DOWNLOAD_PASSWORD="secret"):
                bootstrap.download(url, self.root / "download", False)
            self.assertEqual((self.root / "download").read_bytes(), payload)
            self.assertEqual(requests, ["/internal", "/internal"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    @unittest.skipIf(os.name == "nt", "POSIX fake launcher; archive checks also run on Windows")
    def test_launcher_selects_jdk_and_preserves_arguments_in_offline_mode(self):
        jdk, jdk_sha = self.archive("jdk", {"bin/javac": b"compiler", "bin/java": b"runtime"})
        gradle, gradle_sha = self.archive(
            "gradle", {"bin/gradle": b'#!/bin/sh\nprintf "%s\\n" "$JAVA_HOME" "$@" > "$TRACE_FILE"\n'}
        )
        trace = self.root / "trace"
        with patch.dict(
            os.environ,
            {
                "TDIGEST_JDK_URL": str(jdk),
                "TDIGEST_JDK_SHA256": jdk_sha,
                "TDIGEST_GRADLE_URL": str(gradle),
                "TDIGEST_GRADLE_SHA256": gradle_sha,
                "TDIGEST_TOOLCHAIN_CACHE": str(self.cache),
                "TDIGEST_OFFLINE": "1",
                "TRACE_FILE": str(trace),
            },
            clear=True,
        ):
            self.assertEqual(bootstrap.main(["-p", "project with spaces", "test"]), 0)
        lines = trace.read_text().splitlines()
        self.assertTrue((Path(lines[0]) / "bin/javac").is_file())
        self.assertEqual(lines[1:], ["--offline", "-p", "project with spaces", "test"])


if __name__ == "__main__":
    unittest.main()
