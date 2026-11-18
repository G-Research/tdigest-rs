"""The authenticated Maven fixture must stay inside its staged repository."""

import base64
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

import pytest


@pytest.fixture
def maven_fixture(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "artifact.jar").write_bytes(b"staged artifact")
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private fixture data")
    (root / "outside-link").symlink_to(outside)
    receipt = tmp_path / "receipt.json"
    script = Path(__file__).resolve().parents[3] / "scripts/consumer_maven_fixture.py"
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), "--root", str(root), "--receipt", str(receipt)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while not receipt.exists():
                assert process.poll() is None, "Maven fixture failed to start"
                assert time.monotonic() < deadline, "Maven fixture startup timed out"
                time.sleep(0.01)
            port = urlsplit(json.loads(receipt.read_text())["url"]).port

            def request(method, path, *, authenticated=True):
                headers = {}
                if authenticated:
                    token = base64.b64encode(b"fixture:fixture-password").decode()
                    headers["Authorization"] = "Basic " + token
                connection = HTTPConnection("127.0.0.1", port, timeout=5)
                try:
                    connection.request(method, path, headers=headers)
                    response = connection.getresponse()
                    return response.status, response.read(), response.getheader("Content-Length")
                finally:
                    connection.close()

            yield request, outside, receipt.with_suffix(".jsonl")
        finally:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("escape", ["absolute", "symlink"])
def test_repository_escape_is_rejected(maven_fixture, method, escape):
    request, outside, events = maven_fixture
    path = "/maven/" + (str(outside) if escape == "absolute" else "outside-link")
    status, body, _ = request(method, path)
    assert status == 400
    assert outside.read_bytes() not in body
    assert not events.exists()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_staged_artifacts_and_authentication_still_work(maven_fixture, method):
    request, _, events = maven_fixture
    assert request(method, "/maven/artifact.jar", authenticated=False)[0] == 401
    status, body, length = request(method, "/maven/artifact.jar")
    assert status == 200
    assert length == str(len(b"staged artifact"))
    assert body == (b"staged artifact" if method == "GET" else b"")
    assert json.loads(events.read_text())["path"] == "artifact.jar"


@pytest.mark.parametrize("path", ["/maven/", "/maven/.", "/maven/../private.txt", "/other/artifact.jar"])
def test_invalid_repository_paths_are_rejected(maven_fixture, path):
    request, _, events = maven_fixture
    assert request("GET", path)[0] == 400
    assert not events.exists()
