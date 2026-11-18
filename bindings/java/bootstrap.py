#!/usr/bin/env python3
"""Opt-in, checksum-verified JDK/Gradle bootstrap for internal and offline builds.

Only the standard library is needed (Python 3.10+). Without these options the
normal Gradle wrapper remains usable without Python.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


class BootstrapError(Exception):
    pass


def enabled(value: str) -> bool:
    if value.lower() not in ("", "0", "false", "no", "1", "true", "yes"):
        raise BootstrapError("TDIGEST_OFFLINE must be 0/1, false/true or no/yes")
    return value.lower() in ("1", "true", "yes")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root: Path, path: Path) -> None:
    if not path.resolve().is_relative_to(root.resolve()):
        raise BootstrapError("Archive contains a path outside its destination")


def extract(archive: Path, target: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                inside(target, target / item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise BootstrapError("ZIP archives with symbolic links are unsupported")
            bundle.extractall(target)
            for item in bundle.infolist():
                mode = (item.external_attr >> 16) & 0o777
                if mode and not item.is_dir():
                    (target / item.filename).chmod(mode)
    else:
        with tarfile.open(archive) as bundle:
            for item in bundle.getmembers():
                path = target / item.name
                inside(target, path)
                if item.issym():
                    inside(target, path.parent / item.linkname)
                elif item.islnk():
                    inside(target, target / item.linkname)
                elif not (item.isfile() or item.isdir()):
                    raise BootstrapError("Archive contains an unsupported file type")
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(target, filter="data")
            else:
                if any(member.issym() or member.islnk() for member in bundle.getmembers()):
                    raise BootstrapError("Tool archives with links require a Python tarfile data filter")
                bundle.extractall(target)


def download(source: str, target: Path, offline: bool) -> None:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme == "" or parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path)) if parsed.scheme else Path(source)
        try:
            shutil.copyfile(path, target)
        except OSError as exc:
            raise BootstrapError("Cannot read the configured local tool archive") from exc
        return
    if offline:
        raise BootstrapError("Offline mode: tool archive is not cached; pre-stage it or prepare the cache first")
    if parsed.scheme not in ("https", "http") or parsed.username or parsed.password:
        raise BootstrapError(
            "Use a local path, file URL or HTTP(S) URL; put download credentials in environment variables"
        )
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    username = os.environ.get("TDIGEST_DOWNLOAD_USERNAME")
    password = os.environ.get("TDIGEST_DOWNLOAD_PASSWORD")
    if username is not None:
        manager.add_password(None, source, username, password or "")
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(manager))
    try:
        with opener.open(source, timeout=60) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as exc:
        # Configured URLs can carry access tokens. Do not print them on failure.
        raise BootstrapError("Cannot download the configured tool archive; no alternate source was attempted") from exc


def install(kind: str, source: str, digest: str, cache: Path, offline: bool) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise BootstrapError(f"A 64-character SHA256 is required for the configured {kind} archive")
    digest = digest.lower()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{kind}-{digest}"
    receipt = destination / "complete.json"
    if receipt.is_file():
        try:
            data = json.loads(receipt.read_text())
            home = destination / data["home"]
            valid_digest = data["sha256"] == digest
        except (ValueError, KeyError, TypeError):
            raise BootstrapError(
                f"Invalid cached {kind}; remove this specific cache entry and prepare it again"
            ) from None
        inside(destination, home)
        if not valid_digest or not home.is_dir():
            raise BootstrapError(f"Invalid cached {kind}; remove this specific cache entry and prepare it again")
        return home
    with tempfile.TemporaryDirectory(prefix=f".{kind}-", dir=cache) as temporary:
        stage = Path(temporary)
        archive = stage / "archive"
        download(source, archive, offline)
        if checksum(archive) != digest:
            raise BootstrapError(f"{kind} archive checksum mismatch; nothing was installed")
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        extract(archive, unpacked)
        binary = "javac.exe" if os.name == "nt" else "javac"
        if kind == "gradle":
            binary = "gradle.bat" if os.name == "nt" else "gradle"
        matches = [p.parent.parent for p in unpacked.rglob(binary) if p.parent.name == "bin"]
        if len(matches) != 1:
            raise BootstrapError(f"Expected exactly one {kind} installation in the configured archive")
        home = matches[0]
        data = {"sha256": digest, "home": str(home.relative_to(unpacked))}
        (unpacked / "complete.json").write_text(json.dumps(data) + "\n")
        try:
            unpacked.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not receipt.is_file():
                raise BootstrapError(f"Incomplete {kind} cache entry; inspect it before retrying")
        return destination / data["home"]


def wrapper_defaults() -> dict[str, str]:
    path = Path(__file__).parent / "gradle/wrapper/gradle-wrapper.properties"
    return {
        key: value.replace("\\:", ":")
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#") and "=" in line
        for key, value in [line.split("=", 1)]
    }


def java_environment(environment: dict[str, str]) -> dict[str, str]:
    """Select the same verified JDK for Gradle and direct Java subprocesses."""
    env = environment.copy()
    offline = enabled(env.get("TDIGEST_OFFLINE", ""))
    cache = Path(env.get("TDIGEST_TOOLCHAIN_CACHE", str(Path.home() / ".cache/tdigest-rs/toolchains"))).expanduser()
    if env.get("TDIGEST_JDK_URL"):
        java_home = install("jdk", env["TDIGEST_JDK_URL"], env.get("TDIGEST_JDK_SHA256", ""), cache, offline)
        env["JAVA_HOME"] = str(java_home)
    java = (
        Path(env["JAVA_HOME"]) / "bin" / ("java.exe" if os.name == "nt" else "java") if env.get("JAVA_HOME") else None
    )
    if java is not None and not java.is_file():
        raise BootstrapError("JAVA_HOME does not contain bin/java")
    if java is None and not shutil.which("java", path=env.get("PATH", os.defpath)):
        raise BootstrapError("Set JAVA_HOME or TDIGEST_JDK_URL with TDIGEST_JDK_SHA256")
    if java is not None:
        env["PATH"] = str(java.parent) + os.pathsep + env.get("PATH", os.defpath)
    return env


def main(arguments: list[str]) -> int:
    env = java_environment(os.environ.copy())
    offline = enabled(env.get("TDIGEST_OFFLINE", ""))
    cache = Path(env.get("TDIGEST_TOOLCHAIN_CACHE", str(Path.home() / ".cache/tdigest-rs/toolchains"))).expanduser()
    if env.get("TDIGEST_GRADLE_HOME"):
        gradle_home = Path(env["TDIGEST_GRADLE_HOME"])
    else:
        defaults = wrapper_defaults()
        custom = env.get("TDIGEST_GRADLE_URL")
        source = custom or defaults["distributionUrl"]
        digest = env.get("TDIGEST_GRADLE_SHA256", "") if custom else defaults.get("distributionSha256Sum", "")
        gradle_home = install("gradle", source, digest, cache, offline)
    launcher = gradle_home / "bin" / ("gradle.bat" if os.name == "nt" else "gradle")
    if not launcher.is_file():
        raise BootstrapError("TDIGEST_GRADLE_HOME does not contain bin/gradle")
    if arguments == ["--tdigest-prepare"]:
        print("JDK and Gradle are ready in the configured local tool cache.")
        return 0
    if offline and "--offline" not in arguments:
        arguments = ["--offline", *arguments]
    return subprocess.call([str(launcher), *arguments], env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (BootstrapError, OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"JVM setup: {error}", file=sys.stderr)
        raise SystemExit(1) from None
