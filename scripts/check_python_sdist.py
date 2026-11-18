"""Check a real Maturin source archive without extracting it."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import tarfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 development dependency
    import tomli as tomllib


def canonical_relative(name: str) -> str:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name.rstrip("/"):
        raise RuntimeError(f"noncanonical archive path: {name!r}")
    return str(path)


def verify_sdist_contents(sdist: Path, root: Path) -> None:
    """Require active source bytes and reject aliases, duplicates and caches."""
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [canonical_relative(member.name) for member in members]
        counts = Counter(names)
        duplicate = sorted(name for name, count in counts.items() if count != 1)
        if duplicate:
            raise RuntimeError("sdist repeats member paths: " + ", ".join(duplicate))
        forbidden = sorted(
            name
            for name in names
            if "__pycache__" in PurePosixPath(name).parts or PurePosixPath(name).suffix in {".pyc", ".pyo"}
        )
        if forbidden:
            raise RuntimeError("sdist contains Python caches: " + ", ".join(forbidden))
        top_levels = {PurePosixPath(name).parts[0] for name in names}
        if len(top_levels) != 1:
            raise RuntimeError("sdist must have exactly one top-level directory")
        prefix = top_levels.pop() + "/"
        hashes = {}
        manifests = {}
        for member, name in zip(members, names):
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"sdist contains unsupported link or special member: {name}")
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"cannot read sdist member: {name}")
                content = stream.read()
                relative = name.removeprefix(prefix)
                hashes[relative] = hashlib.sha256(content).hexdigest()
                if PurePosixPath(relative).name in {"Cargo.toml", "pyproject.toml"}:
                    manifests[relative] = tomllib.loads(content.decode())
        for relative in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
            if relative not in hashes:
                raise RuntimeError(f"sdist missing normalized build input: {relative}")

        def require_same(archive_path: str, source_path: str) -> None:
            expected = hashlib.sha256((root / source_path).read_bytes()).hexdigest()
            if hashes.get(archive_path) != expected:
                raise RuntimeError(f"sdist missing or changed source: {archive_path}")

        for relative in (
            "README.md",
            "RELEASE_NOTES_1.0.0.md",
            "tests/quality/data/README.md",
            "src/tdigest/tdigest.rs",
            "scripts/check_build_tools.py",
            "scripts/consumer_contract_probe.py",
            "bindings/java/gradlew",
            "bindings/java/gradlew.bat",
            "bindings/java/gradle/wrapper/gradle-wrapper.jar",
            "bindings/java/gradle/wrapper/gradle-wrapper.properties",
            "bindings/java/examples/scala/build.sbt",
            "bindings/java/examples/scala/project/build.properties",
            "bindings/java/examples/scala/src/main/scala/Example.scala",
            "bindings/java/examples/kotlin/build.gradle",
            "bindings/java/examples/kotlin/settings.gradle",
            "bindings/java/examples/kotlin/src/main/kotlin/Example.kt",
            "bindings/python/tests/test_consumer_contracts.py",
        ):
            require_same(relative, relative)
        # Include everything needed to run the accuracy tests offline.
        for relative in (
            "pytest.ini",
            "scripts/check_upstream_tests.py",
            "scripts/test_upstream_legacy.py",
            "tests/upstream_legacy/manifest.json",
            "tests/upstream_legacy/bindings/python/tests/__init__.py",
            "tests/upstream_legacy/bindings/python/tests/conftest.py",
            "tests/upstream_legacy/bindings/python/tests/constants.py",
            "tests/upstream_legacy/bindings/python/tests/test_distributions.py",
            "tests/upstream_legacy/bindings/python/tests/test_tdigest.py",
            "tests/upstream_legacy/bindings/python/tests/utils.py",
            "tests/upstream_legacy/bindings/python/benchmarks/run.py",
            "tests/quality/__init__.py",
            "tests/quality/conftest.py",
            "tests/quality/measurements.py",
            "tests/quality/test_accuracy.py",
            "scripts/fetch_quality_data.py",
            "tests/quality/test_reference.py",
            "tests/quality/test_weighted_quality.py",
            "tests/quality/baseline-fast.json.gz",
            "tests/quality/baseline-extended.json.gz",
            "tests/quality/known-limits.json",
            "tests/quality/data/manifest.json",
            "tests/quality/data/bike.csv.gz",
            "tests/quality/data/energy.csv.gz",
        ):
            require_same(relative, relative)
        require_same("LICENSE", "LICENSE")
        require_same("bindings/python/LICENSE", "LICENSE")
        if manifests["pyproject.toml"]["project"]["readme"] != "README.md":
            raise RuntimeError("sdist metadata must use the root README")
        normalized_build = manifests["pyproject.toml"].get("tool", {}).get("maturin", {})
        if normalized_build.get("manifest-path") != "Cargo.toml":
            raise RuntimeError("sdist has an unexpected active Rust manifest")
        if normalized_build.get("python-source", ".") != ".":
            raise RuntimeError("sdist has an unexpected active Python source directory")
        for relative in (
            "tdigest_rs/__init__.py",
            "tdigest_rs/_modern.py",
            "tdigest_rs/_modern.pyi",
            "tdigest_rs/_polars.py",
            "tdigest_rs/_shared.py",
            "tdigest_rs/__init__.pyi",
            "tdigest_rs/_native.pyi",
            "tdigest_rs/py.typed",
            "tdigest_rs/lib.py",
            "tdigest_rs/lib.pyi",
            "tdigest_rs/tdigest_rs.py",
        ):
            source = "bindings/python/" + relative
            require_same(relative, source)
            require_same(source, source)
        source_manifest = tomllib.loads((root / "Cargo.toml").read_text())
        source_library = canonical_relative(source_manifest.get("lib", {}).get("path", "src/lib.rs"))
        archive_library = canonical_relative(manifests["Cargo.toml"].get("lib", {}).get("path", "src/lib.rs"))
        if archive_library != source_library:
            raise RuntimeError("sdist changed the active root Rust library path")
        require_same(archive_library, source_library)
        for source in sorted((root / "src").rglob("*.rs")):
            relative = source.relative_to(root).as_posix()
            require_same(relative, relative)
    print("Source archive contains required active source bytes and no duplicate paths or Python caches.", flush=True)
