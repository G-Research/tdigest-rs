"""Numerical imports must work independently of the Polars Python package."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("first_package", ["tdigest_rs", "tdigest_rs._native"])
def test_numerical_apis_without_polars_and_optional_plugin_use(first_package, tmp_path):
    # A subprocess prevents another test's imports from hiding an eager import.
    script = textwrap.dedent(
        """
        import copy
        import importlib
        import importlib.abc
        import importlib.util
        import pickle
        import sys
        import typing

        class NoPolars(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "polars" or fullname.startswith("polars."):
                    raise ModuleNotFoundError("Polars unavailable for this test", name="polars")

        blocker = NoPolars()
        sys.meta_path.insert(0, blocker)
        importlib.import_module(sys.argv[1])
        import numpy as np
        from tdigest_rs import _modern as modern
        from tdigest_rs import TDigest

        values = np.array([0.0, 1.0, 2.0, 3.0])
        old = TDigest.from_array(values)
        assert old.median() == 1.5
        assert copy.copy(old).median() == 1.5
        assert pickle.loads(pickle.dumps(old)).median() == 1.5
        new = modern.TDigest.from_array(values)
        assert new.median() == 1.5
        assert modern.TDigest.from_bytes(new.to_bytes()).median() == 1.5
        assert not any(name == "polars" or name.startswith("polars.") for name in sys.modules)

        def assert_polars_unavailable():
            for operation in (
                lambda: modern.tdigest("x"),
                lambda: modern.infer_column_precision(None, "digest"),
                lambda: typing.get_type_hints(modern.infer_column_precision),
            ):
                try:
                    operation()
                except ModuleNotFoundError as exc:
                    assert exc.name == "polars"
                    assert "Polars is required" in str(exc)
                else:
                    raise AssertionError("Polars operations must explain the missing dependency")

        assert_polars_unavailable()

        # Numerical-first import order must preserve the installed plugin API.
        sys.meta_path.remove(blocker)
        if importlib.util.find_spec("polars") is None:
            # Exercise the actual absent dependency, without the blocker.
            assert_polars_unavailable()
        else:
            hints = typing.get_type_hints(modern.infer_column_precision)
            import polars as pl
            assert hints["df"] is pl.DataFrame

            digest = pl.DataFrame({"x": values}).select(modern.tdigest("x").alias("digest"))
            assert digest.select(modern.median("digest")).item() == 1.5
            blob = digest.select(modern.to_bytes("digest").alias("blob"))
            assert modern.infer_column_precision(blob, "blob") == "f64"
            restored = blob.select(modern.from_bytes("blob", precision="f64").alias("digest"))
            assert restored.select(modern.median("digest")).item() == 1.5
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, first_package],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
