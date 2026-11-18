"""Public Polars declarations must describe calls accepted by the adapter."""

import ast
import inspect
from pathlib import Path

from tdigest_rs import _modern as gr


def test_tdigest_type_stub_matches_runtime_call_signature():
    stub = ast.parse(Path(gr.__file__).with_suffix(".pyi").read_text())
    declaration = next(node for node in stub.body if isinstance(node, ast.FunctionDef) and node.name == "tdigest")
    runtime = inspect.signature(gr.tdigest)
    positional = [name for name, p in runtime.parameters.items() if p.kind == p.POSITIONAL_OR_KEYWORD]
    keywords = [name for name, p in runtime.parameters.items() if p.kind == p.KEYWORD_ONLY]
    assert [arg.arg for arg in declaration.args.args] == positional
    assert [arg.arg for arg in declaration.args.kwonlyargs] == keywords
    runtime.bind("x", 1000)
    assert "delta" not in runtime.parameters
