import pytest

from xdsl.dialects import wasm

empty = wasm.WasmModule()


def test_empty_module():

    assert empty.wat() == "(module)"


pytest.importorskip("wasmtime", reason="wasmtime is an optional dependency")

from wasmtime import wat2wasm  # noqa: E402


def test_wasmtime_coherency():

    assert wat2wasm(empty.wat()) == empty.wasm()
