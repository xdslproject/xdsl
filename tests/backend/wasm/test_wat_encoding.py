import pytest

from xdsl.dialects import wasm


def test_empty_module():
    module = wasm.WasmModule()

    assert module.wat() == "(module)"


pytest.importorskip("wasmtime", reason="wasmtime is an optional dependency")

from wasmtime import wat2wasm  # noqa: E402


def test_wasmtime_coherency():
    empty = wasm.WasmModule()

    assert wat2wasm(empty.wat()) == empty.wasm()
