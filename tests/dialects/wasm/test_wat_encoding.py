from xdsl.dialects import wasm


def test_empty_module():
    module = wasm.WasmModule()

    assert module.wat() == "(module)"
