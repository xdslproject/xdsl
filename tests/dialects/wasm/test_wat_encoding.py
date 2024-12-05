from xdsl.dialects import wasm


def test_empty_module():
    module = wasm.WasmModuleOp()

    assert module.wat() == "(module)"
