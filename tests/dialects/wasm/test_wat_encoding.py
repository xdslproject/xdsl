from xdsl.dialects.wasm.ops import WasmModuleOp


def test_empty_module():
    module = WasmModuleOp()

    assert module.wat() == "(module)"
