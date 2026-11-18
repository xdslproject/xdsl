from xdsl.dialects.wasm.ops import WasmModuleOp


def test_empty_module():
    module = WasmModuleOp()

    assert module.wasm() == b"\x00asm\x01\x00\x00\x00"
