from xdsl.dialects import wasm


def test_empty_module():
    module = wasm.WasmModuleOp()

    assert module.wasm() == b"\x00asm\x01\x00\x00\x00"
