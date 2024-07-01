from io import StringIO

from xdsl.dialects.builtin import IntegerAttr, NoneAttr, i32
from xdsl.dialects.wasm.attrs import (
    WasmFuncType,
    WasmImport,
    WasmImportDescFunc,
    WasmLimits,
)
from xdsl.dialects.wasm.ops import WasmModule
from xdsl.printer import Printer


def test_empty_module():
    module = WasmModule()
    output = StringIO()
    Printer(stream=output).print_op(module)
    assert output.getvalue() == "wasm.module"


def test_partial_module():
    module = WasmModule(
        mems=[WasmLimits((IntegerAttr(0, 32), NoneAttr()))],
        imports=[
            WasmImport("test", WasmImportDescFunc((WasmFuncType((i32,), (i32,)),)))
        ],
    )
    output = StringIO()
    Printer(stream=output).print_op(module)
    assert output.getvalue() == "wasm.module"
