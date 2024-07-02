from io import StringIO

from xdsl.dialects.builtin import i32
from xdsl.dialects.wasm.attrs import (
    TableIdx,
    WasmExport,
    WasmExportDescTable,
    WasmFuncType,
    WasmImport,
    WasmImportDescFunc,
    WasmLimits,
    WasmRefType,
    WasmRefTypeEnum,
    WasmTableType,
)
from xdsl.dialects.wasm.ops import WasmModule
from xdsl.printer import Printer


def test_empty_module():
    module = WasmModule()
    output = StringIO()
    Printer(stream=output).print_op(module)
    assert output.getvalue() == "wasm.module()"


def test_partial_module():
    module = WasmModule(
        mems=[WasmLimits(0, None)],
        imports=[WasmImport("test", WasmImportDescFunc(WasmFuncType((i32,), (i32,))))],
    )
    output = StringIO()
    Printer(stream=output).print_op(module)

    EXPECTED = """wasm.module(
  mems [#wasm.limits<0 : i32, unit>]
  imports [#wasm.import<"test", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>]
)"""

    assert output.getvalue() == EXPECTED


def test_full_module():
    # FIXME: add start and other sections once they are implemented
    module = WasmModule(
        tables=[WasmTableType(WasmRefType(WasmRefTypeEnum.FuncRef), WasmLimits(0, 8))],
        mems=[WasmLimits(0, None)],
        imports=[
            WasmImport("test_im", WasmImportDescFunc(WasmFuncType((i32,), (i32,))))
        ],
        exports=[WasmExport("test_ex", WasmExportDescTable(TableIdx(0, 32)))],
    )
    output = StringIO()
    Printer(stream=output).print_op(module)

    EXPECTED = """wasm.module(
  tables [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>]
  mems [#wasm.limits<0 : i32, unit>]
  imports [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>]
  exports [#wasm.export<"test_ex", #wasm.export_desc_table<0 : i32>>]
)"""

    assert output.getvalue() == EXPECTED
