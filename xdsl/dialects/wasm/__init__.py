from xdsl.ir import Dialect

from .attrs import (
    WasmExport,
    WasmExportDescFunc,
    WasmExportDescGlobal,
    WasmExportDescMem,
    WasmExportDescTable,
    WasmFuncType,
    WasmGlobalType,
    WasmImport,
    WasmImportDescFunc,
    WasmImportDescGlobal,
    WasmImportDescMem,
    WasmImportDescTable,
    WasmLimits,
    WasmMemoryType,
    WasmMut,
    WasmRefType,
    WasmTableType,
)
from .ops import (
    WasmModule,
)

Wasm = Dialect(
    "wasm",
    [
        WasmModule,
    ],
    [
        WasmFuncType,
        WasmLimits,
        WasmMemoryType,
        WasmTableType,
        WasmGlobalType,
        WasmExportDescFunc,
        WasmExportDescTable,
        WasmExportDescMem,
        WasmExportDescGlobal,
        WasmExport,
        WasmImportDescFunc,
        WasmImportDescTable,
        WasmImportDescMem,
        WasmImportDescGlobal,
        WasmImport,
        WasmRefType,
        WasmMut,
    ],
)
"""
The WebAssembly dialect.
"""
