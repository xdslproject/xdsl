from xdsl.ir import Dialect

from .ops import (
    WasmModuleOp,
)

Wasm = Dialect(
    "wasm",
    [
        WasmModuleOp,
    ],
)
"""
The WebAssembly dialect.
"""
