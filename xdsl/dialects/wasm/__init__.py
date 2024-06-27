from xdsl.ir import Dialect

from .ops import (
    WasmModule,
)

Wasm = Dialect(
    "wasm",
    [
        WasmModule,
    ],
)
"""
The WebAssembly dialect.
"""
