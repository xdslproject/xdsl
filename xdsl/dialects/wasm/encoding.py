"""
Helpers for encoding modules in the `wasm` dialect to the WebAssembly binary format.
"""

import abc
from typing import BinaryIO


class WasmBinaryEncodingContext:
    """
    A class to store the state of encoding.
    """


class EncodingException(Exception): ...


class WasmBinaryEncodable(abc.ABC):
    @abc.abstractmethod
    def encode(self, ctx: WasmBinaryEncodingContext, io: BinaryIO) -> None:
        raise NotImplementedError()
