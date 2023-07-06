from enum import Enum
from math import prod
from typing import Any, cast

from xdsl.dialects import memref
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir.core import Attribute


class MemrefValue(Enum):
    Allocated = 1
    Deallocated = 2


@register_impls
class MemrefFunctions(InterpreterFunctions):
    @impl(memref.Alloc)
    def run_alloc(self, interpreter: Interpreter, op: memref.Alloc, args: tuple[()]):
        memref_typ = cast(memref.MemRefType[Attribute], op.memref.typ)

        shape = memref_typ.get_shape()
        size = prod(shape)
        data = [MemrefValue.Allocated] * size

        shaped_array = ShapedArray(data, list(shape))
        return (shaped_array,)

    @impl(memref.Dealloc)
    def run_dealloc(
        self, interpreter: Interpreter, op: memref.Dealloc, args: tuple[Any, ...]
    ):
        (shaped_array,) = args
        for i in range(len(shaped_array.data)):
            shaped_array.data[i] = MemrefValue.Deallocated
        return ()

    @impl(memref.Store)
    def run_store(
        self, interpreter: Interpreter, op: memref.Store, args: tuple[Any, ...]
    ):
        value, memref, *indices = args

        memref = cast(ShapedArray[Any], memref)

        indices = tuple(indices)
        memref.store(indices, value)

        return ()

    @impl(memref.Load)
    def run_load(
        self, interpreter: Interpreter, op: memref.Load, args: tuple[Any, ...]
    ):
        memref, *indices = args

        memref = cast(ShapedArray[Any], memref)

        indices = tuple(indices)
        value = memref.load(indices)

        return (value,)
