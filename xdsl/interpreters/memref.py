from math import prod
from typing import Any, cast

from xdsl.dialects import builtin, memref
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.ptr import TypedPtr
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir import Attribute
from xdsl.traits import SymbolTable


@register_impls
class MemrefFunctions(InterpreterFunctions):
    @impl(memref.Alloc)
    def run_alloc(
        self, interpreter: Interpreter, op: memref.Alloc, args: PythonValues
    ) -> PythonValues:
        memref_type = cast(memref.MemRefType[Attribute], op.memref.type)

        shape = memref_type.get_shape()
        size = prod(shape)
        xtype = xtype_for_el_type(
            memref_type.get_element_type(), interpreter.index_bitwidth
        )

        shaped_array = ShapedArray(TypedPtr[Any].zeros(size, xtype=xtype), list(shape))
        return (shaped_array,)

    @impl(memref.Dealloc)
    def run_dealloc(
        self, interpreter: Interpreter, op: memref.Dealloc, args: PythonValues
    ) -> PythonValues:
        return ()

    @impl(memref.Store)
    def run_store(
        self, interpreter: Interpreter, op: memref.Store, args: PythonValues
    ) -> PythonValues:
        value, memref, *indices = args

        memref = cast(ShapedArray[Any], memref)

        indices = tuple(indices)
        memref.store(indices, value)

        return ()

    @impl(memref.Load)
    def run_load(
        self, interpreter: Interpreter, op: memref.Load, args: tuple[Any, ...]
    ):
        shaped_array, *indices = args

        shaped_array = cast(ShapedArray[Any], shaped_array)

        indices = tuple(indices)
        value = shaped_array.load(indices)

        return (value,)

    @impl(memref.GetGlobal)
    def run_get_global(
        self, interpreter: Interpreter, op: memref.GetGlobal, args: PythonValues
    ) -> PythonValues:
        mem = SymbolTable.lookup_symbol(op, op.name_)
        assert isinstance(mem, memref.Global)
        initial_value = mem.initial_value
        if not isinstance(initial_value, builtin.DenseIntOrFPElementsAttr):
            raise NotImplementedError(
                "Memrefs that are not dense int or float arrays are not implemented"
            )
        data = [el.value.data for el in initial_value.data]
        shape = initial_value.get_shape()
        assert shape is not None
        xtype = xtype_for_el_type(
            initial_value.get_element_type(), interpreter.index_bitwidth
        )
        shaped_array = ShapedArray(TypedPtr[Any].new(data, xtype=xtype), list(shape))
        return (shaped_array,)
