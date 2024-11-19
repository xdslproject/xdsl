from typing import Any

from xdsl.dialects import ml_program
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    register_impls,
)
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.ptr import TypedPtr
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.traits import SymbolTable


@register_impls
class MLProgramFunctions(InterpreterFunctions):
    @impl(ml_program.GlobalLoadConstant)
    def run_global_load_constant(
        self,
        interpreter: Interpreter,
        op: ml_program.GlobalLoadConstant,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        global_op = SymbolTable.lookup_symbol(op, op.global_attr)
        assert isinstance(global_op, ml_program.Global)
        global_value = global_op.value
        assert isinstance(global_value, DenseIntOrFPElementsAttr)
        shape = global_value.get_shape()
        if shape is None:
            raise NotImplementedError()
        xtype = xtype_for_el_type(
            global_value.get_element_type(), interpreter.index_bitwidth
        )
        data = TypedPtr[Any].new(
            [el.value.data for el in global_value.data], xtype=xtype
        )
        shaped_array = ShapedArray(data, list(shape))
        return (shaped_array,)
