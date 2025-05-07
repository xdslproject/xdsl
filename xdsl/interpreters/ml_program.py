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
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa


@register_impls
class MLProgramFunctions(InterpreterFunctions):
    @impl(ml_program.GlobalLoadConstantOp)
    def run_global_load_constant(
        self,
        interpreter: Interpreter,
        op: ml_program.GlobalLoadConstantOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        global_op = SymbolTable.lookup_symbol(op, op.global_attr)
        assert isinstance(global_op, ml_program.GlobalOp)
        global_value = global_op.value
        assert isa(global_value, DenseIntOrFPElementsAttr)
        shape = global_value.get_shape()
        xtype = xtype_for_el_type(
            global_value.get_element_type(), interpreter.index_bitwidth
        )
        data = TypedPtr[Any].new(global_value.get_values(), xtype=xtype)
        shaped_array = ShapedArray(data, list(shape))
        return (shaped_array,)
