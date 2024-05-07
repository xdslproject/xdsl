from typing import Any, Literal

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    AnyFloatAttr,
    AnyIntegerAttr,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    UnrealizedConversionCastOp,
)
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    impl_attr,
    register_impls,
)
from xdsl.interpreters import ptr
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.ir import Attribute


def xtype_for_el_type(
    el_type: Attribute, index_bitwidth: Literal[32, 64]
) -> ptr.XType[Any]:
    match el_type:
        case builtin.i32:
            return ptr.int32
        case builtin.i64:
            return ptr.int64
        case builtin.IndexType():
            return ptr.index(index_bitwidth)
        case builtin.f32:
            return ptr.float32
        case builtin.f64:
            return ptr.float64
        case _:
            raise NotImplementedError(f"Unknown format for element type {el_type}")


@register_impls
class BuiltinFunctions(InterpreterFunctions):
    @impl(UnrealizedConversionCastOp)
    def run_cast(
        self,
        interpreter: Interpreter,
        op: UnrealizedConversionCastOp,
        args: tuple[Any, ...],
    ):
        return tuple(
            interpreter.cast_value(o.type, r.type, arg)
            for (o, r, arg) in zip(op.operands, op.results, args)
        )

    @impl_attr(FloatAttr)
    def float_attr_value(self, interpreter: Interpreter, attr: AnyFloatAttr) -> float:
        return attr.value.data

    @impl_attr(IntegerAttr)
    def integer_attr_value(
        self, interpreter: Interpreter, attr: AnyIntegerAttr
    ) -> float:
        return attr.value.data

    @impl_attr(DenseIntOrFPElementsAttr)
    def dense_int_or_fp_elements_value(
        self, interpreter: Interpreter, attr: DenseIntOrFPElementsAttr
    ) -> ShapedArray[Any]:
        shape = attr.get_shape()
        data = [el.value.data for el in attr.data]
        data_ptr = ptr.TypedPtr.new(
            data,
            xtype=xtype_for_el_type(
                attr.get_element_type(), interpreter.index_bitwidth
            ),
        )
        return ShapedArray(data_ptr, list(shape) if shape is not None else [])
