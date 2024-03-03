from typing import Any

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
from xdsl.interpreters.shaped_array import ShapedArray


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
    def float_attr_value(self, attr: AnyFloatAttr) -> float:
        return attr.value.data

    @impl_attr(IntegerAttr)
    def integer_attr_value(self, attr: AnyIntegerAttr) -> float:
        return attr.value.data

    @impl_attr(DenseIntOrFPElementsAttr)
    def dense_int_or_fp_elements_value(
        self, attr: DenseIntOrFPElementsAttr
    ) -> ShapedArray[Any]:
        shape = attr.get_shape()
        data = [el.value.data for el in attr.data]
        return ShapedArray(data, list(shape) if shape is not None else [])
