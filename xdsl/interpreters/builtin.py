from typing import Any, Literal, cast

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    PackableType,
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
from xdsl.interpreters.utils import ptr
from xdsl.ir import Attribute
from xdsl.utils.hints import isa


def xtype_for_el_type(
    el_type: Attribute, index_bitwidth: Literal[32, 64]
) -> PackableType[Any]:
    """
    Returns the datatype to use during interpretation for a given value.
    For `index`, uses either `i32` or `i64`, depending on the `index_bitwidth`.
    For all other types returns the input, as long as it's a subclass of
    `PackableType`, raising an error otherwise.
    """
    if isinstance(el_type, builtin.IndexType):
        return ptr.index(index_bitwidth)
    if not isinstance(el_type, PackableType):
        raise NotImplementedError(f"Unknown format for element type {el_type}")
    return el_type  # pyright: ignore[reportUnknownVariableType]


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

    @impl_attr(Float64Type)
    def float64_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: Float64Type
    ) -> float:
        interpreter.interpreter_assert(isa(attr, FloatAttr))
        attr = cast(FloatAttr, attr)
        return attr.value.data

    @impl_attr(Float32Type)
    def float32_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: Float32Type
    ) -> float:
        interpreter.interpreter_assert(isa(attr, FloatAttr))
        attr = cast(FloatAttr, attr)
        return attr.value.data

    @impl_attr(IntegerType)
    def integer_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: IntegerType
    ) -> float:
        interpreter.interpreter_assert(isa(attr, IntegerAttr))
        attr = cast(IntegerAttr, attr)
        return attr.value.data

    @impl_attr(builtin.MemRefType)
    def dense_int_or_fp_elements_value(
        self,
        interpreter: Interpreter,
        attr: Attribute,
        type_attr: builtin.MemRefType[Any],
    ) -> ShapedArray[Any]:
        assert isa(attr, builtin.DenseIntOrFPElementsAttr)
        shape = attr.get_shape()
        data = attr.get_values()
        data_ptr = ptr.TypedPtr[Any].new(
            data,
            xtype=xtype_for_el_type(
                attr.get_element_type(), interpreter.index_bitwidth
            ),
        )
        return ShapedArray(data_ptr, list(shape))
