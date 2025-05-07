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
