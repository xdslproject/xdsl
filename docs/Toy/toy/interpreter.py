from __future__ import annotations


from typing import Any, cast

from xdsl.dialects.builtin import TensorType, VectorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl_terminator,
    register_impls,
    impl,
)
from xdsl.interpreters.shaped_array import ShapedArray

from .dialects import toy as toy


@register_impls
class ToyFunctions(InterpreterFunctions):
    @impl(toy.PrintOp)
    def run_print(
        self, interpreter: Interpreter, op: toy.PrintOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        interpreter.print(f"{args[0]}")
        return ()

    @impl(toy.ConstantOp)
    def run_const(
        self, interpreter: Interpreter, op: toy.ConstantOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert not len(args)
        data = op.get_data()
        shape = op.get_shape()
        result = ShapedArray(data, shape)
        return (result,)

    @impl(toy.ReshapeOp)
    def run_reshape(
        self, interpreter: Interpreter, op: toy.ReshapeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (arg,) = args
        assert isinstance(arg, ShapedArray)
        arg = cast(ShapedArray[float], arg)
        result_typ = op.results[0].typ
        assert isinstance(result_typ, (VectorType, TensorType))
        new_shape = list(result_typ.get_shape())

        return (ShapedArray(arg.data, new_shape),)

    @impl(toy.AddOp)
    def run_add(
        self, interpreter: Interpreter, op: toy.AddOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape

        return (ShapedArray([l + r for l, r in zip(lhs.data, rhs.data)], lhs.shape),)

    @impl(toy.MulOp)
    def run_mul(
        self, interpreter: Interpreter, op: toy.MulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape

        return (ShapedArray([l * r for l, r in zip(lhs.data, rhs.data)], lhs.shape),)

    @impl_terminator(toy.ReturnOp)
    def run_return(
        self, interpreter: Interpreter, op: toy.ReturnOp, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        assert len(args) < 2
        return ReturnedValues(args), ()

    @impl(toy.GenericCallOp)
    def run_generic_call(
        self, interpreter: Interpreter, op: toy.GenericCallOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return interpreter.call_op(op.callee.string_value(), args)

    @impl(toy.TransposeOp)
    def run_transpose(
        self, interpreter: Interpreter, op: toy.TransposeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (arg,) = args
        assert isinstance(arg, ShapedArray)
        arg = cast(ShapedArray[float], arg)
        assert len(arg.shape) == 2

        return (arg.transposed(),)
