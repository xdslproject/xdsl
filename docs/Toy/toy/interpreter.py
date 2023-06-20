from __future__ import annotations

import operator
from typing import Any
from itertools import accumulate

from dataclasses import dataclass

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

from .dialects import toy as toy


@dataclass
class Tensor:
    data: list[float]
    shape: list[int]

    def __format__(self, __format_spec: str) -> str:
        prod_shapes: list[int] = list(accumulate(reversed(self.shape), operator.mul))
        assert prod_shapes[-1] == len(self.data)
        result = "[" * len(self.shape)

        for i, d in enumerate(self.data):
            if i:
                n = sum(not i % p for p in prod_shapes)
                result += "]" * n
                result += ", "
                result += "[" * n
            result += f"{d}"

        result += "]" * len(self.shape)
        return result


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
        result = Tensor(data, shape)
        return (result,)

    @impl(toy.ReshapeOp)
    def run_reshape(
        self, interpreter: Interpreter, op: toy.ReshapeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (arg,) = args
        assert isinstance(arg, Tensor)
        result_typ = op.results[0].typ
        assert isinstance(result_typ, (VectorType, TensorType))
        new_shape = list(result_typ.get_shape())

        return (Tensor(arg.data, new_shape),)

    @impl(toy.AddOp)
    def run_add(
        self, interpreter: Interpreter, op: toy.AddOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args
        assert isinstance(lhs, Tensor)
        assert isinstance(rhs, Tensor)
        assert lhs.shape == rhs.shape

        return (Tensor([l + r for l, r in zip(lhs.data, rhs.data)], lhs.shape),)

    @impl(toy.MulOp)
    def run_mul(
        self, interpreter: Interpreter, op: toy.MulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args
        assert isinstance(lhs, Tensor)
        assert isinstance(rhs, Tensor)
        assert lhs.shape == rhs.shape

        return (Tensor([l * r for l, r in zip(lhs.data, rhs.data)], lhs.shape),)

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
        assert isinstance(arg, Tensor)
        assert len(arg.shape) == 2

        cols = arg.shape[0]
        rows = arg.shape[1]

        new_data = [
            arg.data[row * cols + col] for col in range(cols) for row in range(rows)
        ]

        result = Tensor(new_data, arg.shape[::-1])

        return (result,)
