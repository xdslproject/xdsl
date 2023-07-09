from __future__ import annotations

from typing import Any

from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls

from ..dialects import toy_accelerator


@register_impls
class ToyAcceleratorFunctions(InterpreterFunctions):
    @impl(toy_accelerator.Transpose)
    def run_transpose(
        self,
        interpreter: Interpreter,
        op: toy_accelerator.Transpose,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        dest, source = args

        source_rows = op.source_rows.value.data
        source_cols = op.source_cols.value.data

        for row in range(source_rows):
            for col in range(source_cols):
                value = source.load((row, col))
                dest.store((col, row), value)

        return ()

    @impl(toy_accelerator.Add)
    def run_add(
        self, interpreter: Interpreter, op: toy_accelerator.Add, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        dest, lhs, rhs = args

        for i, (l, r) in enumerate(zip(lhs.data, rhs.data)):
            dest.data[i] = l + r

        return ()

    @impl(toy_accelerator.Mul)
    def run_mul(
        self, interpreter: Interpreter, op: toy_accelerator.Mul, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        dest, lhs, rhs = args

        for i, (l, r) in enumerate(zip(lhs.data, rhs.data)):
            dest.data[i] = l * r

        return ()
