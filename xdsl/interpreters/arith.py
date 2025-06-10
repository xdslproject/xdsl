from math import copysign, isnan
from typing import cast

from xdsl.dialects import arith
from xdsl.dialects.builtin import FloatAttr, IntegerAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.utils.exceptions import InterpretationError


@register_impls
class ArithFunctions(InterpreterFunctions):
    @impl(arith.ConstantOp)
    def run_constant(
        self, interpreter: Interpreter, op: arith.ConstantOp, args: PythonValues
    ) -> PythonValues:
        interpreter.interpreter_assert(
            isinstance(op.value, IntegerAttr | FloatAttr),
            f"arith.constant not implemented for {type(op.value)}",
        )
        value = cast(IntegerAttr, op.value)
        return (value.value.data,)

    @impl(arith.SubiOp)
    def run_subi(self, interpreter: Interpreter, op: arith.SubiOp, args: PythonValues):
        return (args[0] - args[1],)

    @impl(arith.AddiOp)
    def run_addi(self, interpreter: Interpreter, op: arith.AddiOp, args: PythonValues):
        return (args[0] + args[1],)

    @impl(arith.MuliOp)
    def run_muli(self, interpreter: Interpreter, op: arith.MuliOp, args: PythonValues):
        return (args[0] * args[1],)

    @impl(arith.SubfOp)
    def run_subf(self, interpreter: Interpreter, op: arith.SubfOp, args: PythonValues):
        return (args[0] - args[1],)

    @impl(arith.AddfOp)
    def run_addf(self, interpreter: Interpreter, op: arith.AddfOp, args: PythonValues):
        return (args[0] + args[1],)

    @impl(arith.MulfOp)
    def run_mulf(self, interpreter: Interpreter, op: arith.MulfOp, args: PythonValues):
        return (args[0] * args[1],)

    @impl(arith.MinimumfOp)
    def run_minimumf(
        self, interpreter: Interpreter, op: arith.MinimumfOp, args: PythonValues
    ):
        if isnan(args[0]) or isnan(args[1]):
            return (float("NaN"),)
        if args[0] == 0 and args[1] == 0:
            if copysign(1.0, args[0]) < 0 or copysign(1.0, args[1]) < 0:
                return (-0.0,)
            else:
                return (0.0,)
        return (min(args[0], args[1]),)

    @impl(arith.MaximumfOp)
    def run_maximumf(
        self, interpreter: Interpreter, op: arith.MaximumfOp, args: PythonValues
    ):
        if isnan(args[0]) or isnan(args[1]):
            return (float("NaN"),)
        if args[0] == 0 and args[1] == 0:
            if copysign(1.0, args[0]) > 0 or copysign(1.0, args[1]) > 0:
                return (0.0,)
            else:
                return (-0.0,)
        return (max(args[0], args[1]),)

    @impl(arith.CmpiOp)
    def run_cmpi(self, interpreter: Interpreter, op: arith.CmpiOp, args: PythonValues):
        match op.predicate.value.data:
            case 0:  # "eq"
                return (args[0] == args[1],)
            case 1:  # "ne"
                return (args[0] != args[1],)
            case 2:  # "slt"
                return (args[0] < args[1],)
            case 3:  # "sle"
                return (args[0] <= args[1],)
            case 4:  # "sgt"
                return (args[0] > args[1],)
            case 5:  # "sge"
                return (args[0] >= args[1],)
            case 6:  # "ult"
                return (args[0] < args[1],)
            case 7:  # "ule"
                return (args[0] <= args[1],)
            case 8:  # "ugt"
                return (args[0] > args[1],)
            case 9:  # "uge"
                return (args[0] >= args[1],)
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )

    @impl(arith.ShLIOp)
    def run_shlsi(self, interpreter: Interpreter, op: arith.ShLIOp, args: PythonValues):
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs >= 0
        return (lhs << rhs,)

    @impl(arith.ShRSIOp)
    def run_shrsi(
        self, interpreter: Interpreter, op: arith.ShRSIOp, args: PythonValues
    ):
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs >= 0
        return (lhs >> rhs,)

    @impl(arith.DivSIOp)
    def run_divsi(
        self, interpreter: Interpreter, op: arith.DivSIOp, args: PythonValues
    ):
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        div = abs(lhs) // abs(rhs)
        if (lhs > 0) != (rhs > 0):
            div = -div
        return (div,)

    @impl(arith.RemSIOp)
    def run_remsi(
        self, interpreter: Interpreter, op: arith.RemSIOp, args: PythonValues
    ):
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        div = abs(lhs) // abs(rhs)
        if (lhs > 0) != (rhs > 0):
            div = -div
        return (lhs - div * rhs,)

    @impl(arith.FloorDivSIOp)
    def run_floordivsi(
        self, interpreter: Interpreter, op: arith.FloorDivSIOp, args: PythonValues
    ):
        lhs: int
        rhs: int
        (lhs, rhs) = args
        assert rhs != 0
        return (lhs // rhs,)
