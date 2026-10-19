from math import copysign, isnan
from typing import Any, cast

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import FloatAttr, IntegerAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.utils.comparisons import to_signed, to_unsigned
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


def _int_bitwidth(
    interpreter: Interpreter, typ: builtin.IndexType | builtin.IntegerType
) -> int:
    if isa(typ, builtin.IntegerType):
        return typ.width.data
    if isa(typ, builtin.IndexType):
        return interpreter.index_bitwidth
    raise ValueError("unexpected integer type")


def _sign_extend(value: int, from_bitwidth: int) -> int:
    """
    Canonicalizes the value to positive or negative.
    """
    sign_bit = 1 << (from_bitwidth - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def _truncate(value: int, to_bitwidth: int) -> int:
    truncated = value & ((1 << to_bitwidth) - 1)
    if truncated & (1 << (to_bitwidth - 1)):
        return truncated - (1 << to_bitwidth)
    return truncated


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
        value = cast(IntegerAttr | FloatAttr, op.value)
        if isa(value, IntegerAttr[builtin.IndexType]):
            return (to_signed(value.value.data, interpreter.index_bitwidth),)
        return (value.value.data,)

    @impl(arith.SubiOp)
    def run_subi(self, interpreter: Interpreter, op: arith.SubiOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = to_signed(args[1], _int_bitwidth(interpreter, op.result.type))
        return (to_signed(lhs - rhs, _int_bitwidth(interpreter, op.result.type)),)

    @impl(arith.AddiOp)
    def run_addi(self, interpreter: Interpreter, op: arith.AddiOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = to_signed(args[1], _int_bitwidth(interpreter, op.result.type))
        return (to_signed(lhs + rhs, _int_bitwidth(interpreter, op.result.type)),)

    @impl(arith.MuliOp)
    def run_muli(self, interpreter: Interpreter, op: arith.MuliOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = to_signed(args[1], _int_bitwidth(interpreter, op.result.type))
        return (to_signed(lhs * rhs, _int_bitwidth(interpreter, op.result.type)),)

    @impl(arith.AndIOp)
    def run_andi(self, interpreter: Interpreter, op: arith.AndIOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = to_signed(args[1], _int_bitwidth(interpreter, op.result.type))
        return (to_signed(lhs & rhs, _int_bitwidth(interpreter, op.result.type)),)

    @impl(arith.OrIOp)
    def run_ori(self, interpreter: Interpreter, op: arith.OrIOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = to_signed(args[1], _int_bitwidth(interpreter, op.result.type))
        return (to_signed(lhs | rhs, _int_bitwidth(interpreter, op.result.type)),)

    @impl(arith.XOrIOp)
    def run_xori(self, interpreter: Interpreter, op: arith.XOrIOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = to_signed(args[1], _int_bitwidth(interpreter, op.result.type))
        return (to_signed(lhs ^ rhs, _int_bitwidth(interpreter, op.result.type)),)

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
        assert isa(op.lhs.type, builtin.IndexType | builtin.IntegerType)
        bitwidth = _int_bitwidth(interpreter, op.lhs.type)
        predicate = op.predicate.value.data
        interpret = to_unsigned if predicate >= 6 else to_signed
        lhs = interpret(args[0], bitwidth)
        rhs = interpret(args[1], bitwidth)
        match predicate:
            case 0:  # "eq"
                return (lhs == rhs,)
            case 1:  # "ne"
                return (lhs != rhs,)
            case 2:  # "slt"
                return (lhs < rhs,)
            case 3:  # "sle"
                return (lhs <= rhs,)
            case 4:  # "sgt"
                return (lhs > rhs,)
            case 5:  # "sge"
                return (lhs >= rhs,)
            case 6:  # "ult"
                return (lhs < rhs,)
            case 7:  # "ule"
                return (lhs <= rhs,)
            case 8:  # "ugt"
                return (lhs > rhs,)
            case 9:  # "uge"
                return (lhs >= rhs,)
            case _:
                raise InterpretationError(
                    f"arith.cmpi predicate {op.predicate} mot implemented yet."
                )

    @impl(arith.CmpfOp)
    def run_cmpf(self, interpreter: Interpreter, op: arith.CmpfOp, args: PythonValues):
        x = args[0]
        y = args[1]

        o = not isnan(x) and not isnan(y)
        u = isnan(x) or isnan(y)

        match op.predicate.value.data:
            case 0:
                return (False,)
            case 1:
                return ((x == y) and o,)
            case 2:
                return ((x > y) and o,)
            case 3:
                return ((x >= y) and o,)
            case 4:
                return ((x < y) and o,)
            case 5:
                return ((x <= y) and o,)
            case 6:
                return ((x != y) and o,)
            case 7:
                return (o,)
            case 8:
                return ((x == y) or u,)
            case 9:
                return ((x > y) or u,)
            case 10:
                return ((x >= y) or u,)
            case 11:
                return ((x < y) or u,)
            case 12:
                return ((x <= y) or u,)
            case 13:
                return ((x != y) or u,)
            case 14:
                return (u,)
            case 15:
                return (True,)
            case _:
                raise InterpretationError(
                    f"arith.cmpf predicate {op.predicate} mot implemented yet."
                )

    @impl(arith.ShLIOp)
    def run_shlsi(self, interpreter: Interpreter, op: arith.ShLIOp, args: PythonValues):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        bitwidth = _int_bitwidth(interpreter, op.result.type)
        lhs = to_signed(args[0], bitwidth)
        rhs = args[1]
        assert rhs >= 0
        return (to_signed(lhs << rhs, bitwidth),)

    @impl(arith.ShRSIOp)
    def run_shrsi(
        self, interpreter: Interpreter, op: arith.ShRSIOp, args: PythonValues
    ):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        lhs = to_signed(args[0], _int_bitwidth(interpreter, op.result.type))
        rhs = args[1]
        assert rhs >= 0
        return (lhs >> rhs,)

    @impl(arith.DivSIOp)
    def run_divsi(
        self, interpreter: Interpreter, op: arith.DivSIOp, args: PythonValues
    ):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        bitwidth = _int_bitwidth(interpreter, op.result.type)
        lhs = to_signed(args[0], bitwidth)
        rhs = to_signed(args[1], bitwidth)
        assert rhs != 0
        div = abs(lhs) // abs(rhs)
        if (lhs > 0) != (rhs > 0):
            div = -div
        return (to_signed(div, bitwidth),)

    @impl(arith.RemSIOp)
    def run_remsi(
        self, interpreter: Interpreter, op: arith.RemSIOp, args: PythonValues
    ):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        bitwidth = _int_bitwidth(interpreter, op.result.type)
        lhs = to_signed(args[0], bitwidth)
        rhs = to_signed(args[1], bitwidth)
        assert rhs != 0
        div = abs(lhs) // abs(rhs)
        if (lhs > 0) != (rhs > 0):
            div = -div
        return (lhs - div * rhs,)

    @impl(arith.FloorDivSIOp)
    def run_floordivsi(
        self, interpreter: Interpreter, op: arith.FloorDivSIOp, args: PythonValues
    ):
        assert isa(op.result.type, builtin.IndexType | builtin.IntegerType)
        bitwidth = _int_bitwidth(interpreter, op.result.type)
        lhs = to_signed(args[0], bitwidth)
        rhs = to_signed(args[1], bitwidth)
        assert rhs != 0
        return (to_signed(lhs // rhs, bitwidth),)

    @impl(arith.SelectOp)
    def run_select(
        self, interpreter: Interpreter, op: arith.SelectOp, args: PythonValues
    ):
        cond, lhs, rhs = args

        if not isinstance(cond, ShapedArray):
            # A scalar condition selects the whole value, shaped or not.
            return (lhs if cond else rhs,)

        # A shaped condition selects elementwise, all shapes must be identical.
        shaped_cond = cast(ShapedArray[int], cond)
        shaped_lhs = cast(ShapedArray[Any], lhs)
        shaped_rhs = cast(ShapedArray[Any], rhs)
        interpreter.interpreter_assert(
            shaped_cond.shape == shaped_lhs.shape == shaped_rhs.shape,
            f"arith.select shape mismatch: {shaped_cond.shape}, "
            f"{shaped_lhs.shape}, {shaped_rhs.shape}",
        )
        data = [
            l if c else r
            for c, l, r in zip(
                shaped_cond.data, shaped_lhs.data, shaped_rhs.data, strict=True
            )
        ]
        return (
            ShapedArray(
                TypedPtr[Any].new(data, xtype=shaped_lhs.element_type),
                list(shaped_lhs.shape),
            ),
        )

    @impl(arith.IndexCastOp)
    def run_indexcast(
        self, interpreter: Interpreter, op: arith.IndexCastOp, args: PythonValues
    ):
        assert len(args) == 1
        result = args[0]

        assert isa(op.input.type, builtin.IndexType | builtin.IntegerType)

        input_bitwidth = _int_bitwidth(interpreter, op.input.type)
        result_bitwidth = _int_bitwidth(interpreter, op.result.type)

        if input_bitwidth > result_bitwidth:
            result = _truncate(result, result_bitwidth)
        elif input_bitwidth < result_bitwidth:
            result = _sign_extend(result, input_bitwidth)
        else:
            result = to_signed(result, result_bitwidth)

        return (result,)
