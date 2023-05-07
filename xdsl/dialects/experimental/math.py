from __future__ import annotations

from typing import Annotated, Union

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import IntegerType, AnyFloat
from xdsl.ir import Operation, SSAValue, OpResult, Dialect
from xdsl.irdl import irdl_op_definition, OptOpAttr, Operand, IRDLOperation


@irdl_op_definition
class AbsFOp(IRDLOperation):
    """
    The absf operation computes the absolute value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type.

    Example:

    // Scalar absolute value.
    %a = math.absf %b : f64
    """

    name = "math.absf"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> AbsFOp:
        operand = SSAValue.get(operand)
        return AbsFOp.build(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.typ],
        )


@irdl_op_definition
class AbsIOp(IRDLOperation):
    """
    The absi operation computes the absolute value. It takes one operand of
    integer type (i.e., scalar, tensor or vector) and returns one result of the
    same type.

    Example:

    // Scalar absolute value.
    %a = math.absi %b : i64
    """

    name = "math.absi"
    operand: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(operand: Union[Operation, SSAValue]) -> AbsIOp:
        operand = SSAValue.get(operand)
        return AbsIOp.build(operands=[operand], result_types=[operand.typ])


@irdl_op_definition
class Atan2Op(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.atan2` ssa-use `,` ssa-use `:` type

    The atan2 operation takes two operands and returns one result, all of
    which must be of the same type.  The operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
    Euclidian plane between the positive x-axis and the ray through the point
    (x, y).  It is a generalization of the 1-argument arcus tangent which
    returns the angle on the basis of the ratio y/x.

    See also https://en.wikipedia.org/wiki/Atan2

    Example:

    // Scalar variant.
    %a = math.atan2 %b, %c : f32
    """

    name = "math.atan2"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    lhs: Annotated[Operand, AnyFloat]
    rhs: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        lhs: Union[Operation, SSAValue],
        rhs: Union[Operation, SSAValue],
        fastmath: FastMathFlagsAttr | None = None,
    ) -> Atan2Op:
        attributes = {"fastmath": fastmath}

        lhs = SSAValue.get(lhs)
        rhs = SSAValue.get(rhs)
        return Atan2Op.build(
            attributes=attributes, operands=[lhs, rhs], result_types=[lhs.typ]
        )


@irdl_op_definition
class AtanOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.atan` ssa-use `:` type

    The atan operation computes the arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    // Arcus tangent of scalar value.
    %a = math.atan %b : f64
    """

    name = "math.atan"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> AtanOp:
        operand = SSAValue.get(operand)
        return AtanOp.build(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.typ],
        )


@irdl_op_definition
class CbrtOp(IRDLOperation):
    """
    The cbrt operation computes the cube root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type. It has no standard attributes.

    Example:

    // Scalar cube root value.
    %a = math.cbrt %b : f64

    Note: This op is not equivalent to powf(..., 1/3.0).
    """

    name = "math.cbrt"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> CbrtOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return CbrtOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class CeilOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.ceil` ssa-use `:` type

    The ceil operation computes the ceiling of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar ceiling value.
    %a = math.ceil %b : f64
    """

    name = "math.ceil"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> CeilOp:
        operand = SSAValue.get(operand)
        return CeilOp.build(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.typ],
        )


@irdl_op_definition
class CopySignOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.copysign` ssa-use `,` ssa-use `:` type

    The copysign returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one result of
    the same type. The operands must be of floating point type (i.e., scalar,
    tensor or vector). It has no standard attributes.

    Example:

    // Scalar copysign value.
    %a = math.copysign %b, %c : f64
    """

    name = "math.copysign"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    lhs: Annotated[Operand, AnyFloat]
    rhs: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        lhs: Union[Operation, SSAValue],
        rhs: Union[Operation, SSAValue],
        fastmath: FastMathFlagsAttr | None = None,
    ) -> CopySignOp:
        attributes = {"fastmath": fastmath}

        lhs = SSAValue.get(lhs)
        rhs = SSAValue.get(rhs)
        return CopySignOp.build(
            attributes=attributes, operands=[lhs, rhs], result_types=[lhs.typ]
        )


@irdl_op_definition
class CosOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.cos` ssa-use `:` type

    The `cos` operation computes the cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar cosine value.
    %a = math.cos %b : f64
    """

    name = "math.cos"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> CosOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return CosOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class CountLeadingZerosOp(IRDLOperation):
    """
    The ctlz operation computes the number of leading zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    // Scalar ctlz function value.
    %a = math.ctlz %b : i32
    """

    name = "math.ctlz"
    operand: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(operand: Union[Operation, SSAValue]) -> CountLeadingZerosOp:
        operand = SSAValue.get(operand)
        return CountLeadingZerosOp.build(operands=[operand], result_types=[operand.typ])


@irdl_op_definition
class CountTrailingZerosOp(IRDLOperation):
    """
    The cttz operation computes the number of trailing zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    // Scalar cttz function value.
    %a = math.cttz %b : i32
    """

    name = "math.cttz"
    operand: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(operand: Union[Operation, SSAValue]) -> CountTrailingZerosOp:
        operand = SSAValue.get(operand)
        return CountTrailingZerosOp.build(
            operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class CtPopOp(IRDLOperation):
    """
    The ctpop operation computes the number of set bits of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    // Scalar ctpop function value.
    %a = math.ctpop %b : i32
    """

    name = "math.ctpop"
    operand: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(operand: Union[Operation, SSAValue]) -> CtPopOp:
        operand = SSAValue.get(operand)
        return CtPopOp.build(operands=[operand], result_types=[operand.typ])


@irdl_op_definition
class ErfOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.erf` ssa-use `:` type

    The erf operation computes the error function. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    // Scalar error function value.
    %a = math.erf %b : f64
    """

    name = "math.erf"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> ErfOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return ErfOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class Exp2Op(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.exp2` ssa-use `:` type

    The exp operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    // Scalar natural exponential.
    %a = math.exp2 %b : f64
    """

    name = "math.exp2"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> Exp2Op:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return Exp2Op.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class ExpM1Op(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.expm1` ssa-use `:` type

    expm1(x) := exp(x) - 1

    The expm1 operation takes one operand of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    // Scalar natural exponential minus 1.
    %a = math.expm1 %b : f64
    """

    name = "math.expm1"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> ExpM1Op:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return ExpM1Op.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class ExpOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.exp` ssa-use `:` type

    The exp operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    // Scalar natural exponential.
    %a = math.exp %b : f64
    """

    name = "math.exp"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> ExpOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return ExpOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class FPowIOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.fpowi` ssa-use `,` ssa-use `:` type

    The fpowi operation takes a `base` operand of floating point type
    (i.e. scalar, tensor or vector) and a `power` operand of integer type
    (also scalar, tensor or vector) and returns one result of the same type
    as `base`. The result is `base` raised to the power of `power`.
    The operation is elementwise for non-scalars, e.g.:

    %v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32

    The result is a vector of:

    [<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]

    Example:

    // Scalar exponentiation.
    %a = math.fpowi %base, %power : f64, i32
    """

    name = "math.fpowi"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    lhs: Annotated[Operand, AnyFloat]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        lhs: Union[Operation, SSAValue],
        rhs: Union[Operation, SSAValue],
        fastmath: FastMathFlagsAttr | None = None,
    ) -> FPowIOp:
        attributes = {"fastmath": fastmath}

        lhs = SSAValue.get(lhs)
        rhs = SSAValue.get(rhs)
        return FPowIOp.build(
            attributes=attributes, operands=[lhs, rhs], result_types=[lhs.typ]
        )


@irdl_op_definition
class FloorOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.floor` ssa-use `:` type

    The floor operation computes the floor of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar floor value.
    %a = math.floor %b : f64
    """

    name = "math.floor"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> FloorOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return FloorOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class FmaOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.fma` ssa-use `,` ssa-use `,` ssa-use `:` type

    The fma operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    Example:

    // Scalar fused multiply-add: d = a*b + c
    %d = math.fma %a, %b, %c : f64

    The semantics of the operation correspond to those of the `llvm.fma`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
    particular case of lowering to LLVM, this is guaranteed to lower
    to the `llvm.fma.*` intrinsic.
    """

    name = "math.fma"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    a: Annotated[Operand, AnyFloat]
    b: Annotated[Operand, AnyFloat]
    c: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        a: Union[Operation, SSAValue],
        b: Union[Operation, SSAValue],
        c: Union[Operation, SSAValue],
        fastmath: FastMathFlagsAttr | None = None,
    ) -> FmaOp:
        attributes = {"fastmath": fastmath}

        a = SSAValue.get(a)
        b = SSAValue.get(b)
        c = SSAValue.get(c)
        return FmaOp.build(
            attributes=attributes, operands=[a, b, c], result_types=[a.typ]
        )


@irdl_op_definition
class IPowIOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.ipowi` ssa-use `,` ssa-use `:` type

    The ipowi operation takes two operands of integer type (i.e., scalar,
    tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:
    // Scalar signed integer exponentiation.
    %a = math.ipowi %b, %c : i32
    """

    name = "math.ipowi"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(
        lhs: Union[Operation, SSAValue], rhs: Union[Operation, SSAValue]
    ) -> IPowIOp:
        lhs = SSAValue.get(lhs)
        rhs = SSAValue.get(rhs)
        return IPowIOp.build(operands=[lhs, rhs], result_types=[lhs.typ])


@irdl_op_definition
class Log10Op(IRDLOperation):
    """
    Computes the base-10 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    // Scalar log10 operation.
    %y = math.log10 %x : f64
    """

    name = "math.log10"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> Log10Op:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return Log10Op.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class Log1pOp(IRDLOperation):
    """
    Computes the base-e logarithm of one plus the given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.

    log1p(x) := log(1 + x)

    Example:

    // Scalar log1p operation.
    %y = math.log1p %x : f64
    """

    name = "math.log1p"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> Log1pOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return Log1pOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class Log2Op(IRDLOperation):
    """
    Computes the base-2 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    // Scalar log2 operation.
    %y = math.log2 %x : f64
    """

    name = "math.log2"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> Log2Op:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return Log2Op.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class LogOp(IRDLOperation):
    """
    Computes the base-e logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    // Scalar log operation.
    %y = math.log %x : f64
    """

    name = "math.log"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> LogOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return LogOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class PowFOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.powf` ssa-use `,` ssa-use `:` type

    The powf operation takes two operands of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    // Scalar exponentiation.
    %a = math.powf %b, %c : f64
    """

    name = "math.powf"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    lhs: Annotated[Operand, AnyFloat]
    rhs: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        lhs: Union[Operation, SSAValue],
        rhs: Union[Operation, SSAValue],
        fastmath: FastMathFlagsAttr | None = None,
    ) -> PowFOp:
        attributes = {"fastmath": fastmath}

        lhs = SSAValue.get(lhs)
        rhs = SSAValue.get(rhs)
        return PowFOp.build(
            attributes=attributes, operands=[lhs, rhs], result_types=[lhs.typ]
        )


@irdl_op_definition
class RoundEvenOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.roundeven` ssa-use `:` type

    The roundeven operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases to even, regardless of the current
    rounding direction.

    Example:

    // Scalar round operation.
    %a = math.roundeven %b : f64
    """

    name = "math.roundeven"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> RoundEvenOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return RoundEvenOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class RoundOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.round` ssa-use `:` type

    The round operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases away from zero, regardless of the current
    rounding direction.

    Example:

    // Scalar round operation.
    %a = math.round %b : f64
    """

    name = "math.round"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> RoundOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return RoundOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class RsqrtOp(IRDLOperation):
    """
    The rsqrt operation computes the reciprocal of the square root. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:
    // Scalar reciprocal square root value.
    %a = math.rsqrt %b : f64
    """

    name = "math.rsqrt"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> RsqrtOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return RsqrtOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class SinOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.sin` ssa-use `:` type

    The sin operation computes the sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar sine value.
    %a = math.sin %b : f64
    """

    name = "math.sin"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> SinOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return SinOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class SqrtOp(IRDLOperation):
    """
    The sqrt operation computes the square root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:
    // Scalar square root value.
    %a = math.sqrt %b : f64
    """

    name = "math.sqrt"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> SqrtOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return SqrtOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class TanOp(IRDLOperation):
    """
    The tan operation computes the tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    // Scalar tangent value.
    %a = math.tan %b : f64
    """

    name = "math.tan"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> TanOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return TanOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class TanhOp(IRDLOperation):
    """
    The tanh operation computes the hyperbolic tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    // Scalar hyperbolic tangent value.
    %a = math.tanh %b : f64
    """

    name = "math.tanh"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> TanhOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return TanhOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


@irdl_op_definition
class TruncOp(IRDLOperation):
    """
    Syntax:
    operation ::= ssa-id `=` `math.trunc` ssa-use `:` type

    The trunc operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.
    The operation always rounds to the nearest integer not larger in magnitude
    than the operand, regardless of the current rounding direction.

    Example:

    // Scalar trunc operation.
    %a = math.trunc %b : f64
    """

    name = "math.trunc"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(
        operand: Union[Operation, SSAValue], fastmath: FastMathFlagsAttr | None = None
    ) -> TruncOp:
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return TruncOp.build(
            attributes=attributes, operands=[operand], result_types=[operand.typ]
        )


Math = Dialect(
    [
        AbsFOp,
        AbsIOp,
        Atan2Op,
        AtanOp,
        CbrtOp,
        CeilOp,
        CopySignOp,
        CosOp,
        CountLeadingZerosOp,
        CountTrailingZerosOp,
        CtPopOp,
        ErfOp,
        Exp2Op,
        ExpM1Op,
        ExpOp,
        FPowIOp,
        FloorOp,
        FmaOp,
        IPowIOp,
        Log10Op,
        Log1pOp,
        Log2Op,
        LogOp,
        PowFOp,
        RoundEvenOp,
        RoundOp,
        RsqrtOp,
        SinOp,
        SqrtOp,
        TanOp,
        TanhOp,
        TruncOp,
    ]
)
