from __future__ import annotations

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import AnyFloatConstr, IntegerType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure


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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.type],
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
    operand = operand_def(IntegerType)
    result = result_def(IntegerType)

    traits = traits_def(Pure())

    def __init__(self, operand: Operation | SSAValue):
        operand = SSAValue.get(operand)
        return super().__init__(operands=[operand], result_types=[operand.type])


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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    lhs = operand_def(AnyFloatConstr)
    rhs = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}
        return super().__init__(
            attributes=attributes,
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.type],
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        operand = SSAValue.get(operand)
        return super().__init__(
            attributes={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.type],
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    lhs = operand_def(AnyFloatConstr)
    rhs = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}

        return super().__init__(
            attributes=attributes,
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    operand = operand_def(IntegerType)
    result = result_def(IntegerType)

    traits = traits_def(Pure())

    def __init__(self, operand: Operation | SSAValue):
        operand = SSAValue.get(operand)
        return super().__init__(operands=[operand], result_types=[operand.type])


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
    operand = operand_def(IntegerType)
    result = result_def(IntegerType)

    traits = traits_def(Pure())

    def __init__(self, operand: Operation | SSAValue):
        operand = SSAValue.get(operand)
        return super().__init__(operands=[operand], result_types=[operand.type])


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
    operand = operand_def(IntegerType)
    result = result_def(IntegerType)

    traits = traits_def(Pure())

    def __init__(self, operand: Operation | SSAValue):
        operand = SSAValue.get(operand)
        return super().__init__(operands=[operand], result_types=[operand.type])


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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    lhs = operand_def(AnyFloatConstr)
    rhs = operand_def(IntegerType)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}

        return super().__init__(
            attributes=attributes,
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    a = operand_def(AnyFloatConstr)
    b = operand_def(AnyFloatConstr)
    c = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        c: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}

        return super().__init__(
            attributes=attributes,
            operands=[a, b, c],
            result_types=[SSAValue.get(a).type],
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
    lhs = operand_def(IntegerType)
    rhs = operand_def(IntegerType)
    result = result_def(IntegerType)

    traits = traits_def(Pure())

    def __init__(self, lhs: Operation | SSAValue, rhs: Operation | SSAValue):
        lhs = SSAValue.get(lhs)
        rhs = SSAValue.get(rhs)
        return super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    lhs = operand_def(AnyFloatConstr)
    rhs = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}

        return super().__init__(
            attributes=attributes,
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
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
    fastmath = opt_prop_def(FastMathFlagsAttr)
    operand = operand_def(AnyFloatConstr)
    result = result_def(AnyFloatConstr)

    traits = traits_def(Pure())

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        attributes = {"fastmath": fastmath}

        operand = SSAValue.get(operand)
        return super().__init__(
            attributes=attributes, operands=[operand], result_types=[operand.type]
        )


Math = Dialect(
    "math",
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
    ],
)
