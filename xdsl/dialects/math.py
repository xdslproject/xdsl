"""
The math dialect is intended to hold mathematical operations on integer and floating
types beyond simple arithmetics.

See external [documentation](https://mlir.llvm.org/docs/Dialects/MathOps/).
"""

from __future__ import annotations

import abc
import math
from typing import ClassVar

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    AnyFloatConstr,
    ContainerOf,
    FloatAttr,
    IndexType,
    SignlessIntegerConstraint,
)
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure, SameOperandsAndResultType
from xdsl.utils.hints import isa

signlessIntegerLike = ContainerOf(AnyOf([SignlessIntegerConstraint, IndexType]))
floatingPointLike = ContainerOf(AnyFloatConstr)


class SignlessIntegerLikeUnaryMathOperation(IRDLOperation, abc.ABC):
    """A generic signless integer-like unary math operation."""

    T: ClassVar = VarConstraint("T", signlessIntegerLike)

    operand = operand_def(T)
    result = result_def(T)

    assembly_format = "$operand attr-dict `:` type($result)"

    def __init__(self, operand: Operation | SSAValue):
        operand = SSAValue.get(operand)
        super().__init__(
            operands=[operand],
            result_types=[operand.type],
        )


class FloatingPointLikeUnaryMathOperation(IRDLOperation, abc.ABC):
    """A generic floating-point-like unary math operation with fastmath flags."""

    T: ClassVar = VarConstraint("T", floatingPointLike)

    operand = operand_def(T)
    result = result_def(T)

    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    assembly_format = "$operand (`fastmath` `` $fastmath^)? attr-dict `:` type($result)"

    def __init__(
        self, operand: Operation | SSAValue, fastmath: FastMathFlagsAttr | None = None
    ):
        operand = SSAValue.get(operand)
        super().__init__(
            properties={"fastmath": fastmath},
            operands=[operand],
            result_types=[operand.type],
        )


class SignlessIntegerLikeBinaryMathOperation(IRDLOperation, abc.ABC):
    """A generic signless integer-like binary math operation."""

    T: ClassVar = VarConstraint("T", signlessIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
        )


class FloatingPointLikeBinaryMathOperation(IRDLOperation, abc.ABC):
    """A generic floating-point-like binary math operation."""

    T: ClassVar = VarConstraint("T", floatingPointLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)

    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    assembly_format = (
        "$lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)"
    )

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        super().__init__(
            properties={"fastmath": fastmath},
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
        )


@irdl_op_definition
class AbsFOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The absf operation computes the absolute value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type.

    Example:

    // Scalar absolute value.
    %a = math.absf %b : f64
    """

    name = "math.absf"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(abs(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class AbsIOp(SignlessIntegerLikeUnaryMathOperation):
    """
    The absi operation computes the absolute value. It takes one operand of
    integer type (i.e., scalar, tensor or vector) and returns one result of the
    same type.

    Example:

    // Scalar absolute value.
    %a = math.absi %b : i64
    """

    name = "math.absi"

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class AcosOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Arcus cosine of the specified value

    The acos operation computes the arcus cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:
    // Scalar arcus cosine value.
    %a = math.acos %b : f64
    """

    name = "math.acos"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.acos(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class AcoshOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Hyperbolic arcus cosine of the given value

    The acosh operation computes the arcus cosine of a given value. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and
    returns one result of the same type. It has no standard attributes.

    Example:
    // Hyperbolic arcus cosine of scalar value.
    %a = math.acosh %b : f64
    """

    name = "math.acosh"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.acosh(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class AsinOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Arcus sine of the given value

    The asin operation computes the arcus sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:
    // Arcus sine of scalar value.
    %a = math.asin %b : f64
    """

    name = "math.asin"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.asin(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class AsinhOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Hyperbolic arcus sine of the given value

    The asinh operation computes the hyperbolic arcus sine of a given value. It
    takes one operand of floating point type (i.e., scalar, tensor or vector) and
    returns one result of the same type. It has no standard attributes.

    Example:
    // Hyperbolic arcus sine of scalar value.
    %a = math.asinh %b : f64
    """

    name = "math.asinh"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.asinh(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class AtanhOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Hyperbolic arcus tangent of the given value

    The atanh operation computes the hyperbolic arcus tangent of a given value. It
    takes one operand of floating point type (i.e., scalar, tensor or vector) and
    returns one result of the same type. It has no standard attributes.

    Example:
    // Hyperbolic arcus tangent of scalar value.
    %a = math.atanh %b : f64
    """

    name = "math.atanh"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.atanh(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class Atan2Op(FloatingPointLikeBinaryMathOperation, HasFolderInterface):
    """
    The atan2 operation takes two operands and returns one result, all of
    which must be of the same type.  The operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
    Euclidian plane between the positive x-axis and the ray through the point
    (x, y).  It is a generalization of the 1-argument arcus tangent which
    returns the angle on the basis of the ratio y/x.

    See also [wikipedia](https://en.wikipedia.org/wiki/Atan2).

    Example:

    // Scalar variant.
    %a = math.atan2 %b, %c : f32
    """

    name = "math.atan2"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        lhs_cst = self.get_constant(self.lhs)
        rhs_cst = self.get_constant(self.rhs)
        if isa(lhs_cst, FloatAttr) and isa(rhs_cst, FloatAttr):
            return [
                FloatAttr(
                    math.atan2(lhs_cst.value.data, rhs_cst.value.data), lhs_cst.type
                )
            ]
        return None


@irdl_op_definition
class AtanOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The atan operation computes the arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    // Arcus tangent of scalar value.
    %a = math.atan %b : f64
    """

    name = "math.atan"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.atan(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class CbrtOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
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

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.cbrt(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class CeilOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The ceil operation computes the ceiling of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar ceiling value.
    %a = math.ceil %b : f64
    """

    name = "math.ceil"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.ceil(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class CopySignOp(FloatingPointLikeBinaryMathOperation, HasFolderInterface):
    """
    The copysign returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one result of
    the same type. The operands must be of floating point type (i.e., scalar,
    tensor or vector). It has no standard attributes.

    Example:

    // Scalar copysign value.
    %a = math.copysign %b, %c : f64
    """

    name = "math.copysign"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        lhs_cst = self.get_constant(self.lhs)
        rhs_cst = self.get_constant(self.rhs)
        if isa(lhs_cst, FloatAttr) and isa(rhs_cst, FloatAttr):
            return [
                FloatAttr(
                    math.copysign(lhs_cst.value.data, rhs_cst.value.data), lhs_cst.type
                )
            ]
        return None


@irdl_op_definition
class CosOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The `cos` operation computes the cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar cosine value.
    %a = math.cos %b : f64
    """

    name = "math.cos"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.cos(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class CoshOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The cosh operation computes the hyperbolic cosine. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    // Scalar hyperbolic cosine value.
    %a = math.cosh %b : f64
    """

    name = "math.cosh"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.cosh(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class CountLeadingZerosOp(SignlessIntegerLikeUnaryMathOperation):
    """
    The ctlz operation computes the number of leading zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    // Scalar ctlz function value.
    %a = math.ctlz %b : i32
    """

    name = "math.ctlz"

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class CountTrailingZerosOp(SignlessIntegerLikeUnaryMathOperation):
    """
    The cttz operation computes the number of trailing zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    // Scalar cttz function value.
    %a = math.cttz %b : i32
    """

    name = "math.cttz"

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class CtPopOp(SignlessIntegerLikeUnaryMathOperation):
    """
    The ctpop operation computes the number of set bits of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    // Scalar ctpop function value.
    %a = math.ctpop %b : i32
    """

    name = "math.ctpop"

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class ErfOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The erf operation computes the error function. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    // Scalar error function value.
    %a = math.erf %b : f64
    """

    name = "math.erf"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.erf(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class Exp2Op(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The exp operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    // Scalar natural exponential.
    %a = math.exp2 %b : f64
    """

    name = "math.exp2"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.exp2(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class ExpM1Op(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    expm1(x) := exp(x) - 1

    The expm1 operation takes one operand of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    // Scalar natural exponential minus 1.
    %a = math.expm1 %b : f64
    """

    name = "math.expm1"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.expm1(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class ExpOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The exp operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    // Scalar natural exponential.
    %a = math.exp %b : f64
    """

    name = "math.exp"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.exp(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class FPowIOp(IRDLOperation):
    """
    The fpowi operation takes a `base` operand of floating point type
    (i.e. scalar, tensor or vector) and a `power` operand of integer type
    (also scalar, tensor or vector) and returns one result of the same type
    as `base`. The result is `base` raised to the power of `power`.
    The operation is elementwise for non-scalars, e.g.:

    %v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32>

    The result is a vector of:

    [<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]

    Example:

    // Scalar exponentiation.
    %a = math.fpowi %base, %power : f64, i32
    """

    name = "math.fpowi"

    T: ClassVar = VarConstraint("T1", floatingPointLike)

    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))
    lhs = operand_def(T)
    rhs = operand_def(signlessIntegerLike)
    result = result_def(T)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($lhs) `,` type($rhs)"

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}
        super().__init__(
            attributes=attributes,
            operands=[lhs, rhs],
            result_types=[SSAValue.get(lhs).type],
        )


@irdl_op_definition
class FloorOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The floor operation computes the floor of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar floor value.
    %a = math.floor %b : f64
    """

    name = "math.floor"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.floor(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class FmaOp(IRDLOperation):
    """
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

    T: ClassVar = VarConstraint("T", floatingPointLike)

    name = "math.fma"

    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))
    a = operand_def(T)
    b = operand_def(T)
    c = operand_def(T)
    result = result_def(T)

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = (
        "$a `,` $b `,` $c (`fastmath` `` $fastmath^)? attr-dict `:` type($result)"
    )

    def __init__(
        self,
        a: Operation | SSAValue,
        b: Operation | SSAValue,
        c: Operation | SSAValue,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        attributes = {"fastmath": fastmath}

        super().__init__(
            attributes=attributes,
            operands=[a, b, c],
            result_types=[SSAValue.get(a).type],
        )


@irdl_op_definition
class IPowIOp(SignlessIntegerLikeBinaryMathOperation):
    """
    The ipowi operation takes two operands of integer type (i.e., scalar,
    tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:
    // Scalar signed integer exponentiation.
    %a = math.ipowi %b, %c : i32
    """

    name = "math.ipowi"

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class Log10Op(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Computes the base-10 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    // Scalar log10 operation.
    %y = math.log10 %x : f64
    """

    name = "math.log10"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.log10(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class Log1pOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
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

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.log1p(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class Log2Op(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Computes the base-2 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    // Scalar log2 operation.
    %y = math.log2 %x : f64
    """

    name = "math.log2"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.log2(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class LogOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    Computes the base-e logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    // Scalar log operation.
    %y = math.log %x : f64
    """

    name = "math.log"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.log(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class PowFOp(FloatingPointLikeBinaryMathOperation, HasFolderInterface):
    """
    The powf operation takes two operands of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    // Scalar exponentiation.
    %a = math.powf %b, %c : f64
    """

    name = "math.powf"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        lhs_cst = self.get_constant(self.lhs)
        rhs_cst = self.get_constant(self.rhs)
        if isa(lhs_cst, FloatAttr) and isa(rhs_cst, FloatAttr):
            return [
                FloatAttr(
                    math.pow(lhs_cst.value.data, rhs_cst.value.data), lhs_cst.type
                )
            ]
        return None


@irdl_op_definition
class RoundEvenOp(FloatingPointLikeUnaryMathOperation):
    """
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

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class RoundOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
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

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(round(cst.value.data, 0), cst.type)]
        return None


@irdl_op_definition
class RsqrtOp(FloatingPointLikeUnaryMathOperation):
    """
    The rsqrt operation computes the reciprocal of the square root. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:
    // Scalar reciprocal square root value.
    %a = math.rsqrt %b : f64
    """

    name = "math.rsqrt"

    traits = traits_def(Pure(), SameOperandsAndResultType())


@irdl_op_definition
class SinOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The sin operation computes the sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    // Scalar sine value.
    %a = math.sin %b : f64
    """

    name = "math.sin"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.sin(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class SinhOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The sinh operation computes the hyperbolic sine. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and
    returns one result of the same type. It has no standard attributes.

    Example:

    // Scalar hyperbolic sine value.
    %a = math.sinh %b : f64
    """

    name = "math.sinh"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.sinh(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class SqrtOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The sqrt operation computes the square root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:
    // Scalar square root value.
    %a = math.sqrt %b : f64
    """

    name = "math.sqrt"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.sqrt(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class TanOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The tan operation computes the tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    // Scalar tangent value.
    %a = math.tan %b : f64
    """

    name = "math.tan"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.tan(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class TanhOp(FloatingPointLikeUnaryMathOperation, HasFolderInterface):
    """
    The tanh operation computes the hyperbolic tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    // Scalar hyperbolic tangent value.
    %a = math.tanh %b : f64
    """

    name = "math.tanh"

    traits = traits_def(Pure(), SameOperandsAndResultType())

    def fold(self):
        cst = self.get_constant(self.operand)
        if isa(cst, FloatAttr):
            return [FloatAttr(math.tanh(cst.value.data), cst.type)]
        return None


@irdl_op_definition
class TruncOp(FloatingPointLikeUnaryMathOperation):
    """
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

    traits = traits_def(Pure(), SameOperandsAndResultType())


Math = Dialect(
    "math",
    [
        AbsFOp,
        AbsIOp,
        AcosOp,
        AcoshOp,
        AsinOp,
        AsinhOp,
        Atan2Op,
        AtanOp,
        AtanhOp,
        CbrtOp,
        CeilOp,
        CopySignOp,
        CosOp,
        CoshOp,
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
        SinhOp,
        SqrtOp,
        TanOp,
        TanhOp,
        TruncOp,
    ],
)
