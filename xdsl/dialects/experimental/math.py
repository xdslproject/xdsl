from __future__ import annotations

from typing import Annotated, Union

from xdsl.dialects.builtin import IntegerType, AnyFloat, Attribute
from xdsl.ir import Operation, SSAValue, OpResult
from xdsl.irdl import irdl_op_definition, OptOpAttr, Operand
from xdsl.dialects.arith import FastMathFlagsAttr


@irdl_op_definition
class FPowIOp(Operation):
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
    name: str = "math.fpowi"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    lhs: Annotated[Operand, AnyFloat]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(lhs: Union[Operation, SSAValue],
            rhs: Union[Operation, SSAValue],
            fastmath: FastMathFlagsAttr | None = None) -> FPowIOp:
        attributes: dict[str, Attribute] = {}
        if fastmath is not None:
            attributes["fastmath"] = fastmath

        lhs = SSAValue.get(lhs)
        return FPowIOp.build(attributes=attributes,
                             operands=[lhs, rhs],
                             result_types=[lhs.typ])


@irdl_op_definition
class IPowIOp(Operation):
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
    name: str = "math.ipowi"
    lhs: Annotated[Operand, IntegerType]
    rhs: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]

    @staticmethod
    def get(lhs: Union[Operation, SSAValue], rhs: Union[Operation,
                                                        SSAValue]) -> IPowIOp:
        lhs = SSAValue.get(lhs)
        return IPowIOp.build(operands=[lhs, rhs], result_types=[lhs.typ])


@irdl_op_definition
class PowFOp(Operation):
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
    name: str = "math.powf"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    lhs: Annotated[Operand, AnyFloat]
    rhs: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(lhs: Union[Operation, SSAValue],
            rhs: Union[Operation, SSAValue],
            fastmath: FastMathFlagsAttr | None = None) -> PowFOp:
        attributes: dict[str, Attribute] = {}
        if fastmath is not None:
            attributes["fastmath"] = fastmath

        lhs = SSAValue.get(lhs)
        return PowFOp.build(attributes=attributes,
                            operands=[lhs, rhs],
                            result_types=[lhs.typ])


@irdl_op_definition
class RsqrtOp(Operation):
    """
    The rsqrt operation computes the reciprocal of the square root. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:
    // Scalar reciprocal square root value.
    %a = math.rsqrt %b : f64
    """
    name: str = "math.rsqrt"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(operand: Union[Operation, SSAValue],
            fastmath: FastMathFlagsAttr | None = None) -> RsqrtOp:
        attributes: dict[str, Attribute] = {}
        if fastmath is not None:
            attributes["fastmath"] = fastmath

        operand = SSAValue.get(operand)
        return RsqrtOp.build(attributes=attributes,
                             operands=[operand],
                             result_types=[operand.typ])


@irdl_op_definition
class SqrtOp(Operation):
    """
    The sqrt operation computes the square root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:
    // Scalar square root value.
    %a = math.sqrt %b : f64
    """
    name: str = "math.sqrt"
    fastmath: OptOpAttr[FastMathFlagsAttr]
    operand: Annotated[Operand, AnyFloat]
    result: Annotated[OpResult, AnyFloat]

    @staticmethod
    def get(operand: Union[Operation, SSAValue],
            fastmath: FastMathFlagsAttr | None = None) -> SqrtOp:
        attributes: dict[str, Attribute] = {}
        if fastmath is not None:
            attributes["fastmath"] = fastmath

        operand = SSAValue.get(operand)
        return SqrtOp.build(attributes=attributes,
                            operands=[operand],
                            result_types=[operand.typ])
