from __future__ import annotations

import abc
from typing import ClassVar, cast

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    AnyFloatConstr,
    ArrayAttr,
    ComplexType,
    FloatAttr,
    IntegerType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import ConstantLike, Pure, SameOperandsAndResultType
from xdsl.utils.exceptions import VerifyException

# Type constraint for our new ComplexType
ComplexTypeConstr = base(ComplexType)


class ComplexUnaryOp(IRDLOperation, abc.ABC):
    """Base class for unary operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    complex = operand_def(T)
    result = result_def(T)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = (
        "$complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)"
    )

    def __init__(
        self,
        operand: SSAValue | Operation,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        operand_ssa = SSAValue.get(operand)
        super().__init__(
            operands=[operand_ssa],
            result_types=[operand_ssa.type],
            properties={"fastmath": fastmath},
        )


class ComplexUnaryRealResultOp(IRDLOperation, abc.ABC):
    """Base class for unary operations on complex numbers that return a float."""

    T: ClassVar = VarConstraint("T", AnyFloatConstr)

    complex = operand_def(ComplexType.constr(T))
    result = result_def(T)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    traits = traits_def(Pure())

    assembly_format = (
        "$complex (`fastmath` `` $fastmath^)? attr-dict `:` type($complex)"
    )

    def __init__(
        self,
        operand: SSAValue | Operation,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        operand_ssa = SSAValue.get(operand)
        element_type = cast(ComplexType, operand_ssa.type).element_type
        super().__init__(
            operands=[operand_ssa],
            result_types=[element_type],
            properties={"fastmath": fastmath},
        )

    def verify_(self):
        element_type = cast(ComplexType, self.complex.type).element_type
        if self.result.type != element_type:
            raise VerifyException(
                f"result type {self.result.type} does not match "
                f"complex element type {element_type}"
            )


class ComplexBinaryOp(IRDLOperation, abc.ABC):
    """Base class for binary operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(T)
    fastmath = prop_def(FastMathFlagsAttr, default_value=FastMathFlagsAttr("none"))

    traits = traits_def(Pure())

    assembly_format = (
        "$lhs `,` $rhs (`fastmath` `` $fastmath^)? attr-dict `:` type($result)"
    )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        fastmath: FastMathFlagsAttr | None = None,
    ):
        lhs_ssa = SSAValue.get(lhs)
        super().__init__(
            operands=[lhs_ssa, rhs],
            result_types=[lhs_ssa.type],
            properties={"fastmath": fastmath},
        )

    def verify_(self):
        if self.lhs.type != self.rhs.type:
            raise VerifyException("lhs and rhs of binary op must have the same type")
        if self.lhs.type != self.result.type:
            raise VerifyException("result type must match operand types")


class ComplexCompareOp(IRDLOperation, abc.ABC):
    """Base class for comparison operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    lhs = operand_def(T)
    rhs = operand_def(T)
    result = result_def(IntegerType(1))

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($lhs)"

    def __init__(self, lhs: SSAValue | Operation, rhs: SSAValue | Operation):
        super().__init__(operands=[lhs, rhs], result_types=[IntegerType(1)])

    def verify_(self):
        if self.lhs.type != self.rhs.type:
            raise VerifyException("lhs and rhs of compare op must have the same type")


@irdl_op_definition
class AbsOp(ComplexUnaryRealResultOp):
    name = "complex.abs"


@irdl_op_definition
class AddOp(ComplexBinaryOp):
    name = "complex.add"


@irdl_op_definition
class AngleOp(ComplexUnaryRealResultOp):
    name = "complex.angle"


@irdl_op_definition
class Atan2Op(ComplexBinaryOp):
    name = "complex.atan2"


@irdl_op_definition
class BitcastOp(IRDLOperation):
    name = "complex.bitcast"
    operand = operand_def(AnyAttr())
    result = result_def(AnyAttr())

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict `:` type($operand) `to` type($result)"

    def __init__(self, operand: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[operand], result_types=[result_type])


@irdl_op_definition
class ConjOp(ComplexUnaryOp):
    name = "complex.conj"


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "complex.constant"
    value = prop_def(ArrayAttr)
    complex = result_def(ComplexTypeConstr)

    traits = traits_def(Pure(), ConstantLike())

    assembly_format = "$value attr-dict `:` type($complex)"

    def __init__(self, value: ArrayAttr, result_type: ComplexType):
        super().__init__(properties={"value": value}, result_types=[result_type])

    def verify_(self):
        if len(self.value.data) != 2:
            raise VerifyException("complex.constant must have an array of 2 values")
        elem_type = self.complex.type.element_type
        # TODO: this check might be wrong, x can be something other than float?
        if any(cast(FloatAttr, x).type != elem_type for x in self.value.data):
            raise VerifyException(
                "complex.constant value types must match the complex element type"
            )


@irdl_op_definition
class CosOp(ComplexUnaryOp):
    name = "complex.cos"


@irdl_op_definition
class CreateOp(IRDLOperation):
    name = "complex.create"
    T: ClassVar = VarConstraint("T", AnyFloatConstr)
    real = operand_def(T)
    imaginary = operand_def(T)
    complex = result_def(ComplexType.constr(T))

    traits = traits_def(Pure())

    assembly_format = "$real `,` $imaginary attr-dict `:` type($complex)"

    def __init__(
        self,
        real: SSAValue | Operation,
        imaginary: SSAValue | Operation,
        result_type: ComplexType,
    ):
        super().__init__(operands=[real, imaginary], result_types=[result_type])

    def verify_(self):
        real_type = self.real.type
        imag_type = self.imaginary.type
        if real_type != imag_type:
            raise VerifyException(
                "real and imaginary parts of complex.create must have the same type"
            )
        complex_element_type = self.complex.type.element_type
        if real_type != complex_element_type:
            raise VerifyException(
                "operands of complex.create must match the element type of the result"
            )


@irdl_op_definition
class DivOp(ComplexBinaryOp):
    name = "complex.div"


@irdl_op_definition
class EqualOp(ComplexCompareOp):
    name = "complex.eq"


@irdl_op_definition
class ExpOp(ComplexUnaryOp):
    name = "complex.exp"


@irdl_op_definition
class Expm1Op(ComplexUnaryOp):
    name = "complex.expm1"


@irdl_op_definition
class ImOp(ComplexUnaryRealResultOp):
    name = "complex.im"


@irdl_op_definition
class LogOp(ComplexUnaryOp):
    name = "complex.log"


@irdl_op_definition
class Log1pOp(ComplexUnaryOp):
    name = "complex.log1p"


@irdl_op_definition
class MulOp(ComplexBinaryOp):
    name = "complex.mul"


@irdl_op_definition
class NegOp(ComplexUnaryOp):
    name = "complex.neg"


@irdl_op_definition
class NotEqualOp(ComplexCompareOp):
    name = "complex.neq"


@irdl_op_definition
class PowOp(ComplexBinaryOp):
    name = "complex.pow"


@irdl_op_definition
class ReOp(ComplexUnaryRealResultOp):
    name = "complex.re"


@irdl_op_definition
class RsqrtOp(ComplexUnaryOp):
    name = "complex.rsqrt"


@irdl_op_definition
class SignOp(ComplexUnaryOp):
    name = "complex.sign"


@irdl_op_definition
class SinOp(ComplexUnaryOp):
    name = "complex.sin"


@irdl_op_definition
class SqrtOp(ComplexUnaryOp):
    name = "complex.sqrt"


@irdl_op_definition
class SubOp(ComplexBinaryOp):
    name = "complex.sub"


@irdl_op_definition
class TanOp(ComplexUnaryOp):
    name = "complex.tan"


@irdl_op_definition
class TanhOp(ComplexUnaryOp):
    name = "complex.tanh"


Complex = Dialect(
    "complex",
    [
        AbsOp,
        AddOp,
        AngleOp,
        Atan2Op,
        BitcastOp,
        ConjOp,
        ConstantOp,
        CosOp,
        CreateOp,
        DivOp,
        EqualOp,
        ExpOp,
        Expm1Op,
        ImOp,
        LogOp,
        Log1pOp,
        MulOp,
        NegOp,
        NotEqualOp,
        PowOp,
        ReOp,
        RsqrtOp,
        SignOp,
        SinOp,
        SqrtOp,
        SubOp,
        TanOp,
        TanhOp,
    ],
)
