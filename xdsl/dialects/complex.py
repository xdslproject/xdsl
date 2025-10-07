from __future__ import annotations

import abc
from typing import ClassVar, cast

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    AnyAttr,
    AnyFloat,
    AnyFloatConstr,
    ArrayAttr,
    ArrayOfConstraint,
    ComplexType,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ParamAttrConstraint,
)
from xdsl.interfaces import ConstantLikeInterface
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    BaseAttr,
    EqIntConstraint,
    IRDLOperation,
    RangeOf,
    VarConstraint,
    base,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.pattern_rewriter import RewritePattern
from xdsl.traits import HasCanonicalizationPatternsTrait, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

ComplexTypeConstr = ComplexType.constr(AnyFloat)


class ComplexUnaryComplexResultOperation(IRDLOperation, abc.ABC):
    """Base class for unary operations on complex numbers."""

    T: ClassVar = VarConstraint("T", ComplexTypeConstr)
    complex = operand_def(T)
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
        super().__init__(
            operands=[operand_ssa],
            result_types=[operand_ssa.type],
            properties={"fastmath": fastmath},
        )


class ComplexUnaryRealResultOperation(IRDLOperation, abc.ABC):
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


class ComplexBinaryOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import (
            FoldConstConstOp,
        )

        return (FoldConstConstOp(),)


@irdl_op_definition
class AbsOp(ComplexUnaryRealResultOperation):
    name = "complex.abs"


@irdl_op_definition
class AddOp(ComplexBinaryOp):
    name = "complex.add"
    traits = traits_def(
        Pure(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class AngleOp(ComplexUnaryRealResultOperation):
    name = "complex.angle"


@irdl_op_definition
class Atan2Op(ComplexBinaryOp):
    name = "complex.atan2"


@irdl_op_definition
class BitcastOp(IRDLOperation):
    name = "complex.bitcast"
    operand = operand_def(
        ComplexType.constr(AnyFloatConstr) | AnyFloatConstr | BaseAttr(IntegerType)
    )
    result = result_def(
        ComplexType.constr(AnyFloatConstr) | AnyFloatConstr | BaseAttr(IntegerType)
    )

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict `:` type($operand) `to` type($result)"

    def __init__(self, operand: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[operand], result_types=[result_type])

    def verify_(self) -> None:
        in_type = self.operand.type
        res_type = self.result.type
        if not BitcastOp._are_types_bitcastable(in_type, res_type):
            raise VerifyException(
                f"Expected ('{in_type}', '{res_type}') to be bitcast between complex and equal arith types"
            )

    @staticmethod
    def _have_compatible_types(type_a: Attribute, type_b: Attribute) -> bool:
        if (
            isa(type_a, ComplexType[AnyFloat]) and isa(type_b, AnyFloat | IntegerType)
        ) or (
            isa(type_a, AnyFloat | IntegerType) and isa(type_b, ComplexType[AnyFloat])
        ):
            return True
        return False

    @staticmethod
    def _are_types_bitcastable(type_a: Attribute, type_b: Attribute) -> bool:
        if not BitcastOp._have_compatible_types(type_a, type_b):
            return False
        complex_type = type_a if isa(type_a, ComplexType) else type_b
        arith_type = type_a if not isa(type_a, ComplexType) else type_b
        return (complex_type.get_element_type().bitwidth << 1) == arith_type.bitwidth


@irdl_op_definition
class ConjOp(ComplexUnaryComplexResultOperation):
    name = "complex.conj"


@irdl_op_definition
class ConstantOp(IRDLOperation, ConstantLikeInterface):
    name = "complex.constant"
    T: ClassVar = VarConstraint("T", AnyFloatConstr | base(IntegerType))
    value = prop_def(
        ArrayOfConstraint(
            RangeOf(
                AnyOf(
                    [
                        ParamAttrConstraint(IntegerAttr, (AnyAttr(), T)),
                        ParamAttrConstraint(FloatAttr, (AnyAttr(), T)),
                    ]
                )
            ).of_length(EqIntConstraint(2))
        )
    )

    # In contrast to other operations, `complex.constant` can
    # have any complex result type, not just floating point:
    complex = result_def(ComplexType.constr(T))

    traits = traits_def(Pure())

    assembly_format = "$value attr-dict `:` type($complex)"

    def __init__(self, value: ArrayAttr, result_type: ComplexType):
        super().__init__(properties={"value": value}, result_types=[result_type])

    def get_constant_value(self) -> Attribute:
        return self.value

    @staticmethod
    def from_tuple_and_width(
        value: tuple[int | float, int | float], width: int
    ) -> ConstantOp:
        if width == 1 and isa(value, tuple[int, int]):
            result_type = ComplexType(IntegerType(1))
            value = ArrayAttr([IntegerAttr(value[0], 1), IntegerAttr(value[1], 1)])
        elif width > 1:
            value = ArrayAttr([FloatAttr(value[0], width), FloatAttr(value[1], width)])
            result_type = ComplexType(value.data[0].type)
        else:
            raise ValueError(
                f"Not expected 'width'={width} with 'tuple'=[{type(value[0])}, {type(value[1])}"
            )
        return ConstantOp(value, result_type)


@irdl_op_definition
class CosOp(ComplexUnaryComplexResultOperation):
    name = "complex.cos"


class CreateOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import (
            RedundantCreateOpPattern,
        )

        return (RedundantCreateOpPattern(),)


@irdl_op_definition
class CreateOp(IRDLOperation):
    name = "complex.create"
    T: ClassVar = VarConstraint("T", AnyFloatConstr)
    real = operand_def(T)
    imaginary = operand_def(T)
    complex = result_def(ComplexType.constr(T))

    traits = traits_def(
        Pure(),
        CreateOpCanonicalizationPatternsTrait(),
    )

    assembly_format = "$real `,` $imaginary attr-dict `:` type($complex)"

    def __init__(
        self,
        real: SSAValue | Operation,
        imaginary: SSAValue | Operation,
        result_type: ComplexType,
    ):
        super().__init__(operands=[real, imaginary], result_types=[result_type])


@irdl_op_definition
class DivOp(ComplexBinaryOp):
    name = "complex.div"
    traits = traits_def(
        Pure(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class EqualOp(ComplexCompareOp):
    name = "complex.eq"


@irdl_op_definition
class ExpOp(ComplexUnaryComplexResultOperation):
    name = "complex.exp"


@irdl_op_definition
class Expm1Op(ComplexUnaryComplexResultOperation):
    name = "complex.expm1"


class ReImOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import (
            ReImNegOpPattern,
            ReImRedundantOpPattern,
        )

        return (
            ReImRedundantOpPattern(),
            ReImNegOpPattern(),
        )


@irdl_op_definition
class ImOp(ComplexUnaryRealResultOperation):
    name = "complex.im"
    traits = traits_def(
        Pure(),
        ReImOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class LogOp(ComplexUnaryComplexResultOperation):
    name = "complex.log"


@irdl_op_definition
class Log1pOp(ComplexUnaryComplexResultOperation):
    name = "complex.log1p"


@irdl_op_definition
class MulOp(ComplexBinaryOp):
    name = "complex.mul"
    traits = traits_def(
        Pure(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class NegOp(ComplexUnaryComplexResultOperation):
    name = "complex.neg"


@irdl_op_definition
class NotEqualOp(ComplexCompareOp):
    name = "complex.neq"


@irdl_op_definition
class PowOp(ComplexBinaryOp):
    name = "complex.pow"


@irdl_op_definition
class ReOp(ComplexUnaryRealResultOperation):
    name = "complex.re"
    traits = traits_def(
        Pure(),
        ReImOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class RsqrtOp(ComplexUnaryComplexResultOperation):
    name = "complex.rsqrt"


@irdl_op_definition
class SignOp(ComplexUnaryComplexResultOperation):
    name = "complex.sign"


@irdl_op_definition
class SinOp(ComplexUnaryComplexResultOperation):
    name = "complex.sin"


@irdl_op_definition
class SqrtOp(ComplexUnaryComplexResultOperation):
    name = "complex.sqrt"


@irdl_op_definition
class SubOp(ComplexBinaryOp):
    name = "complex.sub"
    traits = traits_def(
        Pure(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class TanOp(ComplexUnaryComplexResultOperation):
    name = "complex.tan"


@irdl_op_definition
class TanhOp(ComplexUnaryComplexResultOperation):
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
