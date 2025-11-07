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
from xdsl.interfaces import ConstantLikeInterface, HasFolderInterface
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
from xdsl.traits import (
    Commutative,
    HasCanonicalizationPatternsTrait,
    Pure,
)
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


class ComplexBinaryOp(IRDLOperation, HasFolderInterface, abc.ABC):
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

    @staticmethod
    def py_operation(
        lhs: tuple[float, float], rhs: tuple[float, float]
    ) -> tuple[float, float] | None:
        """
        Performs a python function corresponding to this operation.

        If `i := py_operation(lhs, rhs)` is an int, then this operation can be
        canonicalized to a constant with value `i` when the inputs are constants
        with values `lhs` and `rhs`.
        """
        return None

    def fold(self):
        lhs = self.get_constant(self.lhs)
        rhs = self.get_constant(self.rhs)
        if lhs is not None and rhs is not None:
            if isa(lhs, ArrayAttr[FloatAttr[AnyFloat]]) and isa(
                rhs, ArrayAttr[FloatAttr[AnyFloat]]
            ):
                assert lhs.data[0].type == rhs.data[0].type
                assert lhs.data[1].type == rhs.data[1].type
                re_lhs, im_lhs = lhs.data[0].value.data, lhs.data[1].value.data
                re_rhs, im_rhs = rhs.data[0].value.data, rhs.data[1].value.data
                res = self.py_operation((re_lhs, im_lhs), (re_rhs, im_rhs))
                if res is not None:
                    return (
                        ArrayAttr(
                            [
                                FloatAttr(res[0], lhs.data[0].type),
                                FloatAttr(res[1], lhs.data[0].type),
                            ]
                        ),
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


class UnaryOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import (
            RedundantUnaryOpOpPattern,
        )

        return (RedundantUnaryOpOpPattern(),)


@irdl_op_definition
class AbsOp(ComplexUnaryRealResultOperation):
    name = "complex.abs"


class AddSubCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import AddSubOpPattern

        return (AddSubOpPattern(),)


@irdl_op_definition
class AddOp(ComplexBinaryOp):
    name = "complex.add"
    traits = traits_def(
        Pure(),
        Commutative(),
        AddSubCanonicalizationPatternsTrait(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )

    @staticmethod
    def py_operation(
        lhs: tuple[float, float], rhs: tuple[float, float]
    ) -> tuple[float, float] | None:
        return (lhs[0] + rhs[0], lhs[1] + rhs[1])


@irdl_op_definition
class AngleOp(ComplexUnaryRealResultOperation):
    name = "complex.angle"


@irdl_op_definition
class Atan2Op(ComplexBinaryOp):
    name = "complex.atan2"


class BitcastOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import BitcastOpPattern

        return (BitcastOpPattern(),)


@irdl_op_definition
class BitcastOp(IRDLOperation):
    name = "complex.bitcast"
    operand = operand_def(
        ComplexType.constr(AnyFloatConstr) | AnyFloatConstr | BaseAttr(IntegerType)
    )
    result = result_def(
        ComplexType.constr(AnyFloatConstr) | AnyFloatConstr | BaseAttr(IntegerType)
    )

    traits = traits_def(
        Pure(),
        BitcastOpCanonicalizationPatternsTrait(),
    )

    assembly_format = "$operand attr-dict `:` type($operand) `to` type($result)"

    def __init__(self, operand: SSAValue | Operation, result_type: Attribute):
        super().__init__(operands=[operand], result_types=[result_type])


@irdl_op_definition
class ConjOp(ComplexUnaryComplexResultOperation):
    name = "complex.conj"
    traits = traits_def(
        Pure(),
        UnaryOpCanonicalizationPatternsTrait(),
    )


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
            value_arr = ArrayAttr([IntegerAttr(value[0], 1), IntegerAttr(value[1], 1)])
        elif width > 1:
            value_arr = ArrayAttr(
                [FloatAttr(value[0], width), FloatAttr(value[1], width)]
            )
            result_type = ComplexType(value_arr.data[0].type)
        else:
            raise ValueError(
                f"Not expected 'width'={width} with 'tuple'=[{type(value[0])}, {type(value[1])}"
            )
        return ConstantOp(value_arr, result_type)


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

    @staticmethod
    def py_operation(
        lhs: tuple[float, float], rhs: tuple[float, float]
    ) -> tuple[float, float] | None:
        re_lhs, im_lhs = lhs[0], lhs[1]
        re_rhs, im_rhs = rhs[0], rhs[1]
        if re_rhs == 0.0 and im_rhs == 0.0:
            if re_lhs == 0.0:
                real = float("nan")
            else:
                real = float("inf") if re_lhs > 0 else float("-inf")
            if im_lhs == 0.0:
                imag = float("nan")
            else:
                imag = float("inf") if im_lhs > 0 else float("-inf")
        else:
            real = (re_lhs * re_rhs + im_lhs * im_rhs) / (re_rhs**2 + im_rhs**2)
            imag = (im_lhs * re_rhs - re_lhs * im_rhs) / (re_rhs**2 + im_rhs**2)
        return (real, imag)


@irdl_op_definition
class EqualOp(ComplexCompareOp):
    name = "complex.eq"


@irdl_op_definition
class ExpOp(ComplexUnaryComplexResultOperation):
    name = "complex.exp"
    traits = traits_def(
        Pure(),
        UnaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class Expm1Op(ComplexUnaryComplexResultOperation):
    name = "complex.expm1"


class ImOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import (
            ImNegOpPattern,
            ImRedundantOpPattern,
        )

        return (
            ImRedundantOpPattern(),
            ImNegOpPattern(),
        )


@irdl_op_definition
class ImOp(ComplexUnaryRealResultOperation):
    name = "complex.im"
    traits = traits_def(
        Pure(),
        ImOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class LogOp(ComplexUnaryComplexResultOperation):
    name = "complex.log"
    traits = traits_def(
        Pure(),
        UnaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class Log1pOp(ComplexUnaryComplexResultOperation):
    name = "complex.log1p"


@irdl_op_definition
class MulOp(ComplexBinaryOp):
    name = "complex.mul"
    traits = traits_def(
        Pure(),
        Commutative(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )

    @staticmethod
    def py_operation(
        lhs: tuple[float, float], rhs: tuple[float, float]
    ) -> tuple[float, float] | None:
        re_lhs, im_lhs = lhs[0], lhs[1]
        re_rhs, im_rhs = rhs[0], rhs[1]
        return (re_lhs * re_rhs - im_lhs * im_rhs, re_lhs * im_rhs + im_lhs * re_rhs)


@irdl_op_definition
class NegOp(ComplexUnaryComplexResultOperation):
    name = "complex.neg"
    traits = traits_def(
        Pure(),
        UnaryOpCanonicalizationPatternsTrait(),
    )


@irdl_op_definition
class NotEqualOp(ComplexCompareOp):
    name = "complex.neq"


@irdl_op_definition
class PowOp(ComplexBinaryOp):
    name = "complex.pow"


class ReOpCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import (
            ReNegOpPattern,
            ReRedundantOpPattern,
        )

        return (
            ReRedundantOpPattern(),
            ReNegOpPattern(),
        )


@irdl_op_definition
class ReOp(ComplexUnaryRealResultOperation):
    name = "complex.re"
    traits = traits_def(
        Pure(),
        ReOpCanonicalizationPatternsTrait(),
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


class SubAddCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.complex import SubAddOpPattern

        return (SubAddOpPattern(),)


@irdl_op_definition
class SubOp(ComplexBinaryOp):
    name = "complex.sub"
    traits = traits_def(
        Pure(),
        SubAddCanonicalizationPatternsTrait(),
        ComplexBinaryOpCanonicalizationPatternsTrait(),
    )

    @staticmethod
    def py_operation(
        lhs: tuple[float, float], rhs: tuple[float, float]
    ) -> tuple[float, float] | None:
        return (lhs[0] - rhs[0], lhs[1] - rhs[1])


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
