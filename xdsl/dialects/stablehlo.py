"""
https://github.com/openxla/stablehlo/blob/main/docs/spec.md

StableHLO is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that consume StableHLO programs.
"""

import abc
from collections.abc import Sequence
from typing import Annotated, ClassVar, TypeAlias, cast

from xdsl.dialects.builtin import (
    I32,
    I64,
    AnyTensorType,
    AnyTensorTypeConstr,
    ArrayAttr,
    DenseArrayBase,
    IntegerAttr,
    IntegerType,
    TensorType,
    i64,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    StrEnum,
    TypeAttribute,
)
from xdsl.irdl import (
    BaseAttr,
    ConstraintVar,
    IRDLOperation,
    ParameterDef,
    VarConstraint,
    attr_def,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException

# region Abstract Base Classes


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    # TODO: Remove this constraint for complex types.
    T: ClassVar[VarConstraint[AnyTensorType]] = VarConstraint("T", base(AnyTensorType))

    lhs = operand_def(T)
    rhs = operand_def(T)

    result = result_def(T)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


# endregion

# region Attributes


class Precision(StrEnum):
    """
    XLA precision for an operand. Has backend specific meaning.
    """

    DEFAULT = "DEFAULT"
    HIGH = "HIGH"
    HIGHEST = "HIGHEST"


@irdl_attr_definition
class PrecisionAttr(EnumAttribute[Precision], SpacedOpaqueSyntaxAttribute):
    """
    XLA precision for an operand. Has backend specific meaning.

    https://github.com/openxla/stablehlo/blob/b075e948092d8a27ed0be48f4f8dbaa6df7e2e3e/stablehlo/dialect/StablehloEnums.td#L46
    """

    name = "stablehlo.precision"


@irdl_attr_definition
class TokenType(TypeAttribute, ParametrizedAttribute):
    """
    Token types represent tokens, i.e. opaque values produced and consumed by some operations.
    Tokens are used for imposing execution order on operations as described in the Execution section.

    E.g.,

      // %input0: !stablehlo.token
      // %input1: !stablehlo.token
      %result = "stablehlo.after_all"(%input0, %input1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
    """

    name = "stablehlo.token"


@irdl_attr_definition
class DotAttr(ParametrizedAttribute):
    """
    Attribute that models the dimension information for dot.

    https://github.com/openxla/stablehlo/blob/b075e948092d8a27ed0be48f4f8dbaa6df7e2e3e/stablehlo/dialect/StablehloAttrs.td#L82
    """

    name = "stablehlo.dot"

    lhs_batching_dimensions: ParameterDef[ArrayAttr[IntegerAttr[I64]]]
    rhs_batching_dimensions: ParameterDef[ArrayAttr[IntegerAttr[I64]]]
    lhs_contracting_dimensions: ParameterDef[ArrayAttr[IntegerAttr[I64]]]
    rhs_contracting_dimensions: ParameterDef[ArrayAttr[IntegerAttr[I64]]]

    @staticmethod
    def _print_parameter(
        name: str, value: ArrayAttr[IntegerAttr[I64]], printer: Printer
    ):
        printer.print_string(f"\n{name} = [")
        printer.print_list(
            value.data,
            lambda dim: printer.print_string(f"{dim.value.data}"),
        )
        printer.print_string("]")

    @staticmethod
    def _parse_parameter(name: str, parser: AttrParser) -> ArrayAttr[IntegerAttr[I64]]:
        parser.parse_characters(name)
        parser.parse_punctuation("=")
        value = parser.parse_comma_separated_list(
            AttrParser.Delimiter.SQUARE,
            lambda: IntegerAttr(parser.parse_integer(), i64),
        )
        return ArrayAttr(value)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            with printer.indented():
                DotAttr._print_parameter(
                    "lhs_batching_dimensions", self.lhs_batching_dimensions, printer
                )
                printer.print_string(",")
                DotAttr._print_parameter(
                    "rhs_batching_dimensions", self.rhs_batching_dimensions, printer
                )
                printer.print_string(",")
                DotAttr._print_parameter(
                    "lhs_contracting_dimensions",
                    self.lhs_contracting_dimensions,
                    printer,
                )
                printer.print_string(",")
                DotAttr._print_parameter(
                    "rhs_contracting_dimensions",
                    self.rhs_contracting_dimensions,
                    printer,
                )
            printer.print_string("\n")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            lhs_batching_dimensions = DotAttr._parse_parameter(
                "lhs_batching_dimensions", parser
            )
            parser.parse_punctuation(",")
            rhs_batching_dimensions = DotAttr._parse_parameter(
                "rhs_batching_dimensions", parser
            )
            parser.parse_punctuation(",")
            lhs_contracting_dimensions = DotAttr._parse_parameter(
                "lhs_contracting_dimensions", parser
            )
            parser.parse_punctuation(",")
            rhs_contracting_dimensions = DotAttr._parse_parameter(
                "rhs_contracting_dimensions", parser
            )

            return (
                lhs_batching_dimensions,
                rhs_batching_dimensions,
                lhs_contracting_dimensions,
                rhs_contracting_dimensions,
            )


# endregion


@irdl_op_definition
class AbsOp(IRDLOperation):
    """
    Performs element-wise abs operation on operand tensor and produces a result tensor.
    Depending on the element type, does the following:

    * For signed integers: integer modulus.
    * For floats: abs from IEEE-754.
    * For complex numbers: complex modulus.
    * For quantized types: dequantize_op_quantize(abs, operand, type(result)).

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs
    """

    name = "stablehlo.abs"

    # TODO: Remove this constraint for complex types.
    T: ClassVar[VarConstraint[AnyTensorType]] = VarConstraint("T", base(AnyTensorType))

    operand = operand_def(T)
    result = result_def(T)

    def __init__(self, operand: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            # TODO: Constraints for complex types.
            result_type = operand.type
        super().__init__(operands=(operand,), result_types=(result_type,))


@irdl_op_definition
class AddOp(ElementwiseBinaryOperation):
    """
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For booleans: logical OR.
    * For integers: integer addition.
    * For floats: `addition` from IEEE-754.
    * For complex numbers: complex addition.
    * For quantized types: `dequantize_op_quantize(add, lhs, rhs, type(result))`.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add
    """

    name = "stablehlo.add"


@irdl_op_definition
class AfterAllOp(IRDLOperation):
    """
    Ensures that the operations producing the inputs are executed before any operations that depend on result.
    Execution of this operation does nothing, it only exists to establish data dependencies from result to inputs.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all
    """

    name = "stablehlo.after_all"
    inputs = var_operand_def(TokenType)
    result = result_def(TokenType)

    def __init__(self, inputs: Sequence[SSAValue]):
        super().__init__(operands=[inputs], result_types=(TokenType(),))


IntegerTensorType: TypeAlias = TensorType[IntegerType]


@irdl_op_definition
class AndOp(IRDLOperation):
    """
    Performs element-wise AND of two tensors lhs and rhs and produces a result tensor. Depending on the element type, does the following:

    For booleans: logical AND.
    For integers: bitwise AND.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and
    """

    name = "stablehlo.and"

    T: ClassVar[VarConstraint[IntegerTensorType]] = VarConstraint(
        "T", base(IntegerTensorType)
    )

    lhs = operand_def(T)
    rhs = operand_def(T)

    result = result_def(T)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


# TODO: Change to SI32 once StableHLO adopts signful integer semantics
# See: https://github.com/openxla/stablehlo/issues/22
# https://github.com/openxla/stablehlo/issues/2489
SI32TensorType: TypeAlias = TensorType[I32]


@irdl_op_definition
class CaseOp(IRDLOperation):
    """
    Semantics

    Produces the output from executing exactly one function from branches depending on the value of index.
    More formally, result = selected_branch() where:

    selected_branch = branches[index] if 0 <= index < size(branches).
    selected_branch = branches[-1] otherwise.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
    """

    name = "stablehlo.case"
    index = operand_def(SI32TensorType)
    branches = var_region_def("single_block")
    _results = var_result_def(AnyTensorTypeConstr | BaseAttr(TokenType))

    def __init__(
        self,
        index: SSAValue,
        branches: Sequence[Region],
        result_types: Sequence[AnyTensorType | TokenType],
    ):
        super().__init__(
            operands=(index,), result_types=(result_types,), regions=(branches,)
        )


@irdl_op_definition
class BitcastConvertOp(IRDLOperation):
    """
    Performs a bitcast operation on operand tensor and produces a result tensor
    where the bits of the entire operand tensor are reinterpreted using the type of the result tensor.

    More formally, given E = element_type(operand), E' = element_type(result), and R = rank(operand):

    If num_bits(E') < num_bits(E), bits(result[i0, ..., iR-1, :]) = bits(operand[i0, ..., iR-1]).
    If num_bits(E') > num_bits(E), bits(result[i0, ..., iR-2]) = bits(operand[i0, ..., iR-2, :]).
    If num_bits(E') = num_bits(E), bits(result[i0, ..., iR-1]) = bits(operand[i0, ..., iR-1]).

    bits returns in-memory representation of a given value,
    and its behavior is implementation-defined because the exact representation of tensors is implementation-defined,
    and the exact representation of element types is implementation-defined as well.
    """

    name = "stablehlo.bitcast_convert"
    input = operand_def(AnyTensorType)
    result = result_def(AnyTensorType)

    def __init__(self, input: SSAValue, result: Attribute):
        super().__init__(operands=(input,), result_types=(result,))


@irdl_op_definition
class MultiplyOp(ElementwiseBinaryOperation):
    """
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For booleans: logical AND.
    * For integers: integer multiplication.
    * For floats: `multiplication` from IEEE-754.
    * For complex numbers: complex multiplication.
    * For quantized types:
    * `dequantize_op_quantize(multiply, lhs, rhs, type(result))`.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply
    """

    name = "stablehlo.multiply"


@irdl_op_definition
class SubtractOp(ElementwiseBinaryOperation):
    """
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For integers: integer subtraction.
    * For floats: `subtraction` from IEEE-754.
    * For complex numbers: complex subtraction.
    * For quantized types:
    * `dequantize_op_quantize(subtract, lhs, rhs, type(result))`.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract
    """

    name = "stablehlo.subtract"


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """This op is un-documented.

    StableHLO's return is used inside of the bodies of StableHLO ops.
    It behaves like func.return but for StableHLO ops.
    The func.return op is used inside of func.func op.

    https://discord.com/channels/999073994483433573/1259494021269688360/1259992088565645312
    """

    name = "stablehlo.return"

    input = var_operand_def(AnyTensorType)
    traits = frozenset([IsTerminator()])

    def __init__(self, input: list[SSAValue]):
        super().__init__(operands=(input,))


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    Permutes the dimensions of `operand` tensor using `permutation` and produces a
    `result` tensor. More formally, `result[result_index] = operand[operand_index]`
    where `result_index[d] = operand_index[permutation[d]]`.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose
    """

    name = "stablehlo.transpose"

    ElementType = Annotated[Attribute, ConstraintVar("ElementType")]

    operand = operand_def(TensorType[ElementType])
    result = result_def(TensorType[ElementType])
    permutation = attr_def(DenseArrayBase)

    def __init__(
        self, operand: SSAValue, permutation: DenseArrayBase, result_type: Attribute
    ):
        super().__init__(
            operands=(operand,),
            result_types=(result_type,),
            attributes={"permutation": permutation},
        )

    def get_permutation(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.permutation.as_tuple())

    def verify_(self) -> None:
        # Operand and result types are checked before the custom `verify_`
        o_type = cast(TensorType[Attribute], self.operand.type)
        r_type = cast(TensorType[Attribute], self.result.type)

        o_shape = o_type.get_shape()
        r_shape = r_type.get_shape()

        # TODO: Quantization constraints
        # `permutation` is a permutation of `range(rank(operand))`
        permutation = self.get_permutation()
        if sorted(permutation) != list(range(len(o_shape))):
            raise VerifyException(
                f"Permutation {permutation} of transpose must be a permutation of "
                f"range({len(o_shape)})"
            )

        # `shape(result) = dim(operand, permutation...)`
        for i, dim in enumerate(permutation):
            if r_shape[i] != o_shape[dim]:
                raise VerifyException(
                    f"Permutation mismatch at dimension {i}, expected {o_shape[dim]}"
                )


StableHLO = Dialect(
    "stablehlo",
    [
        AbsOp,
        AddOp,
        AfterAllOp,
        AndOp,
        BitcastConvertOp,
        CaseOp,
        MultiplyOp,
        ReturnOp,
        SubtractOp,
        TransposeOp,
    ],
    [
        DotAttr,
        PrecisionAttr,
        TokenType,
    ],
)
