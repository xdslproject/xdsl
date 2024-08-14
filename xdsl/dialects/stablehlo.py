"""
https://github.com/openxla/stablehlo/blob/main/docs/spec.md

StableHLO is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that consume StableHLO programs.
"""

import abc
from typing import Annotated

from xdsl.dialects.builtin import AnyTensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)

# region Abstract Base Classes


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    # TODO: Remove this constraint for complex types.
    T = Annotated[AnyTensorType, ConstraintVar("T")]

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
    T = Annotated[AnyTensorType, ConstraintVar("T")]

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


StableHLO = Dialect(
    "stablehlo",
    [
        AbsOp,
        AddOp,
        MultiplyOp,
        SubtractOp,
    ],
    [],
)
