"""
https://github.com/openxla/stablehlo/blob/main/docs/spec.md

StableHLO is an operation set for high-level operations (HLO) in machine learning (ML) models.
StableHLO works as a portability layer between different ML frameworks and ML compilers:
ML frameworks that produce StableHLO programs are compatible with ML compilers that consume StableHLO programs.
"""

from typing import Annotated

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class AbsOp(IRDLOperation):
    """https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs

    Performs element-wise abs operation on operand tensor and produces a result tensor.
    Depending on the element type, does the following:

    * For signed integers: integer modulus.
    * For floats: abs from IEEE-754.
    * For complex numbers: complex modulus.
    * For quantized types: dequantize_op_quantize(abs, operand, type(result)).
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


StableHLO = Dialect(
    "stablehlo",
    [
        AbsOp,
    ],
    [],
)
