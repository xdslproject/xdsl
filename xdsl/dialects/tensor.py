from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import IndexType, TensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def, var_operand_def


@irdl_op_definition
class EmptyOp(IRDLOperation):
    name = "tensor.empty"

    dynamic_sizes = var_operand_def(IndexType)

    tensor = result_def(TensorType[Attribute])

    assembly_format = (
        " `(` $dynamic_sizes `)` attr-dict `:` type($tensor) type($dynamic_sizes)"
    )

    def __init__(self, dynamic_sizes: Sequence[SSAValue], tensor_type: Attribute):
        super().__init__(
            operands=(dynamic_sizes,),
            result_types=(tensor_type,),
        )


Tensor = Dialect(
    "tensor",
    [
        EmptyOp,
    ],
    [],
)
