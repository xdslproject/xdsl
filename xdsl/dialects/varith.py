from typing import ClassVar

from xdsl.dialects.builtin import (
    BFloat16Type,
    ContainerOf,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    IndexType,
    IntegerType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    result_def,
    var_operand_def,
)
from xdsl.traits import Pure

integerOrFloatLike: ContainerOf = ContainerOf(
    AnyOf(
        [
            IntegerType,
            IndexType,
            BFloat16Type,
            Float16Type,
            Float32Type,
            Float64Type,
            Float80Type,
            Float128Type,
        ]
    )
)


class VarithOp(IRDLOperation):
    """
    Variadic arithmetic operation
    """

    T: ClassVar[VarConstraint[Attribute]] = VarConstraint("T", integerOrFloatLike)

    args = var_operand_def(T)
    res = result_def(T)

    traits = frozenset((Pure(),))

    assembly_format = "$args attr-dict `:` type($res)"

    def __init__(self, *args: SSAValue | Operation):
        assert len(args) > 0
        super().__init__(operands=[args], result_types=[SSAValue.get(args[-1]).type])


@irdl_op_definition
class VarithAddOp(VarithOp):
    name = "varith.add"


@irdl_op_definition
class VarithMulOp(VarithOp):
    name = "varith.mul"


Varith = Dialect(
    "varith",
    [
        VarithAddOp,
        VarithMulOp,
    ],
)
