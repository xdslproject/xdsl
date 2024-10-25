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
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
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

    T: ClassVar = VarConstraint("T", integerOrFloatLike)

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


@irdl_op_definition
class VarithSelectOp(IRDLOperation):
    """
    Variadic selection operation

    Returns the ith element of `args` where i is given by the `cond` argument.
    Behaviour is undefined if `cond` > len(`args`).
    """

    name = "varith.select"

    T: ClassVar = VarConstraint("T", AnyAttr())

    cond = operand_def(IntegerType)

    args = var_operand_def(T)

    result = result_def(T)

    assembly_format = "`(` $cond `:` type($cond) `)` $args attr-dict `:` type($result)"

    traits = frozenset((Pure(),))

    def __init__(self, cond: SSAValue | Operation, *args: SSAValue | Operation):
        assert len(args) > 0
        super().__init__(
            operands=[cond, args], result_types=(SSAValue.get(args[-1]).type,)
        )


Varith = Dialect(
    "varith",
    [
        VarithAddOp,
        VarithMulOp,
        VarithSelectOp,
    ],
)
