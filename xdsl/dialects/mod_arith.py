from abc import ABC
from typing import Annotated, Generic, TypeVar

from xdsl.dialects.arith import signlessIntegerLike
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.traits import Pure


class ModArithOp(IRDLOperation, ABC):
    pass


_T = TypeVar("_T", bound=Attribute)


class BinaryOp(ModArithOp, ABC, Generic[_T]):
    T = Annotated[Attribute, ConstraintVar("T"), _T]
    lhs = operand_def(T)
    rhs = operand_def(T)
    output = result_def(T)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($output)"
    traits = frozenset((Pure(),))

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        result_type: Attribute | None,
    ):
        if result_type is None:
            result_type = SSAValue.get(lhs).type

        super().__init__(operands=[lhs, rhs], result_types=[result_type])

    def verify_(self) -> None:
        # todo
        return super().verify_()


class BinaryModuloOp(BinaryOp[_T], ABC, Generic[_T]):
    modulus = attr_def(AnyIntegerAttr)


@irdl_op_definition
class AddOp(BinaryModuloOp[Annotated[Attribute, signlessIntegerLike]]):
    name = "mod_arith.add"


ModArith = Dialect(
    "mod_arith",
    [AddOp],
)
