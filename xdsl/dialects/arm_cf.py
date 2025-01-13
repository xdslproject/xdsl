from xdsl.dialects import arm
from xdsl.dialects.arm.assembly import AssemblyInstructionArg
from xdsl.dialects.arm.ops import LabelOp
from xdsl.dialects.arm.register import IntRegisterType
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    Successor,
    irdl_op_definition,
    operand_def,
    successor_def,
    traits_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class BEqOp(arm.ops.ARMInstruction):
    """
    Branch if equal.
    """

    name = "arm_cf.beq"

    s1 = operand_def(IntRegisterType)
    s2 = operand_def(IntRegisterType)

    then_block = successor_def()

    assembly_format = (
        "$s1 `,` $s2  `,` $then_block attr-dict `:` `(` type($s1) `,` type($s2) `)`"
    )

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        s1: Operation | SSAValue,
        s2: Operation | SSAValue,
        then_block: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[s1, s2],
            attributes={
                "comment": comment,
            },
            successors=(then_block,),
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        then_label = self.then_block.first_op
        assert isinstance(then_label, LabelOp)
        return self.s1, self.s2, then_label.label


ARM_CF = Dialect(
    "arm_cf",
    [
        BEqOp,
    ],
)
