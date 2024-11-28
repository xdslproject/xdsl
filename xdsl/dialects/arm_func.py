from xdsl.dialects import arm
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import (
    irdl_op_definition,
    traits_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class RetOp(arm.ops.ARMInstruction):
    """
    Return from subroutine.

    Equivalent to `bx lr`
    """

    name = "arm_func.return"

    assembly_format = "attr-dict"

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            attributes={
                "comment": comment,
            },
        )

    def assembly_line_args(self):
        return ()

    def assembly_instruction_name(self) -> str:
        return "bx lr"


ARM_FUNC = Dialect(
    "arm_func",
    [
        RetOp,
    ],
)
