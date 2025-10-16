"""
A dialect that extends pdl_interp with eqsat specific operations.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.ir import Block, Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    successor_def,
    traits_def,
    var_successor_def,
)
from xdsl.traits import IsTerminator


@irdl_op_definition
class ChooseOp(IRDLOperation):
    """
    This operation can be used in pdl_interp matchers and
    integrates with the backtracking mechanism. It holds multiple
    "choices" (successors). When this operation is encountered,
    a BacktrackPoint is stored, and the choice is visited.
    When this execution of this choice eventually finalizes, the
    backtracking logic will jump to the next choice, until all
    choices are exhausted. Finally, the default successor is visited.
    """

    name = "eqsat_pdl_interp.choose"
    default_dest = successor_def()
    choices = var_successor_def()
    traits = traits_def(IsTerminator())
    assembly_format = "`from` $choices `then` $default_dest attr-dict"

    def __init__(self, choices: Sequence[Block], default: Block):
        super().__init__(
            successors=[default, choices],
        )


EqSatPDLInterp = Dialect(
    "eqsat_pdl_interp",
    [
        ChooseOp,
    ],
)
