"""
An embedding of equivalence classes in IR, for use in equality saturation with
non-destructive rewrites.

Please see the [Equality Saturation Project](https://github.com/orgs/xdslproject/projects/23) for details.

See the overview [notebook](https://xdsl.readthedocs.io/stable/marimo/eqsat.html).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Attribute, Block, Dialect, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    lazy_traits_def,
    opt_attr_def,
    region_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_result_def,
    var_successor_def,
)
from xdsl.traits import HasParent, IsTerminator, Pure, SingleBlockImplicitTerminator
from xdsl.utils.exceptions import DiagnosticException, VerifyException

EQSAT_COST_LABEL = "eqsat_cost"
"""
Key used to store the cost of computing the result of an operation.
"""


@irdl_op_definition
class EClassOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "eqsat.eclass"
    arguments = var_operand_def(T)
    result = result_def(T)
    min_cost_index = opt_attr_def(IntAttr)
    traits = traits_def(Pure())

    assembly_format = "$arguments attr-dict `:` type($result)"

    def __init__(
        self,
        *arguments: SSAValue,
        min_cost_index: IntAttr | None = None,
        res_type: Attribute | None = None,
    ):
        if not arguments:
            raise DiagnosticException("eclass op must have at least one operand")
        if res_type is None:
            res_type = arguments[0].type

        super().__init__(
            operands=[arguments],
            result_types=[res_type],
            attributes={"min_cost_index": min_cost_index},
        )

    def verify_(self) -> None:
        if not self.operands:
            raise VerifyException("Eclass operations must have at least one operand.")

        for operand in self.operands:
            if isinstance(operand.owner, EClassOp):
                # The two ops should have been merged into one.
                raise VerifyException(
                    "A result of an eclass operation cannot be used as an operand of "
                    "another eclass."
                )

            if not operand.has_one_use():
                if len(set(use.operation for use in operand.uses)) == 1:
                    raise VerifyException(
                        "Eclass operands must only be used once by the eclass."
                    )
                else:
                    raise VerifyException(
                        "Eclass operands must only be used by the eclass."
                    )


@irdl_op_definition
class EGraphOp(IRDLOperation):
    name = "eqsat.egraph"

    outputs = var_result_def()
    body = region_def()

    traits = lazy_traits_def(lambda: (SingleBlockImplicitTerminator(YieldOp),))

    assembly_format = "`->` type($outputs) $body attr-dict"

    def __init__(
        self,
        result_types: Sequence[Attribute] | None,
        body: Region,
    ):
        super().__init__(
            result_types=(result_types,),
            regions=[body],
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "eqsat.yield"
    values = var_operand_def()

    traits = traits_def(HasParent(EGraphOp), IsTerminator())

    assembly_format = "$values `:` type($values) attr-dict"

    def __init__(
        self,
        *values: SSAValue,
    ):
        super().__init__(operands=[values])


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

    name = "eqsat.choose"
    default_dest = successor_def()
    choices = var_successor_def()
    traits = traits_def(IsTerminator())
    assembly_format = "`from` $choices `then` $default_dest attr-dict"

    def __init__(self, choices: Sequence[Block], default: Block):
        super().__init__(
            successors=[default, choices],
        )


EqSat = Dialect(
    "eqsat",
    [
        EClassOp,
        YieldOp,
        EGraphOp,
        ChooseOp,
    ],
)
