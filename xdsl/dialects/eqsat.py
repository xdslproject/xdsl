"""
An embedding of equivalence classes in IR, for use in equality saturation with
non-destructive rewrites.

Please see the Equality Saturation Project for details:
https://github.com/orgs/xdslproject/projects/23

TODO: add documentation once we have end-to-end flow working:
https://github.com/xdslproject/xdsl/issues/3174
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    opt_attr_def,
    result_def,
    var_operand_def,
)
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
        # Check that none of the operands are produced by another eclass op.
        # In that case the two ops should have been merged into one.
        for operand in self.operands:
            if isinstance(operand.owner, EClassOp):
                raise VerifyException(
                    "A result of an eclass operation cannot be used as an operand of "
                    "another eclass."
                )


EqSat = Dialect(
    "eqsat",
    [
        EClassOp,
    ],
)
