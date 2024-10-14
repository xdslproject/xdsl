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

from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import DiagnosticException


@irdl_op_definition
class EClassOp(IRDLOperation):
    T: ClassVar[VarConstraint[Attribute]] = VarConstraint("T", AnyAttr())

    name = "eqsat.eclass"
    arguments = var_operand_def(T)
    result = result_def(T)

    assembly_format = "$arguments attr-dict `:` type($result)"

    def __init__(self, *arguments: SSAValue, res_type: Attribute | None = None):
        if not arguments:
            raise DiagnosticException("eclass op must have at least one operand")
        if res_type is None:
            res_type = arguments[0].type

        super().__init__(operands=[arguments], result_types=[res_type])


EqSat = Dialect(
    "eqsat",
    [
        EClassOp,
    ],
)
