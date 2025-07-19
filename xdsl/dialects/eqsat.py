"""
An embedding of equivalence classes in IR, for use in equality saturation with
non-destructive rewrites.

Please see the Equality Saturation Project for details:
https://github.com/orgs/xdslproject/projects/23

TODO: add documentation once we have end-to-end flow working:
https://github.com/xdslproject/xdsl/issues/3174
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import ArrayAttr, IntAttr, SymbolRefAttr
from xdsl.ir import Attribute, Block, Dialect, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    lazy_traits_def,
    opt_attr_def,
    prop_def,
    region_def,
    result_def,
    successor_def,
    traits_def,
    var_operand_def,
    var_result_def,
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
        # Check that none of the operands are produced by another eclass op.
        # In that case the two ops should have been merged into one.
        for operand in self.operands:
            if isinstance(operand.owner, EClassOp):
                raise VerifyException(
                    "A result of an eclass operation cannot be used as an operand of "
                    "another eclass."
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
class MarkUnreachableOp(IRDLOperation):
    """
    Marks a rewrite rule as unreachable from that point in the program.
    This is used to prune the search space of the equality saturation algorithm.
    """

    name = "eqsat.mark_unreachable"

    rules = prop_def(ArrayAttr[SymbolRefAttr])

    assembly_format = "$rules attr-dict"

    def __init__(self, rules: Sequence[SymbolRefAttr]):
        super().__init__(
            operands=[], successors=[], properties={"rules": ArrayAttr(rules)}
        )


@irdl_op_definition
class MarkReachableOp(IRDLOperation):
    """
    Marks a rewrite rule as reachable from that point in the program.
    Technically, an unreachable rule cannot become reachable again,
    but this is used as an optimization where we mark rules as unreachable
    in a parent block and then mark them as reachable in a child block in
    order to generate a more compact IR.
    """

    name = "eqsat.mark_reachable"

    rules = prop_def(ArrayAttr[SymbolRefAttr])

    assembly_format = "$rules attr-dict"

    def __init__(self, rules: Sequence[SymbolRefAttr]):
        super().__init__(
            operands=[], successors=[], properties={"rules": ArrayAttr(rules)}
        )


@irdl_op_definition
class CheckAllUnreachableOp(IRDLOperation):
    """
    Jumps to `unreachable_dest` if all rules are unreachable (either banned or unreachable in the matcher.)
    """

    name = "eqsat.check_all_unreachable"

    traits = traits_def(IsTerminator())

    unreachable_dest = successor_def()
    reachable_dest = successor_def()

    assembly_format = "attr-dict `->` $unreachable_dest `, ` $reachable_dest"

    def __init__(
        self,
        unreachable_dest: Block,
        reachable_dest: Block,
    ):
        super().__init__(
            successors=[unreachable_dest, reachable_dest],
        )


EqSat = Dialect(
    "eqsat",
    [
        EClassOp,
        YieldOp,
        EGraphOp,
        MarkUnreachableOp,
        CheckAllUnreachableOp,
    ],
)
