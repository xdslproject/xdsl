from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import eqsat, pdl
from xdsl.interpreters.pdl import PDLMatcher
from xdsl.ir import SSAValue


@dataclass
class EqsatPDLMatcher(PDLMatcher):
    def match_operand(
        self, ssa_val: SSAValue, pdl_op: pdl.OperandOp, xdsl_val: SSAValue
    ):
        owner = xdsl_val.owner
        assert isinstance(owner, eqsat.EClassOp)
        assert len(owner.operands) == 1, (
            "newly converted eqsat always has 1 element in eclass"
        )
        arg = owner.operands[0]
        res = super().match_operand(ssa_val, pdl_op, arg)
        return res
