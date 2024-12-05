from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Any, cast

from xdsl.context import MLContext
from xdsl.dialects import eqsat, pdl
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, impl, register_impls
from xdsl.interpreters.pdl import PDLMatcher, PDLRewriteFunctions
from xdsl.ir import Attribute, Operation, OpResult, SSAValue, TypeAttribute
from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint
from xdsl.transforms.convert_onnx_to_linalg import get_root_op
from xdsl.utils.exceptions import InterpretationError


@dataclass
class EqsatPDLMatcher(PDLMatcher):
    def match_operand(
        self, ssa_val: SSAValue, pdl_op: pdl.OperandOp, xdsl_val: SSAValue
    ):
        owner = xdsl_val.owner
        assert isinstance(owner, eqsat.EClassOp)
        assert (
            len(owner.operands) == 1
        ), "newly converted eqsat always has 1 element in eclass"
        arg = owner.operands[0]
        res = super().match_operand(ssa_val, pdl_op, arg)
        return res
