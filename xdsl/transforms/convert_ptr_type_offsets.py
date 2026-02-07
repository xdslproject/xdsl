from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, ptr
from xdsl.dialects.builtin import FixedBitwidthType, IndexType, ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException


@dataclass
class ConvertTypeOffsetOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.TypeOffsetOp, rewriter: PatternRewriter, /):
        if not issubclass(type(op.elem_type), FixedBitwidthType):
            raise DiagnosticException(
                "Type offset is currently only supported for fixed size types"
            )
        elem_type = cast(FixedBitwidthType, op.elem_type)
        rewriter.replace_op(
            op, arith.ConstantOp.from_int_and_width(elem_type.size, IndexType())
        )


class ConvertPtrTypeOffsetsPass(ModulePass):
    name = "convert-ptr-type-offsets"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertTypeOffsetOp(),
        ).rewrite_module(op)
