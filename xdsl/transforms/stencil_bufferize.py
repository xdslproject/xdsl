from dataclasses import dataclass
from typing import TypeVar, cast

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.stencil import (
    AllocOp,
    ApplyOp,
    BufferOp,
    FieldType,
    LoadOp,
    TempType,
)
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    Region,
    SSAValue,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def field_from_temp(temp: TempType[_TypeElement]) -> FieldType[_TypeElement]:
    return FieldType[_TypeElement].new(temp.parameters)


class ApplyBufferizePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        if not op.res:
            return

        bounds = cast(TempType[Attribute], op.res[0].type).bounds

        dests = [
            AllocOp(result_types=[field_from_temp(cast(TempType[Attribute], r.type))])
            for r in op.res
        ]
        operands = [
            (
                BufferOp.create(
                    operands=[o],
                    result_types=[field_from_temp(o.type)],
                )
                if isa(o.type, TempType[Attribute])
                else o
            )
            for o in op.operands
        ]

        loads = [
            LoadOp(operands=[d], result_types=[r.type]) for r, d in zip(op.res, dests)
        ]

        new = ApplyOp(
            operands=[operands, dests],
            regions=[Region(Block(arg_types=[SSAValue.get(a).type for a in operands]))],
            result_types=[[]],
            properties={"bounds": bounds},
        )
        rewriter.inline_block(
            op.region.block,
            InsertPoint.at_start(new.region.block),
            new.region.block.args,
        )

        rewriter.replace_matched_op(
            [*(o for o in operands if isinstance(o, Operation)), *dests, new, *loads],
            [SSAValue.get(l) for l in loads],
        )


@dataclass(frozen=True)
class StencilBufferize(ModulePass):
    name = "stencil-bufferize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([ApplyBufferizePattern()])
        )
        walker.rewrite_module(op)
