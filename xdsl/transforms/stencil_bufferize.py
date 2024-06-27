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
    StoreOp,
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
from xdsl.traits import is_side_effect_free
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
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


def walk_from(a: Operation):
    while True:
        yield from a.walk()
        if a.next_op is None:
            break
        a = a.next_op


def walk_from_to(a: Operation, b: Operation):
    for o in walk_from(a):
        if o == b:
            return
        yield o


class ApplyLoadStoreFoldPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter):
        temp = op.temp

        if not isinstance(load := temp.owner, LoadOp):
            return

        infield = load.field

        other_uses = [u for u in infield.uses if u.operation is not load]

        if len(other_uses) != 1:
            print("other uses")
            return

        other_use = other_uses.pop()

        if not isinstance(
            apply := other_use.operation, ApplyOp
        ) or other_use.index < len(apply.args):
            print(other_use)
            print()
            return

        # I don't want to deal with replacing block arguments cleanly yet
        # I do think we really want a helper for this..
        field_owner = op.field.owner
        if isinstance(field_owner, Block):
            return
        effecting = [
            o
            for o in walk_from_to(field_owner, op)
            if infield in o.operands
            and (not is_side_effect_free(o))
            and (o not in (load, apply))
        ]
        if effecting:
            print("effecting: ", effecting)
            print(load)
            return

        new_operands = list(apply.operands)
        new_operands[other_use.index] = op.field

        new_apply = ApplyOp.create(
            operands=new_operands,
            result_types=[],
            properties=apply.properties.copy(),
            attributes=apply.attributes.copy(),
            regions=[
                Region(Block(arg_types=[SSAValue.get(a).type for a in apply.args])),
            ],
        )

        rewriter.inline_block(
            apply.region.block,
            InsertPoint.at_start(new_apply.region.block),
            new_apply.region.block.args,
        )

        rewriter.replace_op(apply, new_apply)
        rewriter.erase_op(op)


@dataclass(frozen=True)
class StencilBufferize(ModulePass):
    name = "stencil-bufferize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ApplyBufferizePattern(),
                    ApplyLoadStoreFoldPattern(),
                    RemoveUnusedOperations(),
                ]
            )
        )
        walker.rewrite_module(op)
