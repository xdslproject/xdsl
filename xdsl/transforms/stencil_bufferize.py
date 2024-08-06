from collections.abc import Generator
from dataclasses import dataclass
from itertools import chain
from typing import Any, TypeVar, cast

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.stencil import (
    AllocOp,
    ApplyOp,
    BufferOp,
    CombineOp,
    FieldType,
    IndexAttr,
    LoadOp,
    StencilBoundsAttr,
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
from xdsl.transforms.canonicalization_patterns.stencil import AllocUnused
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.utils.hints import isa

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def field_from_temp(temp: TempType[_TypeElement]) -> FieldType[_TypeElement]:
    return FieldType[_TypeElement].new(temp.parameters)


class ApplyBufferizePattern(RewritePattern):
    """
    Naive partial stencil.apply bufferization.

    Just replace all operands with the field result of a stencil.buffer on them, meaning
    "The buffer those value are allocated to"; and allocate buffers for every result,
    loading them back after the apply, to keep the pattern local.
    """

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
            LoadOp(operands=[d], result_types=[r.type]) for d, r in zip(dests, op.res)
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


def walk_from(a: Operation) -> Generator[Operation, Any, None]:
    """
    Walk through all operations recursively inside a or its block.
    """
    while True:
        yield from a.walk()
        if a.next_op is None:
            break
        a = a.next_op


def walk_from_to(a: Operation, b: Operation):
    """
    Walk through all operations recursively inside a or its block, until b is met, if
    ever.
    """
    for o in walk_from(a):
        if o == b:
            return
        yield o


class LoadBufferFoldPattern(RewritePattern):
    """
    The underlying buffer of values resulting from a load of a buffer is the buffer
    loaded from. That is, if nobody annoyed the said buffer in-between.

    If someone did, we should probably insert a copy there. Right now, we just bail out.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):

        handle = op.res
        # If this is a value-semantic buffer, we can't fold it
        if not isinstance(handle.type, FieldType):
            return

        temp = op.temp
        load = temp.owner
        # We are interested in folding buffers of loaded values
        if not isinstance(load, LoadOp):
            return

        underlying = load.field

        # TODO: propery analysis of effects in between
        # For illustration, only fold a single use of the handle
        # (Requires more boilerplate to analyse the whole live range otherwise)
        uses = op.res.uses.copy()
        if len(uses) > 1:
            return
        user = uses.pop().operation

        effecting = [
            o
            for o in walk_from_to(load, user)
            if underlying in o.operands
            and (not is_side_effect_free(o))
            and (o not in (load, op, user))
        ]
        if effecting:
            return

        rewriter.replace_matched_op(new_ops=[], new_results=[underlying])


class ApplyLoadStoreFoldPattern(RewritePattern):
    """
    If an apply is acting on a buffer, that is *only* used to be copied in another
    buffer, just make the apply work on the latter directly.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter):
        temp = op.temp

        # We are looking for a loaded destination of an apply
        if not isinstance(load := temp.owner, LoadOp):
            return

        infield = load.field

        other_uses = [u for u in infield.uses if u.operation is not load]

        if len(other_uses) != 1:
            return

        other_use = other_uses.pop()

        if not isinstance(
            apply := other_use.operation, ApplyOp
        ) or other_use.index < len(apply.args):
            print(other_use)
            print()
            return

        # Get first occurence of the field, to walk from it
        start = op.field.owner
        if isinstance(start, Block):
            if start is not op.parent:
                return
            start = cast(Operation, start.first_op)
        effecting = [
            o
            for o in walk_from_to(start, op)
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

        new_load = LoadOp.create(
            operands=[op.field],
            result_types=[r.type for r in load.results],
            attributes=load.attributes.copy(),
            properties=load.properties.copy(),
        )

        rewriter.replace_op(apply, new_apply)
        rewriter.replace_op(load, new_load)
        rewriter.erase_op(op)


@dataclass(frozen=True)
class UpdateApplyArgs(RewritePattern):
    """
    Stencil bufferization will often replace a temporary apply's argument with a wider
    one.
    This pattern simply updates block arguments accordingly.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        new_arg_types = [o.type for o in op.args]
        if new_arg_types == [a.type for a in op.region.block.args]:
            return

        new_block = Block(arg_types=new_arg_types)
        new_apply = ApplyOp.create(
            operands=op.operands,
            result_types=[r.type for r in op.results],
            properties=op.properties.copy(),
            attributes=op.attributes.copy(),
            regions=[Region(new_block)],
        )

        rewriter.inline_block(
            op.region.block, InsertPoint.at_start(new_block), new_block.args
        )

        rewriter.replace_matched_op(new_apply)


@dataclass(frozen=True)
class BufferAlloc(RewritePattern):
    """
    A value-semantics stencil.buffer is meant to simply pass along an underlying buffer.
    Just pass along the value and let bufferization do the rest.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):
        # If it's not a value-semantic buffer, we let other patterns work on it.
        if not isinstance(op.res.type, TempType):
            return

        temp_t = cast(TempType[Attribute], op.temp.type)
        if not isinstance(temp_t.bounds, StencilBoundsAttr):
            raise ValueError(
                "Stencil shape inference must be ran before bufferization."
            )
        alloc = AllocOp(result_types=[field_from_temp(temp_t)])
        rewriter.insert_op(alloc, InsertPoint.at_start(cast(Block, op.parent)))

        rewriter.replace_matched_op(
            new_ops=[
                StoreOp.get(op.temp, alloc.field, temp_t.bounds),
                LoadOp.get(alloc.field, temp_t.bounds.lb, temp_t.bounds.ub),
            ]
        )


@dataclass(frozen=True)
class CombineStoreFold(RewritePattern):
    """
    A stored combine result is folded into stores of the matching operand in the
    destination field.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CombineOp, rewriter: PatternRewriter):
        for i, r in enumerate(op.results):
            if len(r.uses) != 1:
                continue
            store = next(iter(r.uses)).operation
            if not isinstance(store, StoreOp):
                continue

            new_lower = op.lower
            new_upper = op.upper
            new_lowerext = op.lowerext
            new_upperext = op.upperext
            new_results_types = [
                r.type for r in chain(op.results[:i], op.results[i + 1 :])
            ]

            bounds = cast(StencilBoundsAttr, cast(TempType[Attribute], r.type).bounds)
            newub = list(bounds.ub)
            newub[op.dim.value.data] = op.index.value.data
            lower_bounds = StencilBoundsAttr.new((bounds.lb, IndexAttr.get(*newub)))
            newlb = list(bounds.lb)
            newlb[op.dim.value.data] = op.index.value.data
            upper_bounds = StencilBoundsAttr.new((IndexAttr.get(*newlb), bounds.ub))

            rewriter.erase_op(store)

            # If it corresponds to a lower/upper result
            if i < len(op.lower):
                new_lower = op.lower[:i] + op.lower[i + 1 :]
                new_upper = op.upper[:i] + op.upper[i + 1 :]
                rewriter.insert_op(
                    (
                        StoreOp.get(
                            op.lower[i],
                            store.field,
                            lower_bounds,
                        ),
                        StoreOp.get(
                            op.upper[i],
                            store.field,
                            upper_bounds,
                        ),
                    ),
                    InsertPoint.before(op),
                )
            # If it corresponds to a lowerext result
            elif i < len(op.lower) + len(op.lowerext):
                new_lowerext = (
                    op.lowerext[: i - len(op.lower)]
                    + op.lowerext[i - len(op.lower) + 1 :]
                )
                rewriter.insert_op(
                    (
                        StoreOp.get(
                            op.lower[i],
                            store.field,
                            lower_bounds,
                        ),
                        StoreOp.get(
                            op.upper[i],
                            store.field,
                            upper_bounds,
                        ),
                    ),
                    InsertPoint.before(op),
                )
            else:
                new_upperext = (
                    op.upperext[: i - len(op.lower) - len(op.lowerext)]
                    + op.upperext[i - len(op.lower) - len(op.lowerext) + 1 :]
                )
                rewriter.insert_op(
                    (
                        StoreOp.get(
                            op.lower[i],
                            store.field,
                            lower_bounds,
                        ),
                        StoreOp.get(
                            op.upper[i],
                            store.field,
                            upper_bounds,
                        ),
                    ),
                    InsertPoint.before(op),
                )

            new_combine = CombineOp(
                operands=[new_lower, new_upper, new_lowerext, new_upperext],
                result_types=[new_results_types],
                attributes=op.attributes.copy(),
                properties=op.properties.copy(),
            )
            rewriter.replace_matched_op(
                new_combine,
                new_results=new_combine.results[:i] + (None,) + new_combine.results[i:],
            )
            return


@dataclass(frozen=True)
class StencilBufferize(ModulePass):
    name = "stencil-bufferize"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    UpdateApplyArgs(),
                    ApplyBufferizePattern(),
                    BufferAlloc(),
                    CombineStoreFold(),
                    LoadBufferFoldPattern(),
                    ApplyLoadStoreFoldPattern(),
                    RemoveUnusedOperations(),
                    AllocUnused(),
                ]
            )
        )
        walker.rewrite_module(op)
