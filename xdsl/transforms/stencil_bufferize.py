from collections.abc import Generator
from dataclasses import dataclass
from itertools import chain
from typing import Any, cast

from typing_extensions import TypeVar

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.experimental.dmp import SwapOp
from xdsl.dialects.stencil import (
    AccessOp,
    AllocOp,
    ApplyOp,
    BufferOp,
    CombineOp,
    DynAccessOp,
    FieldType,
    IndexAttr,
    LoadOp,
    ReturnOp,
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
from xdsl.traits import MemoryEffectKind, get_effects
from xdsl.transforms.canonicalization_patterns.stencil import ApplyUnusedResults
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.utils.hints import isa

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def field_from_temp(temp: TempType[_TypeElement]) -> FieldType[_TypeElement]:
    return FieldType[_TypeElement].new(temp.parameters)


def might_effect(
    operation: Operation, effects: set[MemoryEffectKind], value: SSAValue
) -> bool:
    """
    Return True if the operation might have any of the given effects on the given value.
    """
    op_effects = get_effects(operation)
    return op_effects is None or any(
        e.kind in effects and e.value in (None, value) for e in op_effects
    )


class ApplyBufferizePattern(RewritePattern):
    """
    Naive partial `stencil.apply` bufferization.

    Just replace all temp arguments with the field result of a stencil.buffer on them, meaning
    "The buffer those value are allocated to".

    Example:
    ```mlir
    %out = stencil.apply(%0 = %in : !stencil.temp<[0,32]xf64>) -> (!stencil.temp<[0,32]>xf64) {
        // [...]
    }
    ```
    yields:
    ```mlir
    %in_buf = stencil.buffer %in : !stencil.temp<[0,32]xf64> -> !stencil.field<[0,32]xf64>
    stencil.apply(%0 = %in_buf : !stencil.field<[0,32]>xf64) outs (%out_buf : !stencil.field<[0,32]>xf64) {
        // [...]
    }
    ```
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        if all(not isinstance(o.type, TempType) for o in op.args):
            return

        bounds = op.get_bounds()

        args = [
            (
                BufferOp.create(
                    operands=[o],
                    result_types=[field_from_temp(o.type)],
                )
                if isa(o.type, TempType[Attribute])
                else o
            )
            for o in op.args
        ]

        new = ApplyOp(
            operands=[args, op.dest],
            regions=[op.detach_region(0)],
            result_types=[op.res.types],
            properties={"bounds": bounds},
        )

        rewriter.replace_op(op, [*(o for o in args if isinstance(o, Operation)), new])


def walk_from(a: Operation) -> Generator[Operation, Any, None]:
    """
    Walk through all operations recursively inside a or its block.
    """
    while True:
        yield from a.walk()
        if a.next_op is None:
            break
        a = a.next_op


def walk_from_to(a: Operation, b: Operation, *, inclusive: bool = False):
    """
    Walk through all operations recursively inside a or its block, until b is met, if
    ever.
    """
    for o in walk_from(a):
        if o == b:
            if inclusive:
                yield o
            return
        yield o


def is_inplace(apply: ApplyOp, field: SSAValue):
    """
    Check if the passed `stencil.apply` has any non-zero offset access to the passed
    `stencil.field`.
    """
    # Get all block arguments matching this field
    field_args = set(
        apply.region.block.args[i] for (i, a) in enumerate(apply.args) if a is field
    )
    # Is there any non-zero access on those arguments?
    return not any(
        access
        for access in apply.walk()
        if isinstance(access, AccessOp)
        and access.temp in field_args
        and any(o != 0 for o in access.offset)
        or isinstance(access, DynAccessOp)
        and access.temp in field_args
        and any(o != 0 for o in chain(access.lb, access.ub))
    )


class LoadBufferFoldPattern(RewritePattern):
    """
    Fold a reference-semantic `stencil.buffer` of a `stencil.load` to the underlying
    field if safe.

    Example:
    ```mlir
    %temp = stencil.load %field : !stencil.field<[-2,34]> -> !stencil.temp<[0,32]>
    // [... No changes on %field]
    %temp_f = stencil.buffer %temp : !stencil.temp<[0,32]> -> !stencil.field<[0,32]>
    // [... No changes on %field]
    // Last use of temp_f
    ```
    yields:
    ```mlir
    // Will be simplified away or folded again
    %temp = stencil.load %field : !stencil.field<[-2,34]> -> !stencil.temp<[0,32]>
    // [... %temp_f replaced by %field]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):
        # If this is a value-semantic buffer, we can't fold it
        if not isinstance(op.res.type, FieldType):
            return

        temp = op.temp
        load = temp.owner
        # We are interested in folding buffers of loaded values
        if not isinstance(load, LoadOp):
            return

        underlying = load.field

        # TODO: further analysis
        # For now, only handle usages in the same block
        uses = tuple(op.res.uses)
        block = op.parent
        if not block or any(use.operation.parent is not block for use in uses):
            return
        last_user = max(
            uses, key=lambda u: block.get_operation_index(u.operation)
        ).operation

        effecting = [
            o
            for o in walk_from_to(load, last_user, inclusive=True)
            if might_effect(o, {MemoryEffectKind.WRITE}, underlying)
        ]

        # If the last effecting op is a stencil, handle the safe inplace case
        if (
            effecting
            and isinstance(effecting[-1], ApplyOp)
            and is_inplace(effecting[-1], op.res)
        ):
            effecting.pop()
        if effecting:
            return

        rewriter.replace_op(op, new_ops=[], new_results=[underlying])


class ApplyStoreFoldPattern(RewritePattern):
    """
    Fold stores of applys result

    Example:
    ```mlir
    %temp = stencil.apply() -> (!stencil.temp<[0,32]>) {
        // [...]
    }
    // [... %dest not read]
    stencil.store %temp to %dest (<[0], [32]>) : !stencil.temp<[0,32]> to !stencil.field<[-2,34]>
    ```
    yields:
    ```mlir
    // Outputs on dest directly
    stencil.apply() outs (%dest : !stencil.field<[-2,34]>) {
        // [...]
    }
    ```
    """

    @staticmethod
    def is_dest_safe(apply: ApplyOp, store: StoreOp) -> bool:
        # Check that the destination is not used between the apply and store.
        dest = store.field
        effecting = [
            o
            for o in walk_from_to(apply, store)
            if might_effect(o, {MemoryEffectKind.READ, MemoryEffectKind.WRITE}, dest)
        ]
        return not effecting

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        apply = op
        for temp_index, stored in enumerate(op.res):
            # We are looking for a result that is stored and foldable
            stores = [
                use.operation
                for use in stored.uses
                if isinstance(use.operation, StoreOp)
                and self.is_dest_safe(apply, use.operation)
            ]
            if not stores:
                continue

            bounds = apply.get_bounds()
            if not isinstance(bounds, StencilBoundsAttr):
                raise ValueError(
                    "Stencil shape inference must be ran before bufferization."
                )

            new_apply = ApplyOp.build(
                # We add new destinations for each store of the removed result
                operands=[
                    apply.args,
                    (*apply.dest, *(store.field for store in stores)),
                ],
                # We only remove the considered result
                result_types=[
                    [
                        r.type
                        for r in apply.results[:temp_index]
                        + apply.results[temp_index + 1 :]
                    ]
                ],
                properties=apply.properties.copy() | {"bounds": bounds},
                attributes=apply.attributes.copy(),
                # The block signature is the same
                regions=[
                    Region(Block(arg_types=[SSAValue.get(a).type for a in apply.args])),
                ],
            )

            # The body is the same
            rewriter.inline_block(
                apply.region.block,
                InsertPoint.at_start(new_apply.region.block),
                new_apply.region.block.args,
            )

            # We swap the return's operand order, to make sure the order still matches destinations
            # after bufferization
            old_return = new_apply.region.block.last_op
            assert isinstance(old_return, ReturnOp)
            uf = old_return.unroll_factor
            new_return_args = list(
                old_return.arg[: uf * temp_index]
                + old_return.arg[uf * (temp_index + 1) :]
                + old_return.arg[uf * temp_index : uf * (temp_index + 1)] * len(stores)
            )
            new_return = ReturnOp.create(
                operands=new_return_args,
                properties=old_return.properties.copy(),
                attributes=old_return.attributes.copy(),
            )
            rewriter.replace_op(old_return, new_return)

            # Create a load of a destination, for any other user of the result
            load = LoadOp.get(stores[0].field, bounds.lb, bounds.ub)

            rewriter.replace_op(
                op,
                [new_apply, load],
                new_apply.results[:temp_index]
                + (load.res,)
                + new_apply.results[temp_index:],
            )
            for store in stores:
                rewriter.erase_op(store)
            return


@dataclass(frozen=True)
class UpdateApplyArgs(RewritePattern):
    """
    Stencil bufferization will often replace a temporary apply's argument with a wider
    one.
    This pattern simply updates block arguments accordingly.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        new_arg_types = op.args.types
        if new_arg_types == op.region.block.arg_types:
            return

        new_block = Block(arg_types=new_arg_types)
        new_apply = ApplyOp.create(
            operands=op.operands,
            result_types=op.result_types,
            properties=op.properties.copy(),
            attributes=op.attributes.copy(),
            regions=[Region(new_block)],
        )

        rewriter.inline_block(
            op.region.block, InsertPoint.at_start(new_block), new_block.args
        )

        rewriter.replace_op(op, new_apply)


@dataclass(frozen=True)
class BufferAlloc(RewritePattern):
    """
    Replace a value semantic `stencil.buffer` by a load from an allocated field, after
    a store of the input values on it.

    This matches the orginal dialect's lowering for this operation.

    Example:
    ```mlir
    // [...]
    %forward = stencil.buffer %in : !stencil.temp<[0,32]> -> !stencil.temp<[0,32]>
    // [...]
    ```
    yields:
    ```mlir
    %alloc = stencil.alloc : !stencil.field<[0,32]>xf64
    // [...]
    // This should be folded in the above computation
    stencil.store %in to %alloc (<[0], [32]>) : !stencil.temp<[0,32]> to !stencil.field<[0,32]>
    // This should be folded in the below computation
    %forward = stencil.load %alloc : !stencil.field<[0,32]>xf64 -> !stencil.temp<[0,32]>
    // [...]
    ```
    """  # noqa: E501

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

        rewriter.replace_op(
            op,
            new_ops=[
                StoreOp.get(op.temp, alloc.field, temp_t.bounds),
                LoadOp.get(alloc.field, temp_t.bounds.lb, temp_t.bounds.ub),
            ],
        )


@dataclass(frozen=True)
class CombineStoreFold(RewritePattern):
    """
    A stored combine result is folded into stores of the matching operand in the
    destination field.

    Example:
    ```mlir
    %res1, %res2 = stencil.combine 1 at 11 lower = (%0 : !stencil.temp<[0,16]xf64>) upper = (%1 : !stencil.temp<[16,32]xf64>) lowerext = (%2 : !stencil.temp<[0,16]xf64>): !stencil.temp<[0,32]xf64>, !stencil.temp<[0,32]xf64>
    stencil.store %res1 to %dest1 (<[0], [32]>) : !stencil.temp<[0,32]xf64> to !stencil.field<[-2,34]xf64>
    stencil.store %res2 to %dest2 (<[0], [32]>) : !stencil.temp<[0,32]xf64> to !stencil.field<[-2,34]xf64>
    ```
    yields:
    ```mlir
    stencil.store %0 to %dest1 (<[0], [16]>) : !stencil.temp<[0,16]xf64> to !stencil.field<[-2,34]xf64>
    stencil.store %1 to %dest1 (<[16], [32]>) : !stencil.temp<[16,32]xf64> to !stencil.field<[-2,34]xf64>
    stencil.store %2 to %dest2 (<[0], [16]>) : !stencil.temp<[0,16]xf64> to !stencil.field<[-2,34]xf64>
    ```
    """  # noqa: E501

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CombineOp, rewriter: PatternRewriter):
        for i, r in enumerate(op.results):
            if not r.has_one_use():
                continue
            store = next(iter(r.uses)).operation
            if not isinstance(store, StoreOp):
                continue

            new_lower = op.lower
            new_upper = op.upper
            new_lowerext = op.lowerext
            new_upperext = op.upperext
            new_results_types = list(op.result_types)
            new_results_types.pop(i)

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
            rewriter.replace_op(
                op,
                new_combine,
                new_results=new_combine.results[:i] + (None,) + new_combine.results[i:],
            )
            return


class SwapBufferize(RewritePattern):
    """
    Bufferize a dmp.swap operation.

    NB: This should most likely consider a shared pass following canonicalize and
    shape-inference.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SwapOp, rewriter: PatternRewriter):
        temp = op.input_stencil

        if not isa(temp_t := temp.type, TempType[Attribute]):
            return

        load = temp.owner
        if not isinstance(load, LoadOp):
            return

        buffer = BufferOp.create(
            operands=[temp], result_types=[field_from_temp(temp_t)]
        )
        new_swap = SwapOp.get(buffer.res, op.strategy)
        new_swap.swaps = op.swaps
        load = LoadOp(operands=[buffer.res], result_types=[temp_t])

        rewriter.replace_op(
            op,
            new_ops=[buffer, new_swap, load],
        )


@dataclass(frozen=True)
class StencilBufferize(ModulePass):
    """
    Bufferize the stencil dialect, i.e., try to fold all loads, sotres, buffer and
    combines, and to output stencils working directly on buffers (fields) with
    hopefully few allocations.
    """

    name = "stencil-bufferize"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    UpdateApplyArgs(),
                    ApplyBufferizePattern(),
                    BufferAlloc(),
                    CombineStoreFold(),
                    LoadBufferFoldPattern(),
                    ApplyStoreFoldPattern(),
                    RemoveUnusedOperations(),
                    ApplyUnusedResults(),
                    SwapBufferize(),
                ]
            ),
            apply_recursively=True,
        )
        walker.rewrite_module(op)
