from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, scf
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    DynAccessOp,
    ResultType,
    ReturnOp,
    StencilBoundsAttr,
    StoreResultOp,
    TempType,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OpResult,
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
from xdsl.transforms.canonicalization_patterns.stencil import (
    ApplyRedundantOperands,
    ApplyUnusedOperands,
    ApplyUnusedResults,
)
from xdsl.transforms.shape_inference_patterns.stencil import update_result_size
from xdsl.transforms.stencil_unroll import offseted_block_clone


def is_before_in_block(op1: Operation, op2: Operation):
    """
    Check if op1 is before op2 in the same block.
    """
    block = op1.parent
    assert block is not None
    assert block is op2.parent
    return block.get_operation_index(op1) < block.get_operation_index(op2)


class StencilStoreResultForwardPattern(RewritePattern):
    """
    Replace non-empty `stencil.store_result`s by their argument.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreResultOp, rewriter: PatternRewriter, /):
        if op.arg is None:
            return
        rewriter.replace_op(op, [], [op.arg])


class StencilIfResultForwardPattern(RewritePattern):
    """
    Replace `!stencil.result<T>`-typed scf.if by `T`-typed scf.if.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.IfOp, rewriter: PatternRewriter, /):
        result_types = [r.type for r in op.output]
        new_result_types = [
            t.elem if isinstance(t, ResultType) else t for t in result_types
        ]
        if new_result_types == result_types:
            return
        rewriter.replace_op(
            op,
            scf.IfOp(
                op.cond,
                new_result_types,
                op.detach_region(0),
                op.detach_region(0),
            ),
        )


def has_single_consumer(producer: ApplyOp, consumer: ApplyOp):
    """
    Check if the producer has a single consumer.
    """
    return all(
        isinstance(u.operation, ApplyOp) and u.operation == consumer
        for r in producer.results
        for u in r.uses
    )


def is_rerouting_possible(producer: ApplyOp, consumer: ApplyOp):
    """
    Check if rerouting is possible.
    """
    # Perform producer consumer inlining instead
    if has_single_consumer(producer, consumer):
        return False
    return not any(
        isinstance(operand.owner, Operation)
        and (operand.owner is not producer)
        and is_before_in_block(producer, operand.owner)
        for operand in consumer.operands
    )


def is_inlining_possible(producer: ApplyOp, consumer: ApplyOp):
    """
    Check if inlining is possible.
    """
    # Don't inline any producer with conditional writes.
    r = not any(
        store_result.arg is None
        for store_result in producer.walk()
        if isinstance(store_result, StoreResultOp)
    ) and not any(
        # Don't inline any dynamic accesses.
        isinstance(use.operation, DynAccessOp)
        for consumer_operand in consumer.operands
        if consumer_operand.owner is producer
        for use in consumer.region.block.args[
            consumer.operands.index(consumer_operand)
        ].uses
    )

    return r


class StencilReroutingPattern(RewritePattern):
    """
        Reroute the producer's results through the consumer to enable inlining:
        ```
        a                           b                     a                           b
        │                           │                     │                           │
    ┌───▼─────────────┐             │                 ┌───▼─────────────┐             │
    │   producer      ├─┐           │                 │   producer      ├─┐           │
    └──┬──────────────┘ │e          │     rerouting   └─────────────┬───┘ │e          │
       │                │           │    ──────────►                │c'   │           │
       │              ┌─▼───────────▼──┐                            └──►┌─▼───────────▼──┐
       │              │    consumer    │                                │    consumer    │
       │              └────────────┬───┘                 ┌──────────────┴────────────┬───┘
       │                           │                     │                           │
       ▼                           ▼                     ▼                           ▼
       c                           d                     c                           d
       ```
    """

    def redirect_store(
        self, producer: ApplyOp, consumer: ApplyOp, rewriter: PatternRewriter
    ):
        # We want to replace the consumer adding the producer's results to its operands
        # and results
        new_operands = list(consumer.args) + list(producer.results)
        new_results = list(r.type for r in consumer.res + producer.res)

        new_consumer = ApplyOp.get(
            new_operands,
            Block(arg_types=[o.type for o in new_operands]),
            cast(Sequence[TempType[Attribute]], new_results),
        )

        # The new consumer contains the computation of the inital one
        rewriter.inline_block(
            consumer.region.block,
            InsertPoint.at_end(new_consumer.region.block),
            new_consumer.region.block.args[: len(consumer.args)],
        )

        # Update the bounds if needed
        producer_bounds = producer.res[0].type.bounds
        consumer_bounds = consumer.res[0].type.bounds
        if isinstance(producer_bounds, StencilBoundsAttr):
            new_bounds = producer_bounds | consumer_bounds
        elif isinstance(consumer_bounds, StencilBoundsAttr):
            new_bounds = producer_bounds | consumer_bounds
        else:
            new_bounds = None
        if isinstance(new_bounds, StencilBoundsAttr):
            update_result_size(new_consumer.res[0], new_bounds, rewriter)

        # Reroute new arguments to the new apply's return
        return_op = cast(ReturnOp, new_consumer.region.block.last_op)
        return_operands = list(return_op.arg)
        zero_offset = [0] * new_consumer.get_rank()
        for arg in new_consumer.region.block.args[-len(producer.res) :]:
            access = AccessOp.get(arg, zero_offset)
            rewriter.insert_op(access, InsertPoint.before(return_op))
            return_operands.append(access.res)
        rewriter.replace_op(return_op, ReturnOp.get(return_operands))

        # Replace the producer's results by the rerouted consumer results
        rerouted_results = new_consumer.res[-len(producer.res) :]
        for pres, rres in zip(producer.res, rerouted_results, strict=True):
            for use in list(pres.uses):
                if use.operation is new_consumer:
                    continue
                use.operation.operands[use.index] = rres

        rewriter.replace_op(
            consumer, new_consumer, new_consumer.res[: len(consumer.res)]
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter):
        consumer = op

        # Reroute input dependency
        # That is, two applys share some operands but have no direct def-use link.
        for operand in consumer.operands:
            if isinstance(operand.owner, Operation):
                for res in operand.owner.results:
                    for use in res.uses:
                        # Only consider other apply operations
                        if isinstance(producer := use.operation, ApplyOp):
                            # Only consider other consumers before the apply op
                            if consumer is producer:
                                continue
                            if not is_before_in_block(producer, consumer):
                                continue

                            if is_inlining_possible(
                                producer, consumer
                            ) and is_rerouting_possible(producer, consumer):
                                return self.redirect_store(producer, consumer, rewriter)

        # Reroute output dependency
        # That is, the consumer is already using some of the producer's results
        for operand in consumer.operands:
            producer = operand.owner
            if isinstance(producer, ApplyOp):
                if is_inlining_possible(producer, consumer) and is_rerouting_possible(
                    producer, consumer
                ):
                    return self.redirect_store(producer, consumer, rewriter)


@dataclass
class StencilInliningPattern(RewritePattern):
    """
    Inline a producer apply in a consumer apply, to use in the simple case where the
    consumer is the only user of the producer's results:
    ```
     a b          c
     │ │          │
    ┌▼─▼────────┐ │                   a b       c
    │ producer  │ │                   │ │       │
    └─────┬─┬───┘ │     inlining     ┌▼─▼───────▼┐
         d│ │e    │    ─────────►    │  inlined  │
        ┌─▼─▼─────▼┐                 └─────────┬─┘
        │ consumer │                           │
        └────────┬─┘                           ▼
                 │                          output
                 ▼
               output
    ```
    """

    result_type_cleaner = PatternRewriteWalker(
        GreedyRewritePatternApplier(
            [StencilIfResultForwardPattern(), StencilStoreResultForwardPattern()]
        )
    )

    def inline_producer(
        self, producer: ApplyOp, consumer: ApplyOp, rewriter: PatternRewriter
    ):
        """
        Inline the producer into the consumer.
        """

        self.result_type_cleaner.rewrite_region(producer.region)

        # Concatenate both applies operands lists.
        operands = list(consumer.operands) + list(producer.operands)

        # Create a new apply with the concatenated operands
        # Corresponding block arguments, and only the consumer's results.
        # (The producer's results are only used in the consumer by assumption)
        merged_block = Block(arg_types=[o.type for o in operands])

        # Prepare the list of block arguments corresponding to the producer's operands.
        merged_producer_arguments = merged_block.args[len(consumer.operands) :]

        # Inline the consumer's block to begin with.
        rewriter.inline_block(
            consumer.region.block,
            InsertPoint.at_start(merged_block),
            merged_block.args[: len(consumer.operands)],
        )

        # Store the list of consumer accesses
        consumer_accesses = [
            op for op in merged_block.walk(reverse=True) if isinstance(op, AccessOp)
        ]

        # Start inlining accesses to the producer
        for access in consumer_accesses:
            # Skip if it is another access
            temp = consumer.args[cast(BlockArgument, access.temp).index]
            if temp.owner is not producer:
                continue
            # Make pyright happy about temp being an OpResult
            temp = cast(OpResult, temp)
            # Find the index of the producer's result
            producer_index = producer.res.index(temp)

            # Clone the producer's block offseted according to the access offset.
            offsetted_block = offseted_block_clone(producer, list(access.offset))

            # Get the returnop's accessed operand.
            return_op = cast(ReturnOp, offsetted_block.last_op)
            accessed = return_op.arg[producer_index]

            # Remove the return, inline the computation, replace the access.
            rewriter.erase_op(return_op)
            rewriter.inline_block(
                offsetted_block, InsertPoint.before(access), merged_producer_arguments
            )
            rewriter.replace_op(access, [], [accessed])

        new_operands = operands
        for arg in reversed(list(merged_block.args)):
            if not arg.uses:
                new_operands.pop(arg.index)
                merged_block.erase_arg(arg)
        new_apply = ApplyOp.get(
            new_operands,
            merged_block,
            [cast(TempType[Attribute], r.type) for r in consumer.results],
        )
        rewriter.replace_op(consumer, new_apply)
        rewriter.erase_op(producer)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        for operand in (consumer := op).operands:
            if isinstance(producer := operand.owner, ApplyOp):
                if has_single_consumer(producer, consumer) and is_inlining_possible(
                    producer, consumer
                ):
                    return self.inline_producer(producer, consumer, rewriter)


@dataclass(frozen=True)
class StencilInliningPass(ModulePass):
    name = "stencil-inlining"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StencilReroutingPattern(),
                    StencilInliningPattern(),
                    ApplyUnusedResults(),
                    ApplyUnusedOperands(),
                    ApplyRedundantOperands(),
                ]
            )
        )
        walker.rewrite_module(op)
