from dataclasses import dataclass

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import MLContext, Block, OpResult, SSAValue, Use
from xdsl.irdl import Region, Attribute
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, i64

from xdsl.dialects.experimental.stencil import (AccessOp, ApplyOp, IndexAttr,
                                                ReturnOp, StoreOp,
                                                StoreResultOp)
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


# Base class for stencil inlining and rerouting.
@dataclass
class StencilInliningPattern(RewritePattern):
    # Check if there is a single apply_op consumer for current producer apply_op.
    def has_single_consumer(self, producer_op: ApplyOp) -> bool:
        return any(
            isinstance(use.operation, ApplyOp) for res in producer_op.res
            for use in res.uses)

    # Check if inlining is possible.
    def is_stencil_inlining_possible(self, producer_op: ApplyOp) -> bool:
        # Do not inline producer ops that do not store stuff.
        for res in producer_op.res:
            for use in res.uses:
                if (isinstance(use.operation, StoreOp)):
                    return True

        for op in producer_op.region.blocks[0].ops:
            if (isinstance(op, StoreResultOp)):
                return True

        return False

        # Not adding the case for dealing with dynamic offsets since we do not support them
        # as of now.

    # Get the consumer apply op for considered producer apply op.
    def get_single_consumer_apply_op(self,
                                     producer_op: ApplyOp) -> ApplyOp | None:
        for res in list(producer_op.res):
            for use in res.uses:
                if (isinstance(use.operation, ApplyOp)):
                    return use.operation
        return None


# Replace all block arguments by their definition and erase them from the block.
def replace_and_erase_block_args_in_apply_op(entry_point: Block,
                                             apply_op: ApplyOp):
    for idx, arg in enumerate(entry_point.args):
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.replace_operand(use.index, apply_op.args[idx])
        entry_point.erase_arg(arg)


def remove_unused_store_result_op_and_return_op_from_producer_op(
        producer_op: ApplyOp, consumer_op: ApplyOp,
        producer_op_external_uses: list[Use],
        inlined_op_res_list: list[Attribute], rewriter: PatternRewriter):
    # Remove ReturnOp and StoreResultOp from producer which do not have another
    # use apart from the consumer_op.
    for op in producer_op.region.ops:
        if isinstance(op, ReturnOp):
            # Flag for signifying an external use of the producer_op result apart from
            # inside the consumer_op.
            external_use_flag = 0
            # Container for storing those results of producer_op which have an external use
            # apart from inside the consumer_op.
            conserved_return_val: list[OpResult] = []
            op_args = list(op.operands)
            for i, return_val in enumerate(op_args):
                for use in list(producer_op.results[i].uses):
                    external_use_flag = 0
                    if not consumer_op is use.operation and not consumer_op.region.blocks[
                            0] is use.operation.parent:
                        external_use_flag = 1
                        producer_op_external_uses.append(use)
                        break
                if external_use_flag:
                    assert isinstance(return_val, OpResult)
                    conserved_return_val.append(return_val)
                    inlined_op_res_list.append(
                        producer_op.results[0].op.results[i].typ)
            if not len(conserved_return_val):
                # Erase the producer_op's return op if no result is needed to be conserved
                # after inlining.
                producer_op.region.blocks[0].erase_op(op)
            else:
                # Replace the producer_op's return op with a new return op containing results
                # which are needed to be conserved after inlining.
                new_return_op = ReturnOp.get(conserved_return_val)
                assert isinstance(new_return_op, ReturnOp)
                rewriter.replace_op(op, new_return_op)

    # Remove unused StoreResult ops from inside the producer_op.
    for op in producer_op.region.ops:
        if isinstance(op, StoreResultOp) and not len(op.results[0].uses):
            producer_op.region.blocks[0].erase_op(op)


def insert_inlined_op_block_args(inlined_op_operands: list[SSAValue],
                                 inlined_op_block: Block,
                                 rewriter: PatternRewriter):
    for i, operand in enumerate(inlined_op_operands):
        rewriter.insert_block_argument(inlined_op_block, i, operand.typ)
        uses = list(operand.uses)
        for use in uses:
            use.operation.replace_operand(use.index, inlined_op_block.args[i])


def find_producer_op_result_traces(producer_op: ApplyOp,
                                   producer_op_result_traces: list[SSAValue]):
    for i in range(len(producer_op.res)):
        for use in producer_op.res[i].uses:
            if (isinstance(use.operation, ApplyOp)):
                producer_op_result_traces.append(use.operation.args[use.index])


@dataclass
class InliningRewrite(StencilInliningPattern):

    def inline_producer(self, producer_op: ApplyOp, rewriter: PatternRewriter,
                        /):
        # Get consumer apply op corresponding to the producer apply op.
        consumer_op = super().get_single_consumer_apply_op(producer_op)
        assert (isinstance(consumer_op, ApplyOp))

        # Obtain entry points for both producer_op and consumer_op.
        entry_producer = producer_op.region.blocks[0]
        entry_consumer = consumer_op.region.blocks[0]

        replace_and_erase_block_args_in_apply_op(entry_producer, producer_op)
        replace_and_erase_block_args_in_apply_op(entry_consumer, consumer_op)

        # Obtain operands for the final inlined op.
        inlined_op_operands = list(producer_op.operands)
        for operand in list(consumer_op.operands):
            # Check if this operand is not a result or operand of producer_op.
            if operand not in list(
                    producer_op.res) and operand not in producer_op.operands:
                inlined_op_operands.append(operand)

        # Initiate result list of the inlined op with results of consumer op.
        inlined_op_res_list = [
            consumer_op_res.typ for consumer_op_res in consumer_op.res
        ]

        # A container to store uses of producer op which do not overlap with consumer op.
        producer_op_external_uses: list[Use] = []
        remove_unused_store_result_op_and_return_op_from_producer_op(
            producer_op, consumer_op, producer_op_external_uses,
            inlined_op_res_list, rewriter)

        # Instantiate inlined op region and corresponding block.
        inlined_op_region = Region()
        inlined_op_block = Block()

        # Insert inlined op block arguments in inlined op block and replace corresponding uses.
        insert_inlined_op_block_args(inlined_op_operands, inlined_op_block,
                                     rewriter)

        # Container to store traces of use of producer_op results in other apply ops.
        producer_op_result_traces: list[SSAValue] = []
        find_producer_op_result_traces(producer_op, producer_op_result_traces)

        # Container to store return values of the resultant inlined op.
        inlined_op_return_arguments: list[OpResult] = []

        # Start inlining ops depending on their use in consumer op.
        for op in consumer_op.region.ops:
            if isinstance(op,
                          AccessOp) and op.temp in producer_op_result_traces:
                for i, producer_op_unit in enumerate(producer_op.region.ops):
                    if isinstance(producer_op_unit, AccessOp):
                        producer_op_unit_clone = producer_op_unit.clone()
                        new_offset = IndexAttr.add_offsets(
                            producer_op_unit_clone.offset, op.offset)
                        new_offset_integer_attr_array: list[
                            IntegerAttr[IntegerType]] = [
                                IntegerAttr(offset_val, i64)
                                for offset_val in new_offset
                            ]
                        new_offset_attr = IndexAttr(
                            [ArrayAttr(new_offset_integer_attr_array)])
                        producer_op_unit_clone.offset = new_offset_attr
                        inlined_op_block.add_op(producer_op_unit_clone)

                        uses = list(producer_op_unit.res.uses)
                        for use in uses:
                            use.operation.replace_operand(
                                use.index, producer_op_unit_clone.res)
                    elif not isinstance(producer_op_unit, ReturnOp):
                        producer_op_unit_clone_normal_op = producer_op_unit.clone(
                        )
                        inlined_op_block.add_op(
                            producer_op_unit_clone_normal_op)

                        if isinstance(producer_op_unit, StoreResultOp):
                            uses = list(producer_op_unit.res.uses)
                            for use in uses:
                                use.operation.replace_operand(
                                    use.index,
                                    producer_op_unit_clone_normal_op.results[0]
                                )
                        else:
                            use_other_than_store_or_return = 0
                            for use in producer_op_unit.results[0].uses:
                                if not isinstance(use.operation,
                                                  ReturnOp) and not isinstance(
                                                      use.operation,
                                                      StoreResultOp):
                                    use_other_than_store_or_return = 1
                                    break

                            if not use_other_than_store_or_return:
                                res_final = producer_op_unit_clone_normal_op.results[
                                    0]

                                uses = list(op.results[0].uses)
                                for use in uses:
                                    use.operation.replace_operand(
                                        use.index, res_final)

                                uses = list(producer_op_unit.results[0].uses)
                                for use in uses:
                                    use.operation.replace_operand(
                                        use.index, res_final)
                            else:
                                res_final = producer_op_unit_clone_normal_op.results[
                                    0]

                                uses = list(producer_op_unit.results[0].uses)
                                for use in uses:
                                    use.operation.replace_operand(
                                        use.index, res_final)
                    elif isinstance(producer_op_unit, ReturnOp):
                        if not len(inlined_op_return_arguments):
                            for x in list(producer_op_unit.operands):
                                assert isinstance(x, OpResult)
                                inlined_op_return_arguments.append(x)
            elif not isinstance(op, ReturnOp):
                op_clone = op.clone()
                inlined_op_block.add_op(op_clone)

                for i, res in enumerate(op.results):
                    res_uses = list(res.uses)
                    for use in res_uses:
                        use.operation.replace_operand(use.index,
                                                      op_clone.results[i])
            else:
                if not len(inlined_op_return_arguments):
                    op_clone = op.clone()
                    inlined_op_block.add_op(op_clone)
                else:
                    combined_list: list[SSAValue] = [
                        *inlined_op_return_arguments, *list(op.operands)
                    ]

                    inlined_op_return = ReturnOp.get(combined_list)
                    inlined_op_block.add_op(inlined_op_return)

        # Attach inlined op block to the inlined op region.
        inlined_op_region.add_block(inlined_op_block)

        assert isinstance(consumer_op.lb, IndexAttr)
        assert isinstance(consumer_op.ub, IndexAttr)

        # Get the final op.
        InlinedOp = ApplyOp.get(inlined_op_operands, inlined_op_region,
                                consumer_op.lb, consumer_op.ub,
                                [inlined_op_res_list])

        rewriter.insert_op_before_matched_op([InlinedOp])

        # Replace producer op's result with inlined op's results.
        for i, use in enumerate(producer_op_external_uses):
            use.operation.replace_operand(use.index, InlinedOp.res[i])

        # Replace consumer op's result with inlined op's results.
        consumer_op_res_list = list(consumer_op.res)
        for i, consumer_op_res in enumerate(consumer_op_res_list):
            consumer_op_res_uses = list(consumer_op_res.uses)
            for use in consumer_op_res_uses:
                use.operation.replace_operand(
                    use.index,
                    InlinedOp.res[i + len(producer_op_external_uses)])

        # Remove consumer op from the IR.
        consumer_op_parent = consumer_op.parent
        assert isinstance(consumer_op_parent, Block)
        consumer_op_parent.erase_op(consumer_op, False)

        # Remove producer op from the IR.
        rewriter.erase_matched_op(False)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if super().has_single_consumer(
                op) and super().is_stencil_inlining_possible(op):
            self.inline_producer(op, rewriter)


class StencilInlining(ModulePass):
    name = 'stencil-inlining'

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier(
            [InliningRewrite()]),
                                            apply_recursively=False,
                                            walk_reverse=False)
        the_one_pass.rewrite_module(op)
