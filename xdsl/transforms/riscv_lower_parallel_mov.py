from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp, SSAValue
from xdsl.ir import Attribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import PassFailedException


class ParallelMovPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.ParallelMovOp, rewriter: PatternRewriter):
        input_types = cast(Sequence[riscv.RISCVRegisterType], op.inputs.types)
        output_types = cast(Sequence[riscv.RISCVRegisterType], op.outputs.types)

        if not (
            all(i.is_allocated for i in input_types)
            and all(i.is_allocated for i in output_types)
        ):
            raise PassFailedException("All registers must be allocated")

        if not (
            all(isinstance(i, riscv.IntRegisterType) for i in input_types)
            and all(isinstance(i, riscv.IntRegisterType) for i in output_types)
        ):
            raise PassFailedException("Not implemented: non-integer support")

        # make free registers a list so we can add to it later
        free_registers: list[riscv.IntRegisterType] = []
        if op.free_registers is not None:
            free_registers = [
                i for i in op.free_registers if isinstance(i, riscv.IntRegisterType)
            ]

        num_operands = len(op.operands)

        results: list[SSAValue | None] = [None] * num_operands

        # We have a graph with nodes as registers and directed edges as moves,
        # pointing from source to destination.
        # Every node has at most 1 in edge since we can't write to a register twice.
        # Therefore the graph forms a directed pseudoforest, which is a group of
        # connected components with at most 1 cycle each.

        # If we ignore the cycles, we will have a forest.
        # For each tree, we need to perform each move such that all out edges of a node
        # are before the in edge, so a post-order traversal.
        # We can do this iteratively by storing processed edges for each node.
        # Then we iterate up the tree from every leaf, stopping whenever we encounter
        # a node where all out edges haven't been processed yet.

        # store the back edges of the graph
        dst_to_src: dict[riscv.RegisterType, SSAValue] = {}
        leaves: set[Attribute] = set(op.outputs.types)
        unprocessed_children: Counter[SSAValue] = Counter()

        for idx, src, dst in zip(
            range(num_operands), op.inputs, op.outputs, strict=True
        ):
            # src.type points to something so it can't be a leaf
            leaves.discard(src.type)

            if src.type == dst.type:
                # Trivial case of moving register to itself.
                # We can ignore all instances of this
                results[idx] = src
            else:
                dst_to_src[dst.type] = src
                unprocessed_children[src] += 1

        for dst in op.outputs.types:
            if dst not in leaves:
                continue
            # Iterate up the tree by traversing back edges.
            while dst in dst_to_src:
                src = dst_to_src[dst]
                mvop = riscv.MVOp(src, rd=dst)
                rewriter.insert_op(mvop, InsertPoint.before(rewriter.current_operation))
                # sanity check since we should only have 1 result per output
                assert results[op.outputs.types.index(dst)] is None
                results[op.outputs.types.index(dst)] = mvop.results[0]
                unprocessed_children[src] -= 1
                # only continue up the tree if all children were processed
                if unprocessed_children[src]:
                    break
                dst = src.type

            # if dst is a register that has no input, we can use it as a free register.
            if dst not in dst_to_src and isinstance(dst, riscv.IntRegisterType):
                free_registers.append(dst)

        # If we have a cycle in the graph, all trees pointing into the cycle cannot
        # enter the cycle because it will have an unprocessed node from its previous
        # node in the cycle.
        # Therefore, all nodes in the cycle will be unprocessed, and their results
        # will still be None

        for idx, val in enumerate(results):
            if val is None:
                # Find a free integer register.
                # We don't have to modify its value since all the cycles
                # can use the same register.
                if not free_registers:
                    raise PassFailedException(
                        "Not implemented: cyclic moves without free int register."
                    )
                temp_reg = free_registers[0]

                # Break the cycle by using free register
                # split the current mov
                cur_input = op.inputs[idx]
                cur_output = op.outputs[idx]
                temp_ssa = riscv.MVOp(cur_input, rd=temp_reg)
                rewriter.insert_op(
                    temp_ssa, InsertPoint.before(rewriter.current_operation)
                )
                # iterate up the chain until we reach the current output
                dst = cur_input.type
                while dst != cur_output.type:
                    src = dst_to_src[dst]
                    mvop = riscv.MVOp(src, rd=dst)
                    rewriter.insert_op(
                        mvop, InsertPoint.before(rewriter.current_operation)
                    )
                    results[op.outputs.types.index(dst)] = mvop.results[0]
                    dst = src.type
                # finish the split mov
                # this assert is already checked at start, but is used for type checking
                assert isinstance(cur_output.type, riscv.IntRegisterType)
                mvop = riscv.MVOp(temp_ssa, rd=cur_output.type)
                rewriter.insert_op(mvop, InsertPoint.before(rewriter.current_operation))
                results[idx] = mvop.results[0]

        rewriter.replace_matched_op([], results)


@dataclass(frozen=True)
class RISCVLowerParallelMovPass(ModulePass):
    """Lowers ParallelMovOp in a module into separate moves."""

    name = "riscv-lower-parallel-mov"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ParallelMovPattern()).rewrite_module(op)
