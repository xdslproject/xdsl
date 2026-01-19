from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.backend.riscv.lowering.utils import move_ops_for_value
from xdsl.context import Context
from xdsl.dialects import builtin, riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, SSAValues
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import PassFailedException


def _insert_mv_op(
    rewriter: PatternRewriter,
    src: SSAValue | Operation,
    dst: riscv.RISCVRegisterType,
    src_width: int,
):
    if isinstance(src, Operation):
        src = SSAValue.get(src)

    # We only need to specify the type in case of floats
    # otherwise, the type is ignored, so we just pass an arbitrary attribute here
    src_type = builtin.IntegerAttr(0, 1)
    if isinstance(src.type, riscv.FloatRegisterType):
        match src_width:
            case 32:
                src_type = builtin.Float32Type()
            case 64:
                src_type = builtin.Float64Type()
            case _:
                pass

    op, _ = move_ops_for_value(src, src_type, dst)
    rewriter.insert_op(op)
    return op


def _insert_swap_ops(
    rewriter: PatternRewriter,
    a: SSAValue[riscv.IntRegisterType],
    b: SSAValue[riscv.IntRegisterType],
) -> tuple[SSAValue[riscv.IntRegisterType], SSAValue[riscv.IntRegisterType]]:
    """Add swap using xors. returns the new SSAValues."""
    op1 = rewriter.insert_op(riscv.XorOp(a, b, rd=a.type))
    op2 = rewriter.insert_op(riscv.XorOp(op1, b, rd=b.type))
    op3 = rewriter.insert_op(riscv.XorOp(op1, op2, rd=a.type))
    return op2.rd, op3.rd


class ParallelMovPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.ParallelMovOp, rewriter: PatternRewriter):
        srcs = cast(SSAValues[SSAValue[riscv.RISCVRegisterType]], op.inputs)
        dsts = cast(SSAValues[SSAValue[riscv.RISCVRegisterType]], op.outputs)
        src_types = cast(Sequence[riscv.RISCVRegisterType], op.inputs.types)
        dst_types = cast(Sequence[riscv.RISCVRegisterType], op.outputs.types)

        if not (
            all(i.is_allocated for i in src_types)
            and all(i.is_allocated for i in dst_types)
        ):
            raise PassFailedException("All registers must be allocated")

        # make a list of free registers for each type so we can add to it later
        free_registers: dict[
            type[riscv.RISCVRegisterType], list[riscv.RISCVRegisterType]
        ] = defaultdict(list)
        if op.free_registers is not None:
            for reg in op.free_registers:
                free_registers[type(reg)].append(reg)

        num_operands = len(op.operands)

        results: list[SSAValue | None] = [None] * num_operands

        # cache the indices from output register type to the index in the outputs array
        # this is typed as Attribute to ensure we can index by input type
        output_index = {register: idx for idx, register in enumerate(dst_types)}

        src_type_by_src = {
            src: src_type
            for src, src_type in zip(srcs, op.input_widths.iter_values(), strict=True)
        }

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
        src_by_dst_type: dict[
            riscv.RISCVRegisterType, SSAValue[riscv.RISCVRegisterType]
        ] = {}
        leaves = set(dst_types)
        unprocessed_children = Counter[SSAValue]()

        for idx, src, dst in zip(range(num_operands), srcs, dsts, strict=True):
            # src.type points to something so it can't be a leaf
            leaves.discard(src.type)

            if src.type == dst.type:
                # Trivial case of moving register to itself.
                # We can ignore all instances of this
                results[idx] = src
            else:
                src_by_dst_type[dst.type] = src
                unprocessed_children[src] += 1

        for dst_type in dst_types:
            if dst_type not in leaves:
                continue
            # Iterate up the tree by traversing back edges.
            while dst_type in src_by_dst_type:
                src = src_by_dst_type[dst_type]
                mvop = _insert_mv_op(rewriter, src, dst_type, src_type_by_src[src])
                # sanity check since we should only have 1 result per output
                assert results[output_index[dst_type]] is None
                results[output_index[dst_type]] = mvop.results[0]
                unprocessed_children[src] -= 1
                # only continue up the tree if all children were processed
                if unprocessed_children[src]:
                    break
                dst_type = src.type

            # if dst is a register that has no input, we can use it as a free register.
            if dst_type not in src_by_dst_type:
                free_registers[type(dst_type)].append(dst_type)

        # If we have a cycle in the graph, all trees pointing into the cycle cannot
        # enter the cycle because it will have an unprocessed node from its previous
        # node in the cycle.
        # Therefore, all nodes in the cycle will be unprocessed, and their results
        # will still be None

        for idx, val in enumerate(results):
            if val is None:
                reg_type = type(dst_types[idx])
                # Find a free register.
                # We don't have to modify its value since all the cycles
                # can use the same register.
                if not free_registers[reg_type]:
                    if reg_type != riscv.IntRegisterType:
                        raise PassFailedException(
                            "Float cyclic move without free register"
                        )

                    # Otherwise if the registers are all integers, we can use the xor swapping
                    # trick to repeatedly swap values to perform the cyclic move.

                    # we don't take srcs[idx] -> dsts[idx] since we need
                    # the SSAValue for both input and output
                    out = srcs[idx]
                    inp = src_by_dst_type[out.type]

                    while inp.type != out.type:
                        # we know these are ints since input and output are of the same type
                        inp = cast(SSAValue[riscv.IntRegisterType], inp)
                        out = cast(SSAValue[riscv.IntRegisterType], out)
                        nw_out, nw_inp = _insert_swap_ops(rewriter, inp, out)
                        # after the swap, the input is in the right place, the input's input
                        # needs to be moved to the new output
                        results[output_index[nw_inp.type]] = nw_inp
                        inp = src_by_dst_type[inp.type]
                        out = nw_out

                    results[output_index[src_types[idx]]] = out
                    continue

                # Break the cycle by using free register
                temp_reg = free_registers[reg_type][0]
                # split the current mov
                cur_input = srcs[idx]
                cur_output = dsts[idx]
                temp_ssa_type = op.input_widths.get_values()[idx]
                temp_ssa = _insert_mv_op(rewriter, cur_input, temp_reg, temp_ssa_type)
                # iterate up the chain until we reach the current output
                dst_type = cur_input.type
                while dst_type != cur_output.type:
                    src = src_by_dst_type[dst_type]
                    mvop = _insert_mv_op(rewriter, src, dst_type, src_type_by_src[src])
                    results[output_index[dst_type]] = mvop.results[0]
                    dst_type = src.type
                # finish the split mov
                mvop = _insert_mv_op(rewriter, temp_ssa, cur_output.type, temp_ssa_type)
                results[idx] = mvop.results[0]

        rewriter.replace_matched_op((), results)


@dataclass(frozen=True)
class RISCVLowerParallelMovPass(ModulePass):
    """Lowers ParallelMovOp in a module into separate moves."""

    name = "riscv-lower-parallel-mov"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(ParallelMovPattern()).rewrite_module(op)
