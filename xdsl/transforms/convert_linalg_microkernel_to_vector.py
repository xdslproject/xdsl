from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg, memref, vector
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa


@dataclass
class MatmulToVector(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MatmulOp, rewriter: PatternRewriter):
        A = op.inputs[0]
        B = op.inputs[1]
        C = op.outputs[0]
        if not (
            isa(A.type, builtin.MemRefType)
            and isa(B.type, builtin.MemRefType)
            and isa(C.type, builtin.MemRefType)
        ):
            raise DiagnosticException("linalg.matmul on tensors is not yet implemented")
        if (
            len(A.type.get_shape()) != 2
            or len(B.type.get_shape()) != 2
            or len(C.type.get_shape()) != 2
        ):
            raise DiagnosticException(
                "MemRefs with ranks higher than 2 are not supported"
            )
        I, J, K = C.type.get_shape()[0], C.type.get_shape()[1], A.type.get_shape()[1]
        i_consts = [
            arith.ConstantOp.from_int_and_width(i, builtin.IndexType())
            for i in range(I)
        ]
        j_consts = [
            arith.ConstantOp.from_int_and_width(j, builtin.IndexType())
            for j in range(J)
        ]
        k_consts = [
            arith.ConstantOp.from_int_and_width(k, builtin.IndexType())
            for k in range(K)
        ]
        rewriter.insert_op_before_matched_op(i_consts + j_consts + k_consts)
        # Load B and C
        vect_type = builtin.VectorType(
            element_type=A.type.get_element_type(), shape=(J,)
        )
        B_loads = [
            vector.LoadOp(B, [k_const, j_consts[0]], vect_type) for k_const in k_consts
        ]
        C_loads = [
            vector.LoadOp(C, [i_const, j_consts[0]], vect_type) for i_const in i_consts
        ]
        rewriter.insert_op_before_matched_op(C_loads + B_loads)
        # Load A, perform the reduction and store C
        C_stores: list[vector.StoreOp] = []
        for i in range(I):
            i_const = i_consts[i]
            A_loads = [memref.LoadOp.get(A, [i_const, k_const]) for k_const in k_consts]
            A_broadcasts = [vector.BroadcastOp(a_scal, vect_type) for a_scal in A_loads]
            rewriter.insert_op_before_matched_op(A_loads + A_broadcasts)
            acc = C_loads[i]
            for a, b in zip(A_broadcasts, B_loads):
                acc = vector.FMAOp(lhs=a, rhs=b, acc=acc)
                rewriter.insert_op_before_matched_op(acc)
            C_stores.append(
                vector.StoreOp(vector=acc, base=C, indices=[i_const, j_consts[0]])
            )
        rewriter.insert_op_before_matched_op(C_stores)

        rewriter.erase_matched_op()


@dataclass(frozen=True)
class ConvertLinalgMicrokernelToVectorPass(ModulePass):
    name = "convert-linalg-microkernel-to-vector"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MatmulToVector(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
