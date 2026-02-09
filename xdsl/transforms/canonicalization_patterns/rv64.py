from xdsl.dialects import riscv, rv64
from xdsl.dialects.builtin import I64, IntegerAttr, i64
from xdsl.ir import OpResult, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ShiftLeftImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv64.SlliOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv64.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_op(
                op,
                rv64.LiOp(
                    op.rs1.op.immediate.value.data << op.immediate.value.data, rd=rd
                ),
            )


class ShiftLeftbyZero(RewritePattern):
    """
    x << 0 -> x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv64.SlliOp, rewriter: PatternRewriter) -> None:
        # check if the shift amount is zero
        if isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0:
            rewriter.replace_op(op, riscv.MVOp(op.rs1, rd=op.rd.type))


class ShiftRightImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv64.SrliOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv64.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_op(
                op,
                rv64.LiOp(
                    op.rs1.op.immediate.value.data >> op.immediate.value.data, rd=rd
                ),
            )


class ShiftRightbyZero(RewritePattern):
    """
    x >> 0 -> x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv64.SrliOp, rewriter: PatternRewriter) -> None:
        # check if the shift amount is zero
        if isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0:
            rewriter.replace_op(op, riscv.MVOp(op.rs1, rd=op.rd.type))


def get_constant_value(value: SSAValue) -> IntegerAttr[I64] | None:
    if value.type == riscv.Registers.ZERO:
        return IntegerAttr(0, i64)

    if not isinstance(value, OpResult):
        return

    if isinstance(value.op, riscv.MVOp):
        return get_constant_value(value.op.rs)

    if isinstance(value.op, rv64.LiOp) and isinstance(value.op.immediate, IntegerAttr):
        return value.op.immediate
