from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import I32, IntegerAttr, i32
from xdsl.ir import OpResult, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class ShiftLeftImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.SlliOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv32.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_op(
                op,
                rv32.LiOp(
                    op.rs1.op.immediate.value.data << op.immediate.value.data, rd=rd
                ),
            )


class ShiftLeftbyZero(RewritePattern):
    """
    x << 0 -> x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.SlliOp, rewriter: PatternRewriter) -> None:
        # check if the shift amount is zero
        if isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0:
            rewriter.replace_op(op, riscv.MVOp(op.rs1, rd=op.rd.type))


class ShiftRightImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.SrliOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv32.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_op(
                op,
                rv32.LiOp(
                    op.rs1.op.immediate.value.data >> op.immediate.value.data, rd=rd
                ),
            )


class ShiftRightbyZero(RewritePattern):
    """
    x >> 0 -> x
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.SrliOp, rewriter: PatternRewriter) -> None:
        # check if the shift amount is zero
        if isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0:
            rewriter.replace_op(op, riscv.MVOp(op.rs1, rd=op.rd.type))


class AddImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.AddiOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0:
            rd = op.rd.type
            rewriter.replace_op(op, riscv.MVOp(op.rs1, rd=rd))


class AddImmediateConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.AddiOp, rewriter: PatternRewriter) -> None:
        if (rs1 := get_constant_value(op.rs1)) is not None and isinstance(
            op.immediate, IntegerAttr
        ):
            rd = op.rd.type
            rewriter.replace_op(
                op,
                rv32.LiOp(
                    rs1.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                ),
            )


class AndiImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.AndiOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv32.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_matched_op(
                rv32.LiOp(
                    op.rs1.op.immediate.value.data & op.immediate.value.data, rd=rd
                )
            )


class OriImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.OriOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv32.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_matched_op(
                rv32.LiOp(
                    op.rs1.op.immediate.value.data | op.immediate.value.data, rd=rd
                )
            )


class XoriImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.XoriOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv32.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_matched_op(
                rv32.LiOp(
                    op.rs1.op.immediate.value.data ^ op.immediate.value.data, rd=rd
                )
            )


class LoadWordWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: rv32.LwOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, rv32.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = op.rd.type
            rewriter.replace_op(
                op,
                rv32.LwOp(
                    op.rs1.op.rs1,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                ),
            )


def get_constant_value(value: SSAValue) -> IntegerAttr[I32] | None:
    if value.type == riscv.Registers.ZERO:
        return IntegerAttr(0, i32)

    if not isinstance(value, OpResult):
        return

    if isinstance(value.op, riscv.MVOp):
        return get_constant_value(value.op.rs)

    if isinstance(value.op, rv32.LiOp) and isinstance(value.op.immediate, IntegerAttr):
        return value.op.immediate
