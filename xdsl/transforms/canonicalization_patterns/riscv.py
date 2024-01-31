from typing import cast

from xdsl.dialects import riscv, riscv_snitch
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RemoveRedundantMv(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MVOp, rewriter: PatternRewriter) -> None:
        if (
            op.rd.type == op.rs.type
            and isinstance(op.rd.type, riscv.RISCVRegisterType)
            and op.rd.type.is_allocated
        ):
            rewriter.replace_matched_op([], [op.rs])


class ImmediateMoveToCopy(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MVOp, rewriter: PatternRewriter, /):
        if isinstance(op.rd.type, riscv.IntRegisterType) and isinstance(
            op.rs.owner, riscv.LiOp
        ):
            rewriter.replace_matched_op(
                riscv.LiOp(op.rs.owner.immediate, rd=op.rd.type)
            )


class RemoveRedundantFMv(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FMVOp, rewriter: PatternRewriter) -> None:
        if (
            op.rd.type == op.rs.type
            and isinstance(op.rd.type, riscv.RISCVRegisterType)
            and op.rd.type.is_allocated
        ):
            rewriter.replace_matched_op([], [op.rs])


class RemoveRedundantFMvD(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FMvDOp, rewriter: PatternRewriter) -> None:
        if (
            op.rd.type == op.rs.type
            and isinstance(op.rd.type, riscv.RISCVRegisterType)
            and op.rd.type.is_allocated
        ):
            rewriter.replace_matched_op([], [op.rs])


class MultiplyImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MulOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.rs2, OpResult)
            and isinstance(op.rs2.op, riscv.LiOp)
            and isinstance(op.rs2.op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LiOp(
                    op.rs1.op.immediate.value.data * op.rs2.op.immediate.value.data,
                    rd=rd,
                )
            )


class MultiplyImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MulOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and op.rs1.op.immediate.value.data == 0
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs1, rd=rd))
        elif (
            isinstance(op.rs2, OpResult)
            and isinstance(op.rs2.op, riscv.LiOp)
            and isinstance(op.rs2.op.immediate, IntegerAttr)
            and op.rs2.op.immediate.value.data == 0
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs2, rd=rd))


class AddImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AddOp, rewriter: PatternRewriter) -> None:
        lhs: int | None = None
        rhs: int | None = None
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
        ):
            lhs = op.rs1.op.immediate.value.data

        if (
            isinstance(op.rs2, OpResult)
            and isinstance(op.rs2.op, riscv.LiOp)
            and isinstance(op.rs2.op.immediate, IntegerAttr)
        ):
            rhs = op.rs2.op.immediate.value.data

        rd = cast(riscv.IntRegisterType, op.rd.type)

        match (lhs, rhs):
            case int(), None:
                rewriter.replace_matched_op(
                    riscv.AddiOp(
                        op.rs2,
                        lhs,
                        rd=rd,
                        comment=op.comment,
                    )
                )
            case None, int():
                rewriter.replace_matched_op(
                    riscv.AddiOp(
                        op.rs1,
                        rhs,
                        rd=rd,
                        comment=op.comment,
                    )
                )
            case int(), int():
                rewriter.replace_matched_op(
                    riscv.LiOp(lhs + rhs, rd=rd, comment=op.comment)
                )
            case _:
                pass


class AddImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AddiOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0:
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs1, rd=rd))


class AddImmediateConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AddiOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(li := op.rs1.owner, riscv.LiOp)
            and isinstance(imm := li.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LiOp(
                    imm.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                )
            )


class SubImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.SubOp, rewriter: PatternRewriter) -> None:
        lhs: int | None = None
        rhs: int | None = None
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
        ):
            lhs = op.rs1.op.immediate.value.data

        if (
            isinstance(op.rs2, OpResult)
            and isinstance(op.rs2.op, riscv.LiOp)
            and isinstance(op.rs2.op.immediate, IntegerAttr)
        ):
            rhs = op.rs2.op.immediate.value.data

        rd = cast(riscv.IntRegisterType, op.rd.type)

        match (lhs, rhs):
            case int(), None:
                # TODO: anything to do here?
                return
            case None, int():
                rewriter.replace_matched_op(
                    riscv.AddiOp(
                        op.rs1,
                        -rhs,
                        rd=rd,
                        comment=op.comment,
                    )
                )
            case int(), int():
                rewriter.replace_matched_op(
                    riscv.LiOp(lhs - rhs, rd=rd, comment=op.comment)
                )
            case _:
                pass


class SubAddi(RewritePattern):
    """
    (a + 4) - a -> 4
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.SubOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and op.rs2 == op.rs1.op.rs1
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.LiOp(op.rs1.op.immediate, rd=rd))


class ShiftLeftImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.SlliOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.LiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LiOp(
                    op.rs1.op.immediate.value.data << op.immediate.value.data, rd=rd
                )
            )


class LoadWordWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.LwOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LwOp(
                    op.rs1.op.rs1,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                )
            )


class StoreWordWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.SwOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
        ):
            rewriter.replace_matched_op(
                riscv.SwOp(
                    op.rs1.op.rs1,
                    op.rs2,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    comment=op.comment,
                )
            )


class LoadFloatWordWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FLwOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.FloatRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.FLwOp(
                    op.rs1.op.rs1,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                )
            )


class StoreFloatWordWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FSwOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
        ):
            rewriter.replace_matched_op(
                riscv.FSwOp(
                    op.rs1.op.rs1,
                    op.rs2,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    comment=op.comment,
                )
            )


class LoadDoubleWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FLdOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and isinstance(op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.FloatRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.FLdOp(
                    op.rs1.op.rs1,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                )
            )


class StoreDoubleWithKnownOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FSdOp, rewriter: PatternRewriter) -> None:
        if (
            isinstance(op.rs1, OpResult)
            and isinstance(op.rs1.op, riscv.AddiOp)
            and isinstance(op.rs1.op.immediate, IntegerAttr)
        ):
            rewriter.replace_matched_op(
                riscv.FSdOp(
                    op.rs1.op.rs1,
                    op.rs2,
                    op.rs1.op.immediate.value.data + op.immediate.value.data,
                    comment=op.comment,
                )
            )


class AdditionOfSameVariablesToMultiplyByTwo(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AddOp, rewriter: PatternRewriter) -> None:
        if op.rs1 == op.rs2:
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                [
                    li_op := riscv.LiOp(2),
                    riscv.MulOp(
                        op.rs1,
                        li_op,
                        rd=rd,
                        comment=op.comment,
                    ),
                ]
            )


class BitwiseAndByZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AndOp, rewriter: PatternRewriter):
        """
        rewrite pattern to optimize bitwise and by 0
        x & 0 = 0
        """

        # check if the first operand is 0
        if (
            isinstance(op.rs1.owner, riscv.LiOp)
            and isinstance(op.rs1.owner.immediate, IntegerAttr)
            and op.rs1.owner.immediate.value.data == 0
        ):
            # if the first operand is 0, set the destination to 0
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs1, rd=rd))

        # check if the second operand is 0
        if (
            isinstance(op.rs2.owner, riscv.LiOp)
            and isinstance(op.rs2.owner.immediate, IntegerAttr)
            and op.rs2.owner.immediate.value.data == 0
        ):
            # if the second operand is 0, set the destination to 0
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs2, rd=rd))


class ScfgwOpUsingImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_snitch.ScfgwOp, rewriter: PatternRewriter
    ) -> None:
        if (
            isinstance(op.rs2, OpResult)
            and isinstance(op.rs2.op, riscv.LiOp)
            and isinstance(op.rs2.op.immediate, IntegerAttr)
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv_snitch.ScfgwiOp(
                    op.rs1,
                    op.rs2.op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                ),
            )
