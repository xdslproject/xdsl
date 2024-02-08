from typing import cast

from xdsl.dialects import llvm, riscv, riscv_snitch
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import OpResult, SSAValue
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
        if (rs1 := get_constant_value(op.rs1)) is not None and (
            rs2 := get_constant_value(op.rs2)
        ) is not None:
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LiOp(
                    rs1.value.data * rs2.value.data,
                    rd=rd,
                )
            )


class MultiplyImmediateZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.MulOp, rewriter: PatternRewriter) -> None:
        if (rs1 := get_constant_value(op.rs1)) is not None and rs1.value.data == 0:
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs1, rd=rd))
        elif (rs2 := get_constant_value(op.rs2)) is not None and rs2.value.data == 0:
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs2, rd=rd))


class AddImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AddOp, rewriter: PatternRewriter) -> None:
        lhs: int | None = None
        rhs: int | None = None
        if (rs1 := get_constant_value(op.rs1)) is not None:
            lhs = rs1.value.data

        if (rs2 := get_constant_value(op.rs2)) is not None:
            rhs = rs2.value.data

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
        if (rs1 := get_constant_value(op.rs1)) is not None and isinstance(
            op.immediate, IntegerAttr
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LiOp(
                    rs1.value.data + op.immediate.value.data,
                    rd=rd,
                    comment=op.comment,
                )
            )


class SubImmediates(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.SubOp, rewriter: PatternRewriter) -> None:
        lhs: int | None = None
        rhs: int | None = None
        if (rs1 := get_constant_value(op.rs1)) is not None:
            lhs = rs1.value.data

        if (rs2 := get_constant_value(op.rs2)) is not None:
            rhs = rs2.value.data

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
            and isinstance(op.rs1.op.immediate, IntegerAttr)
            and op.rs2 == op.rs1.op.rs1
        ):
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv.LiOp(op.rs1.op.immediate.value.data, rd=rd)
            )


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


def _has_contract_flag(op: riscv.RdRsRsFloatOperationWithFastMath) -> bool:
    return (
        op.fastmath is not None
        and llvm.FastMathFlag.ALLOW_CONTRACT in op.fastmath.flags
    )


class FuseMultiplyAddD(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.FAddDOp, rewriter: PatternRewriter) -> None:
        """
        Converts `c = a * b` and `d = e + c` to `d = e + a * b`.
        `c` should not be used anywhere else and both operations must have the
        `contract` fastmath flag set.
        """

        if not _has_contract_flag(op):
            return

        addend = mul = None
        if (
            isinstance(mul := op.rs2.owner, riscv.FMulDOp)
            and _has_contract_flag(mul)
            and len(mul.rd.uses) == 1
        ):
            addend = op.rs1
        elif (
            isinstance(mul := op.rs1.owner, riscv.FMulDOp)
            and _has_contract_flag(mul)
            and len(mul.rd.uses) == 1
        ):
            addend = op.rs2
        else:
            return

        rd = cast(riscv.FloatRegisterType, op.rd.type)
        rewriter.replace_matched_op(
            riscv.FMAddDOp(
                mul.rs1,
                mul.rs2,
                addend,
                rd=rd,
                comment=op.comment,
            )
        )


class BitwiseAndByZero(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.AndOp, rewriter: PatternRewriter):
        """
        rewrite pattern to optimize bitwise and by 0
        x & 0 = 0
        """

        # check if the first operand is 0
        if (rs1 := get_constant_value(op.rs1)) is not None and rs1.value.data == 0:
            # if the first operand is 0, set the destination to 0
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs1, rd=rd))

        # check if the second operand is 0
        if (rs2 := get_constant_value(op.rs2)) is not None and rs2.value.data == 0:
            # if the second operand is 0, set the destination to 0
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(riscv.MVOp(op.rs2, rd=rd))


class ScfgwOpUsingImmediate(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: riscv_snitch.ScfgwOp, rewriter: PatternRewriter
    ) -> None:
        if (rs2 := get_constant_value(op.rs2)) is not None:
            rd = cast(riscv.IntRegisterType, op.rd.type)
            rewriter.replace_matched_op(
                riscv_snitch.ScfgwiOp(
                    op.rs1,
                    rs2.value.data,
                    rd=rd,
                    comment=op.comment,
                ),
            )


class LoadImmediate0(RewritePattern):
    """
    The canonical form of an operation that stores a 0 to register RD is li RD, ZERO.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv.LiOp, rewriter: PatternRewriter) -> None:
        if not (isinstance(op.immediate, IntegerAttr) and op.immediate.value.data == 0):
            return

        rd = cast(riscv.IntRegisterType, op.rd.type)
        if rd == riscv.Registers.ZERO:
            rewriter.replace_matched_op(riscv.GetRegisterOp(riscv.Registers.ZERO))
        else:
            rewriter.replace_matched_op(
                (
                    zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                    riscv.MVOp(zero.res, rd=rd, comment=op.comment),
                )
            )


def get_constant_value(value: SSAValue) -> riscv.Imm32Attr | None:
    if value.type == riscv.Registers.ZERO:
        return IntegerAttr.from_int_and_width(0, 32)

    if isinstance(value.owner, riscv.MVOp):
        return get_constant_value(value.owner.rs)

    if isinstance(value.owner, riscv.LiOp) and isinstance(
        value.owner.immediate, IntegerAttr
    ):
        return value.owner.immediate
