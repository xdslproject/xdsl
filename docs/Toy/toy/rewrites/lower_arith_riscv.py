from xdsl.dialects import arith, riscv
from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir.core import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from .lower_riscv_cf import cast_values_to_registers
from .setup_riscv_pass import DataDirectiveRewritePattern


class LowerConstantOp(DataDirectiveRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        assert isinstance(op.value, (IntegerAttr, FloatAttr))

        value = op.value.value.data
        assert value == float(
            int(value)
        ), f"Only support integer values in arith.Constant, got {value}"
        value = int(value)

        # TODO: get a better label
        label = self.label("constant")

        self.add_data(op, label, [value])

        rewriter.replace_matched_op(
            [
                constant_ptr := riscv.LiOp(label),
                constant := riscv.LwOp(constant_ptr.rd, 0),
                UnrealizedConversionCastOp.get(
                    constant.results, tuple(r.type for r in op.results)
                ),
            ]
        )


class LowerCmpiOp(DataDirectiveRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Cmpi, rewriter: PatternRewriter):
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)

        new_ops: list[Operation]

        match op.predicate.value.data:
            case 0:  # eq
                new_ops = [
                    copy := riscv.MVOp(lhs),
                    xor := riscv.XorOp(copy.rd, rhs),
                    zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                    eq := riscv.SltiuOp(zero.res, 1, rd_operand=xor.rd),
                ]
                res = eq.rd

            # case 1:  # ne

            #     rewriter.replace_matched_op(
            #         riscv_cf.BneOp(
            #             lhs,
            #             rhs,
            #             new_then_arguments,
            #             new_else_arguments,
            #             op.then_block,
            #             op.else_block,
            #         )
            #     )
            case 2:  # slt
                new_ops = [
                    copy := riscv.MVOp(lhs),
                    eq := riscv.SltiuOp(rhs, 1, rd_operand=copy.rd),
                ]
                res = eq.rd
            # case 3:  # sle
            #     # No sle, flip arguments and use ge instead
            #     rewriter.replace_matched_op(
            #         riscv_cf.BgeOp(
            #             rhs,
            #             lhs,
            #             new_else_arguments,
            #             new_then_arguments,
            #             op.else_block,
            #             op.then_block,
            #         )
            #     )
            # case 4:  # sgt
            #     # No sgt, flip arguments and use lt instead
            #     rewriter.replace_matched_op(
            #         riscv_cf.BltOp(
            #             rhs,
            #             lhs,
            #             new_else_arguments,
            #             new_then_arguments,
            #             op.else_block,
            #             op.then_block,
            #         )
            #     )
            # case 5:  # sge
            #     rewriter.replace_matched_op(
            #         riscv_cf.BgeOp(
            #             lhs,
            #             rhs,
            #             new_then_arguments,
            #             new_else_arguments,
            #             op.then_block,
            #             op.else_block,
            #         )
            #     )
            # case 6:  # ult
            #     rewriter.replace_matched_op(
            #         riscv_cf.BltOp(
            #             lhs,
            #             rhs,
            #             new_then_arguments,
            #             new_else_arguments,
            #             op.then_block,
            #             op.else_block,
            #         )
            #     )
            # case 7:  # ule
            #     # No ule, flip arguments and use geu instead
            #     rewriter.replace_matched_op(
            #         riscv_cf.BgeuOp(
            #             rhs,
            #             lhs,
            #             new_else_arguments,
            #             new_then_arguments,
            #             op.else_block,
            #             op.then_block,
            #         )
            #     )
            # case 8:  # ugt
            #     # No ugt, flip arguments and use ltu instead
            #     rewriter.replace_matched_op(
            #         riscv_cf.BltuOp(
            #             rhs,
            #             lhs,
            #             new_else_arguments,
            #             new_then_arguments,
            #             op.else_block,
            #             op.then_block,
            #         )
            #     )
            # case 9:  # uge
            #     rewriter.replace_matched_op(
            #         riscv_cf.BgeuOp(
            #             lhs,
            #             rhs,
            #             new_then_arguments,
            #             new_else_arguments,
            #             op.then_block,
            #             op.else_block,
            #         )
            #     )
            case _:
                assert False, f"Unexpected comparison predicate {op.predicate}"

        cast_op = UnrealizedConversionCastOp.get((res,), (op.result.type,))

        new_ops.append(cast_op)

        rewriter.replace_matched_op(new_ops)


class LowerAddiOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter):
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)

        add = riscv.AddOp(lhs, rhs)
        cast = UnrealizedConversionCastOp.get((add.rd,), (op.result.type,))

        rewriter.replace_matched_op((add, cast))


class LowerAddfOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter):
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)

        # TODO: use floating point addition later
        add = riscv.AddOp(lhs, rhs)
        cast = UnrealizedConversionCastOp.get((add.rd,), (op.result.type,))

        rewriter.replace_matched_op((add, cast))


class LowerMulfOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter):
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)

        # TODO: use floating point multiplication later
        add = riscv.MulOp(lhs, rhs)
        cast = UnrealizedConversionCastOp.get((add.rd,), (op.result.type,))

        rewriter.replace_matched_op((add, cast))


class LowerArithRiscvPass(ModulePass):
    name = "lower-arith-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerConstantOp(),
                    LowerCmpiOp(),
                    LowerAddiOp(),
                    LowerAddfOp(),
                    LowerMulfOp(),
                ]
            )
        ).rewrite_module(op)
