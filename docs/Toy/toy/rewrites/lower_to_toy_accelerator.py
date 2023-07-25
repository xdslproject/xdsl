from xdsl.dialects import affine, arith, memref, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import MLContext, SSAValue
from xdsl.ir.core import OpResult
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..dialects import toy_accelerator


def cast_value_to_register(operand: SSAValue, rewriter: PatternRewriter) -> OpResult:
    types = (riscv.IntRegisterType.unallocated(),)
    cast = UnrealizedConversionCastOp.get((operand,), types)
    rewriter.insert_op_before_matched_op(cast)
    return cast.results[0]


class LowerAffineForOp(RewritePattern):
    """
    Matches on nested loops that implement elementwise addition or transpose, and rewrites
    them as custom toy accelerator instructions.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.For, rewriter: PatternRewriter):
        # Check this is a doubly nested affine for loop, where both loops don't yield
        # values.
        if (
            len(op.body.blocks) != 1
            or len((outer_block := op.body.blocks[0]).ops) != 2
            or not isinstance(inner_for := outer_block.first_op, affine.For)
            or not isinstance(outer_block.last_op, affine.Yield)
            or outer_block.last_op.operands
            or len(inner_for.body.blocks) != 1
            or not (inner_block := inner_for.body.blocks[0])
            or not isinstance(inner_block.last_op, affine.Yield)
            or inner_block.last_op.operands
        ):
            return

        # Check that the indices step from 0 to upper bound by 1

        all_bounds = (
            op.lower_bound.data,
            op.upper_bound.data,
            inner_for.lower_bound.data,
            inner_for.upper_bound.data,
        )
        if any(bound.num_dims or bound.num_symbols for bound in all_bounds):
            # Loop is not made of constant values
            return

        ostep = op.step.value.data
        istep = inner_for.step.value.data

        if ostep != 1 or istep != 1:
            # Loop does not step by 1
            return

        olb, oub, ilb, iub = tuple(bound.eval([], [])[0] for bound in all_bounds)

        if olb != 0 or ilb != 0:
            # Loop doesn't start from 0
            return

        # We now need to check whether it's a transpose or an elementwise
        # operation.

        if len(inner_block.ops) == 3:
            (load, store, _) = inner_block.ops
            if not (isinstance(load, affine.Load) and isinstance(store, affine.Store)):
                return

            if load.indices != store.indices[::-1]:
                return

            # This is a transpose op

            memref_typ = load.memref.type
            assert isinstance(memref_typ, memref.MemRefType)
            rows, cols = memref_typ.shape

            if rows.value.data != iub or cols.value.data != oub:
                # The indices aren't fully enumerated
                return

            rewriter.replace_matched_op(
                toy_accelerator.Transpose(store.memref, load.memref, rows, cols)
            )
        elif len(inner_block.ops) == 5:
            lhs_load, rhs_load, binop, store, _ = inner_block.ops
            if not (
                isinstance(lhs_load, affine.Load)
                and isinstance(rhs_load, affine.Load)
                and isinstance(store, affine.Store)
                and isinstance(binop, (arith.Addf, arith.Mulf))
            ):
                return

            # size = oub * iub

            instr_cls = (
                toy_accelerator.Add
                if isinstance(binop, arith.Addf)
                else toy_accelerator.Mul
            )

            rewriter.replace_matched_op(
                [instr_cls(store.memref, lhs_load.memref, rhs_load.memref)]
            )


class LowerToToyAccelerator(ModulePass):
    name = "lower-to-toy-accelerator"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        ctx.register_dialect(toy_accelerator.ToyAccelerator)
        PatternRewriteWalker(LowerAffineForOp()).rewrite_module(op)


class LowerTransposeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: toy_accelerator.Transpose, rewriter: PatternRewriter
    ):
        destination = cast_value_to_register(op.destination, rewriter)
        source = cast_value_to_register(op.source, rewriter)

        rewriter.replace_matched_op(
            [
                rows_op := riscv.LiOp(op.source_rows.value.data, comment="source rows"),
                cols_op := riscv.LiOp(op.source_cols.value.data, comment="source cols"),
                riscv.CustomAssemblyInstructionOp(
                    "tensor.transpose2d",
                    (destination, source, rows_op.rd, cols_op.rd),
                    (),
                ),
            ]
        )


class LowerBinOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy_accelerator.BinOp, rewriter: PatternRewriter):
        typ = op.dest.type
        assert isinstance(typ, memref.MemRefType)
        size = typ.element_count()

        instruction_name = (
            "buffer.add" if isinstance(op, toy_accelerator.Add) else "buffer.mul"
        )
        dest = cast_value_to_register(op.dest, rewriter)
        lhs = cast_value_to_register(op.lhs, rewriter)
        rhs = cast_value_to_register(op.rhs, rewriter)
        rewriter.replace_matched_op(
            [
                size_op := riscv.LiOp(size, comment="size"),
                riscv.CustomAssemblyInstructionOp(
                    "buffer.add",
                    (size_op.rd, dest, lhs),
                    (),
                ),
                riscv.CustomAssemblyInstructionOp(
                    instruction_name,
                    (size_op.rd, dest, rhs),
                    (),
                ),
            ]
        )


class LowerToyAccelerator(ModulePass):
    name = "lower-toy-accelerator"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTransposeOp(),
                    LowerBinOp(),
                ]
            )
        ).rewrite_module(op)
