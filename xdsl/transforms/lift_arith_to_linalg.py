from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseIntOrFPElementsAttr,
    ModuleOp,
    TensorType,
)
from xdsl.ir import Attribute, BlockArgument, OpResult, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def get_generic_fma(
    mul_op1: SSAValue, mul_op2: SSAValue, add_op: SSAValue, out: SSAValue
) -> linalg.Generic:
    inputs = (mul_op1, mul_op2, add_op)
    outputs = (out,)

    arg_types = linalg.NamedOpBase.body_arg_types((*inputs, *outputs))

    @Builder.implicit_region(arg_types)
    def body(args: tuple[BlockArgument, ...]) -> None:
        m = arith.Mulf(args[0], args[1])
        a = arith.Addf(m, args[2])
        linalg.YieldOp(a)

    return linalg.Generic(
        inputs,
        outputs,
        body,
        4 * [AffineMapAttr(AffineMap.from_callable(lambda i,: (i,)))],
        [linalg.IteratorTypeAttr.parallel()],
        [out.type],
    )


@dataclass
class LiftFMAPass(RewritePattern):
    fma_require_scalar: bool
    fma_require_erasable_mul: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, mul: arith.Mulf, rewriter: PatternRewriter, /):
        if self.fma_require_erasable_mul and len(mul.result.uses) != 1:
            return

        for add in [
            use.operation
            for use in mul.result.uses
            if isinstance(use.operation, arith.Addf)
        ]:
            add_operand = add.lhs if mul.result == add.rhs else add.rhs

            if (
                self.fma_require_scalar
                and not self.is_scalar_constant(mul.lhs)
                and not self.is_scalar_constant(mul.rhs)
            ):
                return

            fma = get_generic_fma(mul.lhs, mul.rhs, add_operand, mul.lhs)

            rewriter.replace_op(add, fma)
            if len(mul.result.uses) == 0:
                rewriter.erase_matched_op()

    @staticmethod
    def is_scalar_constant(op: SSAValue) -> bool:
        return (
            isinstance(op, OpResult)
            and isinstance(op.op, arith.Constant)
            and (
                not isinstance(v := op.op.value, DenseIntOrFPElementsAttr)
                or v.data.data.count(v.data.data[0]) == len(v.data.data)
            )
        )


class LiftAddfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.AddOp(op.operands, [op.lhs], [op.result.type])
            )


class LiftSubfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.SubOp(op.operands, [op.lhs], [op.result.type])
            )


class LiftMulfPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter, /):
        if isa(op.result.type, TensorType[Attribute]):
            rewriter.replace_matched_op(
                linalg.MulOp(op.operands, [op.lhs], [op.result.type])
            )


@dataclass(frozen=True)
class LiftArithToLinalg(ModulePass):
    """
    Pass that lifts arith ops to linalg in order to make use of destination-passing style and bufferization.
    """

    name = "lift-arith-to-linalg"

    generate_fma: bool = True
    """Set to generate fused multiply-add as `linalg.generic`"""

    fma_require_scalar: bool = False
    """Set to require one of the mul factors to be a scalar constant"""

    fma_require_erasable_mul: bool = False
    """Set to only fuse ops if the multiply has no other use and can be erased"""

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        fma_pass = (
            [LiftFMAPass(self.fma_require_scalar, self.fma_require_erasable_mul)]
            if self.generate_fma
            else []
        )

        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    *fma_pass,
                    LiftAddfPass(),
                    LiftSubfPass(),
                    LiftMulfPass(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
