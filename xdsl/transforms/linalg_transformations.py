from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import AffineMapAttr, DenseIntOrFPElementsAttr, ModuleOp
from xdsl.ir import BlockArgument, OpResult, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def build_generic_fma(
    mul_op1: SSAValue, mul_op2: SSAValue, add_op: SSAValue, out: SSAValue
) -> linalg.GenericOp:
    inputs = (mul_op1, mul_op2, add_op)
    outputs = (out,)

    arg_types = linalg.NamedOperation.body_arg_types((*inputs, *outputs))

    @Builder.implicit_region(arg_types)
    def body(args: tuple[BlockArgument, ...]) -> None:
        m = arith.MulfOp(args[0], args[1])
        a = arith.AddfOp(m, args[2])
        linalg.YieldOp(a)

    return linalg.GenericOp(
        inputs,
        outputs,
        body,
        4 * [AffineMapAttr(AffineMap.from_callable(lambda i,: (i,)))],
        [linalg.IteratorTypeAttr.parallel()],
        [out.type],
    )


@dataclass(frozen=True)
class FuseMultiplyAddPass(RewritePattern):
    require_scalar_factor: bool
    require_erasable_mul: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, mul: linalg.MulOp, rewriter: PatternRewriter, /):
        if (
            len(mul.res) != 1
            or self.require_erasable_mul
            and len(set(use.operation for use in mul.res[0].uses)) != 1
        ):
            return

        for add in set(
            use.operation
            for use in mul.res[0].uses
            if isinstance(use.operation, linalg.AddOp)
            and mul.res[0] in use.operation.inputs
        ):
            # if the `require_scalar_factor` flag is set, check if either operand of `mul` is a scalar
            if (
                self.require_scalar_factor
                and not self.is_scalar_constant(mul.inputs[0])
                and not self.is_scalar_constant(mul.inputs[1])
            ):
                return

            # the operand of `add` that is not the `mul` result
            add_operand = (
                add.inputs[0] if mul.res[0] == add.inputs[1] else add.inputs[1]
            )

            # build fma op
            fma = build_generic_fma(
                mul.inputs[0], mul.inputs[1], add_operand, mul.outputs[0]
            )

            # replace in position of the add op
            rewriter.replace_op(add, fma)
            if not mul.res[0].uses:
                rewriter.erase_op(mul)

    @staticmethod
    def is_scalar_constant(op: SSAValue) -> bool:
        """
        Returns if the value is a scalar. This currently checks for scalar constants, and could
        in the future be extended to check for dynamically provided scalar values expanded via linalg.fill
        """
        return (
            isinstance(op, OpResult)
            and isinstance(op.op, arith.ConstantOp)
            and (
                not isinstance(v := op.op.value, DenseIntOrFPElementsAttr)
                or v.is_splat()
            )
        )


@dataclass(frozen=True)
class LinalgFuseMultiplyAddPass(ModulePass):
    """
    Pass that fuses linalg multiply and add ops into a `generic` fma.
    """

    name = "linalg-fuse-multiply-add"

    require_scalar_factor: bool = False
    """Set to require one of the mul factors to be a scalar constant"""

    require_erasable_mul: bool = False
    """Set to only fuse ops if the multiply has no other use and can be erased"""

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            FuseMultiplyAddPass(self.require_scalar_factor, self.require_erasable_mul),
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
