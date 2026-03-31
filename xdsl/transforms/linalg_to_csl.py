from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.dialects.csl import csl
from xdsl.ir import Attribute, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


def match_op_for_precision(
    prec: Attribute, f16: type[csl.BuiltinDsdOp], f32: type[csl.BuiltinDsdOp]
) -> type[csl.BuiltinDsdOp]:
    """Returns the op type matching a given precision."""
    # todo support mixed-precision
    match prec:
        case builtin.f16:
            return f16
        case builtin.f32:
            return f32
        case _:
            raise ValueError(f"Unsupported element type {prec}")


def get_scalar_const(op: SSAValue) -> FloatAttr | IntegerAttr | None:
    """Returns the value of a scalar arith.constant, or None if not a constant or not scalar)."""
    if (
        isinstance(op, OpResult)
        and isinstance(op.op, arith.ConstantOp)
        and isa(val := op.op.value, DenseIntOrFPElementsAttr)
        and val.is_splat()
    ):
        return val.get_attrs()[0]


def transform_op(
    op: linalg.NamedOperation,
    rewriter: PatternRewriter,
    f16: type[csl.BuiltinDsdOp],
    f32: type[csl.BuiltinDsdOp],
):
    if not isa(target_t := op.outputs.types[0], MemRefType):
        return

    builtin = match_op_for_precision(target_t.get_element_type(), f16, f32)

    lhs = op.inputs[0]
    rhs = op.inputs[1]

    # binary functions translated here support mixing scalar and collection operands
    # may need revisiting if more functions are translated
    if scalar_const := get_scalar_const(lhs):
        rewriter.insert_op(
            const_op := arith.ConstantOp(scalar_const), InsertPoint.before(op)
        )
        lhs = const_op.result
    elif scalar_const := get_scalar_const(rhs):
        rewriter.insert_op(
            const_op := arith.ConstantOp(scalar_const), InsertPoint.before(op)
        )
        rhs = const_op.result

    rewriter.replace_op(op, builtin(operands=[[op.outputs[0], lhs, rhs]]))


class ConvertLinalgGenericFMAPass(RewritePattern):
    """Lowers `linalg.generic` fused multiply-adds to csl builtin ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter, /):
        if not self.is_fma(op) or not isa(op.outputs.types[0], MemRefType):
            return

        # one of the factors must be a scalar const, which the csl function signatures require
        if scalar_const := get_scalar_const(op.inputs[0]):
            rewriter.insert_op(
                a := arith.ConstantOp(scalar_const), InsertPoint.before(op)
            )
            x = op.inputs[1]
        elif scalar_const := get_scalar_const(op.inputs[1]):
            rewriter.insert_op(
                a := arith.ConstantOp(scalar_const), InsertPoint.before(op)
            )
            x = op.inputs[0]
        else:
            # if neither factor is a scalar, return
            return

        # fetch the csl op to build depending on the precision
        csl_op = match_op_for_precision(
            op.outputs.types[0].get_element_type(), f16=csl.FmachOp, f32=csl.FmacsOp
        )

        r = op.outputs[0]
        y = op.inputs[2]

        # builds `r = a * x + y`
        rewriter.replace_op(op, csl_op(operands=[[r, y, x, a]]))

    @staticmethod
    def is_fma(op: linalg.GenericOp) -> bool:
        """Returns if a given `generic` op is a fused multiply-add"""
        return (
            len(op.inputs) == 3
            and len(op.outputs) == 1
            and len((block := op.body.block).args) == 4
            and len(block.ops) == 3
            and isinstance(mul := block.first_op, arith.MulfOp)
            and mul.lhs == block.args[0]
            and mul.rhs == block.args[1]
            and isinstance(add := mul.next_op, arith.AddfOp)
            and add.lhs == mul.result
            and add.rhs == block.args[2]
            and isinstance(yld := add.next_op, linalg.YieldOp)
            and yld.operands[0] == add.result
        )


class ConvertLinalgMinPass(RewritePattern):
    """
    Lowers the `linalg.min` op to csl by negating the operands, performing max, and
    again negating the operands as well as the result.

    todo: scalar operands are currently not supported
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MinOp, rewriter: PatternRewriter, /):
        if not isa(op.outputs.types[0], MemRefType):
            return

        # sets of operands to be negated before and after
        negate_before = set(op.inputs)
        negate_after = negate_before | set(op.outputs)

        # builtin op for negating
        neg_op = match_op_for_precision(
            op.outputs.types[0].get_element_type(), f16=csl.FneghOp, f32=csl.FnegsOp
        )

        # constructing in-place negate ops before and after
        before_ops = [neg_op(operands=[(o, o)]) for o in negate_before]
        after_ops = [neg_op(operands=[(o, o)]) for o in negate_after]

        rewriter.insert_op(before_ops, InsertPoint.before(op))
        rewriter.insert_op(after_ops, InsertPoint.after(op))
        transform_op(op, rewriter, f16=csl.FmaxhOp, f32=csl.FmaxsOp)


class ConvertLinalgAddPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.AddOp, rewriter: PatternRewriter, /):
        transform_op(op, rewriter, f16=csl.FaddhOp, f32=csl.FaddsOp)


class ConvertLinalgSubPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.SubOp, rewriter: PatternRewriter, /):
        transform_op(op, rewriter, f16=csl.FsubhOp, f32=csl.FsubsOp)


class ConvertLinalgMulPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MulOp, rewriter: PatternRewriter, /):
        transform_op(op, rewriter, f16=csl.FmulhOp, f32=csl.FmulsOp)


class ConvertLinalgMaxPass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MaxOp, rewriter: PatternRewriter, /):
        transform_op(op, rewriter, f16=csl.FmaxhOp, f32=csl.FmaxsOp)


@dataclass(frozen=True)
class LinalgToCsl(ModulePass):
    """
    Convert linalg ops to csl ops.

    The linalg ops are required to be in 'memref mode', i.e., after bufferization has been applied.
    """

    name = "linalg-to-csl"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertLinalgGenericFMAPass(),
                    ConvertLinalgAddPass(),
                    ConvertLinalgSubPass(),
                    ConvertLinalgMulPass(),
                    ConvertLinalgMaxPass(),
                    ConvertLinalgMinPass(),
                ]
            ),
        )
        module_pass.rewrite_module(op)
