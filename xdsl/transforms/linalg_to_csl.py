from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import (
    AnyFloatAttr,
    AnyIntegerAttr,
    AnyMemRefType,
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    ModuleOp,
)
from xdsl.dialects.csl import csl
from xdsl.ir import OpResult, SSAValue
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


class ConvertBinaryLinalgOp(RewritePattern):
    """
    Base class for converting binary linalg operations.
    """

    def transform_op(
        self,
        op: linalg.NamedOpBase,
        rewriter: PatternRewriter,
        f16: type[csl.BuiltinDsdOp],
        f32: type[csl.BuiltinDsdOp],
    ):
        if not isa(op.outputs.types[0], AnyMemRefType):
            return

        match op.outputs.types[0].get_element_type():
            case Float16Type():
                builtin = f16
            case Float32Type():
                builtin = f32
            case _:
                raise ValueError(
                    f"Unsupported element type {op.outputs.types[0].get_element_type()}"
                )

        lhs = op.inputs[0]
        rhs = op.inputs[1]

        # binary functions translated here support mixing scalar and collection operands
        # may need revisiting if more functions are translated
        if scalar_const := self._get_scalar_const(lhs):
            rewriter.insert_op(
                const_op := arith.Constant(scalar_const), InsertPoint.before(op)
            )
            lhs = const_op.result
        elif scalar_const := self._get_scalar_const(rhs):
            rewriter.insert_op(
                const_op := arith.Constant(scalar_const), InsertPoint.before(op)
            )
            rhs = const_op.result

        rewriter.replace_matched_op(builtin(operands=[[op.outputs[0], lhs, rhs]]))

    @staticmethod
    def _get_scalar_const(op: SSAValue) -> AnyFloatAttr | AnyIntegerAttr | None:
        """Returns the value of a scalar arith.constant, or None if not a constant or not scalar)."""
        if (
            isinstance(op, OpResult)
            and isinstance(op.op, arith.Constant)
            and isa(val := op.op.value, DenseIntOrFPElementsAttr)
            and val.data.data.count(val.data.data[0]) == len(val.data.data)
        ):
            return val.data.data[0]


class ConvertLinalgAddPass(ConvertBinaryLinalgOp):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.AddOp, rewriter: PatternRewriter, /):
        self.transform_op(op, rewriter, f16=csl.FaddhOp, f32=csl.FaddsOp)


class ConvertLinalgSubPass(ConvertBinaryLinalgOp):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.SubOp, rewriter: PatternRewriter, /):
        self.transform_op(op, rewriter, f16=csl.FsubhOp, f32=csl.FsubsOp)


class ConvertLinalgMulPass(ConvertBinaryLinalgOp):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.MulOp, rewriter: PatternRewriter, /):
        self.transform_op(op, rewriter, f16=csl.FmulhOp, f32=csl.FmulsOp)


@dataclass(frozen=True)
class LinalgToCsl(ModulePass):
    """
    Convert linalg ops to csl ops.

    The linalg ops are required to be in 'memref mode', i.e., after bufferization has been applied.
    """

    name = "linalg-to-csl"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertLinalgAddPass(),
                    ConvertLinalgSubPass(),
                    ConvertLinalgMulPass(),
                ]
            ),
        )
        module_pass.rewrite_module(op)
