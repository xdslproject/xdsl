from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.func import FuncOp
from xdsl.ir import BlockArgument, SSAValue
from xdsl.irdl import VarOperand
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.mlir_opt import MLIROptPass


@dataclass
class SubstituteDonatedTensors(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        if op.arg_attrs is None:
            return

        donated_inputs: list[BlockArgument] = []
        for inp, attr in zip(op.args, op.arg_attrs):
            if type(inp.type) is TensorType and "tf.aliasing_output" in attr.data:
                donated_inputs.append(inp)

        for child_op in op.body.ops:
            if (
                hasattr(child_op, "outputs")
                and type(getattr(child_op, "outputs")) is VarOperand
            ):
                value_mapper: dict[SSAValue, SSAValue] = {}
                for output in getattr(child_op, "outputs"):
                    for i, arg in enumerate(donated_inputs):
                        if type(getattr(output, "type")) is TensorType and getattr(
                            arg, "type"
                        ).is_same_type_with(output.type):
                            value_mapper[output] = donated_inputs.pop(i)
                            break
                new_op = child_op.clone(value_mapper)
                rewriter.replace_op(child_op, [new_op])


@dataclass(frozen=True)
class ConvertJaxToLinalgPass(ModulePass):
    name = "convert-jax-to-linalg"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([SubstituteDonatedTensors()]),
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=True,
        )
        the_one_pass.rewrite_module(op)
        MLIROptPass(arguments=("--linalg-fuse-elementwise-ops",)).apply(ctx, op)
