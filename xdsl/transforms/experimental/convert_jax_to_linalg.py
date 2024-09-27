from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import TensorType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.linalg import FillOp
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
        donated_inputs = {}
        for inp, attr in zip(op.regions[0].block._args, op.arg_attrs):
            if (
                type(inp.type) is not TensorType
                or "tf.aliasing_output" not in attr.data
            ):
                continue
            donated_inputs[inp.name_hint] = inp

        for child_op in op.regions[0].ops:
            if type(child_op) is FillOp:
                value_mapper = {}
                for output in child_op.outputs:
                    for arg_name, arg in list(donated_inputs.items()):
                        if arg.type.is_same_type_with(output.type):
                            value_mapper[output] = arg
                            del donated_inputs[arg_name]
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
        MLIROptPass(arguments=["--linalg-fuse-elementwise-ops"]).apply(ctx, op)
