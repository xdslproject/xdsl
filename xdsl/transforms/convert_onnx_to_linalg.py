from dataclasses import dataclass

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, linalg, ml_program, onnx, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    FloatAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    f64,
)
from xdsl.ir import Block, MLContext, Operation, Region
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable


def get_root_op(op: Operation | None) -> Operation | None:
    """
    Recursively finds and returns the root operation associated with the given operation.
    """
    return op if op is None or op.parent_op() is None else get_root_op(op.parent_op())


@dataclass
class AddOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, add: onnx.Add, rewriter: PatternRewriter, /):
        lhs_type = add.lhs.type
        rhs_type = add.rhs.type
        if isinstance(lhs_type, TensorType) and isinstance(rhs_type, TensorType):
            lhs_shape = lhs_type.get_shape()
            rhs_shape = rhs_type.get_shape()

            if 1 in lhs_shape or 1 in rhs_shape:
                raise NotImplementedError()

        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), add.res.type),
                linalg.AddOp((add.lhs, add.rhs), (empty.tensor,), res=(add.res.type,)),
            )
        )


@dataclass
class ReluOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, relu: onnx.Relu, rewriter: PatternRewriter, /):
        body = Region(Block(arg_types=(f64, f64)))
        affine_map = AffineMapAttr(AffineMap.from_callable(lambda d0, d1: (d0, d1)))
        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), relu.res.type),
                zero := arith.Constant(FloatAttr(0, f64)),
                linalg.Generic(
                    (relu.operand,),
                    (empty.tensor,),
                    body,
                    (
                        affine_map,
                        affine_map,
                    ),
                    (
                        linalg.IteratorTypeAttr.parallel(),
                        linalg.IteratorTypeAttr.parallel(),
                    ),
                    (relu.res.type,),
                ),
            )
        )
        with ImplicitBuilder(body) as (a, _):
            max_op = arith.Maximumf(a, zero.result)
            linalg.YieldOp(max_op.result)


@dataclass
class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, constant: onnx.Constant, rewriter: PatternRewriter, /):
        attr_value = list(constant.attributes.values())[1]
        global_op = ml_program.Global(
            StringAttr("global_constant"),
            constant.output.type,
            None,
            attr_value,
            StringAttr("private"),
        )
        root_op = get_root_op(constant)
        if root_op is not None and root_op.has_trait(SymbolTable):
            SymbolTable.insert_or_update(root_op, global_op)
        rewriter.replace_matched_op(
            (
                ml_program.GlobalLoadConstant(
                    SymbolRefAttr(global_op.sym_name),
                    global_op.type,
                ),
            )
        )


@dataclass
class ConvertOnnxToLinalgPass(ModulePass):
    name = "convert-onnx-to-linalg"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    ReluOpLowering(),
                    ConstantOpLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
