from xdsl.ir import MLContext
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    ModuleOp,
    StringAttr,
    i32,
    i64,
)
from xdsl.dialects import llvm
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    op_type_rewrite_pattern,
    RewritePattern,
    PatternRewriter,
)

from xdsl.transforms.dead_code_elimination import dce

from ..dialects import toy, vector


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        # TODO: add support for user defined functions
        assert name == "main", "Only support lowering main function for now"

        region = op.regions[0]

        ftype = llvm.LLVMFunctionType((), llvm.VoidType())

        # create riscv func op with same ops
        llvm_func = llvm.FuncOp(
            name, ftype, rewriter.move_region_contents_to_new_regions(region)
        )

        rewriter.replace_matched_op(llvm_func)


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.ReturnOp, rewriter: PatternRewriter):
        # TODO: add support for optional argument
        assert op.input is None, "Only support return with no arguments for now"

        rewriter.replace_matched_op(llvm.ReturnOp())


class LowerVectorConstantOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.VectorConstantOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.ConstantOp(op.data, op.res.typ))


class LowerVectorAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.VectorAddOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.FAddOp(op.lhs, op.rhs))


class LowerVectorMulOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.VectorMulOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(llvm.FMulOp(op.lhs, op.rhs))


# TODO: add toy_vector, move to pointers/builtin vector for contents
shape_type = i32
data_type = i32
tensor_type = llvm.LLVMStructType(
    [StringAttr("toy_tensor"), ArrayAttr([shape_type, data_type])]
)


class LowerTensorMakeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.TensorMakeOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            [
                t0 := llvm.UndefOp(tensor_type),
                t1 := llvm.InsertValueOp(
                    DenseArrayBase.create_dense_int_or_index(i64, [0]),
                    t0.res,
                    op.shape,
                ),
                llvm.InsertValueOp(
                    DenseArrayBase.create_dense_int_or_index(i64, [1]),
                    t1.res,
                    op.data,
                ),
            ],
        )


class LowerTensorShapeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.TensorShapeOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            llvm.ExtractValueOp(
                DenseArrayBase.create_dense_int_or_index(i64, [0]),
                op.tensor,
                op.data.typ,
            )
        )


class LowerTensorDataOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.TensorDataOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            llvm.ExtractValueOp(
                DenseArrayBase.create_dense_int_or_index(i64, [1]),
                op.tensor,
                op.data.typ,
            )
        )


class LowerVector(ModulePass):
    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerFuncOp()).rewrite_module(op)
        PatternRewriteWalker(LowerReturnOp()).rewrite_module(op)
        PatternRewriteWalker(LowerVectorConstantOp()).rewrite_module(op)
        PatternRewriteWalker(LowerVectorAddOp()).rewrite_module(op)
        PatternRewriteWalker(LowerVectorMulOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorMakeOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorShapeOp()).rewrite_module(op)
        PatternRewriteWalker(LowerTensorDataOp()).rewrite_module(op)
        dce(op)
