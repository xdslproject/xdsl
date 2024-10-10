from dataclasses import dataclass
from typing import cast

from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import arith, linalg, ml_program, onnx, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    ModuleOp,
    NoneType,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    f32,
    f64,
    i64,
)
from xdsl.ir import Attribute, Block, Operation, Region
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

            if -1 in lhs_shape or -1 in rhs_shape:
                raise NotImplementedError()

        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), add.res.type),
                linalg.AddOp((add.lhs, add.rhs), (empty.tensor,), res=(add.res.type,)),
            )
        )


@dataclass
class SubOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, sub: onnx.Sub, rewriter: PatternRewriter, /):
        lhs_type = sub.lhs.type
        rhs_type = sub.rhs.type
        if isinstance(lhs_type, TensorType) and isinstance(rhs_type, TensorType):
            lhs_shape = lhs_type.get_shape()
            rhs_shape = rhs_type.get_shape()

            if -1 in lhs_shape or -1 in rhs_shape:
                raise NotImplementedError()

        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), sub.res.type),
                linalg.SubOp((sub.lhs, sub.rhs), (empty.tensor,), res=(sub.res.type,)),
            )
        )


@dataclass
class ReluOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, relu: onnx.Relu, rewriter: PatternRewriter, /):
        operand = relu.operand.type
        assert isinstance(operand, TensorType)
        operand = cast(TensorType[Attribute], operand)
        operand_rank = len(operand.get_shape())
        body = Region(Block(arg_types=(operand.element_type, operand.element_type)))
        affine_map = AffineMapAttr(AffineMap.identity(operand_rank))
        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), relu.res.type),
                zero := arith.Constant(
                    FloatAttr(0.0, cast(AnyFloat, operand.element_type))
                ),
                linalg.Generic(
                    (relu.operand,),
                    (empty.tensor,),
                    body,
                    (affine_map, affine_map),
                    (linalg.IteratorTypeAttr.parallel(),) * operand_rank,
                    (relu.res.type,),
                ),
            )
        )
        with ImplicitBuilder(body) as (a, _):
            max_op = arith.Maximumf(a, zero.result)
            linalg.YieldOp(max_op.result)


@dataclass
class ConstantOpLowering(RewritePattern):
    constant_count: int = 0

    def make_unique_name(self):
        self.constant_count += 1
        return f"onnx_constant_{self.constant_count}"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, constant: onnx.Constant, rewriter: PatternRewriter, /):
        attr_value = list(constant.attributes.values())[1]
        constant_name = self.make_unique_name()
        global_op = ml_program.Global(
            StringAttr(constant_name),
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
class ReshapeOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, reshape: onnx.Reshape, rewriter: PatternRewriter, /):
        # Dynamic shapes not currently supported
        source_type = reshape.data.type
        shape_type = reshape.shape.type
        if isinstance(source_type, TensorType) and isinstance(shape_type, TensorType):
            source_shape = source_type.get_shape()
            shape_shape = shape_type.get_shape()

            if -1 in source_shape or -1 in shape_shape:
                raise NotImplementedError()

        # Lowering with `allowzero = 1` attribute not supported"
        if reshape.allow_zero is not None and reshape.allow_zero.value.data == 1:
            raise NotImplementedError()

        rewriter.replace_matched_op(
            (
                tensor.ReshapeOp(
                    reshape.data,
                    reshape.shape,
                    reshape.reshaped.type,
                ),
            )
        )


@dataclass
class GemmOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, gemm: onnx.Gemm, rewriter: PatternRewriter, /):
        assert isinstance(tensor_a_type := gemm.tensor_a.type, TensorType)
        assert isinstance(tensor_b_type := gemm.tensor_b.type, TensorType)
        assert isinstance(tensor_c_type := gemm.tensor_c.type, TensorType)

        tensor_a_type = cast(TensorType[Attribute], tensor_a_type)
        tensor_b_type = cast(TensorType[Attribute], tensor_b_type)
        tensor_c_type = cast(TensorType[Attribute], tensor_c_type)

        tensor_a_shape = tensor_a_type.get_shape()
        tensor_b_shape = tensor_b_type.get_shape()
        tensor_c_shape = tensor_c_type.get_shape()

        # Dynamic shapes not currently supported
        if any(
            -1 in shape for shape in [tensor_a_shape, tensor_b_shape, tensor_c_shape]
        ):
            raise NotImplementedError()

        perm: list[int] = [1, 0]
        permutation = DenseArrayBase.create_dense_int(i64, perm)

        # if transA is set, trans_a is changed to this op
        trans_a_res = None
        if gemm.trans_a is not None and gemm.trans_a.value.data == 1:
            shape_type = tensor_a_type.element_type
            # onnx.gemm supports only 2D tensors, hence reversing is acceptable
            shape = tuple(reversed(tensor_a_shape))
            empty_shape = TensorType(shape_type, shape)
            empty = tensor.EmptyOp((), empty_shape)
            trans_a = linalg.TransposeOp(
                gemm.tensor_a, empty.tensor, permutation, empty.tensor.type
            )
            # save the result
            trans_a_res = trans_a.result[0]
            rewriter.insert_op_before_matched_op([empty, trans_a])

        # if transB is set, trans_b is changed to this op
        trans_b_res = None
        if gemm.trans_b is not None and gemm.trans_b.value.data == 1:
            shape_type = tensor_b_type.element_type
            # onnx.gemm supports only 2D tensors, hence reversing is acceptable
            shape = tuple(reversed(tensor_b_shape))
            empty_shape = TensorType(shape_type, shape)
            empty = tensor.EmptyOp((), empty_shape)
            trans_b = linalg.TransposeOp(
                gemm.tensor_b, empty.tensor, permutation, empty.tensor.type
            )
            # save the result
            trans_b_res = trans_b.result[0]
            rewriter.insert_op_before_matched_op([empty, trans_b])

        # if trans_a occurs, else remain
        if trans_a_res is not None:
            trans_a = trans_a_res
        else:
            trans_a = gemm.tensor_a

        # if trans_b occurs, else remain
        if trans_b_res is not None:
            trans_b = trans_b_res
        else:
            trans_b = gemm.tensor_b

        # alpha * A
        alpha_res = None
        if gemm.alpha is not None and gemm.alpha.value.data != 1:
            constant = arith.Constant(FloatAttr(gemm.alpha.value.data, gemm.alpha.type))
            alpha_mul_result = linalg.MulOp(
                (constant.result, trans_a),
                (trans_a,),
                (trans_a.type,),
            )
            alpha_res = alpha_mul_result.res[0]
            rewriter.insert_op_before_matched_op([constant, alpha_mul_result])

        # if alpha * a does not occur remain on previous trans_a else switch
        if alpha_res is not None:
            trans_a = alpha_res

        # beta * C
        beta_mul_result = gemm.tensor_c
        beta_res = None
        if gemm.beta is not None and gemm.beta.value.data != 1:
            constant = arith.Constant(FloatAttr(gemm.beta.value.data, gemm.beta.type))
            beta_mul_result = linalg.MulOp(
                (
                    constant.result,
                    beta_mul_result,
                ),
                (gemm.tensor_c,),
                (gemm.tensor_c.type,),
            )
            beta_res = beta_mul_result.res[0]
            rewriter.insert_op_before_matched_op([constant, beta_mul_result])

        # this is beta * c result else its just c
        if beta_res is not None:
            beta_mul_result = beta_res
        else:
            beta_mul_result = gemm.tensor_c

        # (A * B) + beta * C
        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp(
                    (),
                    gemm.res_tensor.type,
                ),
                # A * B
                mat_mul_res := linalg.MatmulOp(
                    (trans_a, trans_b),
                    (empty.tensor,),
                    res=(gemm.res_tensor.type,),
                ),
                # (A * B) + beta * C
                linalg.AddOp(
                    (mat_mul_res.results[0], beta_mul_result),
                    (mat_mul_res.results[0],),
                    res=(mat_mul_res.results[0].type,),
                ),
            )
        )


@dataclass
class MaxPoolSingleOutOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, max_pool_single_out: onnx.MaxPoolSingleOut, rewriter: PatternRewriter, /
    ):
        kernel: list[int] = [
            value.value.data for value in max_pool_single_out.kernel_shape.data
        ]
        dilations: list[int] = [
            value.value.data for value in max_pool_single_out.dilations.data
        ]
        strides: list[int] = [
            value.value.data for value in max_pool_single_out.strides.data
        ]
        kernel_shape = TensorType(f32, kernel)

        # Lowering with `storage_order = 1` attribute not supported"
        if (
            max_pool_single_out.storage_order.value.data != 0
            and max_pool_single_out.storage_order
        ):
            raise NotImplementedError()

        rewriter.replace_matched_op(
            (
                empty := tensor.EmptyOp((), kernel_shape),
                init := tensor.EmptyOp((), max_pool_single_out.output.type),
                # Since we're unable to represent +/- infinity,
                # we currently use the maximum value by sys
                cst := arith.Constant(FloatAttr(-1e308, f64)),
                fill := linalg.FillOp(
                    (cst.result,),
                    (init.tensor,),
                    (max_pool_single_out.output.type,),
                ),
                linalg.PoolingNchwMaxOp(
                    DenseIntOrFPElementsAttr.tensor_from_list(dilations, i64, [2]),
                    DenseIntOrFPElementsAttr.tensor_from_list(strides, i64, [2]),
                    (
                        max_pool_single_out.data,
                        empty.tensor,
                    ),
                    (fill.results[0],),
                    (max_pool_single_out.output.type,),
                ),
            )
        )


@dataclass
class ConvOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, conv: onnx.Conv, rewriter: PatternRewriter, /):
        dilations = tuple(value.value.data for value in conv.dilations.data)
        strides = tuple(value.value.data for value in conv.strides.data)

        if conv.group.value.data != 1:
            raise NotImplementedError("Only 1 group supported")

        if not all(dilation == 1 for dilation in dilations):
            raise NotImplementedError("Only 1 dilation supported")

        empty = tensor.EmptyOp((), conv.res.type)
        conv_op = linalg.Conv2DNchwFchwOp(
            DenseIntOrFPElementsAttr.tensor_from_list(dilations, i64, [2]),
            DenseIntOrFPElementsAttr.tensor_from_list(strides, i64, [2]),
            (
                conv.data,
                conv.weight,
            ),
            (empty.tensor,),
            (conv.res.type,),
        )
        conv_ops = (
            empty,
            conv_op,
        )
        if not isinstance(conv.bias.type, NoneType):
            add_bias = linalg.AddOp(
                (conv.bias,),
                (conv_op.results[0],),
                res=(conv.res.type,),
            )
            conv_ops += (add_bias,)
        rewriter.replace_matched_op(conv_ops)


@dataclass(frozen=True)
class ConvertOnnxToLinalgPass(ModulePass):
    name = "convert-onnx-to-linalg"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    SubOpLowering(),
                    ReluOpLowering(),
                    ConstantOpLowering(),
                    ReshapeOpLowering(),
                    GemmOpLowering(),
                    MaxPoolSingleOutOpLowering(),
                    ConvOpLowering(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
