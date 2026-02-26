from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AffineMapAttr,
    IntegerAttr,
)
from xdsl.dialects.csl import csl
from xdsl.ir import OpResult
from xdsl.ir.affine import AffineMap
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class GetDsdAndOffsetFolding(RewritePattern):
    """
    Folds a `csl.get_mem_dsd` immediately followed by a `csl.increment_dsd_offset`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        # single use that is `@increment_dsd_offset`
        if not isinstance(
            offset_op := op.result.get_user_of_unique_use(), csl.IncrementDsdOffsetOp
        ):
            return
        # only works on 1d
        if len(op.sizes) > 1:
            return

        # check if we can promote arith.const to property
        if (
            isinstance(offset_op.offset, OpResult)
            and isinstance(cnst := offset_op.offset.op, arith.ConstantOp)
            and isa(attr_val := cnst.value, IntegerAttr)
        ):
            tensor_access = AffineMap.from_callable(
                lambda x: (x + attr_val.value.data,)
            )
            if op.tensor_access:
                tensor_access = tensor_access.compose(op.tensor_access.data)
            rewriter.replace_op(
                op,
                new_op := csl.GetMemDsdOp.build(
                    operands=[op.base_addr, op.sizes],
                    result_types=op.result_types,
                    properties={
                        **op.properties,
                        "tensor_access": AffineMapAttr(tensor_access),
                    },
                ),
            )
            rewriter.replace_op(offset_op, [], new_results=[new_op.result])


class GetDsdAndLengthFolding(RewritePattern):
    """
    Folds a `csl.get_mem_dsd` immediately followed by a `csl.set_dsd_length`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        # single use that is `@set_dsd_length`
        if not isinstance(
            size_op := op.result.get_user_of_unique_use(), csl.SetDsdLengthOp
        ):
            return
        # only works on 1d
        if len(op.sizes) > 1:
            return

        rewriter.replace_op(
            size_op,
            csl.GetMemDsdOp.build(
                operands=[op.base_addr, [size_op.length]],
                result_types=op.result_types,
                properties=op.properties.copy(),
            ),
        )
        rewriter.erase_op(op)


class GetDsdAndStrideFolding(RewritePattern):
    """
    Folds a `csl.get_mem_dsd` immediately followed by a `csl.set_dsd_stride`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        # single use that is `@set_dsd_stride`
        if not isinstance(
            stride_op := op.result.get_user_of_unique_use(), csl.SetDsdStrideOp
        ):
            return
        # only works on 1d and default (unspecified) tensor_access
        if len(op.sizes) > 1 or op.tensor_access:
            return

        # check if we can promote arith.const to property
        if (
            isinstance(stride_op.stride, OpResult)
            and isinstance(cnst := stride_op.stride.op, arith.ConstantOp)
            and isa(attr_val := cnst.value, IntegerAttr)
        ):
            tensor_access = AffineMap.from_callable(
                lambda x: (x * attr_val.value.data,)
            )
            rewriter.replace_op(
                op,
                new_op := csl.GetMemDsdOp.build(
                    operands=[op.base_addr, op.sizes],
                    result_types=op.result_types,
                    properties={
                        **op.properties,
                        "tensor_access": AffineMapAttr(tensor_access),
                    },
                ),
            )
            rewriter.replace_op(stride_op, [], new_results=[new_op.result])


class ChainedDsdOffsetFolding(RewritePattern):
    """
    Folds a chain of `csl.increment_dsd_offset`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl.IncrementDsdOffsetOp, rewriter: PatternRewriter
    ) -> None:
        # single use that is `@increment_dsd_offset`
        if not isinstance(
            next_op := op.result.get_user_of_unique_use(), csl.IncrementDsdOffsetOp
        ):
            return

        # check if we can promote arith.const to property
        if op.elem_type == next_op.elem_type:
            rewriter.replace_op(
                next_op,
                [
                    new_offset := arith.AddiOp(op.offset, next_op.offset),
                    csl.IncrementDsdOffsetOp(
                        operands=[op.op, new_offset.result],
                        properties=op.properties.copy(),
                        result_types=op.result_types,
                    ),
                ],
            )
            rewriter.erase_op(op)


class ChainedDsdLengthFolding(RewritePattern):
    """
    Folds a chain of `csl.set_dsd_length`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl.SetDsdLengthOp, rewriter: PatternRewriter
    ) -> None:
        # single use that is `@set_dsd_length`
        if not isinstance(
            next_op := op.result.get_user_of_unique_use(), csl.SetDsdLengthOp
        ):
            return

        # check if we can promote arith.const to property
        rewriter.replace_op(
            op,
            rebuilt := csl.SetDsdLengthOp(
                operands=[op.op, next_op.length],
                properties=op.properties.copy(),
                result_types=op.result_types,
            ),
        )
        rewriter.replace_op(next_op, [], new_results=[rebuilt.result])


class ChainedDsdStrideFolding(RewritePattern):
    """
    Folds a chain of `csl.set_dsd_stride`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl.SetDsdStrideOp, rewriter: PatternRewriter
    ) -> None:
        # single use that is `@set_dsd_stride`
        if not isinstance(
            next_op := op.result.get_user_of_unique_use(), csl.SetDsdStrideOp
        ):
            return

        # check if we can promote arith.const to property
        rewriter.replace_op(
            op,
            rebuilt := csl.SetDsdStrideOp(
                operands=[op.op, next_op.stride],
                properties=op.properties.copy(),
                result_types=op.result_types,
            ),
        )
        rewriter.replace_op(next_op, [], new_results=[rebuilt.result])
