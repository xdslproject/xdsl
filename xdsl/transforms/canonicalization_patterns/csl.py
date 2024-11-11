from xdsl.dialects import arith
from xdsl.dialects.builtin import AnyIntegerAttrConstr, ArrayAttr
from xdsl.dialects.csl import csl
from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.isattr import isattr


class GetDsdAndOffsetFolding(RewritePattern):
    """
    Folds a `csl.get_mem_dsd` immediately followed by a `csl.increment_dsd_offset`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        # single use that is `@increment_dsd_offset`
        if len(op.result.uses) != 1 or not isinstance(
            offset_op := next(iter(op.result.uses)).operation, csl.IncrementDsdOffsetOp
        ):
            return
        # only works on 1d
        if op.offsets and len(op.offsets) > 1:
            return

        # check if we can promote arith.const to property
        if (
            isinstance(offset_op.offset, OpResult)
            and isinstance(cnst := offset_op.offset.op, arith.Constant)
            and isattr(cnst.value, AnyIntegerAttrConstr)
        ):
            rewriter.replace_matched_op(
                new_op := csl.GetMemDsdOp.build(
                    operands=[op.base_addr, op.sizes],
                    result_types=op.result_types,
                    properties={**op.properties, "offsets": ArrayAttr([cnst.value])},
                )
            )
            rewriter.replace_op(offset_op, [], new_results=[new_op.result])


class GetDsdAndLengthFolding(RewritePattern):
    """
    Folds a `csl.get_mem_dsd` immediately followed by a `csl.set_dsd_length`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        # single use that is `@set_dsd_length`
        if len(op.result.uses) != 1 or not isinstance(
            size_op := next(iter(op.result.uses)).operation, csl.SetDsdLengthOp
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
        rewriter.erase_matched_op()


class GetDsdAndStrideFolding(RewritePattern):
    """
    Folds a `csl.get_mem_dsd` immediately followed by a `csl.set_dsd_stride`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.GetMemDsdOp, rewriter: PatternRewriter) -> None:
        # single use that is `@set_dsd_stride`
        if len(op.result.uses) != 1 or not isinstance(
            stride_op := next(iter(op.result.uses)).operation, csl.SetDsdStrideOp
        ):
            return
        # only works on 1d
        if op.offsets and len(op.offsets) > 1:
            return

        # check if we can promote arith.const to property
        if (
            isinstance(stride_op.stride, OpResult)
            and isinstance(cnst := stride_op.stride.op, arith.Constant)
            and isattr(cnst.value, AnyIntegerAttrConstr)
        ):
            rewriter.replace_matched_op(
                new_op := csl.GetMemDsdOp.build(
                    operands=[op.base_addr, op.sizes],
                    result_types=op.result_types,
                    properties={**op.properties, "strides": ArrayAttr([cnst.value])},
                )
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
        if len(op.result.uses) != 1 or not isinstance(
            next_op := next(iter(op.result.uses)).operation, csl.IncrementDsdOffsetOp
        ):
            return

        # check if we can promote arith.const to property
        if op.elem_type == next_op.elem_type:
            rewriter.replace_op(
                next_op,
                [
                    new_offset := arith.Addi(op.offset, next_op.offset),
                    csl.IncrementDsdOffsetOp(
                        operands=[op.op, new_offset.result],
                        properties=op.properties.copy(),
                        result_types=op.result_types,
                    ),
                ],
            )
            rewriter.erase_matched_op()


class ChainedDsdLengthFolding(RewritePattern):
    """
    Folds a chain of `csl.set_dsd_length`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl.SetDsdLengthOp, rewriter: PatternRewriter
    ) -> None:
        # single use that is `@set_dsd_length`
        if len(op.result.uses) != 1 or not isinstance(
            next_op := next(iter(op.result.uses)).operation, csl.SetDsdLengthOp
        ):
            return

        # check if we can promote arith.const to property
        rewriter.replace_matched_op(
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
        if len(op.result.uses) != 1 or not isinstance(
            next_op := next(iter(op.result.uses)).operation, csl.SetDsdStrideOp
        ):
            return

        # check if we can promote arith.const to property
        rewriter.replace_matched_op(
            rebuilt := csl.SetDsdStrideOp(
                operands=[op.op, next_op.stride],
                properties=op.properties.copy(),
                result_types=op.result_types,
            )
        )
        rewriter.replace_op(next_op, [], new_results=[rebuilt.result])
