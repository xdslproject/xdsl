from typing import TypeVar

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, OpResult, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.memref import Cast as MemRefCast

from xdsl.dialects.experimental.stencil import FieldType, Cast, IndexType

_MemRefTypeElement = TypeVar("_MemRefTypeElement", bound=Attribute)


def GetMemRefFromField(inputFieldType: FieldType) -> MemRefType:
    memref_shape_integer_attr_list = []
    for i in range(len(inputFieldType.parameters[0].data)):
        memref_shape_integer_attr_list.append(
            IntegerAttr.from_int_and_width(
                inputFieldType.parameters[0].data[i].value.data,
                inputFieldType.parameters[0].data[i].typ.width.data))

    memref_shape_array_attr = ArrayAttr.from_list(
        memref_shape_integer_attr_list)

    return MemRefType.from_params(inputFieldType.element_type,
                                  memref_shape_array_attr)


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _MemRefTypeElement,
                                  lb: IndexType, ub: IndexType) -> MemRefType:
    # Assumes lb and ub are of same size and same underlying element types.
    memref_shape_integer_attr_list = []
    for i in range(len(lb.parameters[0].data)):
        memref_shape_integer_attr_list.append(
            IntegerAttr.from_int_and_width(
                abs(lb.parameters[0].data[i].value.data) +
                abs(ub.parameters[0].data[i].value.data),
                lb.parameters[0].data[i].typ.width.data))

    memref_shape_array_attr = ArrayAttr.from_list(
        memref_shape_integer_attr_list)

    return MemRefType.from_params(memref_element_type, memref_shape_array_attr)


class StencilTypeConversionLowering(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if (isinstance(op, FuncOp)):
            for i in range(len(op.body.blocks[0].args)):
                memref_type_equiv = GetMemRefFromField(
                    op.function_type.parameters[0].data[i])
                rewriter.modify_block_argument_type(op.body.blocks[0].args[i],
                                                    memref_type_equiv)
                op.function_type.parameters[0].data[i] = memref_type_equiv


class CastOpLowering(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if (isinstance(op, Cast)):
            res_memref_type = GetMemRefFromFieldWithLBAndUB(
                op.result.typ.element_type, op.lb, op.ub)

            dynamic_dim_memref_type = GetMemRefFromField(op.field.typ)
            dynamic_dim_memref = OpResult(dynamic_dim_memref_type, [], [])

            memref_cast = MemRefCast.build(operands=[dynamic_dim_memref],
                                           result_types=[res_memref_type])

            rewriter.replace_op(op, memref_cast)
            # rewriter.insert_op_before(op, memref_cast)
            # rewriter.insert_op_after(op, memref_cast)


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilTypeConversionLowering(), CastOpLowering()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
