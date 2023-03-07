from typing import TypeVar, Any

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType

from xdsl.dialects.experimental.stencil import FieldType, IndexType

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
        inputFieldType: FieldType[_TypeElement]) -> MemRefType[_TypeElement]:
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


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _TypeElement,
                                  lb: IndexType,
                                  ub: IndexType) -> MemRefType[_TypeElement]:
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


class StencilTypeConversionFuncOp(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if (isinstance(op, FuncOp)):
            for i in range(len(op.body.blocks[0].args)):
                memref_type_equiv = GetMemRefFromField(
                    op.function_type.parameters[0].data[i])
                rewriter.modify_block_argument_type(op.body.blocks[0].args[i],
                                                    memref_type_equiv)
                op.function_type.parameters[0].data[i] = memref_type_equiv


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilTypeConversionFuncOp()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
