from typing import TypeVar

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, AnyIntegerAttr
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType

from xdsl.dialects.experimental.stencil import FieldType, IndexType

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
        inputFieldType: FieldType[_TypeElement]) -> MemRefType[_TypeElement]:
    memref_shape_integer_attr_list: list[AnyIntegerAttr] = []
    for i in range(len(inputFieldType.shape.data)):
        memref_shape_integer_attr_list.append(
            IntegerAttr.from_params(inputFieldType.shape.data[i].value.data,
                                    inputFieldType.shape.data[i].typ))

    memref_shape_array_attr = ArrayAttr[AnyIntegerAttr].from_list(
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
            for i, arg in enumerate(op.body.blocks[0].args):
                if isinstance(arg.typ, FieldType):
                    typ: FieldType[Attribute] = arg.typ
                    memreftyp = GetMemRefFromField(typ)
                    rewriter.modify_block_argument_type(arg, memreftyp)
                    op.function_type.inputs.data[i] = memreftyp


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilTypeConversionFuncOp()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
