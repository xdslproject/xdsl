from typing import TypeVar

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, FunctionType, IntegerAttr, ModuleOp, AnyIntegerAttr, IndexType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType

from xdsl.dialects.experimental.stencil import FieldType, IndexAttr

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
        inputFieldType: FieldType[_TypeElement]) -> MemRefType[_TypeElement]:
    memref_shape_integer_attr_list: list[AnyIntegerAttr] = []
    for i in inputFieldType.shape.data:
        memref_shape_integer_attr_list.append(
            IntegerAttr.from_params(i.value.data, i.typ))

    memref_shape_array_attr = ArrayAttr[AnyIntegerAttr](
        memref_shape_integer_attr_list)

    return MemRefType.from_params(inputFieldType.element_type,
                                  memref_shape_array_attr)


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _TypeElement,
                                  lb: IndexAttr,
                                  ub: IndexAttr) -> MemRefType[_TypeElement]:
    memref_shape_integer_attr_list: list[AnyIntegerAttr] = []
    for i in range(len(lb.array.data)):
        memref_shape_integer_attr_list.append(
            IntegerAttr.from_params(
                ub.array.data[i].value.data - lb.array.data[i].value.data,
                IndexType()))

    memref_shape_array_attr = ArrayAttr(memref_shape_integer_attr_list)

    return MemRefType.from_params(memref_element_type, memref_shape_array_attr)


class StencilTypeConversionFuncOp(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if (isinstance(op, FuncOp)):
            inputs: list[Attribute] = []
            for arg in op.body.blocks[0].args:
                if isinstance(arg.typ, FieldType):
                    typ: FieldType[Attribute] = arg.typ
                    memreftyp = GetMemRefFromField(typ)
                    rewriter.modify_block_argument_type(arg, memreftyp)
                    inputs.append(memreftyp)

            op.attributes["function_type"] = FunctionType(
                [ArrayAttr(inputs),
                 ArrayAttr(op.function_type.outputs.data)])


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilTypeConversionFuncOp()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
