from typing import TypeVar

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier)
from xdsl.ir import MLContext, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, FunctionType, IntegerAttr, ModuleOp, AnyIntegerAttr, IndexType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref

from xdsl.dialects.experimental.stencil import CastOp, FieldType, IndexAttr

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
        inputFieldType: FieldType[_TypeElement]) -> MemRefType[_TypeElement]:
    dims = [i.value.data for i in inputFieldType.shape.data]

    return MemRefType.from_element_type_and_shape(inputFieldType.element_type,
                                                  dims)


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _TypeElement,
                                  lb: IndexAttr,
                                  ub: IndexAttr) -> MemRefType[_TypeElement]:
    dims = [
        ub.value.data - lb.value.data
        for lb, ub in zip(lb.array.data, ub.array.data)
    ]

    return MemRefType.from_element_type_and_shape(memref_element_type, dims)


class CastOpToMemref(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, CastOp):
            return
        if not isinstance(op.field.typ, FieldType):
            return

        field_typ: FieldType[Attribute] = op.field.typ

        result_typ = GetMemRefFromFieldWithLBAndUB(field_typ.element_type,
                                                   op.lb, op.ub)

        rewriter.replace_matched_op(memref.Cast.get(op.field, result_typ))


class StencilTypeConversionFuncOp(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, FuncOp):
            return
        inputs: list[Attribute] = []
        for arg in op.body.blocks[0].args:
            if isinstance(arg.typ, FieldType):
                typ: FieldType[Attribute] = arg.typ
                memreftyp = GetMemRefFromField(typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)

        op.attributes["function_type"] = FunctionType.from_lists(
            inputs, list(op.function_type.outputs.data))


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilTypeConversionFuncOp(),
         CastOpToMemref()]),
                                  walk_regions_first=True,
                                  apply_recursively=False,
                                  walk_reverse=True)
    walker.rewrite_module(module)
