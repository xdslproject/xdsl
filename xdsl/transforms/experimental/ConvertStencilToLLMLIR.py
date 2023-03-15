from typing import TypeVar
from warnings import warn

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import MLContext
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, FunctionType, IntegerAttr, ModuleOp, i64
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin

from xdsl.dialects.experimental.stencil import ApplyOp, CastOp, FieldType, IndexAttr, LoadOp, TempType

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
    inputType: FieldType[_TypeElement] | TempType[_TypeElement]
) -> MemRefType[_TypeElement]:
    dims = [i.value.data for i in inputType.shape.data]

    return MemRefType.from_element_type_and_shape(inputType.element_type, dims)


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _TypeElement,
                                  lb: IndexAttr,
                                  ub: IndexAttr) -> MemRefType[_TypeElement]:
    # lb and ub defines the minimum and maximum coordinates of the resulting memref,
    # so its shape is simply ub - lb, computed here.
    dims = IndexAttr.sizeFromBounds(lb, ub)

    return MemRefType.from_element_type_and_shape(memref_element_type, dims)


class CastOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        if not isinstance(op.field.typ, FieldType):
            return

        field_typ: FieldType[Attribute] = op.field.typ

        result_typ = GetMemRefFromFieldWithLBAndUB(field_typ.element_type,
                                                   op.lb, op.ub)

        rewriter.replace_matched_op(memref.Cast.get(op.field, result_typ))


class LoadOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        if op.lb is None:
            warn("stencil.load should have a lb attribute when lowered.")
            return
        if not isinstance(op.field.owner, memref.Cast):
            warn(
                "stencil.cast should be lowered to memref.cast before the stencil.load lowering."
            )
            return
        offsets: list[Attribute] = [
            IntegerAttr(-i.value.data, i64) for i in op.lb.array.data
        ]

        # TODO: handle with memref.subview or another, cleaner, approach.
        op.field.owner.attributes["stencil_offset"] = ArrayAttr.from_list(
            offsets)
        rewriter.replace_matched_op([], [op.field.owner.dest])


class ApplyOpToParallel(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if (op.lb is None) or (op.ub is None):
            print(f"ub:{op.ub}, lb:{op.ub}")
            return
        # # First replace all TempField arguments by memref.
        entry = op.region.blocks[0]
        # for arg in entry.args:
        #     if isinstance(arg.typ, TempType):
        #         arg_typ: TempType[Attribute] = arg.typ
        #         rewriter.modify_block_argument_type(
        #             arg, GetMemRefFromField(arg_typ))

        for arg in entry.args:
            for use in arg.uses:
                use.operation.replace_operand(use.index, op.args[use.index])
            entry.erase_arg(arg)

        entry.insert_arg(builtin.IndexType(), 0)
        entry.insert_arg(builtin.IndexType(), 0)
        entry.insert_arg(builtin.IndexType(), 0)

        #Then lower to scf.parallel
        dims = IndexAttr.sizeFromBounds(op.lb, op.ub)
        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in dims
        ]

        body = rewriter.move_region_contents_to_new_regions(op.region)
        p = scf.ParallelOp.get(lowerBounds=[zero, zero, zero],
                               upperBounds=upperBounds,
                               steps=[one, one, one],
                               body=body)
        rewriter.replace_matched_op([zero, one, *upperBounds, p])


class StencilTypeConversionFuncOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = []
        for arg in op.body.blocks[0].args:
            if isinstance(arg.typ, FieldType):
                typ: FieldType[Attribute] = arg.typ
                memreftyp = GetMemRefFromField(typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)
            else:
                inputs.append(arg.typ)

        op.attributes["function_type"] = FunctionType.from_lists(
            inputs, list(op.function_type.outputs.data))


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
        StencilTypeConversionFuncOp(),
        CastOpToMemref(),
        LoadOpToMemref(),
        ApplyOpToParallel()
    ]),
                                  walk_regions_first=True)
    walker.rewrite_module(module)
