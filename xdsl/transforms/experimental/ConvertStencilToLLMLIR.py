from typing import TypeVar
from warnings import warn

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import BlockArgument, MLContext
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import ArrayAttr, FunctionType, IntegerAttr, ModuleOp, i64
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin

from xdsl.dialects.experimental.stencil import AccessOp, ApplyOp, CastOp, FieldType, IndexAttr, LoadOp, ReturnOp, StoreOp, TempType

_TypeElement = TypeVar("_TypeElement", bound=Attribute)


def GetMemRefFromField(
    input_type: FieldType[_TypeElement] | TempType[_TypeElement]
) -> MemRefType[_TypeElement]:
    dims = [i.value.data for i in input_type.shape.data]

    return MemRefType.from_element_type_and_shape(input_type.element_type,
                                                  dims)


def GetMemRefFromFieldWithLBAndUB(memref_element_type: _TypeElement,
                                  lb: IndexAttr,
                                  ub: IndexAttr) -> MemRefType[_TypeElement]:
    # lb and ub defines the minimum and maximum coordinates of the resulting memref,
    # so its shape is simply ub - lb, computed here.
    dims = IndexAttr.size_from_bounds(lb, ub)

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


class StoreOpPrepare(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        if op.lb is None:
            warn("stencil.store should have a lb attribute when lowered.")
            return
        if not isinstance(op.field.owner, memref.Cast):
            warn(
                "stencil.cast should be lowered to memref.cast before the stencil.store lowering."
            )
            return
        offsets: list[Attribute] = [
            IntegerAttr(-i.value.data, i64) for i in op.lb.array.data
        ]

        # TODO: handle with memref.subview or another, cleaner, approach.
        op.field.owner.attributes["stencil_offset"] = IndexAttr(
            [ArrayAttr.from_list(offsets)])


class StoreOpCleanup(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()
        pass


class ReturnOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        res = list(apply.res)[0]

        if (len(res.uses) > 1) or (not isinstance(
            (store := list(res.uses)[0].operation), StoreOp)):
            warn("Only single store result atm")
            return

        cast = store.field.owner
        assert isinstance(cast, memref.Cast)

        offsets = cast.attributes['stencil_offset']
        assert isinstance(offsets, IndexAttr)

        block = apply.region.blocks[0]

        off_const_ops = [
            arith.Constant.from_int_and_width(x.value.data,
                                              builtin.IndexType())
            for x in offsets.array.data
        ]

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(block.args, off_const_ops)
        ]

        load = memref.Store.get(op.arg, cast.dest, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load])


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
        op.field.owner.attributes["stencil_offset"] = IndexAttr(
            [ArrayAttr.from_list(offsets)])
        rewriter.replace_matched_op([], [op.field.owner.dest])


class ApplyOpInsertInductionVariables(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        if 'done' in op.attributes.keys():
            return
        rewriter.insert_block_argument(op.region.blocks[0], 0,
                                       builtin.IndexType())
        rewriter.insert_block_argument(op.region.blocks[0], 0,
                                       builtin.IndexType())
        rewriter.insert_block_argument(op.region.blocks[0], 0,
                                       builtin.IndexType())
        op.attributes['done'] = builtin.UnitAttr()


class ApplyOpToParallel(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):

        if (op.lb is None) or (op.ub is None):
            warn(
                "stencil.apply should have lb and ub attributes before being lowered to "
                "scf.parallel.")
            return

        # First replace all current arguments by their definition
        # and erase them from the block. (We are changing the op
        # to a loop, which has access to them either way)
        entry = op.region.blocks[0]

        for arg in entry.args[3:]:
            arg_uses = set(arg.uses)
            for use in arg_uses:
                use.operation.replace_operand(use.index, op.args[use.index])
            entry.erase_arg(arg)

        #Then create the corresponding scf.parallel
        dims = IndexAttr.size_from_bounds(op.lb, op.ub)
        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in dims
        ]

        # Move the body to the loop
        body = rewriter.move_region_contents_to_new_regions(op.region)
        p = scf.ParallelOp.get(lowerBounds=[zero, zero, zero],
                               upperBounds=upperBounds,
                               steps=[one, one, one],
                               body=body)

        # Replace with the loop and necessary constants.
        rewriter.insert_op_before_matched_op([zero, one, *upperBounds, p])
        rewriter.erase_matched_op(False)


class AccessOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        arg = op.temp
        assert isinstance(arg, BlockArgument)
        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        cast = apply.args[arg.index - 3].owner
        if not isinstance(cast, memref.Cast):
            print(cast)
            warn(
                "stencil.load should have been lowered before lowering related "
                "stencil.access")
            return

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        if not isinstance(cast.attributes["stencil_offset"], IndexAttr):
            warn(
                f"Expected IndexAttr-typed stencil_offset, got {cast.attributes['stencil_offset']}"
            )
            return

        access_offset = op.offset.array.data
        memref_offset = cast.attributes["stencil_offset"].array.data

        offsets = [
            a.value.data + m.value.data
            for a, m in zip(access_offset, memref_offset)
        ]

        off_const_ops = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in offsets
        ]

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(block.args, off_const_ops)
        ]

        load = memref.Load.get(cast.dest, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load],
                                    [load.res])


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
    preparation = PatternRewriteWalker(GreedyRewritePatternApplier([
        ApplyOpInsertInductionVariables(),
        StencilTypeConversionFuncOp(),
        CastOpToMemref(),
        LoadOpToMemref(),
        StoreOpPrepare(),
    ]),
                                       walk_regions_first=True)

    lowering = PatternRewriteWalker(GreedyRewritePatternApplier([
        AccessOpToMemref(),
        ReturnOpToMemref(),
    ]),
                                    walk_regions_first=True)

    lowering2 = PatternRewriteWalker(
        GreedyRewritePatternApplier([
            ApplyOpToParallel(),
        ]))

    cleanup = PatternRewriteWalker(
        GreedyRewritePatternApplier([StoreOpCleanup()]))
    preparation.rewrite_module(module)
    lowering.rewrite_module(module)
    lowering2.rewrite_module(module)
    cleanup.rewrite_module(module)
