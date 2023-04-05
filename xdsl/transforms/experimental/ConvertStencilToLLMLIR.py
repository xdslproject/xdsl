from dataclasses import dataclass
from typing import TypeVar, Iterable

from warnings import warn

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import BlockArgument, MLContext, Operation, SSAValue
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin, gpu

from xdsl.dialects.experimental.stencil import AccessOp, ApplyOp, CastOp, FieldType, IndexAttr, LoadOp, ReturnOp, StoreOp, TempType, ExternalLoadOp, ExternalStoreOp, HaloSwapOp
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

from xdsl.transforms.experimental.stencil_global_to_local import LowerHaloExchangeToMpi, HorizontalSlices2D

_TypeElement = TypeVar("_TypeElement", bound=Attribute)

# TODO docstrings and comments


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
    dims.reverse()

    return MemRefType.from_element_type_and_shape(memref_element_type, dims)


@dataclass
class CastOpToMemref(RewritePattern):

    return_target: dict[ReturnOp, CastOp | memref.Cast]
    gpu: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):

        assert isa(op.field.typ, FieldType[Attribute] | MemRefType[Attribute])

        result_typ = GetMemRefFromFieldWithLBAndUB(op.field.typ.element_type,
                                                   op.lb, op.ub)

        cast = memref.Cast.get(op.field, result_typ)

        for k, v in self.return_target.items():
            if v == op:
                self.return_target[k] = cast

        if self.gpu:
            unranked = memref.Cast.get(
                cast.dest,
                memref.UnrankedMemrefType.from_type(op.field.typ.element_type))
            register = gpu.HostRegisterOp.from_memref(unranked.dest)
            rewriter.insert_op_after_matched_op([unranked, register])
        rewriter.replace_matched_op(cast)


class StoreOpCleanup(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):

        rewriter.erase_matched_op()
        pass


class StoreOpShapeInference(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):

        owner = op.temp.owner

        assert isinstance(owner, ApplyOp | LoadOp)

        owner.attributes['lb'] = IndexAttr.min(op.lb, owner.lb)
        owner.attributes['ub'] = IndexAttr.max(op.ub, owner.ub)


@dataclass
class ReturnOpToMemref(RewritePattern):

    return_target: dict[ReturnOp, CastOp | memref.Cast]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp)

        cast = self.return_target[op]

        assert isinstance(cast, CastOp)

        offsets = cast.lb
        assert isinstance(offsets, IndexAttr)

        assert (block := op.parent_block()) is not None

        off_const_ops = [
            arith.Constant.from_int_and_width(-x.value.data,
                                              builtin.IndexType())
            for x in offsets.array.data
        ]
        off_const_ops.reverse()

        args = list(block.args)
        args.reverse()

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(args, off_const_ops)
        ]

        load = memref.Store.get(op.arg, cast.result, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load])


def verify_load_bounds(cast: CastOp, load: LoadOp):

    if [i.value.data for i in IndexAttr.min(cast.lb, load.lb).array.data
        ] != [i.value.data for i in cast.lb.array.data]:
        raise VerifyException(
            "The stencil computation requires a field with lower bound at least "
            f"{load.lb}, got {cast.lb}, min: {IndexAttr.min(cast.lb, load.lb)}"
        )


class LoadOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        cast = op.field.owner
        assert isinstance(cast, CastOp)

        verify_load_bounds(cast, op)

        rewriter.replace_matched_op([], list(cast.results))


class LoadOpShapeInference(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        cast = op.field.owner
        assert isinstance(cast, CastOp)

        verify_load_bounds(cast, op)

        assert op.lb and op.ub
        assert isa(op.res.typ, TempType[Attribute])

        # TODO We need to think about that. Do we want an API for this? Do we just want
        # to recreate the whole operation?
        op.res.typ = TempType.from_shape(
            IndexAttr.size_from_bounds(op.lb, op.ub),
            op.res.typ.element_type,
        )


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter):

    assert (op.lb is not None) and (op.ub is not None)

    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.blocks[0]

    for idx, arg in enumerate(entry.args):
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.replace_operand(use.index, op.args[idx])
        entry.erase_arg(arg)

    dim = len(op.lb.array.data)

    for _ in range(dim):
        rewriter.insert_block_argument(entry, 0, builtin.IndexType())

    return rewriter.move_region_contents_to_new_regions(op.region)


class ApplyOpShapeInference(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):

        def access_shape_infer_walk(access: Operation) -> None:
            assert (op.lb is not None) and (op.ub is not None)
            if not isinstance(access, AccessOp):
                return
            assert isinstance(access.temp, BlockArgument)
            temp_owner = op.args[access.temp.index].owner

            assert isinstance(temp_owner, LoadOp | ApplyOp)

            temp_owner.attributes['lb'] = IndexAttr.min(
                op.lb + access.offset, temp_owner.lb)
            temp_owner.attributes['ub'] = IndexAttr.max(
                op.ub + access.offset, temp_owner.ub)

        op.walk(access_shape_infer_walk)

        assert op.lb and op.ub

        for result in op.results:
            assert isa(result.typ, TempType[Attribute])
            result.typ = TempType.from_shape(
                IndexAttr.size_from_bounds(op.lb, op.ub),
                result.typ.element_type)


class ApplyOpToParallel(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):

        assert (op.lb is not None) and (op.ub is not None)

        body = prepare_apply_body(op, rewriter)
        dim = len(op.lb.array.data)

        #Then create the corresponding scf.parallel
        dims = IndexAttr.size_from_bounds(op.lb, op.ub)
        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in dims
        ]

        # Move the body to the loop
        body.blocks[0].add_op(scf.Yield.get())
        p = scf.ParallelOp.get(lowerBounds=[zero] * dim,
                               upperBounds=upperBounds,
                               steps=[one] * dim,
                               body=body)

        # Replace with the loop and necessary constants.
        rewriter.insert_op_before_matched_op([zero, one, *upperBounds, p])
        rewriter.erase_matched_op(False)


class AccessOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):

        load = op.temp.owner
        assert isinstance(load, LoadOp)

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        assert isinstance(load.lb, IndexAttr)
        assert isinstance(load.field.owner, CastOp)
        memref_offset = (op.offset - load.field.owner.lb).array.data
        off_const_ops = [
            arith.Constant.from_int_and_width(x.value.data,
                                              builtin.IndexType())
            for x in memref_offset
        ]
        off_const_ops.reverse()

        args = list(block.args)
        args.reverse()

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(args, off_const_ops)
        ]

        load = memref.Load.get(load.res, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load],
                                    [load.res])


class StencilTypeConversionFuncOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = []
        for arg in op.body.blocks[0].args:
            if isa(arg.typ, FieldType[Attribute]):
                memreftyp = GetMemRefFromField(arg.typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)
            else:
                inputs.append(arg.typ)

        op.attributes["function_type"] = FunctionType.from_lists(
            inputs, list(op.function_type.outputs.data))


class TrivialExternalLoadOpCleanup(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter,
                          /):
        assert isa(op.result.typ, FieldType[Attribute])
        op.result.typ = GetMemRefFromField(op.result.typ)

        if op.field.typ == op.result.typ:
            rewriter.replace_matched_op([], [op.field])
        pass


class TrivialExternalStoreOpCleanup(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter,
                          /):
        rewriter.erase_matched_op()


def return_target_analysis(module: ModuleOp):

    return_targets: dict[ReturnOp, CastOp | memref.Cast] = {}

    def map_returns(op: Operation) -> None:
        if not isinstance(op, ReturnOp):
            return

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        res = list(apply.res)[0]

        if (len(res.uses) > 1) or (not isinstance(
            (store := list(res.uses)[0].operation), StoreOp)):
            warn("Only single store result atm")
            return

        cast = store.field.owner

        assert isinstance(cast, CastOp)

        return_targets[op] = cast

    module.walk(map_returns)

    return return_targets


def all_matching_uses(op_res: Iterable[SSAValue],
                      typ: type[_TypeElement]) -> _TypeElement | None:
    for res in op_res:
        for use in res.uses:
            if isinstance(use.operation, typ):
                yield use.operation


def infer_halo_from_load_op(op: LoadOp) -> tuple[IndexAttr, IndexAttr]:
    applies: list[LoadOp] = list(all_matching_uses([op.res], ApplyOp))
    assert len(applies) > 0, "Load must be followed by Apply!"

    shape_lb: None | IndexAttr = None
    shape_ub: None | IndexAttr = None

    for apply in applies:
        stores: list[StoreOp] = list(all_matching_uses(apply.res, StoreOp))

        assert len(stores) > 0, "Apply must be followed by store!"
        for store in stores:
            lb, ub = store.lb, store.ub
            if shape_lb is None:
                shape_lb = lb
                shape_ub = ub
                continue

            shape_lb = IndexAttr.min(lb, shape_lb)
            shape_ub = IndexAttr.max(ub, shape_ub)
    return shape_lb, shape_ub


class HaloOpShapeInference(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HaloSwapOp, rewriter: PatternRewriter, /):
        assert isinstance(op.input_stencil.owner, LoadOp)
        load = op.input_stencil.owner
        halo_lb, halo_ub = infer_halo_from_load_op(load)
        op.attributes['core_lb'] = halo_lb
        op.attributes['core_ub'] = halo_ub
        op.attributes['buff_lb'] = load.lb
        op.attributes['buff_ub'] = load.ub


ShapeInference = GreedyRewritePatternApplier([
    ApplyOpShapeInference(),
    LoadOpShapeInference(),
    StoreOpShapeInference(),
    HaloOpShapeInference(),
])


def StencilConversion(return_targets: dict[ReturnOp, CastOp | memref.Cast],
                      gpu: bool):
    return GreedyRewritePatternApplier([
        ApplyOpToParallel(),
        StencilTypeConversionFuncOp(),
        CastOpToMemref(return_targets, gpu),
        LoadOpToMemref(),
        AccessOpToMemref(),
        ReturnOpToMemref(return_targets),
        StoreOpCleanup(),
        TrivialExternalLoadOpCleanup(),
        TrivialExternalStoreOpCleanup()
    ])


# TODO: We probably want to factor that in another file
def StencilShapeInference(ctx: MLContext, module: ModuleOp):

    inference_pass = PatternRewriteWalker(ShapeInference,
                                          apply_recursively=False,
                                          walk_reverse=True)

    inference_pass.rewrite_module(module)


def ConvertStencilToGPU(ctx: MLContext, module: ModuleOp):

    return_targets = return_target_analysis(module)

    the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilConversion(return_targets, gpu=True)]),
                                        apply_recursively=False,
                                        walk_reverse=True)
    the_one_pass.rewrite_module(module)


def ConvertStencilToLLMLIR(ctx: MLContext, module: ModuleOp):

    return_targets = return_target_analysis(module)

    the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier(
        [StencilConversion(return_targets, gpu=False)]),
                                        apply_recursively=False,
                                        walk_reverse=True)
    the_one_pass.rewrite_module(module)
    PatternRewriteWalker(LowerHaloExchangeToMpi(
        HorizontalSlices2D(2))).rewrite_module(module)
