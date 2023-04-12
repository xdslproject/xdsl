from dataclasses import dataclass
from typing import TypeVar, Iterable

from warnings import warn

from xdsl.pattern_rewriter import (PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, GreedyRewritePatternApplier,
                                   op_type_rewrite_pattern)
from xdsl.ir import Block, BlockArgument, MLContext, Operation, Region, SSAValue
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin, gpu

from xdsl.dialects.experimental.stencil import (AccessOp, ApplyOp, CastOp,
                                                FieldType, IndexAttr, LoadOp,
                                                ReturnOp, StoreOp, TempType,
                                                ExternalLoadOp, HaloSwapOp,
                                                ExternalStoreOp)
from xdsl.passes import ModulePass

from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

from xdsl.transforms.experimental.stencil_global_to_local import LowerHaloExchangeToMpi, HorizontalSlices2D, \
    MpiLoopInvariantCodeMotion

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

    return MemRefType.from_element_type_and_shape(memref_element_type, dims)


@dataclass
class CastOpToMemref(RewritePattern):

    gpu: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):

        assert isa(op.field.typ,
                   FieldType[Attribute] | memref.MemRefType[Attribute])

        result_typ = GetMemRefFromFieldWithLBAndUB(op.field.typ.element_type,
                                                   op.lb, op.ub)

        cast = memref.Cast.get(op.field, result_typ)

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


# Collect up to 'number' block arguments by walking up the region tree
# and collecting block arguments as we reach new parent regions.
def collectBlockArguments(number: int, block: Block):
    args = []

    while len(args) < number:
        args = list(block.args) + args

        parent = block.parent_block()
        if parent is None:
            break

        block = parent

    return args


@dataclass
class ReturnOpToMemref(RewritePattern):

    return_target: dict[ReturnOp, CastOp | memref.Subview]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp | scf.For)

        subview = self.return_target[op]

        assert isinstance(subview, memref.Subview)
        assert isa(subview.result.typ, MemRefType[Attribute])

        assert (block := op.parent_block()) is not None

        dims = len(subview.result.typ.shape.data)

        args = collectBlockArguments(dims, block)

        store = memref.Store.get(op.arg, subview.result, args)

        rewriter.replace_matched_op([store])


def verify_load_bounds(cast: CastOp, load: LoadOp):

    if ([i.value.data for i in IndexAttr.min(cast.lb, load.lb).array.data] !=
        [i.value.data for i in cast.lb.array.data]):  # noqa
        raise VerifyException(
            "The stencil computation requires a field with lower bound at least "
            f"{load.lb}, got {cast.lb}, min: {IndexAttr.min(cast.lb, load.lb)}"
        )


class LoadOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        cast = op.field.owner
        assert isinstance(cast, CastOp)
        assert isa(cast.result.typ, FieldType[Attribute])

        verify_load_bounds(cast, op)

        assert op.lb and op.ub

        element_type = cast.result.typ.element_type
        shape = [i.value.data for i in cast.result.typ.shape.data]

        offsets = [i.value.data for i in (op.lb - cast.lb).array.data]
        sizes = [i.value.data for i in (op.ub - op.lb).array.data]
        strides = [1] * len(sizes)

        subview = memref.Subview.from_static_parameters(
            cast.result, element_type, shape, offsets, sizes, strides)

        rewriter.replace_matched_op(subview)


class LoadOpShapeInference(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        cast = op.field.owner
        assert isinstance(cast, CastOp)

        verify_load_bounds(cast, op)

        assert op.lb and op.ub
        assert isa(op.res.typ, TempType[Attribute])

        # TODO: We need to think about that. Do we want an API for this?
        # Do we just want to recreate the whole operation?
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
        body.blocks[0].add_op(scf.Yield.get())
        dim = len(op.lb.array.data)

        # Then create the corresponding scf.parallel
        dims = IndexAttr.size_from_bounds(op.lb, op.ub)
        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in dims
        ]

        # Generate an outer parallel loop as well as two inner sequential
        # loops. The inner sequential loops ensure that the computational
        # kernel itself is not slowed down by the OpenMP runtime.
        current_region = body
        for i in range(1, dim):
            for_op = scf.For.get(lb=zero,
                                 ub=upperBounds[-i],
                                 step=one,
                                 iter_args=[],
                                 body=current_region)
            block = Block(ops=[for_op, scf.Yield()],
                          arg_types=[builtin.IndexType()])
            current_region = Region([block])

        p = scf.ParallelOp.get(lowerBounds=[zero],
                               upperBounds=[upperBounds[0]],
                               steps=[one],
                               body=current_region)

        # Replace with the loop and necessary constants.
        rewriter.insert_op_before_matched_op([zero, one, *upperBounds, p])
        rewriter.erase_matched_op()


class AccessOpToMemref(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):

        load = op.temp.owner
        assert isinstance(load, LoadOp)
        assert load.lb is not None

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        memref_offset = (op.offset - load.lb).array.data
        off_const_ops = [
            arith.Constant.from_int_and_width(x.value.data,
                                              builtin.IndexType())
            for x in memref_offset
        ]

        args = collectBlockArguments(len(memref_offset), block)

        off_sum_ops = [
            arith.Addi.get(i, x) for i, x in zip(args, off_const_ops)
        ]

        load = memref.Load.get(load, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load],
                                    [load.res])


@dataclass
class StencilTypeConversionFuncOp(RewritePattern):

    return_targets: dict[ReturnOp, CastOp | memref.Subview]

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

        stores: list[StoreOp] = []
        op.walk(lambda o: stores.append(o) if isinstance(o, StoreOp) else None)

        for store in stores:
            cast = store.field.owner
            assert isinstance(cast, CastOp)
            assert isa(cast.result.typ, FieldType[Attribute])
            new_cast = cast.clone()
            source_shape = [i.value.data for i in cast.result.typ.shape.data]
            offsets = [i.value.data for i in (store.lb - cast.lb).array.data]
            sizes = [i.value.data for i in (store.ub - store.lb).array.data]
            subview = memref.Subview.from_static_parameters(
                new_cast.result, cast.result.typ.element_type, source_shape,
                offsets, sizes, [1] * len(sizes))
            rewriter.replace_op(cast, [new_cast, subview])

            returns = [r for r, c in self.return_targets.items() if c == cast]
            for r in returns:
                self.return_targets[r] = subview


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


def return_target_analysis(module: builtin.ModuleOp):
    return_targets: dict[ReturnOp, CastOp | memref.Subview] = {}

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


_OpT = TypeVar('_OpT', bound=Operation)


def all_matching_uses(op_res: Iterable[SSAValue],
                      typ: type[_OpT]) -> Iterable[_OpT]:
    for res in op_res:
        for use in res.uses:
            if isinstance(use.operation, typ):
                yield use.operation


def infer_core_size(op: LoadOp) -> tuple[IndexAttr, IndexAttr]:
    """
    This method infers the core size (as used in DimsHelper)
    from an LoadOp by walking the def-use chain down to the `apply`
    """
    applies: list[ApplyOp] = list(all_matching_uses([op.res], ApplyOp))
    assert len(applies) > 0, "Load must be followed by Apply!"

    shape_lb: None | IndexAttr = None
    shape_ub: None | IndexAttr = None

    for apply in applies:
        assert apply.lb is not None and apply.ub is not None
        shape_lb = IndexAttr.min(apply.lb, shape_lb)
        shape_ub = IndexAttr.max(apply.ub, shape_ub)

    assert shape_lb is not None and shape_ub is not None
    return shape_lb, shape_ub


class HaloOpShapeInference(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: HaloSwapOp, rewriter: PatternRewriter, /):
        assert isinstance(op.input_stencil.owner, LoadOp)
        load = op.input_stencil.owner
        halo_lb, halo_ub = infer_core_size(load)
        op.attributes['core_lb'] = halo_lb
        op.attributes['core_ub'] = halo_ub
        assert load.lb is not None
        assert load.ub is not None
        op.attributes['buff_lb'] = load.lb
        op.attributes['buff_ub'] = load.ub


ShapeInference = GreedyRewritePatternApplier([
    ApplyOpShapeInference(),
    LoadOpShapeInference(),
    StoreOpShapeInference(),
    HaloOpShapeInference(),
])


def StencilConversion(return_targets: dict[ReturnOp, CastOp | memref.Subview],
                      gpu: bool):
    """
    List of rewrite passes for stencil
    """
    return GreedyRewritePatternApplier([
        ApplyOpToParallel(),
        StencilTypeConversionFuncOp(return_targets),
        CastOpToMemref(gpu),
        LoadOpToMemref(),
        AccessOpToMemref(),
        ReturnOpToMemref(return_targets),
        StoreOpCleanup(),
        TrivialExternalLoadOpCleanup(),
        TrivialExternalStoreOpCleanup()
    ])


class StencilShapeInferencePass(ModulePass):

    name = 'stencil-shape-inference'

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        inference_walker = PatternRewriteWalker(ShapeInference,
                                                apply_recursively=False,
                                                walk_reverse=True)
        inference_walker.rewrite_module(op)


class ConvertStencilToGPUPass(ModulePass):

    name = 'convert-stencil-to-gpu'

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets = return_target_analysis(op)

        the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier(
            [StencilConversion(return_targets, gpu=True)]),
                                            apply_recursively=False,
                                            walk_reverse=True)
        the_one_pass.rewrite_module(op)


class ConvertStencilToLLMLIRPass(ModulePass):

    name = 'convert-stencil-to-ll-mlir'

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:

        return_targets: dict[ReturnOp, CastOp
                             | memref.Subview] = return_target_analysis(op)

        the_one_pass = PatternRewriteWalker(GreedyRewritePatternApplier(
            [StencilConversion(return_targets, gpu=False)]),
                                            apply_recursively=False,
                                            walk_reverse=True)
        the_one_pass.rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier([
                LowerHaloExchangeToMpi(HorizontalSlices2D(2)),
                MpiLoopInvariantCodeMotion(),
            ])).rewrite_module(op)
