from dataclasses import dataclass
from typing import TypeVar, Iterable

from warnings import warn

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import Block, MLContext, Region, Operation
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin, gpu

from xdsl.dialects.stencil import CastOp
from xdsl.dialects.experimental.stencil import (
    AccessOp,
    ApplyOp,
    FieldType,
    IndexAttr,
    LoadOp,
    ReturnOp,
    StoreOp,
    TempType,
    ExternalLoadOp,
    ExternalStoreOp,
    IndexOp,
)
from xdsl.passes import ModulePass

from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

from xdsl.transforms.experimental.stencil_global_to_local import (
    LowerHaloExchangeToMpi,
    HorizontalSlices2D,
    MpiLoopInvariantCodeMotion,
)

_TypeElement = TypeVar("_TypeElement", bound=Attribute)

# TODO docstrings and comments


def GetMemRefFromField(
    input_type: FieldType[_TypeElement] | TempType[_TypeElement],
) -> MemRefType[_TypeElement]:
    dims = [i.value.data for i in input_type.shape.data]

    return MemRefType.from_element_type_and_shape(input_type.element_type, dims)


def GetMemRefFromFieldWithLBAndUB(
    memref_element_type: _TypeElement, lb: IndexAttr, ub: IndexAttr
) -> MemRefType[_TypeElement]:
    # lb and ub defines the minimum and maximum coordinates of the resulting memref,
    # so its shape is simply ub - lb, computed here.
    dims = IndexAttr.size_from_bounds(lb, ub)

    return MemRefType.from_element_type_and_shape(memref_element_type, dims)


@dataclass
class CastOpToMemref(RewritePattern):
    gpu: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        assert isa(op.field.typ, FieldType[Attribute] | memref.MemRefType[Attribute])

        result_typ = GetMemRefFromFieldWithLBAndUB(
            op.field.typ.element_type, op.lb, op.ub
        )

        cast = memref.Cast.get(op.field, result_typ)

        if self.gpu:
            unranked = memref.Cast.get(
                cast.dest,
                memref.UnrankedMemrefType.from_type(op.field.typ.element_type),
            )
            register = gpu.HostRegisterOp.from_memref(unranked.dest)
            rewriter.insert_op_after_matched_op([unranked, register])
        rewriter.replace_matched_op(cast)


class StoreOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()
        pass


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
    return_target: dict[ReturnOp, list[CastOp | memref.Subview | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        store_list: list[memref.Store] = []

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp | scf.For)

        for j in range(len(op.arg)):
            subview = self.return_target[op][j]

            if subview is None:
                break

            assert isinstance(subview, memref.Subview)
            assert isa(subview.result.typ, MemRefType[Attribute])

            assert (block := op.parent_block()) is not None

            dims = len(subview.result.typ.shape.data)

            args = collectBlockArguments(dims, block)

            store_list.append(memref.Store.get(op.arg[j], subview.result, args))

        rewriter.replace_matched_op([*store_list])


def verify_load_bounds(cast: CastOp, load: LoadOp):
    if [i.value.data for i in IndexAttr.min(cast.lb, load.lb).array.data] != [
        i.value.data for i in cast.lb.array.data
    ]:  # noqa
        raise VerifyException(
            "The stencil computation requires a field with lower bound at least "
            f"{load.lb}, got {cast.lb}, min: {IndexAttr.min(cast.lb, load.lb)}"
        )


class IndexOpToLoopSSA(RewritePattern):
    @staticmethod
    def discover_enclosing_loops(op: Operation) -> Iterable[scf.For | scf.ParallelOp]:
        parent_op = op.parent_op()
        if parent_op is not None:
            yield from IndexOpToLoopSSA.discover_enclosing_loops(parent_op)
        if isa(op, scf.For) or isa(op, scf.ParallelOp):
            yield op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IndexOp, rewriter: PatternRewriter, /):
        # We do not currently support an offset in indexop, therefore check
        # that this is all set to zero as otherwise it will not be handled
        for offset in op.offset:
            assert offset == 0
        enclosing_loops = list(IndexOpToLoopSSA.discover_enclosing_loops(op))
        # The first block argument is the loop iterator
        loop_op = enclosing_loops[op.dim.value.data]
        assert isa(loop_op, scf.For) or isa(loop_op, scf.ParallelOp)
        assert len(loop_op.body.blocks) == 1
        assert len(loop_op.body.block.args) >= 1
        replacement_ssa = loop_op.body.block.args[0]
        op.results[0].replace_by(replacement_ssa)
        rewriter.erase_op(op)


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
            cast.result, element_type, shape, offsets, sizes, strides
        )

        rewriter.replace_matched_op(subview)


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter):
    assert (op.lb is not None) and (op.ub is not None)

    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.block

    for idx, arg in enumerate(entry.args):
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.replace_operand(use.index, op.args[idx])
        entry.erase_arg(arg)

    rewriter.insert_block_argument(entry, 0, builtin.IndexType())

    return rewriter.move_region_contents_to_new_regions(op.region)


class ApplyOpToParallel(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        assert (op.lb is not None) and (op.ub is not None)

        body = prepare_apply_body(op, rewriter)
        body.block.add_op(scf.Yield.get())
        dim = len(op.lb.array.data)

        # Then create the corresponding scf.parallel
        dims = IndexAttr.size_from_bounds(op.lb, op.ub)
        zero = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType()) for x in dims
        ]

        # Generate an outer parallel loop as well as two inner sequential
        # loops. The inner sequential loops ensure that the computational
        # kernel itself is not slowed down by the OpenMP runtime.
        current_region = body
        for i in range(1, dim):
            for_op = scf.For.get(
                lb=zero, ub=upperBounds[-i], step=one, iter_args=[], body=current_region
            )
            block = Block(
                ops=[for_op, scf.Yield.get()], arg_types=[builtin.IndexType()]
            )
            current_region = Region(block)

        p = scf.ParallelOp.get(
            lowerBounds=[zero],
            upperBounds=[upperBounds[0]],
            steps=[one],
            body=current_region,
        )

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
            arith.Constant.from_int_and_width(x.value.data, builtin.IndexType())
            for x in memref_offset
        ]

        args = collectBlockArguments(len(memref_offset), block)

        off_sum_ops = [arith.Addi(i, x) for i, x in zip(args, off_const_ops)]

        load = memref.Load.get(load, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load], [load.res])


@dataclass
class StencilTypeConversionFuncOp(RewritePattern):
    return_targets: dict[ReturnOp, list[CastOp | memref.Subview | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = []
        for arg in op.body.block.args:
            if isa(arg.typ, FieldType[Attribute]):
                memreftyp = GetMemRefFromField(arg.typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)
            else:
                inputs.append(arg.typ)

        op.attributes["function_type"] = FunctionType.from_lists(
            inputs, list(op.function_type.outputs.data)
        )

        stores = [o for o in op.walk() if isinstance(o, StoreOp)]

        for store in stores:
            cast = store.field.owner
            assert isinstance(cast, CastOp)
            assert isa(cast.result.typ, FieldType[Attribute])
            new_cast = cast.clone()
            source_shape = [i.value.data for i in cast.result.typ.shape.data]
            offsets = [i.value.data for i in (store.lb - cast.lb).array.data]
            sizes = [i.value.data for i in (store.ub - store.lb).array.data]
            subview = memref.Subview.from_static_parameters(
                new_cast.result,
                cast.result.typ.element_type,
                source_shape,
                offsets,
                sizes,
                [1] * len(sizes),
            )
            rewriter.replace_op(cast, [new_cast, subview])

            for r, c in self.return_targets.items():
                for i in range(len(c)):
                    if c[i] == cast:
                        self.return_targets[r][i] = subview


class TrivialExternalLoadOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        assert isa(op.result.typ, FieldType[Attribute])
        op.result.typ = GetMemRefFromField(op.result.typ)

        if op.field.typ == op.result.typ:
            rewriter.replace_matched_op([], [op.field])
        pass


class TrivialExternalStoreOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


def return_target_analysis(module: builtin.ModuleOp):
    return_targets: dict[ReturnOp, list[CastOp | memref.Subview | None]] = {}

    for op in module.walk():
        if not isinstance(op, ReturnOp):
            continue

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        return_targets[op] = []
        for res in list(apply.res):
            store = [
                use.operation
                for use in list(res.uses)
                if isinstance(use.operation, StoreOp)
            ]

            if len(store) > 1:
                warn("Each stencil result should be stored only once.")
                continue

            cast = None if len(store) == 0 else store[0].field.owner

            assert isinstance(cast, CastOp | None)

            return_targets[op].append(cast)

    return return_targets


def StencilConversion(
    return_targets: dict[ReturnOp, list[CastOp | memref.Subview | None]], gpu: bool
):
    """
    List of rewrite passes for stencil
    """
    return GreedyRewritePatternApplier(
        [
            ApplyOpToParallel(),
            StencilTypeConversionFuncOp(return_targets),
            CastOpToMemref(gpu),
            LoadOpToMemref(),
            AccessOpToMemref(),
            ReturnOpToMemref(return_targets),
            IndexOpToLoopSSA(),
            StoreOpCleanup(),
            TrivialExternalLoadOpCleanup(),
            TrivialExternalStoreOpCleanup(),
        ]
    )


class ConvertStencilToGPUPass(ModulePass):
    name = "convert-stencil-to-gpu"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets = return_target_analysis(op)

        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([StencilConversion(return_targets, gpu=True)]),
            apply_recursively=False,
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)


class ConvertStencilToLLMLIRPass(ModulePass):
    name = "convert-stencil-to-ll-mlir"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets: dict[
            ReturnOp, list[CastOp | memref.Subview | None]
        ] = return_target_analysis(op)

        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([StencilConversion(return_targets, gpu=False)]),
            apply_recursively=False,
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerHaloExchangeToMpi(HorizontalSlices2D(2)),
                ]
            )
        ).rewrite_module(op)
        MpiLoopInvariantCodeMotion().rewrite_module(op)
