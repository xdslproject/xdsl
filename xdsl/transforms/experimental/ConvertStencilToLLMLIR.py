from dataclasses import dataclass
from typing import Literal, TypeVar, Iterable, cast

from warnings import warn

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import Block, MLContext, Region, Operation, SSAValue
from xdsl.irdl import Attribute
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects import memref, arith, scf, builtin, gpu

from xdsl.dialects.stencil import CastOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    FieldType,
    LoadOp,
    ReturnOp,
    StencilBoundsAttr,
    StencilType,
    StoreOp,
    TempType,
    ExternalLoadOp,
    ExternalStoreOp,
    IndexOp,
)
from xdsl.passes import ModulePass

from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

_TypeElement = TypeVar("_TypeElement", bound=Attribute)

# TODO docstrings and comments


def StencilToMemRefType(
    input_type: StencilType[_TypeElement],
) -> MemRefType[_TypeElement]:
    return MemRefType.from_element_type_and_shape(
        input_type.element_type, input_type.get_shape()
    )


@dataclass
class CastOpToMemref(RewritePattern):
    gpu: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        assert isa(op.result.typ, FieldType[Attribute])
        assert isinstance(op.result.typ.bounds, StencilBoundsAttr)

        result_typ = StencilToMemRefType(op.result.typ)

        cast = memref.Cast.get(op.field, result_typ)

        if self.gpu:
            unranked = memref.Cast.get(
                cast.dest,
                memref.UnrankedMemrefType.from_type(op.result.typ.element_type),
            )
            register = gpu.HostRegisterOp.from_memref(unranked.dest)
            rewriter.insert_op_after_matched_op([unranked, register])
        rewriter.replace_matched_op(cast)


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
    return_target: dict[ReturnOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        store_list: list[memref.Store] = []

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp | scf.For)

        for j in range(len(op.arg)):
            target = self.return_target[op][j]

            if target is None:
                break

            assert isinstance(target.typ, builtin.ShapedType)

            assert (block := op.parent_block()) is not None

            dims = target.typ.get_num_dims()

            args = collectBlockArguments(dims, block)

            store_list.append(memref.Store.get(op.arg[j], target, args))

        rewriter.replace_matched_op([*store_list])


def assert_subset(field: FieldType[Attribute], temp: TempType[Attribute]):
    assert isinstance(field.bounds, StencilBoundsAttr)
    assert isinstance(temp.bounds, StencilBoundsAttr)
    if temp.bounds.lb < field.bounds.lb or temp.bounds.ub > field.bounds.ub:
        raise VerifyException(
            "The stencil computation requires a field with lower bound at least "
            f"{temp.bounds.lb}, got {field.bounds.lb}, min: {min(field.bounds.lb, temp.bounds.lb)}"
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
        field = op.field.typ
        assert isa(field, FieldType[Attribute])
        assert isa(field.bounds, StencilBoundsAttr)
        temp = op.res.typ
        assert isa(temp, TempType[Attribute])
        assert isa(temp.bounds, StencilBoundsAttr)

        assert_subset(field, temp)

        offsets = [i for i in -field.bounds.lb]
        sizes = [i for i in temp.get_shape()]
        strides = [1] * len(sizes)

        subview = memref.Subview.from_static_parameters(
            op.field, StencilToMemRefType(field), offsets, sizes, strides
        )

        rewriter.replace_matched_op(subview)
        name = None
        if subview.source.name_hint:
            name = subview.source.name_hint + "_loadview"
        subview.result.name_hint = name


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter):
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
        res_typ = op.res[0].typ
        assert isa(res_typ, TempType[Attribute])
        assert isinstance(res_typ.bounds, StencilBoundsAttr)

        body = prepare_apply_body(op, rewriter)
        body.block.add_op(scf.Yield.get())
        dim = res_typ.get_num_dims()

        # Then create the corresponding scf.parallel
        lowerBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in res_typ.bounds.lb
        ]
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        upperBounds = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in res_typ.bounds.ub
        ]

        # Generate an outer parallel loop as well as two inner sequential
        # loops. The inner sequential loops ensure that the computational
        # kernel itself is not slowed down by the OpenMP runtime.
        current_region = body
        for i in range(1, dim):
            for_op = scf.For.get(
                lb=lowerBounds[-i],
                ub=upperBounds[-i],
                step=one,
                iter_args=[],
                body=current_region,
            )
            block = Block(
                ops=[for_op, scf.Yield.get()], arg_types=[builtin.IndexType()]
            )
            current_region = Region(block)

        p = scf.ParallelOp.get(
            lowerBounds=[lowerBounds[0]],
            upperBounds=[upperBounds[0]],
            steps=[one],
            body=current_region,
        )

        # Replace with the loop and necessary constants.
        rewriter.insert_op_before_matched_op([*lowerBounds, one, *upperBounds, p])
        rewriter.erase_matched_op()


class AccessOpToMemref(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        temp = op.temp.typ
        assert isa(temp, TempType[Attribute])
        assert isinstance(temp.bounds, StencilBoundsAttr)

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        memref_offset = op.offset
        off_const_ops = [
            arith.Constant.from_int_and_width(x, builtin.IndexType())
            for x in memref_offset
        ]

        args = collectBlockArguments(len(memref_offset), block)

        off_sum_ops = [arith.Addi(i, x) for i, x in zip(args, off_const_ops)]

        load = memref.Load.get(op.temp, off_sum_ops)

        rewriter.replace_matched_op([*off_const_ops, *off_sum_ops, load], [load.res])


class StencilTypeConversionFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = []
        for arg in op.body.block.args:
            if isa(arg.typ, FieldType[Attribute]):
                memreftyp = StencilToMemRefType(arg.typ)
                rewriter.modify_block_argument_type(arg, memreftyp)
                inputs.append(memreftyp)
            else:
                inputs.append(arg.typ)
        outputs: list[Attribute] = [
            StencilToMemRefType(out) if isa(out, FieldType[Attribute]) else out
            for out in op.function_type.outputs
        ]
        op.attributes["function_type"] = FunctionType.from_lists(inputs, outputs)


class UpdateLoopCarriedVarTypes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        for i in range(len(op.iter_args)):
            block_arg = op.body.block.args[i + 1]
            iter_typ = op.iter_args[i].typ
            if block_arg.typ != iter_typ:
                rewriter.modify_block_argument_type(block_arg, iter_typ)
            y = cast(scf.Yield, op.body.ops.last)
            y.arguments[i].typ = iter_typ
            if op.res[i].typ != iter_typ:
                op.res[i].typ = iter_typ


@dataclass
class StencilStoreToSubview(RewritePattern):
    return_targets: dict[ReturnOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        stores = [o for o in op.walk() if isinstance(o, StoreOp)]

        for store in stores:
            field = store.field
            assert isa(field.typ, FieldType[Attribute])
            assert isa(field.typ.bounds, StencilBoundsAttr)
            temp = store.temp
            assert isa(temp.typ, TempType[Attribute])
            offsets = [i for i in -field.typ.bounds.lb]
            sizes = [i for i in temp.typ.get_shape()]
            subview = memref.Subview.from_static_parameters(
                field,
                StencilToMemRefType(field.typ),
                offsets,
                sizes,
                [1] * len(sizes),
            )
            name = None
            if subview.source.name_hint:
                name = subview.source.name_hint + "_storeview"
            subview.result.name_hint = name
            if isinstance(field.owner, Operation):
                rewriter.insert_op_after(subview, field.owner)
            else:
                rewriter.insert_op_at_start(subview, field.owner)

            rewriter.erase_op(store)

            for r, c in self.return_targets.items():
                for i in range(len(c)):
                    if c[i] == field:
                        self.return_targets[r][i] = subview.result


class TrivialExternalLoadOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        assert isa(op.result.typ, FieldType[Attribute])
        op.result.typ = StencilToMemRefType(op.result.typ)

        if op.field.typ == op.result.typ:
            rewriter.replace_matched_op([], [op.field])
        pass


class TrivialExternalStoreOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


def return_target_analysis(module: builtin.ModuleOp):
    return_targets: dict[ReturnOp, list[SSAValue | None]] = {}

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

            field = None if len(store) == 0 else store[0].field

            return_targets[op].append(field)

    return return_targets


@dataclass
class ConvertStencilToLLMLIRPass(ModulePass):
    name = "convert-stencil-to-ll-mlir"

    target: Literal["cpu", "gpu"] = "cpu"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets: dict[ReturnOp, list[SSAValue | None]] = return_target_analysis(
            op
        )

        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ApplyOpToParallel(),
                    StencilStoreToSubview(return_targets),
                    CastOpToMemref(gpu=(self.target == "gpu")),
                    LoadOpToMemref(),
                    AccessOpToMemref(),
                    ReturnOpToMemref(return_targets),
                    IndexOpToLoopSSA(),
                    TrivialExternalLoadOpCleanup(),
                    TrivialExternalStoreOpCleanup(),
                ]
            ),
            apply_recursively=True,
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
        type_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    UpdateLoopCarriedVarTypes(),
                    StencilTypeConversionFuncOp(),
                ]
            )
        )
        type_pass.rewrite_module(op)
