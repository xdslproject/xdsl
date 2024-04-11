from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import TypeVar, cast
from warnings import warn

from xdsl.dialects import arith, builtin, memref, scf
from xdsl.dialects.builtin import MemRefType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
    CastOp,
    ExternalLoadOp,
    ExternalStoreOp,
    FieldType,
    IndexAttr,
    IndexOp,
    LoadOp,
    ResultType,
    ReturnOp,
    StencilBoundsAttr,
    StencilType,
    StoreOp,
    StoreResultOp,
    TempType,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    MLContext,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

_TypeElement = TypeVar("_TypeElement", bound=Attribute)

# TODO docstrings and comments


def StencilToMemRefType(
    input_type: StencilType[_TypeElement],
) -> MemRefType[_TypeElement]:
    return MemRefType(input_type.element_type, input_type.get_shape())


@dataclass
class CastOpToMemref(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        assert isa(op.result.type, FieldType[Attribute])
        assert isinstance(op.result.type.bounds, StencilBoundsAttr)

        result_type = StencilToMemRefType(op.result.type)

        cast = memref.Cast.get(op.field, result_type)

        rewriter.replace_matched_op(cast)


# Collect up to 'number' block arguments by walking up the region tree
# and collecting block arguments as we reach new parent regions.
def collectBlockArguments(number: int, block: Block):
    args = []

    while len(args) < number:
        args = list(block.args[0 : number - len(args)]) + args

        parent = block.parent_block()
        if parent is None:
            break

        block = parent

    return args


def update_return_target(
    return_targets: dict[ReturnOp, list[SSAValue | None]],
    old_target: SSAValue,
    new_target: SSAValue,
):
    for targets in return_targets.values():
        for i, target in enumerate(targets):
            if target == old_target:
                targets[i] = new_target


@dataclass
class ReturnOpToMemref(RewritePattern):
    return_target: dict[ReturnOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        unroll_factor = op.unroll_factor
        n_res = len(op.arg) // unroll_factor

        store_list: list[Operation] = []
        for j in range(n_res):
            target = self.return_target[op][j]
            if target is None:
                continue
            assert isinstance(target.type, builtin.ShapedType)
            dims = target.type.get_num_dims()

            unroll = op.unroll
            if unroll is None:
                unroll = IndexAttr.get(*([1] * dims))

            for k, offset in enumerate(product(*(range(u) for u in unroll))):
                arg = op.arg[j * unroll_factor + k]
                assert (block := op.parent_block()) is not None
                args = cast(list[SSAValue], collectBlockArguments(dims, block))

                for i in range(dims):
                    if offset[i] != 0:
                        constant_op = arith.Constant.from_int_and_width(
                            offset[i], builtin.IndexType()
                        )
                        add_op = arith.Addi(args[i], constant_op)
                        args[i] = add_op.results[0]
                        store_list.append(constant_op)
                        store_list.append(add_op)

                store = memref.Store.get(arg, target, args)
                if isinstance(arg.type, ResultType):
                    if isinstance(arg.owner, StoreResultOp):
                        rewriter.replace_op(
                            arg.owner,
                            store,
                            new_results=[arg.owner.args[0]],
                            safe_erase=False,
                        )
                    else:
                        raise NotImplementedError("Oopsie Woopsie")
                else:
                    store_list.append(store)

        rewriter.replace_matched_op([*store_list])


def assert_subset(field: FieldType[Attribute], temp: TempType[Attribute]):
    assert isinstance(field.bounds, StencilBoundsAttr)
    assert isinstance(temp.bounds, StencilBoundsAttr)
    if temp.bounds.lb < field.bounds.lb:
        raise VerifyException(
            "The stencil computation requires a field with lower bound at least "
            f"{temp.bounds.lb}, got {field.bounds.lb}, min: {min(field.bounds.lb, temp.bounds.lb)}"
        )
    if temp.bounds.ub > field.bounds.ub:
        raise VerifyException(
            "The stencil computation requires a field with upper bound at least "
            f"{temp.bounds.ub}, got {field.bounds.ub}, max: {max(field.bounds.ub, temp.bounds.ub)}"
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
        enclosing_loops = list(IndexOpToLoopSSA.discover_enclosing_loops(op))
        # The first block argument is the loop iterator
        loop_op = enclosing_loops[0]
        assert isa(loop_op, scf.For) or isa(loop_op, scf.ParallelOp)
        assert len(loop_op.body.blocks) == 1
        assert len(loop_op.body.block.args) >= 1
        replacement_ssa = loop_op.body.block.args[op.dim.value.data]
        offset = op.offset.array.data[op.dim.value.data].data
        if offset == 0:
            rewriter.replace_matched_op([], [replacement_ssa])
        else:
            rewriter.replace_matched_op(
                [
                    offset_op := arith.Constant.from_int_and_width(
                        offset, builtin.IndexType()
                    ),
                    arith.Addi(replacement_ssa, offset_op),
                ]
            )


class LoadOpToMemref(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        field = op.field.type
        assert isa(field, FieldType[Attribute])
        assert isa(field.bounds, StencilBoundsAttr)
        temp = op.res.type
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


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter, dim: int):
    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.block

    for idx, arg in enumerate(entry.args):
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.operands[use.index] = op.args[idx]
        entry.erase_arg(arg)

    for _ in range(dim):
        rewriter.insert_block_argument(entry, 0, builtin.IndexType())

    return rewriter.move_region_contents_to_new_regions(op.region)


@dataclass
class ApplyOpToParallel(RewritePattern):
    return_targets: dict[ReturnOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        res_type = op.res[0].type
        assert isa(res_type, TempType[Attribute])
        assert isinstance(res_type.bounds, StencilBoundsAttr)

        # Get this apply's ReturnOp
        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))
        unroll = return_op.unroll

        dim = res_type.get_num_dims()
        body = prepare_apply_body(op, rewriter, dim)
        body.block.add_op(scf.Yield())
        if unroll is None:
            unroll = [1] * dim
        else:
            unroll = [i for i in unroll]

        # Then create the corresponding scf.parallel
        boilerplate_ops = [
            *(
                lowerBounds := [
                    arith.Constant.from_int_and_width(x, builtin.IndexType())
                    for x in res_type.bounds.lb
                ]
            ),
            *(
                steps := [
                    arith.Constant.from_int_and_width(x, builtin.IndexType())
                    for x in unroll
                ]
            ),
            *(
                upperBounds := [
                    arith.Constant.from_int_and_width(x, builtin.IndexType())
                    for x in res_type.bounds.ub
                ]
            ),
        ]

        # Generate an outer parallel loop as well as two inner sequential
        # loops. The inner sequential loops ensure that the computational
        # kernel itself is not slowed down by the OpenMP runtime.
        tiled_steps = steps
        p = scf.ParallelOp(
            lower_bounds=lowerBounds,
            upper_bounds=upperBounds,
            steps=tiled_steps,
            body=Region(),
        )

        p.body.insert_block(body.detach_block(0), 0)

        # Handle returnd values
        for result in op.res:
            assert isa(
                result.type, TempType[Attribute]
            ), f"Expected return value to be a !{TempType.name}"
            assert isinstance(
                result.type.bounds, StencilBoundsAttr
            ), f"Expected output to be sized before lowering. {result.type}"
            shape = result.type.get_shape()
            element_type = result.type.element_type

            # If it is buffered, allocate the buffer
            if any(isinstance(use.operation, BufferOp) for use in result.uses):
                alloc = memref.Alloc.get(element_type, shape=shape)
                alloc_type = alloc.memref.type
                assert isa(alloc_type, MemRefType[Attribute])

                offset = list(-result.type.bounds.lb)

                view = memref.Subview.from_static_parameters(
                    alloc,
                    alloc_type,
                    offset,
                    shape,
                    [1] * result.type.get_num_dims(),
                )
                rewriter.insert_op_before_matched_op((alloc, view))
                update_return_target(self.return_targets, result, view.result)

        deallocs: list[Operation] = []
        # Handle input buffer deallocation
        for input in op.args:
            # Is this input a temp buffer?
            if isinstance(input.type, TempType) and isinstance(input.owner, BufferOp):
                block = op.parent_block()
                assert block is not None
                self_index = block.get_operation_index(op)
                # Is it its last use?
                if not any(
                    use.operation.parent_block() is block
                    and block.get_operation_index(use.operation) > self_index
                    for use in input.uses
                ):
                    # Then deallocate it
                    deallocs.append(memref.Dealloc.get(input))

        # Get the maybe updated results
        new_results: list[SSAValue | None] = []
        new_results = self.return_targets[return_op]
        # Replace with the loop and necessary constants.
        assert isa(boilerplate_ops, list[Operation])
        rewriter.insert_op_before_matched_op([*boilerplate_ops, p])
        rewriter.insert_op_after_matched_op([*deallocs])
        rewriter.replace_matched_op([], new_results)


@dataclass
class AccessOpToMemref(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        temp = op.temp.type
        assert isa(temp, TempType[Attribute])
        assert isinstance(temp.bounds, StencilBoundsAttr)

        # Make pyright happy with the fact that this op has to be in
        # a block.
        assert (block := op.parent_block()) is not None

        memref_offset = op.offset
        if op.offset_mapping is not None:
            max_idx = 0
            for i in op.offset_mapping:
                if i.data + 1 > max_idx:
                    max_idx = i.data + 1
            args = collectBlockArguments(max_idx, block)
            # Reverse the list as arguments are collated in the opposite
            # order to the stencil.apply ordering (e.g. the top most loop is
            # front of the list, rather than at the end)
            args.reverse()
        else:
            args = collectBlockArguments(len(memref_offset), block)

        off_const_ops: list[Operation] = []
        memref_load_args: list[BlockArgument | OpResult] = []

        # This will apply an offset to the index if one is required
        # (e.g the offset is not zero), otherwise will use the index value directly
        for i, x in enumerate(memref_offset):
            block_arg = (
                args[list(op.offset_mapping)[i].data]
                if op.offset_mapping is not None
                else args[i]
            )
            if x != 0:
                constant_op = arith.Constant.from_int_and_width(x, builtin.IndexType())
                add_op = arith.Addi(block_arg, constant_op)
                memref_load_args.append(add_op.results[0])
                off_const_ops += [constant_op, add_op]
            else:
                memref_load_args.append(block_arg)

        load = memref.Load.get(op.temp, memref_load_args)

        rewriter.replace_matched_op([*off_const_ops, load], [load.res])


@dataclass
class StencilStoreToSubview(RewritePattern):
    return_targets: dict[ReturnOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        stores = [o for o in op.walk() if isinstance(o, StoreOp)]

        for store in stores:
            field = store.field
            assert isa(field.type, FieldType[Attribute])
            assert isa(field.type.bounds, StencilBoundsAttr)
            temp = store.temp
            assert isa(temp.type, TempType[Attribute])
            offsets = [i for i in -field.type.bounds.lb]
            sizes = [i for i in temp.type.get_shape()]
            subview = memref.Subview.from_static_parameters(
                field,
                StencilToMemRefType(field.type),
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

            update_return_target(self.return_targets, field, subview.result)


class BufferOpCleanUp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op([], [op.temp])


class TrivialExternalLoadOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        assert isa(op.result.type, FieldType[Attribute])
        op.result.type = StencilToMemRefType(op.result.type)

        if op.field.type == op.result.type:
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
                if isinstance(use.operation, StoreOp | BufferOp)
            ]

            if len(store) > 1:
                warn("Each stencil result should be stored only once.")
                continue

            elif len(store) == 0:
                field = None
            elif isinstance(store[0], StoreOp):
                field = store[0].field
            # then it's a BufferOp
            else:
                field = store[0].temp

            return_targets[op].append(field)

    return return_targets


class StencilTypeConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: StencilType[Attribute]) -> MemRefType[Attribute]:
        return StencilToMemRefType(typ)


@dataclass(frozen=True)
class ConvertStencilToLLMLIRPass(ModulePass):
    name = "convert-stencil-to-ll-mlir"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets: dict[ReturnOp, list[SSAValue | None]] = return_target_analysis(
            op
        )

        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ApplyOpToParallel(return_targets),
                    StencilStoreToSubview(return_targets),
                    CastOpToMemref(),
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
                    StencilTypeConversion(recursive=True),
                    BufferOpCleanUp(),
                ]
            )
        )
        type_pass.rewrite_module(op)
