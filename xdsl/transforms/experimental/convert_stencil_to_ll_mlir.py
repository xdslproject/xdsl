from dataclasses import dataclass
from typing import Iterable, Literal, TypeVar, cast
from warnings import warn

from xdsl.dialects import arith, builtin, gpu, memref, scf
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp
from xdsl.dialects.memref import MemRefType
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
    CastOp,
    ExternalLoadOp,
    ExternalStoreOp,
    FieldType,
    IndexOp,
    LoadOp,
    ReturnOp,
    StencilBoundsAttr,
    StencilType,
    StoreOp,
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
    op_type_rewrite_pattern,
)
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
    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        assert isa(op.result.type, FieldType[Attribute])
        assert isinstance(op.result.type.bounds, StencilBoundsAttr)

        result_type = StencilToMemRefType(op.result.type)

        cast = memref.Cast.get(op.field, result_type)

        if self.target == "gpu":
            unranked = memref.Cast.get(
                cast.dest,
                memref.UnrankedMemrefType.from_type(op.result.type.element_type),
            )
            register = gpu.HostRegisterOp(unranked.dest)
            rewriter.insert_op_after_matched_op([unranked, register])
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

    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        store_list: list[memref.Store] = []

        parallel = op.parent_op()
        assert isinstance(parallel, scf.ParallelOp | gpu.LaunchOp | scf.For)

        for j in range(len(op.arg)):
            target = self.return_target[op][j]

            if target is None:
                break

            assert isinstance(target.type, builtin.ShapedType)

            assert (block := op.parent_block()) is not None

            dims = target.type.get_num_dims()

            args = collectBlockArguments(dims, block)

            if self.target == "gpu":
                args = list(reversed(args))

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


def prepare_apply_body(op: ApplyOp, rewriter: PatternRewriter):
    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.block

    for idx, arg in enumerate(entry.args):
        arg_uses = set(arg.uses)
        for use in arg_uses:
            use.operation.operands[use.index] = op.args[idx]
        entry.erase_arg(arg)

    rewriter.insert_block_argument(entry, 0, builtin.IndexType())

    return rewriter.move_region_contents_to_new_regions(op.region)


@dataclass
class ApplyOpToParallel(RewritePattern):
    return_targets: dict[ReturnOp, list[SSAValue | None]]

    target: Literal["cpu", "gpu"] = "cpu"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        res_type = op.res[0].type
        assert isa(res_type, TempType[Attribute])
        assert isinstance(res_type.bounds, StencilBoundsAttr)

        # Get this apply's ReturnOp
        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))

        body = prepare_apply_body(op, rewriter)
        body.block.add_op(scf.Yield.get())
        dim = res_type.get_num_dims()

        # Then create the corresponding scf.parallel
        boilerplate_ops = [
            *(
                lowerBounds := [
                    arith.Constant.from_int_and_width(x, builtin.IndexType())
                    for x in res_type.bounds.lb
                ]
            ),
            one := arith.Constant.from_int_and_width(1, builtin.IndexType()),
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
        match self.target:
            case "cpu":
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
            case "gpu":
                stencil_rank = len(upperBounds)
                boilerplate_ops.insert(
                    1, zero := arith.Constant.from_int_and_width(0, builtin.IndexType())
                )
                p = scf.ParallelOp.get(
                    lowerBounds=list(reversed(lowerBounds))
                    + [zero] * (3 - stencil_rank),
                    upperBounds=list(reversed(upperBounds))
                    + [one] * (3 - stencil_rank),
                    steps=[one] * 3,
                    body=body,
                )
                for _ in range(3 - 1):
                    rewriter.insert_block_argument(p.body.block, 0, builtin.IndexType())

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
        rewriter.insert_op_before_matched_op([*boilerplate_ops, p])
        rewriter.insert_op_after_matched_op([*deallocs])
        rewriter.replace_matched_op([], new_results)


@dataclass
class AccessOpToMemref(RewritePattern):
    target: Literal["cpu", "gpu"] = "cpu"

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

        if self.target == "gpu":
            args.reverse()

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


class StencilTypeConversionFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        inputs: list[Attribute] = [
            StencilToMemRefType(inp) if isa(inp, FieldType[Attribute]) else inp
            for inp in op.function_type.inputs
        ]
        outputs: list[Attribute] = [
            StencilToMemRefType(out) if isa(out, FieldType[Attribute]) else out
            for out in op.function_type.outputs
        ]
        op.attributes["function_type"] = FunctionType.from_lists(inputs, outputs)
        if op.body.blocks:
            for inp, arg in zip(inputs, op.body.blocks[0].args):
                if inp != arg.type:
                    rewriter.modify_block_argument_type(arg, inp)


class UpdateLoopCarriedVarTypes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter, /):
        for i in range(len(op.iter_args)):
            block_arg = op.body.block.args[i + 1]
            iter_type = op.iter_args[i].type
            if block_arg.type != iter_type:
                rewriter.modify_block_argument_type(block_arg, iter_type)
            y = cast(scf.Yield, op.body.ops.last)
            y.arguments[i].type = iter_type
            if op.res[i].type != iter_type:
                op.res[i].type = iter_type


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
                    ApplyOpToParallel(return_targets, self.target),
                    StencilStoreToSubview(return_targets),
                    CastOpToMemref(self.target),
                    LoadOpToMemref(),
                    AccessOpToMemref(self.target),
                    ReturnOpToMemref(return_targets, self.target),
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
                    BufferOpCleanUp(),
                ]
            )
        )
        type_pass.rewrite_module(op)
