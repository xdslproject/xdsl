from dataclasses import dataclass
from itertools import product
from typing import TypeVar
from warnings import warn

from xdsl.dialects import arith, builtin, memref, scf
from xdsl.dialects.builtin import (
    MemRefType,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    BufferOp,
    CastOp,
    CombineOp,
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
    Use,
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
from xdsl.rewriter import InsertPoint
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
    return_targets: dict[ApplyOp, list[SSAValue | None]],
    old_target: SSAValue,
    new_target: SSAValue | None,
):
    for targets in return_targets.values():
        for i, target in enumerate(targets):
            if target == old_target:
                targets[i] = new_target


def _find_result_store(result: SSAValue) -> tuple[StoreResultOp, ...]:
    while True:
        match owner := result.owner:
            case StoreResultOp():
                return (owner,)
            case scf.If():
                assert isinstance(result, OpResult)
                index = result.index
                yield_true = owner.true_region.ops.last
                assert isinstance(yield_true, scf.Yield)
                yield_false = owner.false_region.ops.last
                assert isinstance(yield_false, scf.Yield)
                true_stores = _find_result_store(yield_true.arguments[index])
                false_stores = _find_result_store(yield_false.arguments[index])
                return true_stores + false_stores

            case _:
                raise ValueError(
                    "Could not find the corresponding stencil.store_result"
                )


@dataclass
class ReturnOpToMemref(RewritePattern):
    return_target: dict[ApplyOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        unroll_factor = op.unroll_factor
        n_res = len(op.arg) // unroll_factor

        store_list: list[Operation] = []
        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)
        if op.unroll is not None:
            apply.attributes["__unroll__"] = op.unroll

        n_dims = apply.get_rank()

        for j in range(n_res):
            target = self.return_target[apply][j]

            unroll = op.unroll
            if unroll is None:
                unroll = IndexAttr.get(*([1] * n_dims))

            for k, offset in enumerate(product(*(range(u) for u in unroll))):
                arg = op.arg[j * unroll_factor + k]
                index_ops: list[Operation] = list(
                    IndexOp(
                        attributes={
                            "dim": builtin.IntegerAttr.from_index_int_value(i),
                            "offset": IndexAttr.get(*([0] * n_dims)),
                        },
                        result_types=[builtin.IndexType()],
                    )
                    for i in range(n_dims)
                )
                store_list += index_ops

                for i in range(n_dims):
                    if offset[i] != 0:
                        constant_op = arith.Constant.from_int_and_width(
                            offset[i], builtin.IndexType()
                        )
                        add_op = arith.Addi(index_ops[i], constant_op)
                        index_ops[i] = add_op
                        store_list.append(constant_op)
                        store_list.append(add_op)

                if isinstance(arg.type, ResultType):
                    result_owner = _find_result_store(arg)
                    for owner in result_owner:
                        if owner.arg:
                            if target is not None:
                                store = memref.Store.get(owner.arg, target, index_ops)
                            else:
                                store = list[Operation]()
                            rewriter.replace_op(
                                owner,
                                store,
                                new_results=[owner.arg],
                            )
                        else:
                            dummy = UnrealizedConversionCastOp.get([], [arg.type.elem])
                            rewriter.replace_op(owner, dummy)

                else:
                    if target is not None:
                        store = memref.Store.get(arg, target, index_ops)
                        store_list.append(store)

        rewriter.insert_op_before_matched_op([*store_list])
        rewriter.erase_matched_op()


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


def prepare_apply_body(op: ApplyOp):
    # First replace all current arguments by their definition
    # and erase them from the block. (We are changing the op
    # to a loop, which has access to them either way)
    entry = op.region.block

    for operand, arg in zip(op.operands, entry.args):
        arg.replace_by(operand)
        entry.erase_arg(arg)
    entry.add_op(scf.Yield())
    for _ in range(op.get_rank()):
        entry.insert_arg(builtin.IndexType(), 0)

    return op.region.detach_block(entry)


@dataclass
class BufferOpToMemref(RewritePattern):

    return_targets: dict[ApplyOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BufferOp, rewriter: PatternRewriter, /):
        # The current lowering simply allocate at block entry and deallocates at block
        # exit.
        # One could be smarter, e.g., aim precisly at where the first and last need is
        # But first, this requires more code with stencil.combine in the mix.
        # And second, do we want to be that smart? We use it iterated either way, so
        # probably always hoping to hoist (de)allocations out of the loop anyway?
        temp_t = op.temp.type
        assert isa(temp_t, TempType[Attribute])
        temp_bounds = temp_t.bounds
        assert isa(temp_bounds, StencilBoundsAttr)

        block = op.parent_block()
        assert block is not None
        first_op = block.first_op
        assert first_op is not None
        last_op = block.last_op
        assert last_op is not None

        alloc = memref.Alloc.get(temp_t.get_element_type(), shape=temp_t.get_shape())
        alloc_type = alloc.memref.type
        assert isa(alloc_type, MemRefType[Attribute])

        offset = list(-temp_bounds.lb)

        view = memref.Subview.from_static_parameters(
            alloc,
            alloc_type,
            offset,
            temp_t.get_shape(),
            [1] * temp_t.get_num_dims(),
        )

        rewriter.insert_op(alloc, InsertPoint.before(first_op))
        rewriter.insert_op(view, InsertPoint.before(first_op))

        update_return_target(self.return_targets, op.temp, view.result)

        dealloc = memref.Dealloc.get(alloc.memref)

        if not op.res.uses:
            rewriter.insert_op_after_matched_op(dealloc)
            rewriter.erase_matched_op()
            return

        rewriter.insert_op(dealloc, InsertPoint.before(last_op))
        rewriter.replace_matched_op([], [view.result])


@dataclass
class ApplyOpToParallel(RewritePattern):
    return_targets: dict[ApplyOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        res_type = op.res[0].type
        assert isa(res_type, TempType[Attribute])
        assert isinstance(res_type.bounds, StencilBoundsAttr)

        # Get this apply's ReturnOp
        unroll = op.attributes.get("__unroll__", None)
        if unroll is not None:
            assert isinstance(unroll, IndexAttr)
            unroll = list(u for u in unroll)

        rank = op.get_rank()
        body = prepare_apply_body(op)
        if unroll is None:
            unroll = [1] * rank
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
            body=Region(body),
        )
        for index in body.walk():
            if isinstance(index, IndexOp):
                offset = list(index.offset)
                ops: list[Operation] = []
                res: list[SSAValue] = [body.args[index.dim.value.data]]
                if offset[index.dim.value.data] != 0:
                    ops = [
                        cst := arith.Constant.from_int_and_width(
                            offset[index.dim.value.data], builtin.IndexType()
                        ),
                        add := arith.Addi(body.args[index.dim.value.data], cst),
                    ]
                    res = [add.result]
                rewriter.replace_op(index, ops, res)

        # Get the maybe updated results
        new_results = self.return_targets[op]
        # Replace with the loop and necessary constants.
        assert isa(boilerplate_ops, list[Operation])
        rewriter.insert_op_before_matched_op([*boilerplate_ops, p])
        rewriter.replace_matched_op([], new_results)


@dataclass
class AccessOpToMemref(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        temp = op.temp.type
        assert isa(temp, TempType[Attribute])
        assert isinstance(temp.bounds, StencilBoundsAttr)

        memref_offset = op.offset

        mapping = (
            op.offset_mapping
            if op.offset_mapping is not None
            else range(len(memref_offset))
        )

        args = [
            IndexOp(
                attributes={
                    "dim": builtin.IntegerAttr.from_index_int_value(i),
                    "offset": IndexAttr.get(*([0] * op.get_apply().get_rank())),
                },
                result_types=[builtin.IndexType()],
            )
            for i in mapping
        ]

        off_const_ops: list[Operation] = []
        memref_load_args: list[BlockArgument | OpResult] = []

        # This will apply an offset to the index if one is required
        # (e.g the offset is not zero), otherwise will use the index value directly
        for arg, x in zip(args, memref_offset):
            if x != 0:
                constant_op = arith.Constant.from_int_and_width(x, builtin.IndexType())
                add_op = arith.Addi(arg, constant_op)
                memref_load_args.append(add_op.results[0])
                off_const_ops += [constant_op, add_op]
            else:
                memref_load_args.append(arg.idx)

        load = memref.Load.get(op.temp, memref_load_args)

        rewriter.insert_op_before_matched_op(args)
        rewriter.replace_matched_op([*off_const_ops, load], [load.res])


@dataclass
class StencilStoreToSubview(RewritePattern):
    return_targets: dict[ApplyOp, list[SSAValue | None]]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        field = op.field
        assert isa(field.type, FieldType[Attribute])
        assert isa(field.type.bounds, StencilBoundsAttr)
        temp = op.temp
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
            rewriter.insert_op(subview, InsertPoint.after(field.owner))
        else:
            rewriter.insert_op(subview, InsertPoint.at_start(field.owner))

        rewriter.erase_matched_op()

        update_return_target(self.return_targets, field, subview.result)


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


class CombineOpCleanup(RewritePattern):
    """
    Just remove `stencil.combine`s as they are just used for return target analysis.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CombineOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


def _get_use_target(use: Use) -> SSAValue | None:
    """
    Given the use of a `stencil.temp`, return the target `stencil.field` of this use.
    """
    match store := use.operation:
        case StoreOp():
            # If it's a store, just return the field it is stored to. It will be lowered
            # to the corresponding memref.
            return store.field
        case BufferOp():
            # If it's a buffer, return the buffer itself. It will be lowered to the
            # allocated memref.
            return store.temp
        case CombineOp():
            # If it's a combine, recurse to find the target of the combined
            # `stencil.temp`
            if use.index < len(store.lower):
                # If it's the nth lower arg, the combined temp is the nth combined.
                temp = store.results[use.index]
            elif use.index < len(store.lower) + len(store.upper):
                # If it's the nth upper arg, the combined temp is the nth combined.
                temp = store.results[use.index - len(store.lower)]
            elif use.index < len(store.lower) + len(store.upper) + len(store.lowerext):
                # If it's the nth lowerext arg, the combined temp is the (lower+n)th
                # combined.
                temp = store.results[use.index - len(store.lower)]
            else:
                temp = store.results[use.index - len(store.lower) - len(store.lowerext)]
                # If it's the nth upperext arg, the combined temp is the
                # (lower+lowerext+n)th combined.
            temp_uses = temp.uses
            match len(temp_uses):
                case 0:
                    return None
                case 1:
                    target = _get_use_target(list(temp_uses)[0])
                    return target
                case _:
                    raise ValueError("Each stencil result should be stored only once.")
        case _:
            # Should be unreachable
            raise ValueError(f"Unexpected store type {store}")


def return_target_analysis(module: builtin.ModuleOp):
    return_targets: dict[ApplyOp, list[SSAValue | None]] = {}

    for op in module.walk():
        if not isinstance(op, ReturnOp):
            continue

        apply = op.parent_op()
        assert isinstance(apply, ApplyOp)

        return_targets[apply] = []
        for res in list(apply.res):
            store = [
                use
                for use in list(res.uses)
                if isinstance(use.operation, StoreOp | BufferOp | CombineOp)
            ]

            if len(store) > 1:
                warn("Each stencil result should be stored only once.")
                continue

            elif len(store) == 0:
                field = None
            else:
                field = _get_use_target(store[0])

            return_targets[apply].append(field)

    return return_targets


class StencilTypeConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: StencilType[Attribute]) -> MemRefType[Attribute]:
        return StencilToMemRefType(typ)


class ResultTypeConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: ResultType) -> Attribute:
        return typ.elem


@dataclass(frozen=True)
class ConvertStencilToLLMLIRPass(ModulePass):
    name = "convert-stencil-to-ll-mlir"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        return_targets: dict[ApplyOp, list[SSAValue | None]] = return_target_analysis(
            op
        )

        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ApplyOpToParallel(return_targets),
                    BufferOpToMemref(return_targets),
                    StencilStoreToSubview(return_targets),
                    CastOpToMemref(),
                    LoadOpToMemref(),
                    AccessOpToMemref(),
                    ReturnOpToMemref(return_targets),
                    TrivialExternalLoadOpCleanup(),
                    TrivialExternalStoreOpCleanup(),
                ]
            ),
            apply_recursively=True,
            walk_reverse=True,
            walk_regions_first=True,
        )
        the_one_pass.rewrite_module(op)
        type_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CombineOpCleanup(),
                    StencilTypeConversion(recursive=True),
                    ResultTypeConversion(recursive=True),
                ]
            ),
            walk_reverse=True,
        )
        type_pass.rewrite_module(op)
