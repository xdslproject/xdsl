"""
This file implements a partial lowering of Toy operations to a combination of
affine loops, memref operations and standard operations. This lowering
expects that all calls have been inlined, and all shapes have been resolved.
"""

from collections.abc import Callable, Sequence
from itertools import product
from typing import TypeAlias, cast

from typing_extensions import TypeVar

from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, memref, printf
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    ModuleOp,
    ShapedType,
    f64,
)
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..dialects import toy

# region Helpers

MemRefTypeF64: TypeAlias = memref.MemRefType[Float64Type]


def convert_tensor_to_memref(type: toy.TensorTypeF64) -> MemRefTypeF64:
    """
    Convert the given RankedTensorType into the corresponding MemRefType.
    """
    return memref.MemRefType(f64, type.shape)


def insert_alloc_and_dealloc(
    type: MemRefTypeF64, op: Operation, rewriter: PatternRewriter
) -> memref.AllocOp:
    """
    Insert an allocation and deallocation for the given MemRefType.
    """
    block = op.parent

    assert block is not None, "Operation to be allocated must be in a block"
    assert block.last_op is not None

    # Make sure to allocate at the beginning of the block.
    alloc = memref.AllocOp.get(type.element_type, None, type.shape)
    rewriter.insert_op(alloc, InsertPoint.at_start(block))

    # Make sure to deallocate this alloc at the end of the block. This is fine as toy
    # functions have no control flow.
    dealloc = memref.DeallocOp.get(alloc)
    rewriter.insert_op(dealloc, InsertPoint.before(block.last_op))

    return alloc


_ValueRange = Sequence[SSAValue]
_AffineForOpBodyBuilderFn: TypeAlias = Callable[[Builder, SSAValue, _ValueRange], None]


def build_affine_for(
    builder: Builder,
    lb_operands: _ValueRange,
    lb_map: affine.AffineMap,
    ub_operands: _ValueRange,
    ub_map: affine.AffineMap,
    step: int,
    iter_args: _ValueRange,
    body_builder_fn: _AffineForOpBodyBuilderFn,
) -> affine.ForOp:
    """
    `body_builder_fn` is used to build the body of affine.for.
    """

    assert len(lb_operands) == lb_map.num_dims, (
        "lower bound operand count does not match the affine map"
    )
    assert len(ub_operands) == ub_map.num_dims, (
        "upper bound operand count does not match the affine map"
    )
    assert step > 0, "step has to be a positive integer constant"

    # Create a region and a block for the body.
    # The argument of the region is the loop induction variable.
    block_arg_types = (IndexType(), *(arg.type for arg in iter_args))
    block = Block(arg_types=block_arg_types)
    induction_var, *rest = block.args
    region = Region(block)

    op = affine.ForOp.from_region(
        lb_operands,
        ub_operands,
        iter_args,
        tuple(iter_arg.type for iter_arg in iter_args),
        affine.AffineMapAttr(lb_map),
        affine.AffineMapAttr(ub_map),
        region,
        step,
    )
    builder.insert(op)
    body_builder_fn(Builder(InsertPoint.at_end(block)), induction_var, rest)
    return op


def build_affine_for_const(
    builder: Builder,
    lb: int,
    ub: int,
    step: int,
    iter_args: _ValueRange,
    body_builder_fn: _AffineForOpBodyBuilderFn,
) -> affine.ForOp:
    return build_affine_for(
        builder,
        (),
        affine.AffineMap.constant_map(lb),
        (),
        affine.AffineMap.constant_map(ub),
        step,
        iter_args,
        body_builder_fn,
    )


LoopIterationFn: TypeAlias = Callable[[Builder, _ValueRange, _ValueRange], SSAValue]
"""
This defines the function type used to process an iteration of a lowered loop. It takes as
input an OpBuilder, an range of memRefOperands corresponding to the operands of the input
operation, and the range of loop induction variables for the iteration. It returns a value
to store at the current index of the iteration.
"""

_BoundT = TypeVar("_BoundT")

_BodyBuilderFn: TypeAlias = Callable[[Builder, _ValueRange], None]
_LoopCreatorFn: TypeAlias = Callable[
    [Builder, _BoundT, _BoundT, int, _AffineForOpBodyBuilderFn],
    affine.ForOp,
]


def build_affine_loop_nest_impl(
    builder: Builder,
    lbs: Sequence[_BoundT],
    ubs: Sequence[_BoundT],
    steps: Sequence[int],
    body_builder_fn: _BodyBuilderFn,
    loop_creator_fn: _LoopCreatorFn[_BoundT],
) -> None:
    """
    Corresponds to affine::buildAffineLoopNestImpl in MLIR
    """

    assert len(lbs) == len(ubs), "Mismatch in number of arguments"
    assert len(lbs) == len(steps), "Mismatch in number of arguments"

    # if there are no loops to be constructed, construct the body anyway
    if not lbs:
        body_builder_fn(builder, ())
        return

    # Create the loops iteratively and store the induction variables.

    ivs: list[SSAValue] = []

    e = len(lbs)
    for i in range(e):
        # Callback for creating the loop body, always creates the terminator.
        def body(nested_builder: Builder, iv: SSAValue, iter_args: _ValueRange):
            nonlocal ivs

            ivs.append(iv)

            # In the innermost loop, call the body builder
            if i == e - 1:
                body_builder_fn(nested_builder, ivs)

            nested_builder.insert(affine.YieldOp.get())

        # Delegate actual loop creation to the callback in order to dispatch
        # between constant- and variable-bound loops.

        loop = loop_creator_fn(builder, lbs[i], ubs[i], steps[i], body)
        builder = Builder(InsertPoint(loop.body.block, loop.body.block.first_op))


def build_affine_loop_from_constants(
    builder: Builder,
    lb: int,
    ub: int,
    step: int,
    body_builder_fn: _AffineForOpBodyBuilderFn,
) -> affine.ForOp:
    """
    Creates an affine loop from the bounds known to be constants.
    """
    return build_affine_for_const(builder, lb, ub, step, (), body_builder_fn)


def build_affine_loop_from_values(
    builder: Builder,
    lb: SSAValue,
    ub: SSAValue,
    step: int,
    body_builder_fn: _AffineForOpBodyBuilderFn,
) -> affine.ForOp:
    """
    Creates an affine loop from the bounds that may or may not be constants.
    """
    lb_const = lb.owner
    ub_const = ub.owner

    if (
        isinstance(lb_const, arith.ConstantOp)
        and isinstance(lb_const_value := lb_const.value, IntegerAttr)
        and isinstance(ub_const, arith.ConstantOp)
        and isinstance(ub_const_value := ub_const.value, IntegerAttr)
    ):
        lb_val = lb_const_value.value.data
        ub_val = ub_const_value.value.data
        return build_affine_loop_from_constants(
            builder, lb_val, ub_val, step, body_builder_fn
        )
    return build_affine_for(
        builder,
        (lb,),
        affine.AffineMap(1, 0, (affine.AffineExpr.dimension(0),)),
        (ub,),
        affine.AffineMap(1, 0, (affine.AffineExpr.dimension(0),)),
        step,
        (),
        body_builder_fn,
    )


def build_affine_loop_nest_const(
    builder: Builder,
    lbs: Sequence[int],
    ubs: Sequence[int],
    steps: Sequence[int],
    body_builder_fn: _BodyBuilderFn,
) -> None:
    build_affine_loop_nest_impl(
        builder, lbs, ubs, steps, body_builder_fn, build_affine_loop_from_constants
    )


def build_affine_loop_nest(
    builder: Builder,
    lbs: Sequence[SSAValue],
    ubs: Sequence[SSAValue],
    steps: Sequence[int],
    body_builder_fn: _BodyBuilderFn,
) -> None:
    build_affine_loop_nest_impl(
        builder, lbs, ubs, steps, body_builder_fn, build_affine_loop_from_values
    )


def lower_op_to_loops(
    op: toy.AddOp | toy.MulOp | toy.TransposeOp,
    operands: _ValueRange,
    rewriter: PatternRewriter,
    process_iteration: LoopIterationFn,
):
    tensor_type = cast(toy.TensorTypeF64, op.res.type)

    # insert an allocation and deallocation for the result of this operation.
    memref_type = convert_tensor_to_memref(tensor_type)
    alloc = insert_alloc_and_dealloc(memref_type, op, rewriter)

    # Create a nest of affine loops, with one loop per dimension of the shape.
    # The buildAffineLoopNest function takes a callback that is used to construct the body
    # of the innermost loop given a builder, a location and a range of loop induction
    # variables.

    rank = tensor_type.get_num_dims()
    lower_bounds = tuple(0 for _ in range(rank))
    steps = tuple(1 for _ in range(rank))

    def impl_loop(nested_builder: Builder, ivs: _ValueRange):
        # Call the processing function with the rewriter, the memref operands, and the
        # loop induction variables. This function will return the value to store at the
        # current index.
        value_to_store = process_iteration(nested_builder, operands, ivs)
        store_op = affine.StoreOp(value_to_store, alloc.memref, ivs)
        nested_builder.insert(store_op)

    builder = Builder(InsertPoint.before(op))
    build_affine_loop_nest_const(
        builder, lower_bounds, tensor_type.get_shape(), steps, impl_loop
    )
    # Replace this operation with the generated alloc.

    op.res.replace_by(alloc.memref)
    rewriter.erase_op(op)


# endregion Helpers

# region RewritePatterns


class AddOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.AddOp, rewriter: PatternRewriter):
        def body(
            builder: Builder, memref_operands: _ValueRange, loop_ivs: _ValueRange
        ) -> SSAValue:
            # Generate loads for the element of 'lhs' and 'rhs' at the inner loop.
            loaded_lhs = builder.insert(affine.LoadOp(op.lhs, loop_ivs))
            loaded_rhs = builder.insert(affine.LoadOp(op.rhs, loop_ivs))
            new_binop = builder.insert(arith.AddfOp(loaded_lhs, loaded_rhs))
            return new_binop.result

        lower_op_to_loops(op, op.operands, rewriter, body)


class MulOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.MulOp, rewriter: PatternRewriter):
        def body(
            builder: Builder, memref_operands: _ValueRange, loop_ivs: _ValueRange
        ) -> SSAValue:
            # Generate loads for the element of 'lhs' and 'rhs' at the inner loop.
            loaded_lhs = builder.insert(affine.LoadOp(op.lhs, loop_ivs))
            loaded_rhs = builder.insert(affine.LoadOp(op.rhs, loop_ivs))
            new_binop = builder.insert(arith.MulfOp(loaded_lhs, loaded_rhs))
            return new_binop.result

        lower_op_to_loops(op, op.operands, rewriter, body)


class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.ConstantOp, rewriter: PatternRewriter):
        constant_value = op.value

        # When lowering the constant operation, we allocate and assign the constant
        # values to a corresponding memref allocation.

        tensor_type = op.res.type
        memref_type = convert_tensor_to_memref(tensor_type)
        alloc = insert_alloc_and_dealloc(memref_type, op, rewriter)

        value_shape = memref_type.get_shape()

        # Scalar constant values for elements of the tensor
        constants: list[arith.ConstantOp] = [
            arith.ConstantOp(FloatAttr(i, f64)) for i in constant_value.get_values()
        ]

        # n-d indices of elements
        _indices = product(*(range(d) for d in value_shape))

        # For each n-d index into the tensor, store the corresponding scalar
        stores = [
            affine.StoreOp(
                constants[offset].result,
                alloc.memref,
                (),
                map=affine.AffineMapAttr(affine.AffineMap.point_map(*index)),
            )
            for offset, index in enumerate(_indices)
        ]

        # Insert constants used before the alloc, not before matched operation
        rewriter.insert_op(constants, InsertPoint.before(alloc))

        # Replace the constant by the stores, and its result by the allocated value
        rewriter.replace_op(op, stores, (alloc.memref,))


class FuncOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.FuncOp, rewriter: PatternRewriter):
        name = op.sym_name.data

        # We only lower the main function as we expect that all other functions
        # have been inlined.
        assert name == "main", "Only support lowering main function for now"

        # Verify that the given main has no inputs and results.
        if op.function_type.inputs or op.function_type.outputs:
            raise ValueError("expected 'main' to have 0 inputs and 0 results")

        # Create a new non-toy function, with the same region.
        region = op.regions[0]

        new_op = func.FuncOp(
            name, op.function_type, rewriter.move_region_contents_to_new_regions(region)
        )

        rewriter.replace_op(op, new_op)


class PrintOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.PrintOp, rewriter: PatternRewriter):
        assert isinstance(shaped_type := op.input.type, ShapedType)
        shape = shaped_type.get_shape()

        format_str = "{}"

        for dim in reversed(shape):
            format_str = "[" + ", ".join([format_str] * dim) + "]"

        new_vals: list[SSAValue] = []

        for indices in product(*(range(dim) for dim in shape)):
            rewriter.insert_op(
                load := affine.LoadOp(
                    op.input,
                    (),
                    AffineMapAttr(AffineMap.from_callable(lambda: indices)),
                )
            )
            new_vals.append(load.result)

        rewriter.replace_op(op, printf.PrintFormatOp(format_str, *new_vals))


class ReturnOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.ReturnOp, rewriter: PatternRewriter):
        assert op.input is None, (
            "During this lowering, we expect that all function calls have been inlined."
        )

        rewriter.replace_op(op, func.ReturnOp())


class TransposeOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: toy.TransposeOp, rewriter: PatternRewriter):
        def body(
            builder: Builder, mem_ref_operands: _ValueRange, loop_ivs: _ValueRange
        ) -> SSAValue:
            # Transpose the elements by generating a load from the reverse indices.
            load_op = affine.LoadOp(op.arg, tuple(reversed(loop_ivs)))
            builder.insert(load_op)
            return load_op.result

        lower_op_to_loops(op, op.operands, rewriter, body)


# endregion RewritePatterns


class LowerToyPass(ModulePass):
    """
    A pass for lowering operations in the Toy dialect to built-in dialects.
    """

    name = "lower-toy"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    ConstantOpLowering(),
                    FuncOpLowering(),
                    MulOpLowering(),
                    PrintOpLowering(),
                    ReturnOpLowering(),
                    TransposeOpLowering(),
                ]
            )
        ).rewrite_module(op)
