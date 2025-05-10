from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, ptr
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import Any
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa


def offset_calculations(
    memref_type: memref.MemRefType[Any], indices: Iterable[SSAValue]
) -> tuple[list[Operation], SSAValue]:
    """
    Get operations calculating an offset which needs to be added to memref's base
    pointer to access an element referenced by indices.
    """

    assert isinstance(memref_type.element_type, builtin.FixedBitwidthType)

    match memref_type.layout:
        case builtin.NoneAttr():
            strides = builtin.ShapedType.strides_for_shape(memref_type.get_shape())
        case builtin.StridedLayoutAttr():
            strides = memref_type.layout.get_strides()
        case _:
            raise DiagnosticException(f"Unsupported layout type {memref_type.layout}")

    ops: list[Operation] = []

    head: SSAValue | None = None

    for index, stride in zip(indices, strides, strict=True):
        # Calculate the offset that needs to be added through the index of the current
        # dimension.
        increment = index
        match stride:
            case None:
                raise DiagnosticException(
                    f"MemRef {memref_type} with dynamic stride is not yet implemented"
                )
            case 1:
                # Stride 1 is a noop making the index equal to the offset.
                pass
            case _:
                # Otherwise, multiply the stride (which by definition is the number of
                # elements required to be skipped when incrementing that dimension).
                ops.extend(
                    (
                        stride_op := arith.ConstantOp.from_int_and_width(
                            stride, builtin.IndexType()
                        ),
                        offset_op := arith.MuliOp(increment, stride_op),
                    )
                )
                stride_op.result.name_hint = "pointer_dim_stride"
                offset_op.result.name_hint = "pointer_dim_offset"

                increment = offset_op.result

        if head is None:
            # First iteration.
            head = increment
            continue

        # Otherwise sum up the products.
        add_op = arith.AddiOp(head, increment)
        add_op.result.name_hint = "pointer_dim_stride"
        ops.append(add_op)
        head = add_op.result

    if head is None:
        raise DiagnosticException("Got empty indices for offset calculations.")

    ops.extend(
        [
            bytes_per_element_op := ptr.TypeOffsetOp(
                memref_type.element_type, builtin.IndexType()
            ),
            final_offset := arith.MuliOp(head, bytes_per_element_op),
        ]
    )

    bytes_per_element_op.offset.name_hint = "bytes_per_element"
    final_offset.result.name_hint = "scaled_pointer_offset"

    return ops, final_offset.result


def get_target_ptr(
    target_memref: SSAValue,
    memref_type: memref.MemRefType[Any],
    indices: Iterable[SSAValue],
) -> tuple[list[Operation], SSAValue]:
    """Get operations returning a pointer to an element of a memref referenced by indices."""

    ops: list[Operation] = [memref_ptr := ptr.ToPtrOp(target_memref)]

    if not indices:
        return ops, memref_ptr.res

    offset_ops, offset = offset_calculations(memref_type, indices)
    ops = offset_ops + ops
    ops.append(target_ptr := ptr.PtrAddOp(memref_ptr.res, offset))

    target_ptr.result.name_hint = "offset_pointer"
    return ops, target_ptr.result


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)

        ops, target_ptr = get_target_ptr(op.memref, memref_type, op.indices)
        ops.append(ptr.StoreOp(target_ptr, op.value))

        rewriter.replace_matched_op(ops)


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        ops, target_ptr = get_target_ptr(op.memref, memref_type, op.indices)
        ops.append(load_result := ptr.LoadOp(target_ptr, memref_type.element_type))
        rewriter.replace_matched_op(ops, new_results=[load_result.res])


@dataclass
class ConvertSubviewOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.source.type, memref.MemRefType)

        static_offsets = cast(tuple[int, ...], op.static_offsets.get_values())
        offsets: list[SSAValue] = []

        for idx, offset in enumerate(static_offsets):
            if offset == memref.SubviewOp.DYNAMIC_INDEX:
                offsets.append(op.offsets[idx])
            else:
                rewriter.insert_op_before_matched_op(
                    const_op := arith.ConstantOp(
                        builtin.IntegerAttr(offset, builtin.IndexType())
                    )
                )
                const_op.result.name_hint = f"c{offset}"
                offsets.append(const_op.result)

        # We can treat a subview as getting a pointer to the first element in the subview.
        ops, target_ptr = get_target_ptr(op.source, memref_type, offsets)

        rewriter.replace_matched_op(
            (
                *ops,
                builtin.UnrealizedConversionCastOp.get([target_ptr], [op.result.type]),
            )
        )


@dataclass
class LowerMemRefFuncOpPattern(RewritePattern):
    """
    Rewrites function arguments of MemRefType to PtrType.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        # rewrite function declaration
        new_input_types = [
            ptr.PtrType() if isinstance(arg, builtin.MemRefType) else arg
            for arg in op.function_type.inputs
        ]
        new_output_types = [
            ptr.PtrType() if isinstance(arg, builtin.MemRefType) else arg
            for arg in op.function_type.outputs
        ]
        op.function_type = func.FunctionType.from_lists(
            new_input_types,
            new_output_types,
        )

        if op.is_declaration:
            return

        insert_point = InsertPoint.at_start(op.body.blocks[0])

        # rewrite arguments
        for arg in op.args:
            if not isinstance(arg_type := arg.type, memref.MemRefType):
                continue

            old_type = cast(memref.MemRefType[Attribute], arg_type)
            arg = rewriter.replace_value_with_new_type(arg, ptr.PtrType())

            if not arg.uses:
                continue

            rewriter.insert_op(
                cast_op := ptr.FromPtrOp(arg, old_type),
                insert_point,
            )
            arg.replace_by_if(cast_op.res, lambda x: x.operation is not cast_op)


@dataclass
class LowerMemRefFuncReturnPattern(RewritePattern):
    """
    Rewrites all `memref` arguments to `func.return` into `ptr.PtrType`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter, /):
        if not any(isinstance(arg.type, memref.MemRefType) for arg in op.arguments):
            return

        new_arguments: list[SSAValue] = []

        # insert `memref -> ptr` casts for memref return values
        for argument in op.arguments:
            if isinstance(argument.type, memref.MemRefType):
                rewriter.insert_op_before_matched_op(cast_op := ptr.ToPtrOp(argument))
                new_arguments.append(cast_op.res)
                cast_op.res.name_hint = argument.name_hint
            else:
                new_arguments.append(argument)

        rewriter.replace_matched_op(func.ReturnOp(*new_arguments))


@dataclass
class LowerMemRefFuncCallPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter, /):
        if not any(
            isinstance(arg.type, memref.MemRefType) for arg in op.arguments
        ) and not any(isinstance(type, memref.MemRefType) for type in op.result_types):
            return

        # rewrite arguments
        new_arguments: list[SSAValue] = []

        # insert `memref -> ptr` casts for memref arguments values, if necessary
        for argument in op.arguments:
            if isinstance(argument.type, memref.MemRefType):
                rewriter.insert_op_before_matched_op(cast_op := ptr.ToPtrOp(argument))
                new_arguments.append(cast_op.res)
                cast_op.res.name_hint = argument.name_hint
            else:
                new_arguments.append(argument)

        new_return_types = [
            ptr.PtrType() if isinstance(type, memref.MemRefType) else type
            for type in op.result_types
        ]

        new_ops: list[Operation] = [
            call_op := func.CallOp(op.callee, new_arguments, new_return_types)
        ]
        new_results = list(call_op.results)

        #  insert `ptr -> memref` casts for return values, if necessary
        for i, (new_result, old_result) in enumerate(zip(call_op.results, op.results)):
            new_result.name_hint = old_result.name_hint
            if isa(old_result.type, memref.MemRefType):
                new_ops.append(cast_op := ptr.FromPtrOp(new_result, old_result.type))
                new_results[i] = cast_op.res

        rewriter.replace_matched_op(new_ops, new_results)


@dataclass(frozen=True)
class ConvertMemRefToPtr(ModulePass):
    name = "convert-memref-to-ptr"

    lower_func: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStoreOp(),
                    ConvertLoadOp(),
                    ConvertSubviewOp(),
                ]
            )
        ).rewrite_module(op)

        if self.lower_func:
            PatternRewriteWalker(
                GreedyRewritePatternApplier(
                    [
                        LowerMemRefFuncOpPattern(),
                        LowerMemRefFuncCallPattern(),
                        LowerMemRefFuncReturnPattern(),
                    ]
                )
            ).rewrite_module(op)
