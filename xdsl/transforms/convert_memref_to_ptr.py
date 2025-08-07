from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from xdsl.builder import Builder
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
from xdsl.utils.exceptions import DiagnosticException, PassFailedException
from xdsl.utils.hints import isa

_index_type = builtin.IndexType()


def get_constant_strides(memref_type: builtin.MemRefType) -> Sequence[int]:
    """
    If the memref has constant strides and offset, returns them, otherwise raises a
    DiagnosticException.
    """
    match memref_type.layout:
        case builtin.NoneAttr():
            strides = builtin.ShapedType.strides_for_shape(memref_type.get_shape())
        case builtin.StridedLayoutAttr():
            strides = memref_type.layout.get_strides()
            if None in strides:
                raise DiagnosticException(
                    f"MemRef {memref_type} with dynamic stride is not yet implemented"
                )
            strides = cast(Sequence[int], strides)
        case _:
            raise DiagnosticException(f"Unsupported layout type {memref_type.layout}")
    return strides


def get_strides_offset(
    indices: Iterable[SSAValue], strides: Sequence[int], builder: Builder
) -> SSAValue | None:
    """
    Given SSA values for indices, and constant strides, insert the arithmetic ops that
    create the combined index offset.
    The length of indices and strides must be the same.
    Strides must not contain `-1`.
    """
    head: SSAValue | None = None

    for index, stride in zip(indices, strides, strict=True):
        # Calculate the offset that needs to be added through the index of the current
        # dimension.
        increment = index

        # Stride 1 is a noop making the index equal to the offset.
        if stride != 1:
            # Otherwise, multiply the stride (which by definition is the number of
            # elements required to be skipped when incrementing that dimension).
            stride_op = builder.insert_op(
                arith.ConstantOp.from_int_and_width(stride, _index_type)
            )
            offset_op = builder.insert_op(arith.MuliOp(increment, stride_op))
            stride_op.result.name_hint = "pointer_dim_stride"
            offset_op.result.name_hint = "pointer_dim_offset"

            increment = offset_op.result

        if head is None:
            # First iteration.
            head = increment
            continue

        # Otherwise sum up the products.
        add_op = builder.insert_op(arith.AddiOp(head, increment))
        add_op.result.name_hint = "pointer_dim_stride"
        head = add_op.result

    return head


def get_offset_pointer(
    offset_in_indices: SSAValue,
    pointer: SSAValue,
    element_type: Attribute,
    builder: Builder,
) -> SSAValue:
    """
    Returns the offset in indices scaled by the size of element_type.
    """
    bytes_per_element_op = builder.insert_op(
        ptr.TypeOffsetOp(element_type, _index_type)
    )
    final_offset = builder.insert_op(
        arith.MuliOp(offset_in_indices, bytes_per_element_op)
    )
    target_ptr = builder.insert_op(ptr.PtrAddOp(pointer, final_offset.result))

    bytes_per_element_op.offset.name_hint = "bytes_per_element"
    final_offset.result.name_hint = "scaled_pointer_offset"
    target_ptr.result.name_hint = "offset_pointer"
    pointer = target_ptr.result
    return pointer


def get_target_ptr(
    target_memref: SSAValue,
    memref_type: memref.MemRefType[Any],
    indices: Iterable[SSAValue],
    builder: Builder,
) -> SSAValue:
    """
    Get operations returning a pointer to an element of a memref referenced by indices.
    """

    memref_ptr = builder.insert_op(ptr.ToPtrOp(target_memref))
    pointer = memref_ptr.res
    pointer.name_hint = target_memref.name_hint

    if indices:
        strides = get_constant_strides(memref_type)
        head = get_strides_offset(indices, strides, builder)

        if head is not None:
            pointer = get_offset_pointer(
                head, pointer, memref_type.element_type, builder
            )

    return pointer


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        target_ptr = get_target_ptr(op.memref, memref_type, op.indices, rewriter)
        rewriter.replace_matched_op(ptr.StoreOp(target_ptr, op.value))


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        target_ptr = get_target_ptr(op.memref, memref_type, op.indices, rewriter)
        rewriter.replace_matched_op(ptr.LoadOp(target_ptr, memref_type.element_type))


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

            old_type = cast(memref.MemRefType, arg_type)
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


class LowerExtractStridedMetadataOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref.ExtractStridedMetaDataOp, rewriter: PatternRewriter, /
    ):
        if (
            op.offset.has_uses()
            or any(size.has_uses() for size in op.sizes)
            or any(stride.has_uses() for stride in op.strides)
        ):
            raise PassFailedException(
                "Cannot lower op with uses of offset, sizes, or strides."
            )
        rewriter.insert_op(
            (
                to_ptr := ptr.ToPtrOp(op.source),
                from_ptr := ptr.FromPtrOp(to_ptr.res, op.base_buffer.type),
            )
        )
        rewriter.replace_all_uses_with(op.base_buffer, from_ptr.res)


class LowerReinterpretCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref.ReinterpretCastOp, rewriter: PatternRewriter, /
    ):
        static_offsets = op.static_offsets.get_values()
        if len(static_offsets) != 1:
            raise NotImplementedError
        to_ptr = rewriter.insert_op(ptr.ToPtrOp(op.source))
        if static_offsets[0] == memref.ReinterpretCastOp.DYNAMIC_INDEX:
            offset = op.offsets[0]
        else:
            offset = rewriter.insert_op(
                arith.ConstantOp(builtin.IntegerAttr(static_offsets[0], _index_type))
            ).result
        new_ptr = get_offset_pointer(
            offset, to_ptr.res, op.result.type.element_type, rewriter
        )
        rewriter.replace_matched_op(ptr.FromPtrOp(new_ptr, op.result.type))


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
                    LowerExtractStridedMetadataOp(),
                    LowerReinterpretCastOp(),
                ]
            ),
            apply_recursively=False,
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
