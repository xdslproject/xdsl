from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from xdsl.context import MLContext
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


def offset_calculations(
    memref_type: memref.MemRefType[Any], indices: Iterable[SSAValue]
) -> tuple[list[Operation], SSAValue]:
    """Get operations calculating an offset which needs to be added to memref's base pointer to access an element referenced by indices."""

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
                operands=[],
                result_types=[builtin.IndexType()],
                properties={"elem_type": memref_type.element_type},
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

    ops: list[Operation] = [
        memref_ptr := ptr.ToPtrOp(
            operands=[target_memref], result_types=[ptr.PtrType()]
        )
    ]

    if not indices:
        return ops, memref_ptr.res

    offset_ops, offset = offset_calculations(memref_type, indices)
    ops = offset_ops + ops
    ops.append(
        target_ptr := ptr.PtrAddOp(
            operands=[memref_ptr, offset], result_types=[ptr.PtrType()]
        )
    )

    target_ptr.result.name_hint = "offset_pointer"
    return ops, target_ptr.result


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_type = cast(memref.MemRefType[Any], op_memref_type)

        ops, target_ptr = get_target_ptr(op.memref, memref_type, op.indices)
        ops.append(ptr.StoreOp(operands=[target_ptr, op.value]))

        rewriter.replace_matched_op(ops)


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_type = cast(memref.MemRefType[Any], op_memref_type)

        ops, target_ptr = get_target_ptr(op.memref, memref_type, op.indices)
        ops.append(
            load_result := ptr.LoadOp(
                operands=[target_ptr], result_types=[memref_type.element_type]
            )
        )

        rewriter.replace_matched_op(ops, new_results=[load_result.res])


@dataclass
class LowerMemrefFuncOpPattern(RewritePattern):
    """
    Rewrites function arguments of MemRefType to PtrType - leaves IR in invalid state(?)

    Args:
        RewritePattern (_type_): _description_
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
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

        for arg in op.args:
            if not isinstance(arg_type := arg.type, memref.MemRefType):
                continue

            old_type = cast(memref.MemRefType[Attribute], arg_type)
            arg.type = ptr.PtrType()

            if not arg.uses:
                continue

            rewriter.insert_op(
                cast_op := builtin.UnrealizedConversionCastOp.get([arg], [old_type]),
                insert_point,
            )
            arg.replace_by_if(cast_op.results[0], lambda x: x.operation != cast_op)


@dataclass
class LowerMemrefFuncReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter, /):
        if not any(isinstance(arg.type, memref.MemRefType) for arg in op.arguments):
            return

        new_arguments = [
            (arg.owner.inputs[0], arg.owner)
            if isinstance(arg.owner, builtin.UnrealizedConversionCastOp)
            and isinstance(arg.owner.inputs[0].type, ptr.PtrType)
            else (arg, None)
            for arg in op.arguments
        ]

        rewriter.replace_matched_op(func.ReturnOp(*(arg for (arg, _) in new_arguments)))

        for _, cast_op in new_arguments:
            if cast_op is not None and not cast_op.results[0].uses:
                rewriter.erase_op(cast_op)


@dataclass
class LowerMemrefFuncCallPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter, /):
        if not any(
            isinstance(arg.type, memref.MemRefType) for arg in op.arguments
        ) and not any(isinstance(type, memref.MemRefType) for type in op.result_types):
            return

        new_arguments = [
            (arg.owner.inputs[0], arg.owner)
            if isinstance(arg.owner, builtin.UnrealizedConversionCastOp)
            and isinstance(arg.owner.inputs[0].type, ptr.PtrType)
            else (arg, None)
            for arg in op.arguments
        ]
        new_return_types = [
            ptr.PtrType() if isinstance(type, memref.MemRefType) else type
            for type in op.result_types
        ]
        rewriter.replace_matched_op(
            func.CallOp(
                op.callee, [arg for (arg, _) in new_arguments], new_return_types
            )
        )

        for _, cast_op in new_arguments:
            if cast_op is not None and not cast_op.results[0].uses:
                rewriter.erase_op(cast_op)


class ReconcilePtrCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: builtin.UnrealizedConversionCastOp, rewriter: PatternRewriter, /
    ):
        if not isinstance(op.inputs[0].type, ptr.PtrType):
            return

        cast_ops = [use.operation for use in op.outputs[0].uses]
        if not all(isinstance(op, ptr.ToPtrOp) for op in cast_ops):
            return

        for cast_op in cast_ops:
            cast_op.results[0].replace_by(op.inputs[0])
            rewriter.erase_op(cast_op)

        rewriter.erase_op(op)


@dataclass(frozen=True)
class ConvertMemrefToPtr(ModulePass):
    name = "convert-memref-to-ptr"

    convert_func_args: bool = False

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertStoreOp(), ConvertLoadOp()])
        ).rewrite_module(op)

        if self.convert_func_args:
            PatternRewriteWalker(
                GreedyRewritePatternApplier(
                    [
                        LowerMemrefFuncOpPattern(),
                        LowerMemrefFuncCallPattern(),
                        LowerMemrefFuncReturnPattern(),
                        ReconcilePtrCasts(),
                    ]
                )
            ).rewrite_module(op)
