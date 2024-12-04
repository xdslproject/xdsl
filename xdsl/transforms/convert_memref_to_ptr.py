from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, memref, ptr
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import Any
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
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


@dataclass(frozen=True)
class ConvertMemrefToPtr(ModulePass):
    name = "convert-memref-to-ptr"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertStoreOp(), ConvertLoadOp()])
        )
        the_one_pass.rewrite_module(op)
