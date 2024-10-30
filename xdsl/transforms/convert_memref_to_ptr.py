from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, memref, ptr
from xdsl.dialects.builtin import i32
from xdsl.dialects.ptr import TypeOffsetOp
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


# I think we also need to pass the adress width.
def offset_calculations(
    memref_type: memref.MemRefType[Any], indices: Iterable[SSAValue]
) -> tuple[list[Operation], SSAValue]:
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
                        stride_op := arith.Constant.from_int_and_width(
                            stride, builtin.IndexType()
                        ),
                        offset_op := arith.Muli(increment, stride_op),
                    )
                )
                increment = offset_op.result

        if head is None:
            # First iteration.
            head = increment
            continue

        # Otherwise sum up the products.
        add_op = arith.Addi(head, increment)
        ops.append(add_op)
        head = add_op.result

    if head is None:
        ops.append(const_op := arith.Constant.from_int_and_width(0, i32))
        return ops, const_op.result

    ops.extend(
        [
            hack_to_get_type := arith.Constant.from_int_and_width(0, i32),
            bytes_per_element_op := TypeOffsetOp(
                operands=[hack_to_get_type], result_types=[builtin.IndexType()]
            ),
            final_offset := arith.Muli(head, bytes_per_element_op),
        ]
    )

    return ops, final_offset.result


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter, /):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_type = cast(memref.MemRefType[Any], op_memref_type)

        ops, offset = offset_calculations(memref_type, op.indices)

        ops.extend(
            [
                memref_ptr := memref.ToPtrOp(
                    operands=[op.memref], result_types=[ptr.PtrType()]
                ),
                target_ptr := ptr.PtrAddOp(
                    operands=[memref_ptr, offset], result_types=[ptr.PtrType()]
                ),
                ptr.StoreOp(operands=[target_ptr, op.value]),
            ]
        )

        rewriter.replace_matched_op(ops)


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Load, rewriter: PatternRewriter, /):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_type = cast(memref.MemRefType[Any], op_memref_type)

        ops, offset = offset_calculations(memref_type, op.indices)

        ops.extend(
            [
                memref_ptr := memref.ToPtrOp(
                    operands=[op.memref], result_types=[ptr.PtrType()]
                ),
                target_ptr := ptr.PtrAddOp(
                    operands=[memref_ptr, offset], result_types=[ptr.PtrType()]
                ),
                load_result := ptr.LoadOp(
                    operands=[target_ptr], result_types=[memref_type.element_type]
                ),
            ]
        )

        rewriter.replace_matched_op(ops, new_results=[load_result.res])


@dataclass(frozen=True)
class ConvertMemrefToPtr(ModulePass):
    name = "convert-memref-to-ptr"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertStoreOp(), ConvertLoadOp()]),
            apply_recursively=False,
            walk_reverse=True,
            walk_regions_first=True,
        )
        the_one_pass.rewrite_module(op)
