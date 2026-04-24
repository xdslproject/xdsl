"""
Implements a lowering from operations on memrefs to operations on pointers.

The default lowering of memrefs in MLIR is to a structure that carries the base pointer,
an offset, and strides
(See (memref.extract_strided_metadata)[https://mlir.llvm.org/docs/Dialects/MemRef/#memrefextract_strided_metadata-memrefextractstridedmetadataop]).
If the offset and strides are statically known, they can be encoded in the type, but
they can also be dynamic.

When lowering operations on memrefs to operations on pointers, the offset and stride
information must be lowered to operations that perform the required pointer arithmetic.
In order to simplify the lowering of memory accesses, this pass makes the choice to
lower memrefs to a pointer to the buffer including the dynamic offset.
This means that memory accesses can be a simple dot product of statically known strides
and dynamic indices.
On the other hand, operations that create views of memrefs from other memrefs must lower
to the relevant pointer arithmetic to encode the new inner buffer offset, when possible.
"""

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
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa

_index_type = builtin.IndexType()


def build_bytes_offset(
    elements_offset: SSAValue, element_type: Attribute, builder: Builder
) -> SSAValue:
    """
    Returns the offset in bytes given an offset in elements and the element type.
    """
    bytes_per_element_op = builder.insert_op(
        ptr.TypeOffsetOp(element_type, _index_type)
    )
    bytes_offset = builder.insert_op(
        arith.MuliOp(elements_offset, bytes_per_element_op)
    )
    bytes_per_element_op.offset.name_hint = "bytes_per_element"
    bytes_offset.result.name_hint = "scaled_pointer_offset"

    return bytes_offset.result


def build_offset_pointer(
    pointer: SSAValue,
    bytes_offset: SSAValue,
    builder: Builder,
) -> SSAValue:
    """
    Returns the pointer incremented by the given number of bytes.
    """
    target_ptr = builder.insert_op(ptr.PtrAddOp(pointer, bytes_offset))
    target_ptr.result.name_hint = "offset_pointer"
    return target_ptr.result


def get_strides(
    memref_val: SSAValue,
    memref_type: builtin.MemRefType,
    builder: Builder,
) -> Sequence[int | SSAValue]:
    """
    Returns the per-dimension strides of `memref_val`. Each stride is an `int` when
    statically known, or an SSA value when it depends on dynamic source dimensions
    (for which `memref.dim` ops are emitted).
    """
    match memref_type.layout:
        case builtin.StridedLayoutAttr() as layout:
            layout_strides = layout.get_strides()
            if None in layout_strides:
                raise DiagnosticException(
                    f"MemRef {memref_type} with dynamic stride is not yet implemented"
                )
            return cast(Sequence[int], layout_strides)
        case builtin.NoneAttr():
            pass
        case _:
            raise DiagnosticException(f"Unsupported layout type {memref_type.layout}")

    shape = memref_type.get_shape()
    if builtin.DYNAMIC_INDEX not in shape:
        return builtin.ShapedType.strides_for_shape(shape)

    rank = len(shape)
    strides: list[int | SSAValue] = [1] * rank
    for i in range(rank - 2, -1, -1):
        dim_size: int | SSAValue = shape[i + 1]
        if dim_size == builtin.DYNAMIC_INDEX:
            dim_idx = builder.insert_op(
                arith.ConstantOp.from_int_and_width(i + 1, _index_type)
            )
            dim_idx.result.name_hint = "dim_idx"
            dim_size = builder.insert_op(
                memref.DimOp.from_source_and_index(memref_val, dim_idx.result)
            ).result
        prev = strides[i + 1]
        match (prev, dim_size):
            case (int(p), int(d)):
                strides[i] = p * d
                continue
            case (1, _):
                strides[i] = dim_size
                continue
            case (_, 1):
                strides[i] = prev
                continue
            case _:
                pass
        if isinstance(prev, int):
            prev = builder.insert_op(
                arith.ConstantOp.from_int_and_width(prev, _index_type)
            ).result
        if isinstance(dim_size, int):
            dim_size = builder.insert_op(
                arith.ConstantOp.from_int_and_width(dim_size, _index_type)
            ).result
        strides[i] = builder.insert_op(arith.MuliOp(prev, dim_size)).result
    return strides


def build_strides_offset(
    indices: Iterable[SSAValue],
    strides: Sequence[int | SSAValue],
    builder: Builder,
) -> SSAValue | None:
    """
    Given SSA values for indices, and strides (each either an `int` or an SSA value),
    insert the arithmetic ops that create the combined index offset.
    The length of indices and strides must be the same.
    Static strides must be positive.
    """
    head: SSAValue | None = None

    for index, stride in zip(indices, strides, strict=True):
        increment = index
        if isinstance(stride, int):
            assert stride > 0, f"Strides must be positive, got {stride}"
            if stride != 1:
                stride_op = builder.insert_op(
                    arith.ConstantOp.from_int_and_width(stride, _index_type)
                )
                offset_op = builder.insert_op(arith.MuliOp(increment, stride_op))
                stride_op.result.name_hint = "pointer_dim_stride"
                offset_op.result.name_hint = "pointer_dim_offset"
                increment = offset_op.result
        else:
            offset_op = builder.insert_op(arith.MuliOp(increment, stride))
            offset_op.result.name_hint = "pointer_dim_offset"
            increment = offset_op.result

        if head is None:
            head = increment
            continue

        add_op = builder.insert_op(arith.AddiOp(head, increment))
        add_op.result.name_hint = "pointer_dim_stride"
        head = add_op.result

    return head


def build_target_ptr(
    target_memref: SSAValue,
    memref_type: memref.MemRefType[Any],
    indices: Iterable[SSAValue],
    builder: Builder,
) -> SSAValue:
    """
    Build operations returning a pointer to an element of a memref referenced by indices.
    """

    memref_ptr = builder.insert_op(ptr.ToPtrOp(target_memref))
    pointer = memref_ptr.res
    pointer.name_hint = target_memref.name_hint

    strides = get_strides(target_memref, memref_type, builder)
    head = build_strides_offset(indices, strides, builder)

    if head is not None:
        offset = build_bytes_offset(head, memref_type.element_type, builder)
        pointer = build_offset_pointer(pointer, offset, builder)

    return pointer


@dataclass
class ConvertStorePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        target_ptr = build_target_ptr(op.memref, memref_type, op.indices, rewriter)
        rewriter.replace_op(op, ptr.StoreOp(target_ptr, op.value))


@dataclass
class ConvertLoadPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, memref.MemRefType)
        target_ptr = build_target_ptr(op.memref, memref_type, op.indices, rewriter)
        rewriter.replace_op(op, ptr.LoadOp(target_ptr, memref_type.element_type))


class ConvertSubviewPattern(RewritePattern):
    """
    Converts the subview to a pointer offset.

    From the subview op documentation:

    > In the absence of rank reductions, the resulting memref type is computed as
    follows:
    ```
    ...
    result_offset = src_offset + dot_product(offset_operands, src_strides)
    ```

    The pointer that the source memref is lowered to is assumed to incorporate the
    `src_offset`, so this lowering just addds the dot product.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        # The result type of the subview op already encodes the shape, strides, and
        # element type that downstream users need for indexing, so this rewrite only
        # emits the pointer offset. For dynamic source shapes the source strides can
        # be SSA values, which the loop below multiplies directly.

        source_type = op.source.type
        assert isa(source_type, builtin.MemRefType)
        result_type = op.result.type
        element_type = result_type.element_type

        source_strides = get_strides(op.source, source_type, rewriter)

        pointer = rewriter.insert_op(ptr.ToPtrOp(op.source)).res
        pointer.name_hint = op.source.name_hint
        # The new pointer
        head = None
        dynamic_offset_index = 0

        for stride, offset in zip(
            source_strides, op.static_offsets.iter_values(), strict=True
        ):
            if offset == builtin.DYNAMIC_INDEX:
                offset_val = op.offsets[dynamic_offset_index]
                dynamic_offset_index += 1
            else:
                offset_val = rewriter.insert_op(
                    arith.ConstantOp(builtin.IntegerAttr(offset, _index_type))
                ).result
                offset_val.name_hint = f"c{offset}"

            if stride == 1:
                increment = offset_val
            elif isinstance(stride, int):
                stride_val = rewriter.insert_op(
                    arith.ConstantOp(builtin.IntegerAttr(stride, _index_type))
                ).result
                increment = rewriter.insert_op(
                    arith.MuliOp(stride_val, offset_val)
                ).result
                stride_val.name_hint = f"c{stride}"
                increment.name_hint = "increment"
            else:
                increment = rewriter.insert_op(arith.MuliOp(stride, offset_val)).result
                increment.name_hint = "increment"

            if head is None:
                head = increment
            else:
                # Otherwise sum up the products.
                head = rewriter.insert_op(arith.AddiOp(head, increment)).result
                head.name_hint = "subview"

        if head is not None:
            offset = build_bytes_offset(head, element_type, rewriter)
            pointer = build_offset_pointer(pointer, offset, rewriter)

        rewriter.replace_op(op, ptr.FromPtrOp(pointer, result_type))


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
            rewriter.replace_uses_with_if(
                arg,
                cast_op.res,
                lambda x: x.operation is not cast_op,
            )


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
                rewriter.insert_op(cast_op := ptr.ToPtrOp(argument))
                new_arguments.append(cast_op.res)
                cast_op.res.name_hint = argument.name_hint
            else:
                new_arguments.append(argument)

        rewriter.replace_op(op, func.ReturnOp(*new_arguments))


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
                rewriter.insert_op(cast_op := ptr.ToPtrOp(argument))
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

        rewriter.replace_op(op, new_ops, new_results)


@dataclass
class ConvertCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CastOp, rewriter: PatternRewriter, /):
        assert isa(op.source.type, memref.MemRefType)
        rewriter.replace_matched_op((), (op.source,))


@dataclass
class ConvertReinterpretCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref.ReinterpretCastOp, rewriter: PatternRewriter, /
    ):
        pointer = rewriter.insert_op(ptr.ToPtrOp(op.source)).res
        pointer.name_hint = op.source.name_hint

        # reinterpret_cast has exactly one flat element offset
        static_offset = next(iter(op.static_offsets.iter_values()))

        if static_offset != 0:
            # dynamic offset: passed as operand
            if static_offset == builtin.DYNAMIC_INDEX:
                offset_val = op.offsets[0]
            # static non-zero offset: materialize as constant
            else:
                offset_val = rewriter.insert_op(
                    arith.ConstantOp(builtin.IntegerAttr(static_offset, _index_type))
                ).result
                offset_val.name_hint = f"c{static_offset}"

            # scale element offset to bytes and advance pointer
            element_type = op.result.type.element_type
            byte_offset = build_bytes_offset(offset_val, element_type, rewriter)
            pointer = build_offset_pointer(pointer, byte_offset, rewriter)
        rewriter.replace_op(op, ptr.FromPtrOp(pointer, op.result.type))


@dataclass(frozen=True)
class ConvertMemRefToPtr(ModulePass):
    name = "convert-memref-to-ptr"

    lower_func: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStorePattern(),
                    ConvertLoadPattern(),
                    ConvertSubviewPattern(),
                    ConvertCastOp(),
                    ConvertReinterpretCastOp(),
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
