from collections.abc import Iterable, Sequence
from math import prod
from typing import Any, cast

from xdsl.backend.riscv.lowering.utils import (
    cast_operands_to_regs,
    register_type_for_type,
)
from xdsl.builder import ImplicitBuilder
from xdsl.context import MLContext
from xdsl.dialects import memref, riscv, riscv_func
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseIntOrFPElementsAttr,
    Float32Type,
    Float64Type,
    IntegerType,
    MemRefType,
    ModuleOp,
    NoneAttr,
    ShapedType,
    StridedLayoutAttr,
    SymbolRefAttr,
    UnrealizedConversionCastOp,
)
from xdsl.interpreters.ptr import TypedPtr
from xdsl.ir import Attribute, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable
from xdsl.utils.exceptions import DiagnosticException


def bitwidth_of_type(type_attribute: Attribute) -> int:
    """
    Returns the width of an element type in bits, or raises ValueError for unknown inputs.
    """
    if isinstance(type_attribute, AnyFloat):
        return type_attribute.get_bitwidth
    elif isinstance(type_attribute, IntegerType):
        return type_attribute.width.data
    else:
        raise NotImplementedError(
            f"Unsupported memref element type for riscv lowering: {type_attribute}"
        )


class ConvertMemrefAllocOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter) -> None:
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        op_memref_type = cast(memref.MemRefType[Any], op_memref_type)
        width_in_bytes = bitwidth_of_type(op_memref_type.element_type) // 8
        size = prod(op_memref_type.get_shape()) * width_in_bytes
        rewriter.replace_matched_op(
            (
                size_op := riscv.LiOp(size, comment="memref alloc size"),
                move_op := riscv.MVOp(size_op.rd, rd=riscv.Registers.A0),
                call := riscv_func.CallOp(
                    SymbolRefAttr("malloc"),
                    (move_op.rd,),
                    (riscv.Registers.A0,),
                ),
                move_op := riscv.MVOp(call.ress[0], rd=riscv.Registers.UNALLOCATED_INT),
                UnrealizedConversionCastOp.get((move_op.rd,), (op.memref.type,)),
            )
        )


class ConvertMemrefDeallocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Dealloc, rewriter: PatternRewriter) -> None:
        raise DiagnosticException("Lowering memref.dealloc not implemented yet")


def memref_shape_ops(
    src_ptr: SSAValue,
    indices: Sequence[SSAValue],
    shape: Sequence[int],
    element_type: Attribute,
) -> tuple[list[Operation], SSAValue]:
    """
    Returns ssa value representing pointer into the memref at given indices.
    The pointer is byte-indexed, and the indices are strided by element size, so the index
    into the flat memory buffer needs to be multiplied by the size of the element.
    """
    assert len(shape) == len(indices)

    bitwidth = bitwidth_of_type(element_type)
    if bitwidth % 8:
        raise DiagnosticException(
            f"Cannot create offset for element type {element_type}"
            f" with bitwidth {bitwidth}"
        )
    bytes_per_element = bitwidth // 8

    if not shape:
        # Scalar memref
        return ([], src_ptr)

    ops: list[Operation] = []

    head, *tail = indices

    for factor, value in zip(shape[1:], tail):
        if factor == 0:
            continue
        if factor == 1:
            ops.extend(
                (
                    new_head_op := riscv.AddOp(
                        head, value, rd=riscv.IntRegisterType.unallocated()
                    ),
                )
            )
        else:
            ops.extend(
                (
                    factor_op := riscv.LiOp(factor),
                    offset_op := riscv.MulOp(
                        factor_op.rd, head, rd=riscv.IntRegisterType.unallocated()
                    ),
                    new_head_op := riscv.AddOp(
                        offset_op, value, rd=riscv.IntRegisterType.unallocated()
                    ),
                )
            )

        head = new_head_op.rd

    ops.extend(
        [
            bytes_per_element_op := riscv.LiOp(bytes_per_element),
            offset_bytes := riscv.MulOp(
                head,
                bytes_per_element_op.rd,
                rd=riscv.IntRegisterType.unallocated(),
                comment="multiply by element size",
            ),
            ptr := riscv.AddOp(
                src_ptr, offset_bytes, rd=riscv.IntRegisterType.unallocated()
            ),
        ]
    )

    return ops, ptr.rd


def pointer_offsets_ops(
    src_ptr: SSAValue,
    pairs: Iterable[tuple[SSAValue, int]],
    element_type: Attribute,
) -> tuple[list[Operation], SSAValue]:
    """
    Returns an ssa value representing a pointer into the memref at the given indices.
    'pairs' consists of the indices as SSA values and the corresponding stride of the dimension being indexed in
    number of elements. 'element_type' must be the element type of the memref being indexed.
    """
    bitwidth = bitwidth_of_type(element_type)
    if bitwidth % 8:
        raise DiagnosticException(
            f"Cannot create offset for element type {element_type}"
            f" with bitwidth {bitwidth}"
        )
    bytes_per_element = bitwidth // 8

    pairs = tuple((index, factor) for index, factor in pairs if factor != 0)

    if not pairs:
        # Scalar memref
        return ([], src_ptr)

    ops: list[Operation] = []
    res_ptr = src_ptr

    for index, factor in pairs:
        if factor == 1:
            ops.append(
                head_op := riscv.AddOp(
                    res_ptr, index, rd=riscv.IntRegisterType.unallocated()
                )
            )
            res_ptr = head_op.rd
        else:
            ops.extend(
                (
                    factor_op := riscv.LiOp(factor),
                    offset_op := riscv.MulOp(
                        index, factor_op.rd, rd=riscv.IntRegisterType.unallocated()
                    ),
                    head_op := riscv.AddOp(
                        res_ptr, offset_op.rd, rd=riscv.IntRegisterType.unallocated()
                    ),
                )
            )
        res_ptr = head_op.rd

    ops.extend(
        [
            bytes_per_element_op := riscv.LiOp(bytes_per_element),
            offset_bytes := riscv.MulOp(
                res_ptr,
                bytes_per_element_op.rd,
                rd=riscv.IntRegisterType.unallocated(),
                comment="multiply by element size",
            ),
            ptr := riscv.AddOp(
                src_ptr, offset_bytes, rd=riscv.IntRegisterType.unallocated()
            ),
        ]
    )

    return ops, ptr.rd


class ConvertMemrefStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter):
        value, mem, *indices = cast_operands_to_regs(rewriter)

        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_type = cast(memref.MemRefType[Any], op_memref_type)
        shape = memref_type.get_shape()

        ops, ptr = memref_shape_ops(mem, indices, shape, memref_type.element_type)

        rewriter.insert_op_before_matched_op(ops)
        match value.type:
            case riscv.IntRegisterType():
                new_op = riscv.SwOp(
                    ptr, value, 0, comment=f"store int value to memref of shape {shape}"
                )
            case riscv.FloatRegisterType():
                float_type = cast(AnyFloat, memref_type.element_type)
                match float_type:
                    case Float32Type():
                        new_op = riscv.FSwOp(
                            ptr,
                            value,
                            0,
                            comment=f"store float value to memref of shape {shape}",
                        )
                    case Float64Type():
                        new_op = riscv.FSdOp(
                            ptr,
                            value,
                            0,
                            comment=f"store double value to memref of shape {shape}",
                        )
                    case _:
                        assert False, f"Unexpected floating point type {float_type}"

            case _:
                assert False, f"Unexpected register type {value.type}"

        rewriter.replace_matched_op(new_op)


class ConvertMemrefLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Load, rewriter: PatternRewriter):
        mem, *indices = cast_operands_to_regs(rewriter)

        assert isinstance(
            op_memref_type := op.memref.type, memref.MemRefType
        ), f"{op.memref.type}"
        memref_type = cast(memref.MemRefType[Any], op_memref_type)
        shape = memref_type.get_shape()
        ops, ptr = memref_shape_ops(mem, indices, shape, memref_type.element_type)
        rewriter.insert_op_before_matched_op(ops)

        result_register_type = register_type_for_type(op.res.type)

        match result_register_type:
            case riscv.IntRegisterType:
                lw_op = riscv.LwOp(
                    ptr, 0, comment=f"load word from memref of shape {shape}"
                )
            case riscv.FloatRegisterType:
                float_type = cast(AnyFloat, memref_type.element_type)
                match float_type:
                    case Float32Type():
                        lw_op = riscv.FLwOp(
                            ptr, 0, comment=f"load float from memref of shape {shape}"
                        )
                    case Float64Type():
                        lw_op = riscv.FLdOp(
                            ptr, 0, comment=f"load double from memref of shape {shape}"
                        )
                    case _:
                        assert False, f"Unexpected floating point type {float_type}"

            case _:
                assert False, f"Unexpected register type {result_register_type}"

        rewriter.replace_matched_op(
            [
                lw := lw_op,
                UnrealizedConversionCastOp.get(lw.results, (op.res.type,)),
            ],
        )


class ConvertMemrefGlobalOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Global, rewriter: PatternRewriter):
        initial_value = op.initial_value

        if not isinstance(initial_value, DenseIntOrFPElementsAttr):
            raise DiagnosticException(
                f"Unsupported memref.global initial value: {initial_value}"
            )

        memref_type = cast(memref.MemRefType[Any], op.type)
        element_type = memref_type.element_type

        # Only handle a small subset of elements
        # Might be useful as a helper for other passes in the future
        match element_type:
            case IntegerType():
                bitwidth = element_type.width.data
                if bitwidth != 32:
                    raise DiagnosticException(
                        f"Unsupported memref element type for riscv lowering: {element_type}"
                    )
                ints = [d.value.data for d in initial_value.data]
                for i in ints:
                    assert isinstance(i, int)
                ints = cast(list[int], ints)
                ptr = TypedPtr.new_int32(ints).raw
            case Float32Type():
                floats = [d.value.data for d in initial_value.data]
                ptr = TypedPtr.new_float32(floats).raw
            case Float64Type():
                floats = [d.value.data for d in initial_value.data]
                ptr = TypedPtr.new_float64(floats).raw
            case _:
                raise DiagnosticException(
                    f"Unsupported memref element type for riscv lowering: {element_type}"
                )

        text = ",".join(hex(i) for i in ptr.int32.get_list(42))

        section = riscv.AssemblySectionOp(".data")
        with ImplicitBuilder(section.data):
            riscv.LabelOp(op.sym_name.data)
            riscv.DirectiveOp(".word", text)

        rewriter.replace_matched_op(section)


class ConvertMemrefGetGlobalOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.GetGlobal, rewriter: PatternRewriter):
        rewriter.replace_matched_op(
            [
                ptr := riscv.LiOp(op.name_.string_value()),
                UnrealizedConversionCastOp.get((ptr,), (op.memref.type,)),
            ]
        )


class ConvertMemrefSubviewOp(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter):
        # Assumes that the operation is valid, meaning that the subview is indeed a
        # subview, and that if the offset is stated in the layout attribute, then it's
        # correct.

        source = op.source
        result = op.result
        source_type = source.type
        assert isinstance(source_type, MemRefType)
        source_type = cast(MemRefType[Attribute], source_type)
        result_type = cast(MemRefType[Attribute], result.type)

        result_layout_attr = result_type.layout
        if isinstance(result_layout_attr, NoneAttr):
            # When a subview has no layout attr, the result is a perfect subview at offset
            # 0.
            rewriter.replace_matched_op(
                UnrealizedConversionCastOp.get((source,), (result_type,))
            )
            return

        if not isinstance(result_layout_attr, StridedLayoutAttr):
            raise DiagnosticException("Only strided layout attrs implemented")

        # We still need to check that the layout attribute strides are contiguous in
        # memory.
        result_shape = result_type.get_shape()
        result_strides = tuple(result_layout_attr.get_strides())

        identity_strides = ShapedType.strides_for_shape(result_shape)

        if identity_strides[-len(result_strides) :] != result_strides:
            # The subview may be rank reducing, we want the suffix of the original strides
            raise DiagnosticException(
                "Cannot lower strides that do not result in a contiguous subview"
            )

        factor = bitwidth_of_type(result_type.element_type)

        if (static_offset := result_layout_attr.get_offset()) is not None:
            # The offset is known statically, can just add it, scaled by element bitwidth
            rewriter.replace_matched_op(
                (
                    src := UnrealizedConversionCastOp.get(
                        (source,), (riscv.IntRegisterType.unallocated(),)
                    ),
                    res := riscv.AddiOp(
                        src,
                        static_offset * factor,
                        comment="subview into original memref",
                    ),
                    UnrealizedConversionCastOp.get((res,), (result_type,)),
                )
            )
            return

        # Accumulate op offsets into a new register
        src, *offsets = cast_operands_to_regs(rewriter)

        pairs: list[tuple[SSAValue, int]] = []

        dynamic_offset_index = 0
        for static_offset_attr, stride in zip(
            op.static_offsets.data, identity_strides, strict=True
        ):
            static_offset = static_offset_attr.data
            assert isinstance(static_offset, int)
            if static_offset == memref.Subview.DYNAMIC_INDEX:
                pairs.append((offsets[dynamic_offset_index], stride))
                dynamic_offset_index += 1
            else:
                if static_offset:
                    # No need to insert arithmetic ops that will be multiplied by zero
                    rewriter.insert_op_before_matched_op(
                        offset_op := riscv.LiOp(static_offset)
                    )
                    pairs.append((offset_op.rd, stride))

        ops, ptr = pointer_offsets_ops(
            src,
            pairs,
            source_type.element_type,
        )

        rewriter.replace_matched_op(
            (*ops, UnrealizedConversionCastOp.get((ptr,), (result_type,)))
        )


class ConvertMemrefToRiscvPass(ModulePass):
    name = "convert-memref-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        contains_malloc = PatternRewriteWalker(ConvertMemrefAllocOp()).rewrite_module(
            op
        )
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertMemrefDeallocOp(),
                    ConvertMemrefStoreOp(),
                    ConvertMemrefLoadOp(),
                    ConvertMemrefGlobalOp(),
                    ConvertMemrefGetGlobalOp(),
                    ConvertMemrefSubviewOp(),
                ]
            )
        ).rewrite_module(op)
        if contains_malloc:
            func_op = riscv_func.FuncOp(
                "malloc",
                Region(),
                ((riscv.Registers.A0,), (riscv.Registers.A0,)),
                visibility="private",
            )
            SymbolTable.insert_or_update(op, func_op)
