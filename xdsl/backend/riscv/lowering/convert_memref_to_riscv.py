from collections.abc import Iterable
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


def get_strided_pointer(
    src_ptr: SSAValue,
    indices: Iterable[SSAValue],
    memref_type: MemRefType[Any],
) -> tuple[list[Operation], SSAValue]:
    """
    Given a buffer pointer 'src_ptr' which was originally of type 'memref_type', returns
    a new pointer to the element being accessed by the 'indices'.
    """

    bitwidth = bitwidth_of_type(memref_type.element_type)
    if bitwidth % 8:
        raise DiagnosticException(
            f"Cannot create offset for element type {memref_type.element_type}"
            f" with bitwidth {bitwidth}"
        )
    bytes_per_element = bitwidth // 8

    match memref_type.layout:
        case NoneAttr():
            strides = ShapedType.strides_for_shape(memref_type.get_shape())
        case StridedLayoutAttr():
            strides = memref_type.layout.get_strides()
        case _:
            raise DiagnosticException(f"Unsupported layout type {memref_type.layout}")

    ops: list[Operation] = []

    head: SSAValue | None = None

    for index, stride in zip(indices, strides):
        match stride:
            case None:
                raise NotImplementedError(
                    f"MemRef {memref_type} with dynamic stride is not yet implemented"
                )
            case 1:
                pass
            case _:
                ops.extend(
                    (
                        stride_op := riscv.LiOp(stride),
                        offset_op := riscv.MulOp(
                            index, stride_op.rd, rd=riscv.IntRegisterType.unallocated()
                        ),
                    )
                )
                index = offset_op.rd

        if head is None:
            # First iteration.
            head = index
            continue

        # Otherwise sum up the products.
        ops.append(
            add_op := riscv.AddOp(head, index, rd=riscv.IntRegisterType.unallocated())
        )
        head = add_op.rd

    if head is None:
        return ops, src_ptr

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


class ConvertMemrefStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter):
        assert isinstance(op_memref_type := op.memref.type, memref.MemRefType)
        memref_type = cast(memref.MemRefType[Any], op_memref_type)

        value, mem, *indices = cast_operands_to_regs(rewriter)

        shape = memref_type.get_shape()
        ops, ptr = get_strided_pointer(mem, indices, memref_type)

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
        assert isinstance(
            op_memref_type := op.memref.type, memref.MemRefType
        ), f"{op.memref.type}"
        memref_type = cast(memref.MemRefType[Any], op_memref_type)

        mem, *indices = cast_operands_to_regs(rewriter)

        shape = memref_type.get_shape()
        ops, ptr = get_strided_pointer(mem, indices, memref_type)
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
