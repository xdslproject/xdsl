from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, ptr, x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    UnrealizedConversionCastOp,
    VectorType,
)
from xdsl.dialects.x86.register import X86VectorRegisterType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException


def vector_type_to_register_type(
    value_type: VectorType,
    arch: str,
) -> X86VectorRegisterType:
    vector_num_elements = value_type.element_count()
    element_type = cast(FixedBitwidthType, value_type.get_element_type())
    element_size = element_type.bitwidth
    vector_size = vector_num_elements * element_size
    # Choose the x86 vector register according to the
    # target architecture and the abstract vector size
    if vector_size == 128:
        vect_reg_type = x86.register.UNALLOCATED_SSE
    elif vector_size == 256 and (arch == "avx2" or arch == "avx512"):
        vect_reg_type = x86.register.UNALLOCATED_AVX2
    elif vector_size == 512 and arch == "avx512":
        vect_reg_type = x86.register.UNALLOCATED_AVX512
    else:
        raise DiagnosticException(
            "The vector size and target architecture are inconsistent."
        )
    return vect_reg_type


@dataclass
class PtrAddToX86(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter):
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        ptr_cast_op = UnrealizedConversionCastOp.get((op.addr,), (x86_reg_type,))
        offset_cast_op = UnrealizedConversionCastOp.get((op.offset,), (x86_reg_type,))
        add_op = x86.RR_AddOp(ptr_cast_op, offset_cast_op, result=x86_reg_type)
        rewriter.replace_matched_op([ptr_cast_op, offset_cast_op, add_op])


@dataclass
class PtrStoreToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter):
        value_type = op.value.type
        if not isinstance(value_type, VectorType):
            raise DiagnosticException(
                "The lowering of ptr.store is not yet implemented for non-vector types."
            )
        value_type = cast(VectorType, value_type)
        # Pointer casts
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        addr_cast_op = UnrealizedConversionCastOp.get((op.addr,), (x86_reg_type,))
        x86_vect_type = vector_type_to_register_type(value_type, self.arch)
        vect_cast_op = UnrealizedConversionCastOp.get((op.value,), (x86_vect_type,))
        # Choose the x86 vector instruction according to the
        # abstract vector element size
        element_size = cast(FixedBitwidthType, value_type.get_element_type()).bitwidth
        match element_size:
            case 16:
                raise DiagnosticException(
                    "Half-precision vector load is not implemented yet."
                )
            case 32:
                mov = x86.ops.MR_VmovupsOp
            case 64:
                mov = x86.ops.MR_VmovapdOp
            case _:
                raise DiagnosticException(
                    "Float precision must be half, single or double."
                )

        mov_op = mov(addr_cast_op, vect_cast_op, offset=0)
        rewriter.replace_matched_op([addr_cast_op, vect_cast_op, mov_op])


@dataclass
class PtrLoadToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter):
        value_type = op.res.type
        if not isinstance(value_type, VectorType):
            raise DiagnosticException(
                "The lowering of ptr.load is not yet implemented for non-vector types."
            )
        value_type = cast(VectorType, value_type)
        # Pointer cast
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        cast_op = UnrealizedConversionCastOp.get((op.addr,), (x86_reg_type,))
        # Choose the x86 vector instruction according to the
        # abstract vector element size
        element_size = cast(FixedBitwidthType, value_type.get_element_type()).bitwidth
        match element_size:
            case 16:
                raise DiagnosticException(
                    "Half-precision vector load is not implemented yet."
                )
            case 32:
                mov = x86.ops.RM_VmovupsOp
            case 64:
                raise DiagnosticException(
                    "Double precision vector load is not implemented yet."
                )
            case _:
                raise DiagnosticException(
                    "Float precision must be half, single or double."
                )

        mov_op = mov(
            cast_op,
            offset=0,
            result=vector_type_to_register_type(value_type, self.arch),
        )
        rewriter.replace_matched_op([cast_op, mov_op])


@dataclass(frozen=True)
class ConvertPtrToX86Pass(ModulePass):
    name = "convert-ptr-to-x86"

    arch: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PtrLoadToX86(self.arch),
                    PtrStoreToX86(self.arch),
                    PtrAddToX86(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
