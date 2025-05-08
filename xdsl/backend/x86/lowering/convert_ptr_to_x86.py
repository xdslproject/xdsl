from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, ptr, x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    UnrealizedConversionCastOp,
    VectorType,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException


@dataclass
class PtrLoadToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter):
        source = op.addr
        value = op.res
        value_type = value.type
        if not isinstance(value_type, VectorType):
            raise DiagnosticException(
                "The lowering of ptr.load is not yet implemented for non-vector types."
            )
        # Pointer cast
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        cast_op = UnrealizedConversionCastOp.get((source,), (x86_reg_type,))
        # Output vector description
        vector_num_elements = value_type.element_count()
        element_type = cast(FixedBitwidthType, value_type.get_element_type())
        element_size = element_type.bitwidth
        vector_size = vector_num_elements * element_size
        # Choose the x86 vector register according to the
        # target architecture and the abstract vector size
        if vector_size == 128:
            vect_reg_type = x86.register.UNALLOCATED_SSE
        elif vector_size == 256 and (self.arch == "avx2" or self.arch == "avx512"):
            vect_reg_type = x86.register.UNALLOCATED_AVX2
        elif vector_size == 512 and self.arch == "avx512":
            vect_reg_type = x86.register.UNALLOCATED_AVX512
        else:
            raise DiagnosticException(
                "The vector size and target architecture are inconsistent."
            )
        # Choose the x86 vector instruction according to the
        # abstract vector element size
        match element_size:
            case 16:
                raise DiagnosticException(
                    "Half-precision vector load is not implemented yet."
                )
            case 32:
                mov = x86.ops.RM_VmovupsOp
            case 64:
                # mov = x86.ops.RM_VmovapdOp
                raise DiagnosticException(
                    "Double precision vector load is not implemented yet."
                )
            case _:
                raise DiagnosticException(
                    "Float precision must be half, single or double."
                )

        mov_op = mov(cast_op, offset=0, result=vect_reg_type)
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
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
