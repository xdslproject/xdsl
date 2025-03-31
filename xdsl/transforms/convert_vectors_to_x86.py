from dataclasses import dataclass
from functools import reduce

from xdsl.context import Context
from xdsl.dialects import builtin, memref, vector, x86
from xdsl.dialects.builtin import (
    ArrayAttr,
    FixedBitwidthType,
    IntAttr,
    UnrealizedConversionCastOp,
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
class VectorLoadToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.LoadOp, rewriter: PatternRewriter):
        assert self.arch == "sse" or self.arch == "avx2" or self.arch == "avx512"

        # Output vector description
        vector = op.result
        vector_int_shape = [i.data for i in vector.type.shape.data]
        vector_shape = ArrayAttr(
            [IntAttr(reduce(lambda x, y: x * y, vector_int_shape))]
        )
        vector_num_elements = vector_shape.data[0].data
        element_type = vector.type.get_element_type()
        assert isinstance(element_type, FixedBitwidthType)
        element_size = element_type.bitwidth
        vector_size = vector_num_elements * element_size

        # Input memref description
        memory = op.base
        memory_type = memory.type
        assert isinstance(memory_type, memref.MemRefType)

        # Build a subview of the original memref
        strides = memory_type.get_strides()
        static_strides: list[int] = []
        if strides:
            static_strides += [s for s in strides if s is not None]
        subview_type = memref.MemRefType(
            element_type=element_type, shape=vector_int_shape
        )
        subview_op = memref.SubviewOp.get(
            source=memory,
            result_type=subview_type,
            offsets=op.indices,
            strides=static_strides,
            sizes=vector_int_shape,
        )

        subview = subview_op.result
        # Build a pointer from the subview
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        cast_op = UnrealizedConversionCastOp.get((subview,), (x86_reg_type,))

        # Choose the x86 vector register according to the
        # target architecture and the abstract vector size
        if vector_size == 128:
            vect_reg_type = x86.register.UNALLOCATED_SSE
        elif vector_size == 256 and (self.arch == "avx2" or self.arch == "avx512"):
            vect_reg_type = x86.register.UNALLOCATED_AVX2
        elif vector_size == 512 and self.arch == "avx512":
            vect_reg_type = x86.register.UNALLOCATED_AVX512
        else:
            vect_reg_type = None
        assert vect_reg_type is not None

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
                mov = None

        assert mov is not None
        mov_op = mov(cast_op, offset=0, result=vect_reg_type)

        rewriter.replace_matched_op([subview_op, cast_op, mov_op])


@dataclass(frozen=True)
class ConvertVectorsToX86Pass(ModulePass):
    name = "convert-vectors-to-x86"

    arch: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([VectorLoadToX86(self.arch)]),
            apply_recursively=False,
        ).rewrite_module(op)
