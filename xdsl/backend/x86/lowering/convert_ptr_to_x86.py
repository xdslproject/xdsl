from dataclasses import dataclass

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
from xdsl.utils.hints import isa

from .helpers import vector_type_to_register_type


@dataclass
class PtrAddToX86(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter):
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        ptr_cast_op = UnrealizedConversionCastOp.get((op.addr,), (x86_reg_type,))
        offset_cast_op = UnrealizedConversionCastOp.get((op.offset,), (x86_reg_type,))
        add_op = x86.RS_AddOp(ptr_cast_op, offset_cast_op, register_out=x86_reg_type)
        res_cast_op = UnrealizedConversionCastOp.get(
            (add_op.register_out,), (ptr.PtrType(),)
        )
        rewriter.replace_matched_op([ptr_cast_op, offset_cast_op, add_op, res_cast_op])


@dataclass
class PtrStoreToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter):
        value_type = op.value.type
        # Pointer casts
        addr_cast_op, x86_ptr = UnrealizedConversionCastOp.cast_one(
            op.addr, x86.register.UNALLOCATED_GENERAL
        )
        if isa(value_type, VectorType[FixedBitwidthType]):
            x86_vect_type = vector_type_to_register_type(value_type, self.arch)
            cast_op, x86_data = UnrealizedConversionCastOp.cast_one(
                op.value, x86_vect_type
            )
            # Choose the x86 vector instruction according to the
            # abstract vector element size
            match value_type.get_element_type().bitwidth:
                case 16:
                    raise DiagnosticException(
                        "Half-precision floating point vector load is not implemented yet."
                    )
                case 32:
                    mov = x86.ops.MS_VmovupsOp
                case 64:
                    mov = x86.ops.MS_VmovapdOp
                case _:
                    raise DiagnosticException(
                        "Float precision must be half, single or double."
                    )
            mov_op = mov(x86_ptr, x86_data, memory_offset=0)
        else:
            cast_op, x86_data = UnrealizedConversionCastOp.cast_one(
                op.value, x86.register.UNALLOCATED_GENERAL
            )
            mov_op = x86.MS_MovOp(x86_ptr, x86_data, memory_offset=0)

        rewriter.replace_matched_op([addr_cast_op, cast_op, mov_op])


@dataclass
class PtrLoadToX86(RewritePattern):
    arch: str

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter):
        # Pointer cast
        x86_reg_type = x86.register.UNALLOCATED_GENERAL
        cast_op, addr_x86 = UnrealizedConversionCastOp.cast_one(op.addr, x86_reg_type)

        value_type = op.res.type
        if isa(value_type, VectorType[FixedBitwidthType]):
            # Choose the x86 vector instruction according to the
            # abstract vector element size
            match value_type.get_element_type().bitwidth:
                case 16:
                    raise DiagnosticException(
                        "Half-precision floating point vector load is not implemented yet."
                    )
                case 32:
                    mov = x86.ops.DM_VmovupsOp
                case 64:
                    mov = x86.ops.DM_VmovupdOp
                case _:
                    raise DiagnosticException(
                        "Float precision must be half, single or double."
                    )
            mov_op = mov(
                addr_x86,
                memory_offset=0,
                destination=vector_type_to_register_type(value_type, self.arch),
            )
        else:
            mov_op = x86.DM_MovOp(
                addr_x86, memory_offset=0, destination=x86.register.UNALLOCATED_GENERAL
            )

        res_cast_op = UnrealizedConversionCastOp.get(mov_op.results, (value_type,))
        rewriter.replace_matched_op([cast_op, mov_op, res_cast_op])


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
