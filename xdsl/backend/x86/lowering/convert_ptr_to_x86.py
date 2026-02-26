from dataclasses import dataclass

from xdsl.backend.x86.lowering.helpers import Arch
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


@dataclass
class PtrAddToX86(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter):
        x86_reg_type = x86.registers.UNALLOCATED_GENERAL

        rewriter.replace_op(
            op,
            [
                ptr_cast_op := UnrealizedConversionCastOp.get(
                    (op.addr,), (x86_reg_type,)
                ),
                offset_cast_op := UnrealizedConversionCastOp.get(
                    (op.offset,), (x86_reg_type,)
                ),
                ptr_mv_op := x86.DS_MovOp(
                    ptr_cast_op, destination=x86.registers.UNALLOCATED_GENERAL
                ),
                add_op := x86.RS_AddOp(
                    ptr_mv_op.destination, offset_cast_op, register_out=x86_reg_type
                ),
                UnrealizedConversionCastOp.get(
                    (add_op.register_out,), (ptr.PtrType(),)
                ),
            ],
        )


@dataclass
class PtrStoreToX86(RewritePattern):
    arch: Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter):
        value_type = op.value.type
        # Pointer casts
        addr_cast_op, x86_ptr = UnrealizedConversionCastOp.cast_one(
            op.addr, x86.registers.UNALLOCATED_GENERAL
        )
        if isa(value_type, VectorType[FixedBitwidthType]):
            x86_vect_type = self.arch.register_type_for_type(value_type)
            cast_op, x86_data = UnrealizedConversionCastOp.cast_one(
                op.value, x86_vect_type.unallocated()
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
                op.value, x86.registers.UNALLOCATED_GENERAL
            )
            mov_op = x86.MS_MovOp(x86_ptr, x86_data, memory_offset=0)

        rewriter.replace_op(op, [addr_cast_op, cast_op, mov_op])


@dataclass
class PtrLoadToX86(RewritePattern):
    arch: Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter):
        # Pointer cast
        x86_reg_type = x86.registers.UNALLOCATED_GENERAL
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
                destination=self.arch.register_type_for_type(value_type).unallocated(),
            )
        else:
            mov_op = x86.DM_MovOp(
                addr_x86, memory_offset=0, destination=x86.registers.UNALLOCATED_GENERAL
            )

        res_cast_op = UnrealizedConversionCastOp.get(mov_op.results, (value_type,))
        rewriter.replace_op(op, [cast_op, mov_op, res_cast_op])


@dataclass(frozen=True)
class ConvertPtrToX86Pass(ModulePass):
    name = "convert-ptr-to-x86"

    arch: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        arch = Arch.arch_for_name(self.arch)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PtrLoadToX86(arch),
                    PtrStoreToX86(arch),
                    PtrAddToX86(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
