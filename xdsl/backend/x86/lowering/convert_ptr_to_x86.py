from dataclasses import dataclass

from xdsl.backend.x86.arch import X86Arch
from xdsl.context import Context
from xdsl.dialects import asm, builtin, ptr, x86
from xdsl.dialects.builtin import (
    FixedBitwidthType,
    VectorType,
)
from xdsl.dialects.x86.registers import GeneralRegisterType
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
    arch: X86Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter):
        # Pointer-sized GPR for address arithmetic (ptr + offset in the flat address space).
        reg = self.arch.register_type_for_type(ptr.PtrType()).unallocated()
        assert isinstance(reg, GeneralRegisterType)

        rewriter.replace_op(
            op,
            [
                ptr_cast_op := asm.ToRegOp.get(op.addr, reg),
                offset_cast_op := asm.ToRegOp.get(op.offset, reg),
                ptr_mv_op := x86.DS_MovOp(ptr_cast_op, destination=reg),
                add_op := x86.RS_AddOp(
                    ptr_mv_op.destination, offset_cast_op, register_out=reg
                ),
                asm.FromRegOp.get(add_op.register_out, ptr.PtrType()),
            ],
        )


@dataclass
class PtrStoreToX86(RewritePattern):
    arch: X86Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter):
        value_type = op.value.type
        # Pointer casts
        addr_unalloc = self.arch.register_type_for_type(op.addr.type).unallocated()
        addr_cast_op = asm.ToRegOp.get(op.addr, addr_unalloc)
        x86_ptr = addr_cast_op.register
        if isa(value_type, VectorType[FixedBitwidthType]):
            x86_vect_type = self.arch.register_type_for_type(value_type)
            cast_op = asm.ToRegOp.get(op.value, x86_vect_type.unallocated())
            x86_data = cast_op.register
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
            value_unalloc = self.arch.register_type_for_type(value_type).unallocated()
            cast_op = asm.ToRegOp.get(op.value, value_unalloc)
            x86_data = cast_op.register
            mov_op = x86.MS_MovOp(x86_ptr, x86_data, memory_offset=0)

        rewriter.replace_op(op, [addr_cast_op, cast_op, mov_op])


@dataclass
class PtrLoadToX86(RewritePattern):
    arch: X86Arch

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter):
        # Pointer cast
        x86_reg_type = self.arch.register_type_for_type(op.addr.type).unallocated()
        assert isinstance(x86_reg_type, GeneralRegisterType)
        cast_op = asm.ToRegOp.get(op.addr, x86_reg_type)
        addr_x86 = cast_op.register
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
            mov_op = x86.DM_MovOp(addr_x86, memory_offset=0, destination=x86_reg_type)

        res_cast_op = asm.FromRegOp.get(mov_op.results[0], value_type)
        rewriter.replace_op(op, [cast_op, mov_op, res_cast_op])


@dataclass(frozen=True)
class ConvertPtrToX86Pass(ModulePass):
    name = "convert-ptr-to-x86"

    arch: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        arch = X86Arch.arch_for_name(self.arch)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PtrLoadToX86(arch),
                    PtrStoreToX86(arch),
                    PtrAddToX86(arch),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
