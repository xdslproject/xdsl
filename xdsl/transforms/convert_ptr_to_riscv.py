from dataclasses import dataclass
from typing import cast

from xdsl.backend.riscv.lowering.utils import (
    cast_operands_to_regs,
    register_type_for_type,
)
from xdsl.context import Context
from xdsl.dialects import ptr, riscv
from xdsl.dialects.builtin import (
    AnyFloat,
    Float32Type,
    Float64Type,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException


class PtrTypeConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: ptr.PtrType) -> riscv.IntRegisterType:
        return riscv.Registers.UNALLOCATED_INT


@dataclass
class ConvertPtrAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter, /):
        oper1, oper2 = cast_operands_to_regs(rewriter)
        rewriter.replace_op(op, riscv.AddOp(oper1, oper2))


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter, /):
        addr, value = cast_operands_to_regs(rewriter)

        match value.type:
            case riscv.IntRegisterType():
                new_op = riscv.SwOp(
                    addr, value, 0, comment="store int value to pointer"
                )
            case riscv.FloatRegisterType():
                float_type = cast(AnyFloat, op.value.type)
                match float_type:
                    case Float32Type():
                        new_op = riscv.FSwOp(
                            addr,
                            value,
                            0,
                            comment="store float value to pointer",
                        )
                    case Float64Type():
                        new_op = riscv.FSdOp(
                            addr,
                            value,
                            0,
                            comment="store double value to pointer",
                        )
                    case _:
                        raise DiagnosticException(
                            f"Lowering memref.store op with floating point type {float_type} not yet implemented"
                        )
            case _:
                raise ValueError(f"Unexpected register type {op.value.type}")

        rewriter.replace_op(op, new_op)


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter, /):
        casted = cast_operands_to_regs(rewriter)
        addr = casted[0]

        result_register_type = register_type_for_type(op.res.type)

        if issubclass(result_register_type, riscv.IntRegisterType):
            lw_op = riscv.LwOp(addr, 0, comment="load word from pointer")
        else:
            float_type = cast(AnyFloat, op.res.type)
            match float_type:
                case Float32Type():
                    lw_op = riscv.FLwOp(addr, 0, comment="load float from pointer")
                case Float64Type():
                    lw_op = riscv.FLdOp(addr, 0, comment="load double from pointer")
                case _:
                    raise DiagnosticException(
                        f"Lowering memref.load op with floating point type {float_type} not yet implemented"
                    )

        rewriter.replace_op(
            op,
            (lw := lw_op, UnrealizedConversionCastOp.get(lw.results, (op.res.type,))),
        )


@dataclass
class ConvertMemRefToPtrOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.ToPtrOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            UnrealizedConversionCastOp.get(
                (op.source,), (riscv.Registers.UNALLOCATED_INT,)
            ),
        )


class ConvertPtrToRiscvPass(ModulePass):
    name = "convert-ptr-to-riscv"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PtrTypeConversion(),
                    ConvertPtrAddOp(),
                    ConvertStoreOp(),
                    ConvertLoadOp(),
                    ConvertMemRefToPtrOp(),
                ]
            ),
        ).rewrite_module(op)
