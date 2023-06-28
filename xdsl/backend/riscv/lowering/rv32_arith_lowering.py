import ctypes
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    ModuleOp,
)
from xdsl.ir.core import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects import arith, riscv
from xdsl.transforms.dead_code_elimination import dce

# Various helper functions


def convert_float_to_int(value: float) -> int:
    """
    Convert an IEEE 754 float to a raw integer representation, useful for loading constants.
    """
    raw_float = ctypes.c_float(value)
    raw_int = ctypes.c_int.from_address(ctypes.addressof(raw_float)).value
    return raw_int


class LowerArithConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter) -> None:
        if isinstance(op.result.typ, arith.IntegerType) and isinstance(
            op.value, IntegerAttr
        ):
            assert (
                op.result.typ.width.data <= 32
            ), "64 bit operations not supported on RV32 triple"
            rewriter.replace_op(
                op,
                riscv.LiOp(
                    op.value.value.data, rd=riscv.RegisterType(riscv.Register())
                ),
            )
        elif isinstance(op.result.typ, Float32Type) and isinstance(op.value, FloatAttr):
            lui = riscv.LiOp(
                convert_float_to_int(op.value.value.data),
                rd=riscv.RegisterType(riscv.Register()),
            )
            fld = riscv.FCvtSWOp(lui.rd, rd=riscv.FloatRegisterType(riscv.Register()))
            rewriter.replace_op(op, [lui, fld])
        elif isinstance(op.result.typ, IndexType):
            assert isinstance(op.value, IntegerAttr)
            rewriter.replace_op(
                op,
                riscv.LiOp(
                    op.value.value.data, rd=riscv.RegisterType(riscv.Register())
                ),
            )


class LowerArithIndexCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.IndexCastOp, rewriter: PatternRewriter
    ) -> None:
        old_uses = set(op.result.uses)
        for use in old_uses:
            use.operation.replace_operand(use.index, op.input)
        rewriter.erase_matched_op()


class LowerArithAddi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.IntegerType):
            assert (
                op.lhs.width.data <= 32
            ), "64 bit operations not supported on RV32 triple"
            rewriter.replace_op(
                op,
                [riscv.AddOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))],
            )
        else:
            rewriter.replace_op(
                op,
                [riscv.AddOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))],
            )


class LowerArithSubi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subi, rewriter: PatternRewriter) -> None:
        assert isinstance(op.lhs, arith.IntegerType) and isinstance(
            op.rhs, arith.IntegerType
        )
        assert op.lhs.width.data <= 32, "64 bit operations not supported on RV32 triple"
        rewriter.replace_op(
            op, riscv.SubOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))
        )


class LowerArithMuli(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Muli, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.IntegerType):
            assert (
                op.lhs.width.data <= 32
            ), "64 bit operations not supported on RV32 triple"
            rewriter.replace_op(
                op,
                [riscv.MulOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))],
            )
        else:
            rewriter.replace_op(
                op,
                [riscv.MulOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))],
            )


class LowerArithDivUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUI, rewriter: PatternRewriter) -> None:
        assert isinstance(op.lhs, arith.IntegerType) and isinstance(
            op.rhs, arith.IntegerType
        )
        assert op.lhs.width.data <= 32, "64 bit operations not supported on RV32 triple"
        rewriter.replace_op(
            op, riscv.DivuOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))
        )


class LowerArithDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivSI, rewriter: PatternRewriter) -> None:
        assert isinstance(op.lhs, arith.IntegerType) and isinstance(
            op.rhs, arith.IntegerType
        )
        assert op.lhs.width.data <= 32, "64 bit operations not supported on RV32 triple"

        rewriter.replace_op(
            op, riscv.DivOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))
        )


class LowerArithFloorDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.FloorDivSI, rewriter: PatternRewriter
    ) -> None:
        assert isinstance(op.lhs, arith.IntegerType) and isinstance(
            op.rhs, arith.IntegerType
        )
        assert op.lhs.width.data <= 32, "64 bit operations not supported on RV32 triple"

        raise NotImplementedError("FloorDivSI is not supported")


class LowerArithCeilDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CeilDivSI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("CeilDivSI is not supported")


class LowerArithCeilDivUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CeilDivUI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("CeilDivUI is not supported")


class LowerArithRemUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemUI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("RemUI is not supported")


class LowerArithRemSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemSI, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.IntegerType):
            assert (
                op.lhs.width.data <= 32
            ), "64 bit operations not supported on RV32 triple"
            rewriter.replace_matched_op(
                [riscv.RemOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))]
            )
        else:
            rewriter.replace_matched_op(
                [riscv.RemOp(op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register()))]
            )


class LowerArithMinSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MinSI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MinSI is not supported")


class LowerArithMaxSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MaxSI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MaxSI is not supported")


class LowerArithMinUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MinUI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MinUI is not supported")


class LowerArithMaxUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MaxUI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MaxUI is not supported")


class LowerArithCmpi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Cmpi, rewriter: PatternRewriter) -> None:
        # based on https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/RISCV/i32-icmp.ll
        match op.predicate.value.data:
            # eq
            case 0:
                xor_op = riscv.XorOp(op.lhs, op.rhs)
                seqz_op = riscv.SltiuOp(xor_op, 1)
                rewriter.replace_matched_op([xor_op, seqz_op])
            # ne
            case 1:
                xor_op = riscv.XorOp(op.lhs, op.rhs)
                snez_op = riscv.SltuOp(
                    riscv.GetRegisterOp(riscv.Registers.ZERO), xor_op
                )
                rewriter.replace_matched_op([xor_op, snez_op])
                pass
            # slt
            case 2:
                slt = riscv.SltOp(
                    op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register())
                )
                rewriter.replace_matched_op([slt])
            # sle
            case 3:
                slt = riscv.SltOp(
                    op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register())
                )
                xori = riscv.XoriOp(slt, 1, rd=riscv.RegisterType(riscv.Register()))
                rewriter.replace_matched_op([slt, xori])
            # ult
            case 4:
                sltu = riscv.SltuOp(
                    op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register())
                )
                rewriter.replace_matched_op([sltu])
            # ule
            case 5:
                sltu = riscv.SltuOp(
                    op.lhs, op.rhs, rd=riscv.RegisterType(riscv.Register())
                )
                xori = riscv.XoriOp(sltu, 1, rd=riscv.RegisterType(riscv.Register()))
                rewriter.replace_matched_op([sltu, xori])
            # ugt
            case 6:
                sltu = riscv.SltuOp(
                    op.rhs, op.lhs, rd=riscv.RegisterType(riscv.Register())
                )
                rewriter.replace_matched_op([sltu])
            # uge
            case 7:
                sltu = riscv.SltuOp(
                    op.rhs, op.lhs, rd=riscv.RegisterType(riscv.Register())
                )
                xori = riscv.XoriOp(sltu, 1, rd=riscv.RegisterType(riscv.Register()))
                rewriter.replace_matched_op([sltu, xori])
            case _:
                raise NotImplementedError("Cmpi predicate not supported")


class LowerArithSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Select, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Select is not supported")


class LowerArithAndI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AndI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("AndI is not supported")


class LowerArithOrI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.OrI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("OrI is not supported")


class LowerArithXOrI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.XOrI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("XorI is not supported")


class LowerArithShLI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShLI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("ShLI is not supported")


class LowerArithShRUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRUI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("ShRUI is not supported")


class LowerArithShRSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRSI, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("ShRSI is not supported")


class LowerArithAddf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.FloatAttr):
            assert not isinstance(
                op.lhs.typ, Float64Type
            ), "Double precision not supported on RV32 triple"
            rewriter.replace_matched_op(
                [
                    riscv.FAddSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )
        else:
            rewriter.replace_matched_op(
                [
                    riscv.FAddSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )


class LowerArithSubf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.FloatAttr):
            assert not isinstance(
                op.lhs.typ, Float64Type
            ), "Double precision not supported on RV32 triple"
            rewriter.replace_matched_op(
                [
                    riscv.FSubSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )
        else:
            rewriter.replace_matched_op(
                [
                    riscv.FSubSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )


class LowerArithMulf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.FloatAttr):
            assert not isinstance(
                op.lhs.typ, Float64Type
            ), "Double precision not supported on RV32 triple"
            rewriter.replace_matched_op(
                [
                    riscv.FMulSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )
        else:
            rewriter.replace_matched_op(
                [
                    riscv.FMulSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )


class LowerArithDivf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Divf, rewriter: PatternRewriter) -> None:
        if isinstance(op.lhs, arith.FloatAttr):
            assert not isinstance(
                op.lhs.typ, Float64Type
            ), "Double precision not supported on RV32 triple"
            rewriter.replace_matched_op(
                [
                    riscv.FDivSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )
        else:
            rewriter.replace_matched_op(
                [
                    riscv.FDivSOp(
                        op.lhs, op.rhs, rd=riscv.FloatRegisterType(riscv.Register())
                    )
                ]
            )


class LowerArithSIToFPOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SIToFPOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.input, arith.IntegerType):
            assert (
                op.input.width.data <= 32
            ), "64 bit operations not supported on RV32 triple"
            rewriter.replace_matched_op(
                [riscv.FCvtSWOp(op.input, rd=riscv.FloatRegisterType(riscv.Register()))]
            )
        else:
            rewriter.replace_matched_op(
                [riscv.FCvtSWOp(op.input, rd=riscv.FloatRegisterType(riscv.Register()))]
            )


class LowerArithRV32(ModulePass):
    name = "lower-arith-rv32"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # Implemented lowerings
        PatternRewriteWalker(LowerArithConstant()).rewrite_module(op)
        PatternRewriteWalker(LowerArithIndexCast()).rewrite_module(op)

        PatternRewriteWalker(LowerArithAddi()).rewrite_module(op)
        PatternRewriteWalker(LowerArithSubi()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMuli()).rewrite_module(op)
        PatternRewriteWalker(LowerArithDivUI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithDivSI()).rewrite_module(op)

        PatternRewriteWalker(LowerArithFloorDivSI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithRemSI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithCmpi()).rewrite_module(op)

        PatternRewriteWalker(LowerArithAddf()).rewrite_module(op)
        PatternRewriteWalker(LowerArithSubf()).rewrite_module(op)
        PatternRewriteWalker(LowerArithDivf()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMulf()).rewrite_module(op)
        PatternRewriteWalker(LowerArithSIToFPOp()).rewrite_module(op)

        # Unimplemented lowerings
        PatternRewriteWalker(LowerArithCeilDivSI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithCeilDivUI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithRemUI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMinSI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMaxSI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMinUI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMaxUI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithSelect()).rewrite_module(op)
        PatternRewriteWalker(LowerArithAndI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithOrI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithXOrI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithShLI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithShRUI()).rewrite_module(op)
        PatternRewriteWalker(LowerArithShRSI()).rewrite_module(op)

        dce(op)
