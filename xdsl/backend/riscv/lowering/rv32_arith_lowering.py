import ctypes
from xdsl.dialects.builtin import (
    Float32Type,
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
            if op.result.typ.width.data <= 32:
                rewriter.replace_op(
                    op,
                    riscv.LiOp(op.value.value.data),
                )
            else:
                raise NotImplementedError("Only 32 bit integers are supported for now")
        elif isinstance(op.value, FloatAttr):
            if isinstance(op.result.typ, Float32Type):
                lui = riscv.LiOp(
                    convert_float_to_int(op.value.value.data),
                    rd=riscv.RegisterType(riscv.Register()),
                )
                fld = riscv.FCvtSWOp(lui.rd)
                rewriter.replace_op(op, [lui, fld])
            else:
                raise NotImplementedError("Only 32 bit floats are supported")
        elif isinstance(op.result.typ, IndexType) and isinstance(op.value, IntegerAttr):
            rewriter.replace_op(
                op,
                riscv.LiOp(op.value.value.data),
            )
        else:
            raise NotImplementedError(
                f"Unsupported constant type {op.value} of type {type(op.value)}"
            )


class LowerArithIndexCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.IndexCastOp, rewriter: PatternRewriter
    ) -> None:
        """
        On a RV32 triple, the index type is 32 bits, so we can just drop the cast.
        """

        rewriter.replace_matched_op([], [op.input])


class LowerArithAddi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, riscv.AddOp(op.lhs, op.rhs))


class LowerArithSubi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subi, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, riscv.SubOp(op.lhs, op.rhs))


class LowerArithMuli(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Muli, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, riscv.MulOp(op.lhs, op.rhs))


class LowerArithDivUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUI, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, riscv.DivuOp(op.lhs, op.rhs))


class LowerArithDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivSI, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(op, riscv.DivOp(op.lhs, op.rhs))


class LowerArithFloorDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.FloorDivSI, rewriter: PatternRewriter
    ) -> None:
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
        rewriter.replace_matched_op(riscv.RemuOp(op.lhs, op.rhs))


class LowerArithRemSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemSI, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.RemOp(op.lhs, op.rhs)])


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
                zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
                xor_op = riscv.XorOp(op.lhs, op.rhs)
                snez_op = riscv.SltuOp(zero, xor_op)
                rewriter.replace_matched_op([xor_op, snez_op])
                pass
            # slt
            case 2:
                slt = riscv.SltOp(op.lhs, op.rhs)
                rewriter.replace_matched_op([slt])
            # sle
            case 3:
                slt = riscv.SltOp(op.lhs, op.rhs)
                xori = riscv.XoriOp(slt, 1)
                rewriter.replace_matched_op([slt, xori])
            # ult
            case 4:
                sltu = riscv.SltuOp(op.lhs, op.rhs)
                rewriter.replace_matched_op([sltu])
            # ule
            case 5:
                sltu = riscv.SltuOp(op.lhs, op.rhs)
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_matched_op([sltu, xori])
            # ugt
            case 6:
                sltu = riscv.SltuOp(op.rhs, op.lhs)
                rewriter.replace_matched_op([sltu])
            # uge
            case 7:
                sltu = riscv.SltuOp(op.rhs, op.lhs)
                xori = riscv.XoriOp(sltu, 1)
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
        rewriter.replace_matched_op([riscv.AndOp(op.lhs, op.rhs)])


class LowerArithOrI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.OrI, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.OrOp(op.lhs, op.rhs)])


class LowerArithXOrI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.XOrI, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.XorOp(op.lhs, op.rhs)])


class LowerArithShLI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShLI, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.SllOp(op.lhs, op.rhs)])


class LowerArithShRUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRUI, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.SrlOp(op.lhs, op.rhs)])


class LowerArithShRSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRSI, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.SraOp(op.lhs, op.rhs)])


class LowerArithAddf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FAddSOp(op.lhs, op.rhs)])


class LowerArithSubf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FSubSOp(op.lhs, op.rhs)])


class LowerArithMulf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FMulSOp(op.lhs, op.rhs)])


class LowerArithDivf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Divf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FDivSOp(op.lhs, op.rhs)])


class LowerArithNegf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Negf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FSgnJNSOp(op.operand, op.operand)])


class LowerArithMinfOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Minf, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Minf is not supported")


class LowerArithMaxfOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Maxf, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Maxf is not supported")


class LowerArithCmpf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Cmpf, rewriter: PatternRewriter) -> None:
        match op.predicate.value.data:
            # false
            case 0:
                rewriter.replace_matched_op([riscv.LiOp(0)])
            # oeq
            case 1:
                rewriter.replace_matched_op([riscv.FeqSOP(op.lhs, op.rhs)])
            # ogt
            case 2:
                rewriter.replace_matched_op([riscv.FltSOP(op.rhs, op.lhs)])
            # oge
            case 3:
                rewriter.replace_matched_op([riscv.FleSOP(op.rhs, op.lhs)])
            # olt
            case 4:
                rewriter.replace_matched_op([riscv.FltSOP(op.lhs, op.rhs)])
            # ole
            case 5:
                rewriter.replace_matched_op([riscv.FleSOP(op.lhs, op.rhs)])
            # one
            case 6:
                flt1 = riscv.FltSOP(op.lhs, op.rhs)
                flt2 = riscv.FltSOP(op.rhs, op.lhs)
                rewriter.replace_matched_op([flt1, flt2, riscv.OrOp(flt2, flt1)])
            # ord
            case 7:
                feq1 = riscv.FeqSOP(op.lhs, op.lhs)
                feq2 = riscv.FeqSOP(op.rhs, op.rhs)
                rewriter.replace_matched_op([feq1, feq2, riscv.AndOp(feq2, feq1)])
            # ueq
            case 8:
                flt1 = riscv.FltSOP(op.lhs, op.rhs)
                flt2 = riscv.FltSOP(op.rhs, op.lhs)
                or_ = riscv.OrOp(flt2, flt1)
                rewriter.replace_matched_op([flt1, flt2, or_, riscv.XoriOp(or_, 1)])
            # ugt
            case 9:
                fle = riscv.FleSOP(op.lhs, op.rhs)
                rewriter.replace_matched_op([fle, riscv.XoriOp(fle, 1)])
            # uge
            case 10:
                fle = riscv.FltSOP(op.lhs, op.rhs)
                rewriter.replace_matched_op([fle, riscv.XoriOp(fle, 1)])
            # ult
            case 11:
                fle = riscv.FleSOP(op.rhs, op.lhs)
                rewriter.replace_matched_op([fle, riscv.XoriOp(fle, 1)])
            # ule
            case 12:
                flt = riscv.FltSOP(op.rhs, op.lhs)
                rewriter.replace_matched_op([flt, riscv.XoriOp(flt, 1)])
            # une
            case 13:
                feq = riscv.FeqSOP(op.lhs, op.rhs)
                rewriter.replace_matched_op([feq, riscv.XoriOp(feq, 1)])
            # uno
            case 14:
                feq1 = riscv.FeqSOP(op.lhs, op.lhs)
                feq2 = riscv.FeqSOP(op.rhs, op.rhs)
                and_ = riscv.AndOp(feq2, feq1)
                rewriter.replace_matched_op([feq1, feq2, and_, riscv.XoriOp(and_, 1)])
            # true
            case 15:
                rewriter.replace_matched_op([riscv.LiOp(1)])
            case _:
                raise NotImplementedError("Cmpf predicate not supported")


class LowerArithSIToFPOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SIToFPOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FCvtSWOp(op.input)])


class LowerArithFPToSIOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FPToSIOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op([riscv.FCvtWSOp(op.input)])


class LowerArithExtFOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ExtFOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("ExtF is not supported")


class LowerArithTruncFOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.TruncFOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("TruncF is not supported")


class LowerArithRV32(ModulePass):
    name = "lower-arith-rv32"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # Implemented lowerings
        PatternRewriteWalker(LowerArithConstant()).rewrite_module(op)
        PatternRewriteWalker(LowerArithIndexCast()).rewrite_module(op)
        PatternRewriteWalker(LowerArithSIToFPOp()).rewrite_module(op)
        PatternRewriteWalker(LowerArithFPToSIOp()).rewrite_module(op)

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
        PatternRewriteWalker(LowerArithNegf()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMulf()).rewrite_module(op)
        PatternRewriteWalker(LowerArithCmpf()).rewrite_module(op)

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

        PatternRewriteWalker(LowerArithExtFOp()).rewrite_module(op)
        PatternRewriteWalker(LowerArithTruncFOp()).rewrite_module(op)

        PatternRewriteWalker(LowerArithMinfOp()).rewrite_module(op)
        PatternRewriteWalker(LowerArithMaxfOp()).rewrite_module(op)

        dce(op)
