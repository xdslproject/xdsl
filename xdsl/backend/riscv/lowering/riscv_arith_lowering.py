import ctypes

from xdsl.backend.riscv.lowering.lower_utils import cast_values_to_registers
from xdsl.dialects import arith, riscv
from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir.core import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
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
        if isinstance(op.result.type, arith.IntegerType) and isinstance(
            op.value, IntegerAttr
        ):
            if op.result.type.width.data <= 32:
                rewriter.replace_matched_op(
                    [
                        li := riscv.LiOp(op.value.value.data),
                        UnrealizedConversionCastOp.get(li.results, (op.result.type,)),
                    ]
                )
            else:
                raise NotImplementedError("Only 32 bit integers are supported for now")
        elif isinstance(op.value, FloatAttr):
            if isinstance(op.result.type, Float32Type):
                rewriter.replace_matched_op(
                    [
                        lui := riscv.LiOp(
                            convert_float_to_int(op.value.value.data),
                            rd=riscv.RegisterType(riscv.Register()),
                        ),
                        fld := riscv.FCvtSWOp(lui.rd),
                        UnrealizedConversionCastOp.get(fld.results, (op.result.type,)),
                    ]
                )
            else:
                raise NotImplementedError("Only 32 bit floats are supported")
        elif isinstance(op.result.type, IndexType) and isinstance(
            op.value, IntegerAttr
        ):
            rewriter.replace_matched_op(
                [
                    li := riscv.LiOp(op.value.value.data),
                    UnrealizedConversionCastOp.get(li.results, (op.result.type,)),
                ]
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
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                add := riscv.AddOp(lhs, rhs),
                UnrealizedConversionCastOp.get(add.results, (op.result.type,)),
            ]
        )


class LowerArithSubi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subi, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                sub := riscv.SubOp(lhs, rhs),
                UnrealizedConversionCastOp.get(sub.results, (op.result.type,)),
            ]
        )


class LowerArithMuli(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Muli, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                mul := riscv.MulOp(lhs, rhs),
                UnrealizedConversionCastOp.get(mul.results, (op.result.type,)),
            ]
        )


class LowerArithDivUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                divu := riscv.DivuOp(lhs, rhs),
                UnrealizedConversionCastOp.get(divu.results, (op.result.type,)),
            ]
        )


class LowerArithDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivSI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                div := riscv.DivOp(lhs, rhs),
                UnrealizedConversionCastOp.get(div.results, (op.result.type,)),
            ]
        )


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
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                remu := riscv.RemuOp(lhs, rhs),
                UnrealizedConversionCastOp.get(remu.results, (op.result.type,)),
            ]
        )


class LowerArithRemSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemSI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                rem := riscv.RemOp(lhs, rhs),
                UnrealizedConversionCastOp.get(rem.results, (op.result.type,)),
            ]
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
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        # based on https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/RISCV/i32-icmp.ll
        match op.predicate.value.data:
            # eq
            case 0:
                rewriter.replace_matched_op(
                    [
                        xor_op := riscv.XorOp(lhs, rhs),
                        seqz_op := riscv.SltiuOp(xor_op, 1),
                        UnrealizedConversionCastOp.get(
                            seqz_op.results, (op.result.type,)
                        ),
                    ]
                )
            # ne
            case 1:
                rewriter.replace_matched_op(
                    [
                        zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                        xor_op := riscv.XorOp(lhs, rhs),
                        snez_op := riscv.SltuOp(zero, xor_op),
                        UnrealizedConversionCastOp.get(
                            snez_op.results, (op.result.type,)
                        ),
                    ]
                )
                pass
            # slt
            case 2:
                rewriter.replace_matched_op(
                    [
                        slt := riscv.SltOp(lhs, rhs),
                        UnrealizedConversionCastOp.get(slt.results, (op.result.type,)),
                    ]
                )
            # sle
            case 3:
                rewriter.replace_matched_op(
                    [
                        slt := riscv.SltOp(lhs, rhs),
                        xori := riscv.XoriOp(slt, 1),
                        UnrealizedConversionCastOp.get(xori.results, (op.result.type,)),
                    ]
                )
            # ult
            case 4:
                rewriter.replace_matched_op(
                    [
                        sltu := riscv.SltuOp(lhs, rhs),
                        UnrealizedConversionCastOp.get(sltu.results, (op.result.type,)),
                    ]
                )
            # ule
            case 5:
                rewriter.replace_matched_op(
                    [
                        sltu := riscv.SltuOp(lhs, rhs),
                        xori := riscv.XoriOp(sltu, 1),
                        UnrealizedConversionCastOp.get(xori.results, (op.result.type,)),
                    ]
                )
            # ugt
            case 6:
                rewriter.replace_matched_op(
                    [
                        sltu := riscv.SltuOp(rhs, lhs),
                        UnrealizedConversionCastOp.get(sltu.results, (op.result.type,)),
                    ]
                )
            # uge
            case 7:
                rewriter.replace_matched_op(
                    [
                        sltu := riscv.SltuOp(rhs, lhs),
                        xori := riscv.XoriOp(sltu, 1),
                        UnrealizedConversionCastOp.get(xori.results, (op.result.type,)),
                    ]
                )
            case _:
                raise NotImplementedError("Cmpi predicate not supported")


class LowerArithSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Select, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Select is not supported")


class LowerArithAndI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AndI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                and_ := riscv.AndOp(lhs, rhs),
                UnrealizedConversionCastOp.get(and_.results, (op.result.type,)),
            ]
        )


class LowerArithOrI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.OrI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                or_ := riscv.OrOp(lhs, rhs),
                UnrealizedConversionCastOp.get(or_.results, (op.result.type,)),
            ]
        )


class LowerArithXOrI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.XOrI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                xor := riscv.XorOp(lhs, rhs),
                UnrealizedConversionCastOp.get(xor.results, (op.result.type,)),
            ]
        )


class LowerArithShLI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShLI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                sll := riscv.SllOp(lhs, rhs),
                UnrealizedConversionCastOp.get(sll.results, (op.result.type,)),
            ]
        )


class LowerArithShRUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRUI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                srl := riscv.SrlOp(lhs, rhs),
                UnrealizedConversionCastOp.get(srl.results, (op.result.type,)),
            ]
        )


class LowerArithShRSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRSI, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                sra := riscv.SraOp(lhs, rhs),
                UnrealizedConversionCastOp.get(sra.results, (op.result.type,)),
            ]
        )


class LowerArithAddf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addf, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                fadds := riscv.FAddSOp(lhs, rhs),
                UnrealizedConversionCastOp.get(fadds.results, (op.result.type,)),
            ]
        )


class LowerArithSubf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subf, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                fsubs := riscv.FSubSOp(lhs, rhs),
                UnrealizedConversionCastOp.get(fsubs.results, (op.result.type,)),
            ]
        )


class LowerArithMulf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                fmuls := riscv.FMulSOp(lhs, rhs),
                UnrealizedConversionCastOp.get(fmuls.results, (op.result.type,)),
            ]
        )


class LowerArithDivf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Divf, rewriter: PatternRewriter) -> None:
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        rewriter.replace_matched_op(
            [
                fdivs := riscv.FMulSOp(lhs, rhs),
                UnrealizedConversionCastOp.get(fdivs.results, (op.result.type,)),
            ]
        )


class LowerArithNegf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Negf, rewriter: PatternRewriter) -> None:
        operand = cast_values_to_registers([op.operand], rewriter)
        rewriter.replace_matched_op(
            [
                fsgnjns := riscv.FSgnJNSOp(operand[0], operand[0]),
                UnrealizedConversionCastOp.get(fsgnjns.results, (op.result.type,)),
            ]
        )


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
        lhs, rhs = cast_values_to_registers([op.lhs, op.rhs], rewriter)
        match op.predicate.value.data:
            # false
            case 0:
                rewriter.replace_matched_op(
                    [
                        li := riscv.LiOp(0),
                        UnrealizedConversionCastOp.get(li.results, (op.result.type,)),
                    ]
                )
            # oeq
            case 1:
                rewriter.replace_matched_op(
                    [
                        feq := riscv.FeqSOP(lhs, rhs),
                        UnrealizedConversionCastOp.get(feq.results, (op.result.type,)),
                    ]
                )
            # ogt
            case 2:
                rewriter.replace_matched_op(
                    [
                        ogt := riscv.FltSOP(rhs, lhs),
                        UnrealizedConversionCastOp.get(ogt.results, (op.result.type,)),
                    ]
                )
            # oge
            case 3:
                rewriter.replace_matched_op(
                    [
                        fle := riscv.FleSOP(rhs, lhs),
                        UnrealizedConversionCastOp.get(fle.results, (op.result.type,)),
                    ]
                )
            # olt
            case 4:
                rewriter.replace_matched_op(
                    [
                        olt := riscv.FltSOP(lhs, rhs),
                        UnrealizedConversionCastOp.get(olt.results, (op.result.type,)),
                    ]
                )
            # ole
            case 5:
                rewriter.replace_matched_op(
                    [
                        fle := riscv.FleSOP(lhs, rhs),
                        UnrealizedConversionCastOp.get(fle.results, (op.result.type,)),
                    ]
                )
            # one
            case 6:
                rewriter.replace_matched_op(
                    [
                        flt1 := riscv.FltSOP(lhs, rhs),
                        flt2 := riscv.FltSOP(rhs, lhs),
                        or_ := riscv.OrOp(flt2, flt1),
                        UnrealizedConversionCastOp.get(or_.results, (op.result.type,)),
                    ]
                )
            # ord
            case 7:
                rewriter.replace_matched_op(
                    [
                        feq1 := riscv.FeqSOP(lhs, lhs),
                        feq2 := riscv.FeqSOP(rhs, rhs),
                        and_ := riscv.AndOp(feq2, feq1),
                        UnrealizedConversionCastOp.get(and_.results, (op.result.type,)),
                    ]
                )
            # ueq
            case 8:
                rewriter.replace_matched_op(
                    [
                        flt1 := riscv.FltSOP(lhs, rhs),
                        flt2 := riscv.FltSOP(rhs, lhs),
                        or_ := riscv.OrOp(flt2, flt1),
                        xor := riscv.XoriOp(or_, 1),
                        UnrealizedConversionCastOp.get(xor.results, (op.result.type,)),
                    ]
                )
            # ugt
            case 9:
                rewriter.replace_matched_op(
                    [
                        fle := riscv.FleSOP(lhs, rhs),
                        xor := riscv.XoriOp(fle, 1),
                        UnrealizedConversionCastOp.get(xor.results, (op.result.type,)),
                    ]
                )
            # uge
            case 10:
                rewriter.replace_matched_op(
                    [
                        fle := riscv.FltSOP(lhs, rhs),
                        xor := riscv.XoriOp(fle, 1),
                        UnrealizedConversionCastOp.get(xor.results, (op.result.type,)),
                    ]
                )
            # ult
            case 11:
                rewriter.replace_matched_op(
                    [
                        fle := riscv.FleSOP(rhs, lhs),
                        xor := riscv.XoriOp(fle, 1),
                        UnrealizedConversionCastOp.get(xor.results, (op.result.type,)),
                    ]
                )
            # ule
            case 12:
                flt = riscv.FltSOP(rhs, lhs)
                rewriter.replace_matched_op([flt, riscv.XoriOp(flt, 1)])
            # une
            case 13:
                rewriter.replace_matched_op(
                    [
                        feq := riscv.FeqSOP(lhs, rhs),
                        xor := riscv.XoriOp(feq, 1),
                        UnrealizedConversionCastOp.get(xor.results, (op.result.type,)),
                    ]
                )
            # uno
            case 14:
                rewriter.replace_matched_op(
                    [
                        feq1 := riscv.FeqSOP(lhs, lhs),
                        feq2 := riscv.FeqSOP(rhs, rhs),
                        and_ := riscv.AndOp(feq2, feq1),
                        riscv.XoriOp(and_, 1),
                        UnrealizedConversionCastOp.get(and_.results, (op.result.type,)),
                    ]
                )
            # true
            case 15:
                rewriter.replace_matched_op(
                    [
                        li := riscv.LiOp(1),
                        UnrealizedConversionCastOp.get(li.results, (op.result.type,)),
                    ]
                )
            case _:
                raise NotImplementedError("Cmpf predicate not supported")


class LowerArithSIToFPOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SIToFPOp, rewriter: PatternRewriter) -> None:
        input = cast_values_to_registers([op.input], rewriter)
        rewriter.replace_matched_op(
            [
                fcvtsw := riscv.FCvtSWOp(input[0]),
                UnrealizedConversionCastOp.get(fcvtsw.results, (op.result.type,)),
            ]
        )


class LowerArithFPToSIOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FPToSIOp, rewriter: PatternRewriter) -> None:
        input = cast_values_to_registers([op.input], rewriter)
        rewriter.replace_matched_op(
            [
                fcvtws := riscv.FCvtWSOp(input[0]),
                UnrealizedConversionCastOp.get(fcvtws.results, (op.result.type,)),
            ]
        )


class LowerArithExtFOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ExtFOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("ExtF is not supported")


class LowerArithTruncFOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.TruncFOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("TruncF is not supported")


class RISCVLowerArith(ModulePass):
    name = "lower-arith-to-riscv"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerArithConstant(),
                    LowerArithIndexCast(),
                    LowerArithSIToFPOp(),
                    LowerArithFPToSIOp(),
                    LowerArithAddi(),
                    LowerArithSubi(),
                    LowerArithMuli(),
                    LowerArithDivUI(),
                    LowerArithDivSI(),
                    LowerArithFloorDivSI(),
                    LowerArithRemSI(),
                    LowerArithCmpi(),
                    LowerArithAddf(),
                    LowerArithSubf(),
                    LowerArithDivf(),
                    LowerArithNegf(),
                    LowerArithMulf(),
                    LowerArithCmpf(),
                    LowerArithRemUI(),
                    LowerArithAndI(),
                    LowerArithOrI(),
                    LowerArithXOrI(),
                    LowerArithShLI(),
                    LowerArithShRUI(),
                    LowerArithShRSI(),
                    LowerArithCeilDivSI(),
                    LowerArithCeilDivUI(),
                    LowerArithMinSI(),
                    LowerArithMaxSI(),
                    LowerArithMinUI(),
                    LowerArithMaxUI(),
                    LowerArithSelect(),
                    LowerArithExtFOp(),
                    LowerArithTruncFOp(),
                    LowerArithMinfOp(),
                    LowerArithMaxfOp(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)

        dce(op)
