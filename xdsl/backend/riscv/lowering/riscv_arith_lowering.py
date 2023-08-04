import ctypes
from typing import overload

from xdsl.backend.riscv.lowering.utils import (
    cast_matched_op_results,
    cast_operands_to_float_regs,
    cast_operands_to_int_regs,
)
from xdsl.dialects import arith, riscv
from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir.core import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.dead_code_elimination import dce

_INT_REGISTER_TYPE = riscv.IntRegisterType.unallocated()
_FLOAT_REGISTER_TYPE = riscv.FloatRegisterType.unallocated()


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
        op_result_type = op.result.type
        if isinstance(op_result_type, IntegerType) and isinstance(
            op.value, IntegerAttr
        ):
            if op_result_type.width.data <= 32:
                rewriter.replace_matched_op(
                    [
                        constant := riscv.LiOp(op.value.value.data),
                        UnrealizedConversionCastOp.get(
                            constant.results, (op_result_type,)
                        ),
                    ],
                )
            else:
                raise NotImplementedError("Only 32 bit integers are supported for now")
        elif isinstance(op.value, FloatAttr):
            if isinstance(op_result_type, Float32Type):
                rewriter.replace_matched_op(
                    [
                        lui := riscv.LiOp(
                            convert_float_to_int(op.value.value.data),
                            rd=_INT_REGISTER_TYPE,
                        ),
                        fld := riscv.FCvtSWOp(lui.rd),
                        UnrealizedConversionCastOp.get(fld.results, (op_result_type,)),
                    ],
                )
            else:
                raise NotImplementedError("Only 32 bit floats are supported")
        elif isinstance(op_result_type, IndexType) and isinstance(
            op.value, IntegerAttr
        ):
            rewriter.replace_matched_op(
                [
                    constant := riscv.LiOp(op.value.value.data),
                    UnrealizedConversionCastOp.get(constant.results, (op_result_type,)),
                ],
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

        rewriter.replace_matched_op(
            UnrealizedConversionCastOp.get((op.input,), (op.result.type,))
        )


class LowerBinaryOp(RewritePattern):
    arith_op_cls: type[arith.SignlessIntegerBinaryOp] | type[
        arith.FloatingPointLikeBinaryOp
    ]
    riscv_op_cls: type[riscv.RdRsRsIntegerOperation] | type[riscv.RdRsRsFloatOperation]
    register_type: riscv.IntRegisterType | riscv.FloatRegisterType

    @overload
    def __init__(
        self,
        arith_op_cls: type[arith.SignlessIntegerBinaryOp],
        riscv_op_cls: type[riscv.RdRsRsIntegerOperation],
        register_type: riscv.IntRegisterType,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arith_op_cls: type[arith.FloatingPointLikeBinaryOp],
        riscv_op_cls: type[riscv.RdRsRsFloatOperation],
        register_type: riscv.FloatRegisterType,
    ) -> None:
        ...

    def __init__(
        self,
        arith_op_cls: type[arith.SignlessIntegerBinaryOp]
        | type[arith.FloatingPointLikeBinaryOp],
        riscv_op_cls: type[riscv.RdRsRsIntegerOperation]
        | type[riscv.RdRsRsFloatOperation],
        register_type: riscv.IntRegisterType | riscv.FloatRegisterType,
    ) -> None:
        self.arith_op_cls = arith_op_cls
        self.riscv_op_cls = riscv_op_cls
        self.register_type = register_type

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        lhs = UnrealizedConversionCastOp.get((op.lhs,), (self.register_type,))
        rhs = UnrealizedConversionCastOp.get((op.rhs,), (self.register_type,))
        add = self.riscv_op_cls(lhs, rhs)
        cast = UnrealizedConversionCastOp.get((add.rd,), (op.result.type,))

        rewriter.replace_matched_op((lhs, rhs, add, cast))


def lower_signless_integer_binary_op(
    arith_op_cls: type[arith.SignlessIntegerBinaryOp],
    riscv_op_cls: type[riscv.RdRsRsIntegerOperation],
) -> LowerBinaryOp:
    return LowerBinaryOp(arith_op_cls, riscv_op_cls, _INT_REGISTER_TYPE)


def lower_float_binary_op(
    arith_op_cls: type[arith.FloatingPointLikeBinaryOp],
    riscv_op_cls: type[riscv.RdRsRsFloatOperation],
) -> LowerBinaryOp:
    return LowerBinaryOp(arith_op_cls, riscv_op_cls, _FLOAT_REGISTER_TYPE)


lower_arith_addi = lower_signless_integer_binary_op(arith.Addi, riscv.AddOp)
lower_arith_subi = lower_signless_integer_binary_op(arith.Subi, riscv.SubOp)
lower_arith_muli = lower_signless_integer_binary_op(arith.Muli, riscv.MulOp)
lower_arith_divui = lower_signless_integer_binary_op(arith.DivUI, riscv.DivuOp)
lower_arith_divsi = lower_signless_integer_binary_op(arith.DivSI, riscv.DivOp)


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


lower_arith_remui = lower_signless_integer_binary_op(arith.RemUI, riscv.RemuOp)
lower_arith_remsi = lower_signless_integer_binary_op(arith.RemSI, riscv.RemOp)


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
        lhs, rhs = cast_operands_to_int_regs(rewriter)
        cast_matched_op_results(rewriter)

        match op.predicate.value.data:
            # eq
            case 0:
                xor_op = riscv.XorOp(lhs, rhs)
                seqz_op = riscv.SltiuOp(xor_op, 1)
                rewriter.replace_matched_op([xor_op, seqz_op])
            # ne
            case 1:
                zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
                xor_op = riscv.XorOp(lhs, rhs)
                snez_op = riscv.SltuOp(zero, xor_op)
                rewriter.replace_matched_op([zero, xor_op, snez_op])
            # slt
            case 2:
                rewriter.replace_matched_op([riscv.SltOp(lhs, rhs)])
            # sle
            case 3:
                slt = riscv.SltOp(lhs, rhs)
                xori = riscv.XoriOp(slt, 1)
                rewriter.replace_matched_op([slt, xori])
            # ult
            case 4:
                rewriter.replace_matched_op([riscv.SltuOp(lhs, rhs)])
            # ule
            case 5:
                sltu = riscv.SltuOp(lhs, rhs)
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_matched_op([sltu, xori])
            # ugt
            case 6:
                rewriter.replace_matched_op([riscv.SltuOp(rhs, lhs)])
            # uge
            case 7:
                sltu = riscv.SltuOp(rhs, lhs)
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_matched_op([sltu, xori])
            case _:
                raise NotImplementedError("Cmpi predicate not supported")


class LowerArithSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Select, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Select is not supported")


lower_arith_andi = lower_signless_integer_binary_op(arith.AndI, riscv.AndOp)
lower_arith_ori = lower_signless_integer_binary_op(arith.OrI, riscv.OrOp)
lower_arith_xori = lower_signless_integer_binary_op(arith.XOrI, riscv.XorOp)
lower_arith_shli = lower_signless_integer_binary_op(arith.ShLI, riscv.SllOp)
lower_arith_shrui = lower_signless_integer_binary_op(arith.ShRUI, riscv.SrlOp)
lower_arith_shrsi = lower_signless_integer_binary_op(arith.ShRSI, riscv.SraOp)


lower_arith_addf = lower_float_binary_op(arith.Addf, riscv.FAddSOp)
lower_arith_subf = lower_float_binary_op(arith.Subf, riscv.FSubSOp)
lower_arith_mulf = lower_float_binary_op(arith.Mulf, riscv.FMulSOp)
lower_arith_divf = lower_float_binary_op(arith.Divf, riscv.FDivSOp)


class LowerArithNegf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Negf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op(
            (
                operand := UnrealizedConversionCastOp.get(
                    (op.operand,), (_FLOAT_REGISTER_TYPE,)
                ),
                negf := riscv.FSgnJNSOp(operand, operand),
                UnrealizedConversionCastOp.get((negf.rd,), (op.result.type,)),
            )
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
        # https://llvm.org/docs/LangRef.html#id309
        lhs, rhs = cast_operands_to_float_regs(rewriter)
        cast_matched_op_results(rewriter)

        match op.predicate.value.data:
            # false
            case 0:
                rewriter.replace_matched_op([riscv.LiOp(0)])
            # oeq
            case 1:
                rewriter.replace_matched_op([riscv.FeqSOP(lhs, rhs)])
            # ogt
            case 2:
                rewriter.replace_matched_op([riscv.FltSOP(rhs, lhs)])
            # oge
            case 3:
                rewriter.replace_matched_op([riscv.FleSOP(rhs, lhs)])
            # olt
            case 4:
                rewriter.replace_matched_op([riscv.FltSOP(lhs, rhs)])
            # ole
            case 5:
                rewriter.replace_matched_op([riscv.FleSOP(lhs, rhs)])
            # one
            case 6:
                flt1 = riscv.FltSOP(lhs, rhs)
                flt2 = riscv.FltSOP(rhs, lhs)
                rewriter.replace_matched_op([flt1, flt2, riscv.OrOp(flt2, flt1)])
            # ord
            case 7:
                feq1 = riscv.FeqSOP(lhs, lhs)
                feq2 = riscv.FeqSOP(rhs, rhs)
                rewriter.replace_matched_op([feq1, feq2, riscv.AndOp(feq2, feq1)])
            # ueq
            case 8:
                flt1 = riscv.FltSOP(lhs, rhs)
                flt2 = riscv.FltSOP(rhs, lhs)
                or_ = riscv.OrOp(flt2, flt1)
                rewriter.replace_matched_op([flt1, flt2, or_, riscv.XoriOp(or_, 1)])
            # ugt
            case 9:
                fle = riscv.FleSOP(lhs, rhs)
                rewriter.replace_matched_op([fle, riscv.XoriOp(fle, 1)])
            # uge
            case 10:
                fle = riscv.FltSOP(lhs, rhs)
                rewriter.replace_matched_op([fle, riscv.XoriOp(fle, 1)])
            # ult
            case 11:
                fle = riscv.FleSOP(rhs, lhs)
                rewriter.replace_matched_op([fle, riscv.XoriOp(fle, 1)])
            # ule
            case 12:
                flt = riscv.FltSOP(rhs, lhs)
                rewriter.replace_matched_op([flt, riscv.XoriOp(flt, 1)])
            # une
            case 13:
                feq = riscv.FeqSOP(lhs, rhs)
                rewriter.replace_matched_op([feq, riscv.XoriOp(feq, 1)])
            # uno
            case 14:
                feq1 = riscv.FeqSOP(lhs, lhs)
                feq2 = riscv.FeqSOP(rhs, rhs)
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
        rewriter.replace_matched_op(
            (
                cast_input := UnrealizedConversionCastOp.get(
                    (op.input,), (_INT_REGISTER_TYPE,)
                ),
                new_op := riscv.FCvtSWOp(cast_input.results[0]),
                UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,)),
            )
        )


class LowerArithFPToSIOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FPToSIOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op(
            (
                cast_input := UnrealizedConversionCastOp.get(
                    (op.input,), (_FLOAT_REGISTER_TYPE,)
                ),
                new_op := riscv.FCvtWSOp(cast_input.results[0]),
                UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,)),
            )
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
                    lower_arith_addi,
                    lower_arith_subi,
                    lower_arith_muli,
                    lower_arith_divui,
                    lower_arith_divsi,
                    LowerArithFloorDivSI(),
                    lower_arith_remsi,
                    LowerArithCmpi(),
                    lower_arith_addf,
                    lower_arith_subf,
                    lower_arith_divf,
                    LowerArithNegf(),
                    lower_arith_mulf,
                    LowerArithCmpf(),
                    lower_arith_remui,
                    lower_arith_andi,
                    lower_arith_ori,
                    lower_arith_xori,
                    lower_arith_shli,
                    lower_arith_shrui,
                    lower_arith_shrsi,
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
