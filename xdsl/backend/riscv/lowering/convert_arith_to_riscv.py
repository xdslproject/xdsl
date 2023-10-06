import struct
from dataclasses import dataclass

from xdsl.backend.riscv.lowering.utils import (
    cast_matched_op_results,
    cast_operands_to_regs,
)
from xdsl.dialects import arith, riscv
from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.dead_code_elimination import dce
from xdsl.utils.bitwise_casts import convert_f32_to_u32

_INT_REGISTER_TYPE = riscv.IntRegisterType.unallocated()
_FLOAT_REGISTER_TYPE = riscv.FloatRegisterType.unallocated()


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
                            convert_f32_to_u32(op.value.value.data),
                            rd=_INT_REGISTER_TYPE,
                        ),
                        fld := riscv.FMvWXOp(
                            lui.rd, rd=riscv.FloatRegisterType.unallocated()
                        ),
                        UnrealizedConversionCastOp.get(fld.results, (op_result_type,)),
                    ],
                )
            elif isinstance(op_result_type, Float64Type):
                # There is no way to load an immediate value to a float register directly.
                # We have to load the bits into an integer register, store them on the
                # stack, and load again.

                # TODO: check the xlen in this lowering.

                # This lowering assumes that xlen is 32 and flen is 64

                lower, upper = struct.unpack(
                    "<ii", struct.pack("<d", op.value.value.data)
                )
                rewriter.replace_matched_op(
                    [
                        sp := riscv.GetRegisterOp(riscv.Registers.SP),
                        li_upper := riscv.LiOp(upper),
                        riscv.SwOp(sp, li_upper, -4),
                        li_lower := riscv.LiOp(lower),
                        riscv.SwOp(sp, li_lower, -8),
                        fld := riscv.FLdOp(sp, -8, rd=_FLOAT_REGISTER_TYPE),
                        UnrealizedConversionCastOp.get(fld.results, (op_result_type,)),
                    ],
                )
            else:
                raise NotImplementedError("Only 32 or 64 bit floats are supported")
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


RdRsRsIntegerOperation = riscv.RdRsRsOperation[
    riscv.IntRegisterType, riscv.IntRegisterType, riscv.IntRegisterType
]

RdRsRsFloatOperation = riscv.RdRsRsOperation[
    riscv.FloatRegisterType, riscv.FloatRegisterType, riscv.FloatRegisterType
]


@dataclass
class LowerBinaryIntegerOp(RewritePattern):
    arith_op_cls: type[arith.SignlessIntegerBinaryOp]
    riscv_op_cls: type[RdRsRsIntegerOperation]

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        lhs = UnrealizedConversionCastOp.get((op.lhs,), (_INT_REGISTER_TYPE,))
        rhs = UnrealizedConversionCastOp.get((op.rhs,), (_INT_REGISTER_TYPE,))
        add = self.riscv_op_cls(lhs, rhs, rd=_INT_REGISTER_TYPE)
        cast = UnrealizedConversionCastOp.get((add.rd,), (op.result.type,))

        rewriter.replace_matched_op((lhs, rhs, add, cast))


@dataclass
class LowerBinaryFloatOp(RewritePattern):
    arith_op_cls: type[arith.FloatingPointLikeBinaryOp]
    riscv_f_op_cls: type[RdRsRsFloatOperation]
    riscv_d_op_cls: type[RdRsRsFloatOperation]

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        lhs = UnrealizedConversionCastOp.get((op.lhs,), (_FLOAT_REGISTER_TYPE,))
        rhs = UnrealizedConversionCastOp.get((op.rhs,), (_FLOAT_REGISTER_TYPE,))
        match op.lhs.type:
            case Float32Type():
                cls = self.riscv_f_op_cls
            case Float64Type():
                cls = self.riscv_d_op_cls
            case _:
                assert False, f"Unexpected float type {op.lhs.type}"

        new_op = cls(lhs, rhs, rd=_FLOAT_REGISTER_TYPE)
        cast = UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,))

        rewriter.replace_matched_op((lhs, rhs, new_op, cast))


lower_arith_addi = LowerBinaryIntegerOp(arith.Addi, riscv.AddOp)
lower_arith_subi = LowerBinaryIntegerOp(arith.Subi, riscv.SubOp)
lower_arith_muli = LowerBinaryIntegerOp(arith.Muli, riscv.MulOp)
lower_arith_divui = LowerBinaryIntegerOp(arith.DivUI, riscv.DivuOp)
lower_arith_divsi = LowerBinaryIntegerOp(arith.DivSI, riscv.DivOp)


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


lower_arith_remui = LowerBinaryIntegerOp(arith.RemUI, riscv.RemuOp)
lower_arith_remsi = LowerBinaryIntegerOp(arith.RemSI, riscv.RemOp)


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
        lhs, rhs = cast_operands_to_regs(rewriter)
        cast_matched_op_results(rewriter)

        match op.predicate.value.data:
            # eq
            case 0:
                xor_op = riscv.XorOp(lhs, rhs, rd=riscv.IntRegisterType.unallocated())
                seqz_op = riscv.SltiuOp(xor_op, 1)
                rewriter.replace_matched_op([xor_op, seqz_op])
            # ne
            case 1:
                zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
                xor_op = riscv.XorOp(lhs, rhs, rd=riscv.IntRegisterType.unallocated())
                snez_op = riscv.SltuOp(
                    zero, xor_op, rd=riscv.IntRegisterType.unallocated()
                )
                rewriter.replace_matched_op([zero, xor_op, snez_op])
            # slt
            case 2:
                rewriter.replace_matched_op(
                    [riscv.SltOp(lhs, rhs, rd=riscv.IntRegisterType.unallocated())]
                )
            # sle
            case 3:
                slt = riscv.SltOp(lhs, rhs, rd=riscv.IntRegisterType.unallocated())
                xori = riscv.XoriOp(slt, 1)
                rewriter.replace_matched_op([slt, xori])
            # ult
            case 4:
                rewriter.replace_matched_op(
                    [riscv.SltuOp(lhs, rhs, rd=riscv.IntRegisterType.unallocated())]
                )
            # ule
            case 5:
                sltu = riscv.SltuOp(lhs, rhs, rd=riscv.IntRegisterType.unallocated())
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_matched_op([sltu, xori])
            # ugt
            case 6:
                rewriter.replace_matched_op(
                    [riscv.SltuOp(rhs, lhs, rd=riscv.IntRegisterType.unallocated())]
                )
            # uge
            case 7:
                sltu = riscv.SltuOp(rhs, lhs, rd=riscv.IntRegisterType.unallocated())
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_matched_op([sltu, xori])
            case _:
                raise NotImplementedError("Cmpi predicate not supported")


class LowerArithSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Select, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Select is not supported")


lower_arith_andi = LowerBinaryIntegerOp(arith.AndI, riscv.AndOp)
lower_arith_ori = LowerBinaryIntegerOp(arith.OrI, riscv.OrOp)
lower_arith_xori = LowerBinaryIntegerOp(arith.XOrI, riscv.XorOp)
lower_arith_shli = LowerBinaryIntegerOp(arith.ShLI, riscv.SllOp)
lower_arith_shrui = LowerBinaryIntegerOp(arith.ShRUI, riscv.SrlOp)
lower_arith_shrsi = LowerBinaryIntegerOp(arith.ShRSI, riscv.SraOp)


lower_arith_addf = LowerBinaryFloatOp(arith.Addf, riscv.FAddSOp, riscv.FAddDOp)
lower_arith_subf = LowerBinaryFloatOp(arith.Subf, riscv.FSubSOp, riscv.FSubDOp)
lower_arith_mulf = LowerBinaryFloatOp(arith.Mulf, riscv.FMulSOp, riscv.FMulDOp)
lower_arith_divf = LowerBinaryFloatOp(arith.Divf, riscv.FDivSOp, riscv.FDivDOp)


class LowerArithNegf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Negf, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op(
            (
                operand := UnrealizedConversionCastOp.get(
                    (op.operand,), (_FLOAT_REGISTER_TYPE,)
                ),
                negf := riscv.FSgnJNSOp(
                    operand, operand, rd=riscv.FloatRegisterType.unallocated()
                ),
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
        lhs, rhs = cast_operands_to_regs(rewriter)
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
                rewriter.replace_matched_op(
                    [
                        flt1,
                        flt2,
                        riscv.OrOp(flt2, flt1, rd=riscv.IntRegisterType.unallocated()),
                    ]
                )
            # ord
            case 7:
                feq1 = riscv.FeqSOP(lhs, lhs)
                feq2 = riscv.FeqSOP(rhs, rhs)
                rewriter.replace_matched_op(
                    [
                        feq1,
                        feq2,
                        riscv.AndOp(feq2, feq1, rd=riscv.IntRegisterType.unallocated()),
                    ]
                )
            # ueq
            case 8:
                flt1 = riscv.FltSOP(lhs, rhs)
                flt2 = riscv.FltSOP(rhs, lhs)
                or_ = riscv.OrOp(flt2, flt1, rd=riscv.IntRegisterType.unallocated())
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
                and_ = riscv.AndOp(feq2, feq1, rd=riscv.IntRegisterType.unallocated())
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
                new_op := riscv.FCvtSWOp(
                    cast_input.results[0], rd=riscv.FloatRegisterType.unallocated()
                ),
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
                new_op := riscv.FCvtWSOp(
                    cast_input.results[0], rd=riscv.IntRegisterType.unallocated()
                ),
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


class ConvertArithToRiscvPass(ModulePass):
    name = "convert-arith-to-riscv"

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
