import struct
from dataclasses import dataclass

from xdsl.backend.riscv.lowering.utils import (
    cast_matched_op_results,
    cast_operands_to_regs,
)
from xdsl.context import Context
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
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.bitwise_casts import convert_f32_to_u32
from xdsl.utils.comparisons import signed_lower_bound, signed_upper_bound
from xdsl.utils.hints import isa

_INT_REGISTER_TYPE = riscv.Registers.UNALLOCATED_INT
_FLOAT_REGISTER_TYPE = riscv.Registers.UNALLOCATED_FLOAT


class LowerArithConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.ConstantOp, rewriter: PatternRewriter
    ) -> None:
        op_result_type = op.result.type
        if isa(op_result_type, IntegerType) and isinstance(
            op_val := op.value, IntegerAttr
        ):
            if op_result_type.width.data <= 32:
                rewriter.replace_op(
                    op,
                    [
                        constant := riscv.LiOp(op_val.value.data),
                        UnrealizedConversionCastOp.get(
                            constant.results, (op_result_type,)
                        ),
                    ],
                )
            else:
                raise NotImplementedError("Only 32 bit integers are supported for now")
        elif isinstance(op_val := op.value, FloatAttr):
            if isinstance(op_result_type, Float32Type):
                rewriter.replace_op(
                    op,
                    [
                        lui := riscv.LiOp(
                            convert_f32_to_u32(op_val.value.data),
                            rd=_INT_REGISTER_TYPE,
                        ),
                        fld := riscv.FMvWXOp(lui.rd),
                        UnrealizedConversionCastOp.get(fld.results, (op_result_type,)),
                    ],
                )
            elif isinstance(op_result_type, Float64Type):
                # There is no way to load an immediate value to a float register directly.

                s32_min = signed_lower_bound(32)
                s32_max = signed_upper_bound(32)
                # If the value is an integer that fits in s32, then convert.
                if (val_data := op_val.value.data).is_integer() and s32_min <= (
                    int_val := int(val_data)
                ) < s32_max:
                    rewriter.replace_op(
                        op,
                        [
                            lui := riscv.LiOp(
                                int_val,
                                rd=_INT_REGISTER_TYPE,
                            ),
                            fcvtdw := riscv.FCvtDWOp(lui.rd, rd=_FLOAT_REGISTER_TYPE),
                            UnrealizedConversionCastOp.get(
                                fcvtdw.results, (op_result_type,)
                            ),
                        ],
                    )
                else:
                    # We have to load the bits into an integer register, store them on the
                    # stack, and load again.

                    # TODO: check the xlen in this lowering.

                    # This lowering assumes that xlen is 32 and flen is 64

                    lower, upper = struct.unpack(
                        "<ii", struct.pack("<d", op_val.value.data)
                    )
                    rewriter.replace_op(
                        op,
                        [
                            sp := riscv.GetRegisterOp(riscv.Registers.SP),
                            li_upper := riscv.LiOp(upper),
                            riscv.SwOp(sp, li_upper, -4),
                            li_lower := riscv.LiOp(lower),
                            riscv.SwOp(sp, li_lower, -8),
                            fld := riscv.FLdOp(sp, -8, rd=_FLOAT_REGISTER_TYPE),
                            UnrealizedConversionCastOp.get(
                                fld.results, (op_result_type,)
                            ),
                        ],
                    )
            else:
                raise NotImplementedError("Only 32 or 64 bit floats are supported")
        elif isinstance(op_result_type, IndexType) and isinstance(
            op_val := op.value, IntegerAttr
        ):
            rewriter.replace_op(
                op,
                [
                    constant := riscv.LiOp(op_val.value.data),
                    UnrealizedConversionCastOp.get(constant.results, (op_result_type,)),
                ],
            )
        else:
            raise NotImplementedError(
                f"Unsupported constant type {op_val} of type {type(op_val)}"
            )


class LowerArithIndexCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.IndexCastOp, rewriter: PatternRewriter
    ) -> None:
        """
        On a RV32 triple, the index type is 32 bits, so we can just drop the cast.
        """

        rewriter.replace_op(
            op, UnrealizedConversionCastOp.get((op.input,), (op.result.type,))
        )


RdRsRsIntegerOperation = riscv.RdRsRsOperation[
    riscv.IntRegisterType, riscv.IntRegisterType, riscv.IntRegisterType
]

RdRsRsFloatOperation = riscv.RdRsRsOperation[
    riscv.FloatRegisterType, riscv.FloatRegisterType, riscv.FloatRegisterType
]


@dataclass
class LowerBinaryIntegerOp(RewritePattern):
    arith_op_cls: type[arith.SignlessIntegerBinaryOperation]
    riscv_op_cls: type[RdRsRsIntegerOperation]

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        lhs = UnrealizedConversionCastOp.get((op.lhs,), (_INT_REGISTER_TYPE,))
        rhs = UnrealizedConversionCastOp.get((op.rhs,), (_INT_REGISTER_TYPE,))
        add = self.riscv_op_cls(lhs, rhs, rd=_INT_REGISTER_TYPE)
        cast = UnrealizedConversionCastOp.get((add.rd,), (op.result.type,))

        rewriter.replace_op(op, (lhs, rhs, add, cast))


@dataclass
class LowerBinaryFloatOp(RewritePattern):
    arith_op_cls: type[arith.FloatingPointLikeBinaryOperation]
    riscv_f_op_cls: type[riscv.RdRsRsFloatOperationWithFastMath]
    riscv_d_op_cls: type[riscv.RdRsRsFloatOperationWithFastMath]

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
                raise ValueError(f"Unexpected float type {op.lhs.type}")

        rv_flags = riscv.FastMathFlagsAttr(op.fastmath.data)

        new_op = cls(lhs, rhs, rd=_FLOAT_REGISTER_TYPE, fastmath=rv_flags)
        cast = UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,))

        rewriter.replace_op(op, (lhs, rhs, new_op, cast))


lower_arith_addi = LowerBinaryIntegerOp(arith.AddiOp, riscv.AddOp)
lower_arith_subi = LowerBinaryIntegerOp(arith.SubiOp, riscv.SubOp)
lower_arith_muli = LowerBinaryIntegerOp(arith.MuliOp, riscv.MulOp)
lower_arith_divui = LowerBinaryIntegerOp(arith.DivUIOp, riscv.DivuOp)
lower_arith_divsi = LowerBinaryIntegerOp(arith.DivSIOp, riscv.DivOp)


class LowerArithFloorDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.FloorDivSIOp, rewriter: PatternRewriter
    ) -> None:
        raise NotImplementedError("FloorDivSI is not supported")


class LowerArithCeilDivSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.CeilDivSIOp, rewriter: PatternRewriter
    ) -> None:
        raise NotImplementedError("CeilDivSI is not supported")


class LowerArithCeilDivUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.CeilDivUIOp, rewriter: PatternRewriter
    ) -> None:
        raise NotImplementedError("CeilDivUI is not supported")


lower_arith_remui = LowerBinaryIntegerOp(arith.RemUIOp, riscv.RemuOp)
lower_arith_remsi = LowerBinaryIntegerOp(arith.RemSIOp, riscv.RemOp)


class LowerArithMinSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MinSIOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MinSI is not supported")


class LowerArithMaxSI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MaxSIOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MaxSI is not supported")


class LowerArithMinUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MinUIOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MinUI is not supported")


class LowerArithMaxUI(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MaxUIOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("MaxUI is not supported")


class LowerArithCmpi(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter) -> None:
        # based on https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/RISCV/i32-icmp.ll
        lhs, rhs = cast_operands_to_regs(rewriter)
        cast_matched_op_results(rewriter)

        match op.predicate.value.data:
            # eq
            case 0:
                xor_op = riscv.XorOp(lhs, rhs)
                seqz_op = riscv.SltiuOp(xor_op, 1)
                rewriter.replace_op(op, [xor_op, seqz_op])
            # ne
            case 1:
                zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
                xor_op = riscv.XorOp(lhs, rhs)
                snez_op = riscv.SltuOp(zero, xor_op)
                rewriter.replace_op(op, [zero, xor_op, snez_op])
            # slt
            case 2:
                rewriter.replace_op(op, [riscv.SltOp(lhs, rhs)])
            # sle
            case 3:
                slt = riscv.SltOp(lhs, rhs)
                xori = riscv.XoriOp(slt, 1)
                rewriter.replace_op(op, [slt, xori])
            # ult
            case 4:
                rewriter.replace_op(op, [riscv.SltuOp(lhs, rhs)])
            # ule
            case 5:
                sltu = riscv.SltuOp(lhs, rhs)
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_op(op, [sltu, xori])
            # ugt
            case 6:
                rewriter.replace_op(op, [riscv.SltuOp(rhs, lhs)])
            # uge
            case 7:
                sltu = riscv.SltuOp(rhs, lhs)
                xori = riscv.XoriOp(sltu, 1)
                rewriter.replace_op(op, [sltu, xori])
            case _:
                raise NotImplementedError("Cmpi predicate not supported")


class LowerArithSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError("Select is not supported")


lower_arith_andi = LowerBinaryIntegerOp(arith.AndIOp, riscv.AndOp)
lower_arith_ori = LowerBinaryIntegerOp(arith.OrIOp, riscv.OrOp)
lower_arith_xori = LowerBinaryIntegerOp(arith.XOrIOp, riscv.XorOp)
lower_arith_shli = LowerBinaryIntegerOp(arith.ShLIOp, riscv.SllOp)
lower_arith_shrui = LowerBinaryIntegerOp(arith.ShRUIOp, riscv.SrlOp)
lower_arith_shrsi = LowerBinaryIntegerOp(arith.ShRSIOp, riscv.SraOp)


lower_arith_addf = LowerBinaryFloatOp(arith.AddfOp, riscv.FAddSOp, riscv.FAddDOp)
lower_arith_subf = LowerBinaryFloatOp(arith.SubfOp, riscv.FSubSOp, riscv.FSubDOp)
lower_arith_mulf = LowerBinaryFloatOp(arith.MulfOp, riscv.FMulSOp, riscv.FMulDOp)
lower_arith_divf = LowerBinaryFloatOp(arith.DivfOp, riscv.FDivSOp, riscv.FDivDOp)
lower_arith_minf = LowerBinaryFloatOp(arith.MinimumfOp, riscv.FMinSOp, riscv.FMinDOp)
lower_arith_maxf = LowerBinaryFloatOp(arith.MaximumfOp, riscv.FMaxSOp, riscv.FMaxDOp)


class LowerArithNegf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.NegfOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(
            op,
            (
                operand := UnrealizedConversionCastOp.get(
                    (op.operand,), (_FLOAT_REGISTER_TYPE,)
                ),
                negf := riscv.FSgnJNSOp(operand, operand),
                UnrealizedConversionCastOp.get((negf.rd,), (op.result.type,)),
            ),
        )


class LowerArithCmpf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpfOp, rewriter: PatternRewriter) -> None:
        # https://llvm.org/docs/LangRef.html#id309
        lhs, rhs = cast_operands_to_regs(rewriter)
        cast_matched_op_results(rewriter)

        fastmath = riscv.FastMathFlagsAttr(op.fastmath.data)

        match op.predicate.value.data:
            # false
            case 0:
                rewriter.replace_op(op, [riscv.LiOp(0)])
            # oeq
            case 1:
                rewriter.replace_op(op, [riscv.FeqSOp(lhs, rhs, fastmath=fastmath)])
            # ogt
            case 2:
                rewriter.replace_op(op, [riscv.FltSOp(rhs, lhs, fastmath=fastmath)])
            # oge
            case 3:
                rewriter.replace_op(op, [riscv.FleSOp(rhs, lhs, fastmath=fastmath)])
            # olt
            case 4:
                rewriter.replace_op(op, [riscv.FltSOp(lhs, rhs, fastmath=fastmath)])
            # ole
            case 5:
                rewriter.replace_op(op, [riscv.FleSOp(lhs, rhs, fastmath=fastmath)])
            # one
            case 6:
                flt1 = riscv.FltSOp(lhs, rhs, fastmath=fastmath)
                flt2 = riscv.FltSOp(rhs, lhs, fastmath=fastmath)
                rewriter.replace_op(
                    op,
                    [
                        flt1,
                        flt2,
                        riscv.OrOp(flt2, flt1),
                    ],
                )
            # ord
            case 7:
                feq1 = riscv.FeqSOp(lhs, lhs, fastmath=fastmath)
                feq2 = riscv.FeqSOp(rhs, rhs, fastmath=fastmath)
                rewriter.replace_op(
                    op,
                    [
                        feq1,
                        feq2,
                        riscv.AndOp(feq2, feq1),
                    ],
                )
            # ueq
            case 8:
                flt1 = riscv.FltSOp(lhs, rhs, fastmath=fastmath)
                flt2 = riscv.FltSOp(rhs, lhs, fastmath=fastmath)
                or_ = riscv.OrOp(flt2, flt1)
                rewriter.replace_op(op, [flt1, flt2, or_, riscv.XoriOp(or_, 1)])
            # ugt
            case 9:
                fle = riscv.FleSOp(lhs, rhs, fastmath=fastmath)
                rewriter.replace_op(op, [fle, riscv.XoriOp(fle, 1)])
            # uge
            case 10:
                fle = riscv.FltSOp(lhs, rhs, fastmath=fastmath)
                rewriter.replace_op(op, [fle, riscv.XoriOp(fle, 1)])
            # ult
            case 11:
                fle = riscv.FleSOp(rhs, lhs, fastmath=fastmath)
                rewriter.replace_op(op, [fle, riscv.XoriOp(fle, 1)])
            # ule
            case 12:
                flt = riscv.FltSOp(rhs, lhs, fastmath=fastmath)
                rewriter.replace_op(op, [flt, riscv.XoriOp(flt, 1)])
            # une
            case 13:
                feq = riscv.FeqSOp(lhs, rhs, fastmath=fastmath)
                rewriter.replace_op(op, [feq, riscv.XoriOp(feq, 1)])
            # uno
            case 14:
                feq1 = riscv.FeqSOp(lhs, lhs, fastmath=fastmath)
                feq2 = riscv.FeqSOp(rhs, rhs, fastmath=fastmath)
                and_ = riscv.AndOp(feq2, feq1)
                rewriter.replace_op(op, [feq1, feq2, and_, riscv.XoriOp(and_, 1)])
            # true
            case 15:
                rewriter.replace_op(op, [riscv.LiOp(1)])
            case _:
                raise NotImplementedError("Cmpf predicate not supported")


class LowerArithSIToFPOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SIToFPOp, rewriter: PatternRewriter) -> None:
        match op.result.type:
            case Float32Type():
                cls = riscv.FCvtSWOp
            case Float64Type():
                cls = riscv.FCvtDWOp
            case _:
                raise ValueError(f"Unexpected float type {op.result.type}")

        rewriter.replace_op(
            op,
            (
                cast_input := UnrealizedConversionCastOp.get(
                    (op.input,), (_INT_REGISTER_TYPE,)
                ),
                new_op := cls(cast_input.results[0]),
                UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,)),
            ),
        )


class LowerArithFPToSIOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.FPToSIOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_op(
            op,
            (
                cast_input := UnrealizedConversionCastOp.get(
                    (op.input,), (_FLOAT_REGISTER_TYPE,)
                ),
                new_op := riscv.FCvtWSOp(cast_input.results[0]),
                UnrealizedConversionCastOp.get((new_op.rd,), (op.result.type,)),
            ),
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

    def apply(self, ctx: Context, op: ModuleOp) -> None:
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
                    lower_arith_minf,
                    lower_arith_maxf,
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
