"""
Lowers `math` operations to the `llvm` dialect.

This is a port of the upstream MLIR conversion found at
https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp

Operations are converted to the corresponding LLVM intrinsics, forwarding their
fast-math flags.
`math.expm1`, `math.log1p` and `math.rsqrt` have no direct intrinsic counterpart and
are expanded to `exp(x) - 1`, `log(1 + x)` and `1 / sqrt(x)` respectively.

Only operations on LLVM-compatible floating-point scalars and one-dimensional vectors
of them are converted, all other operations are left unchanged.
The expansions additionally require a constant 1.0 of the operand type, so they are
not applied to `f80` and `f128` values, which xDSL cannot yet encode as constants.
"""

from dataclasses import dataclass
from typing import TypeAlias

from xdsl.context import Context
from xdsl.dialects import builtin, llvm, math
from xdsl.dialects.builtin import (
    BFloat16Type,
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    VectorType,
)
from xdsl.ir import Attribute, Operation
from xdsl.irdl import IRDLOperation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa

_LLVMCompatibleFloat: TypeAlias = (
    BFloat16Type | Float16Type | Float32Type | Float64Type | Float80Type | Float128Type
)
"""Floating-point types that map directly to an LLVM floating-point type."""

_ConstantFloat: TypeAlias = BFloat16Type | Float16Type | Float32Type | Float64Type
"""LLVM-compatible floating-point types for which xDSL can encode constant values."""


def _is_llvm_compatible_float_like(type: Attribute) -> bool:
    """
    Returns whether the type is an LLVM-compatible float or a one-dimensional vector of
    LLVM-compatible floats.
    """
    if isinstance(type, VectorType):
        return type.get_num_dims() == 1 and isa(
            type.get_element_type(), _LLVMCompatibleFloat
        )
    return isa(type, _LLVMCompatibleFloat)


def _float_one(type: Attribute) -> llvm.ConstantOp | None:
    """
    Creates an `llvm.mlir.constant` holding 1.0 for a float, or a splat of 1.0 for a
    one-dimensional vector of floats.
    Returns None for other types, and for floats whose constant values cannot yet be
    encoded in xDSL.
    """
    if isinstance(type, VectorType):
        if type.get_num_dims() != 1 or not isa(type, VectorType[_ConstantFloat]):
            return None
        value = DenseIntOrFPElementsAttr.from_list(type, (1.0,))
    elif isa(type, _ConstantFloat):
        value = FloatAttr(1.0, type)
    else:
        return None
    return llvm.ConstantOp(value, type)


_FloatMathOp: TypeAlias = (
    math.FloatingPointLikeUnaryMathOperation
    | math.FloatingPointLikeBinaryMathOperation
    | math.FmaOp
)
"""`math` operations that have a floating-point result and fast-math flags."""

_FLOAT_INTRINSICS: dict[type[_FloatMathOp], type[IRDLOperation]] = {
    math.AbsFOp: llvm.FAbsOp,
    math.CeilOp: llvm.FCeilOp,
    math.CopySignOp: llvm.FCopySignOp,
    math.CosOp: llvm.FCosOp,
    math.Exp2Op: llvm.FExp2Op,
    math.ExpOp: llvm.FExpOp,
    math.FloorOp: llvm.FFloorOp,
    math.FmaOp: llvm.FMAOp,
    math.Log2Op: llvm.FLog2Op,
    math.LogOp: llvm.FLogOp,
    math.PowFOp: llvm.FPowOp,
    math.SinOp: llvm.FSinOp,
    math.SqrtOp: llvm.FSqrtOp,
}
"""Maps `math` operations to the LLVM intrinsics with the same semantics."""


class ConvertFloatIntrinsicOp(RewritePattern):
    """
    Converts a floating-point `math` operation to the equivalent LLVM intrinsic,
    e.g. `math.exp` to `llvm.intr.exp`.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, _FloatMathOp):
            return
        if (intrinsic_type := _FLOAT_INTRINSICS.get(type(op))) is None:
            return
        if not _is_llvm_compatible_float_like(op.result.type):
            return
        intrinsic = intrinsic_type.create(
            operands=op.operands,
            result_types=(op.result.type,),
            properties={"fastmathFlags": llvm.FastMathAttr(op.fastmath.data)},
            attributes=op.attributes.copy(),
        )
        rewriter.replace(op, intrinsic)


class ConvertExpM1Op(RewritePattern):
    """Converts `math.expm1` to `exp(x) - 1`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.ExpM1Op, rewriter: PatternRewriter, /):
        if (one := _float_one(op.result.type)) is None:
            return
        fast_math = llvm.FastMathAttr(op.fastmath.data)
        exp = llvm.FExpOp(op.operand, fast_math)
        exp.attributes = op.attributes.copy()
        sub = llvm.FSubOp(exp.res, one.result, fast_math, op.attributes.copy())
        rewriter.replace(op, (one, exp, sub))


class ConvertLog1pOp(RewritePattern):
    """Converts `math.log1p` to `log(1 + x)`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.Log1pOp, rewriter: PatternRewriter, /):
        if (one := _float_one(op.result.type)) is None:
            return
        fast_math = llvm.FastMathAttr(op.fastmath.data)
        add = llvm.FAddOp(one.result, op.operand, fast_math, op.attributes.copy())
        log = llvm.FLogOp(add.res, fast_math)
        log.attributes = op.attributes.copy()
        rewriter.replace(op, (one, add, log))


class ConvertRsqrtOp(RewritePattern):
    """Converts `math.rsqrt` to `1 / sqrt(x)`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: math.RsqrtOp, rewriter: PatternRewriter, /):
        if (one := _float_one(op.result.type)) is None:
            return
        fast_math = llvm.FastMathAttr(op.fastmath.data)
        sqrt = llvm.FSqrtOp(op.operand, fast_math)
        sqrt.attributes = op.attributes.copy()
        div = llvm.FDivOp(one.result, sqrt.res, fast_math, op.attributes.copy())
        rewriter.replace(op, (one, sqrt, div))


@dataclass(frozen=True)
class ConvertMathToLLVMPass(ModulePass):
    """
    Converts `math` operations on floating-point scalars and one-dimensional vectors
    to LLVM intrinsics.

    `math.log1p` is only converted if `approximate_log1p` is set, as `log(1 + x)` loses
    precision for `x` close to zero.
    """

    name = "convert-math-to-llvm"

    approximate_log1p: bool = True

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        patterns: list[RewritePattern] = [
            ConvertFloatIntrinsicOp(),
            ConvertExpM1Op(),
            ConvertRsqrtOp(),
        ]
        if self.approximate_log1p:
            patterns.append(ConvertLog1pOp())
        PatternRewriteWalker(GreedyRewritePatternApplier(patterns)).rewrite_module(op)
