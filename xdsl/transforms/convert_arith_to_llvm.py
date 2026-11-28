"""
Lowers `arith` operations to the `llvm` dialect.

This is a port of the upstream MLIR conversion found at
https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/ArithToLLVM/ArithToLLVM.cpp

Operations are converted one-to-one to the LLVM operation or intrinsic with the same
semantics, forwarding their fast-math and integer overflow flags.

Only operations whose operands and results all have LLVM-compatible types are
converted: signless integers, LLVM-compatible floating-point types, LLVM dialect types
and one-dimensional vectors of them.
All other operations are left unchanged, including operations on `index` values, which
upstream MLIR converts to integers of a configurable bitwidth.
"""

from dataclasses import dataclass
from typing import TypeAlias

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
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


def _is_llvm_compatible(type: Attribute) -> bool:
    """
    Returns whether the type is compatible with the LLVM dialect, or is a
    one-dimensional, possibly scalable, vector of compatible elements.
    """
    if isa(type, VectorType):
        return type.get_num_dims() == 1 and llvm.is_compatible_type(
            type.get_element_type()
        )
    return llvm.is_compatible_type(type)


_FastMathOp: TypeAlias = (
    arith.FloatingPointLikeBinaryOperation | arith.NegfOp | arith.CmpfOp
)
"""`arith` operations that have fast-math flags."""

_OverflowOp: TypeAlias = arith.SignlessIntegerBinaryOperationWithOverflow
"""`arith` operations that have integer overflow flags."""

_ComparisonOp: TypeAlias = arith.CmpiOp | arith.CmpfOp
"""`arith` operations that have a comparison predicate."""

_ARITH_TO_LLVM: dict[type[Operation], type[IRDLOperation]] = {
    # Integer arithmetic
    arith.AddiOp: llvm.AddOp,
    arith.SubiOp: llvm.SubOp,
    arith.MuliOp: llvm.MulOp,
    arith.DivSIOp: llvm.SDivOp,
    arith.DivUIOp: llvm.UDivOp,
    arith.RemSIOp: llvm.SRemOp,
    arith.RemUIOp: llvm.URemOp,
    arith.AndIOp: llvm.AndOp,
    arith.OrIOp: llvm.OrOp,
    arith.XOrIOp: llvm.XOrOp,
    arith.ShLIOp: llvm.ShlOp,
    arith.ShRSIOp: llvm.AShrOp,
    arith.ShRUIOp: llvm.LShrOp,
    arith.MaxSIOp: llvm.SMaxOp,
    arith.MaxUIOp: llvm.UMaxOp,
    arith.MinSIOp: llvm.SMinOp,
    arith.MinUIOp: llvm.UMinOp,
    # Floating-point arithmetic
    arith.AddfOp: llvm.FAddOp,
    arith.SubfOp: llvm.FSubOp,
    arith.MulfOp: llvm.FMulOp,
    arith.DivfOp: llvm.FDivOp,
    arith.NegfOp: llvm.FNegOp,
    arith.MaximumfOp: llvm.FMaximumOp,
    arith.MaxnumfOp: llvm.VectorFMaxOp,
    arith.MinimumfOp: llvm.FMinimumOp,
    arith.MinnumfOp: llvm.VectorFMinOp,
    # Comparisons and selection
    arith.CmpiOp: llvm.ICmpOp,
    arith.CmpfOp: llvm.FCmpOp,
    arith.SelectOp: llvm.SelectOp,
    # Casts
    arith.BitcastOp: llvm.BitcastOp,
    arith.ExtSIOp: llvm.SExtOp,
    arith.ExtUIOp: llvm.ZExtOp,
    arith.TruncIOp: llvm.TruncOp,
    arith.ExtFOp: llvm.FPExtOp,
    arith.TruncFOp: llvm.FPTruncOp,
    arith.SIToFPOp: llvm.SIToFPOp,
    arith.UIToFPOp: llvm.UIToFPOp,
    arith.FPToSIOp: llvm.FPToSIOp,
    arith.FPToUIOp: llvm.FPToUIOp,
}
"""Maps `arith` operations to the LLVM operations with the same semantics."""


def _convert_overflow_flags(flags: arith.IntegerOverflowAttr) -> IntegerAttr:
    """
    Converts `arith` integer overflow flags to the integer encoding of the `llvm`
    overflow flags.
    """
    overflow = llvm.OverflowAttr(llvm.OverflowFlag(flag) for flag in flags.data)
    return IntegerAttr(overflow.to_int(), 32)


class ConvertArithOp(RewritePattern):
    """
    Converts an `arith` operation to the equivalent LLVM operation, e.g. `arith.addi`
    to `llvm.add`.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if (llvm_op_type := _ARITH_TO_LLVM.get(type(op))) is None:
            return
        if not all(
            _is_llvm_compatible(t) for t in (*op.operand_types, *op.result_types)
        ):
            return

        properties: dict[str, Attribute] = {}
        if isinstance(op, _OverflowOp):
            properties["overflowFlags"] = _convert_overflow_flags(op.overflow_flags)
        if isinstance(op, _FastMathOp):
            properties["fastmathFlags"] = llvm.FastMathAttr(op.fastmath.data)
        if isinstance(op, _ComparisonOp):
            # The `arith` and `llvm` predicates are enumerated in the same order.
            properties["predicate"] = IntegerAttr(op.predicate.value.data, 64)

        rewriter.replace(
            op,
            llvm_op_type.create(
                operands=op.operands,
                result_types=op.result_types,
                properties=properties,
                attributes=op.attributes.copy(),
            ),
        )


class ConvertConstantOp(RewritePattern):
    """Converts `arith.constant` to `llvm.mlir.constant`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter, /):
        if not _is_llvm_compatible(op.result.type):
            return
        if not isinstance(op.value, IntegerAttr | FloatAttr | DenseIntOrFPElementsAttr):
            return
        constant = llvm.ConstantOp(op.value, op.result.type)
        constant.attributes = op.attributes.copy()
        rewriter.replace(op, constant)


@dataclass(frozen=True)
class ConvertArithToLLVMPass(ModulePass):
    """
    Converts `arith` operations on LLVM-compatible scalars and one-dimensional vectors
    to the `llvm` dialect.
    """

    name = "convert-arith-to-llvm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertArithOp(), ConvertConstantOp()])
        ).rewrite_module(op)
