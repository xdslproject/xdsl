"""Passes to manipulate fastmath flags in FP arith operations."""

from dataclasses import dataclass
from typing import Literal

from xdsl.dialects import arith, builtin, llvm
from xdsl.ir import Operation
from xdsl.passes import MLContext, ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)

_FASTMATH_NAMES_TO_ENUM = {str(member.value): member for member in llvm.FastMathFlag}


def _get_flag_list(flags: list[str]):
    try:
        return [_FASTMATH_NAMES_TO_ENUM[flag] for flag in flags]
    except KeyError as e:
        raise ValueError(f"{e} is not a valid fastmath flag.")


@dataclass
class AddArithFastMathFlags(RewritePattern):
    """Adds fastmath flags to FP binary operations from arith dialect."""

    arith_op_cls: type[arith.FloatingPointLikeBinaryOp]
    fastmath_op_attr: arith.FastMathFlagsAttr

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        op.fastmath = self.fastmath_op_attr


@dataclass
class AddArithFastMathFlagsPass(ModulePass):
    """Module pass that adds fastmath flags to FP binary operations from arith dialect.
    It currently does not preserve any existing fastmath flags that may already be part
    of the operation.
    By default (no arguments) it adds the "fast" flag.

    Arguments (all optional):

    - flags: {"fast", "none"} | list[str]: Set specific fastmath flags. Using "fast" or
      "none" enables or disables all flags, respectively.
    """

    name = "arith-add-fastmath"

    flags: Literal["fast", "none"] | list[str] = "fast"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        fm_flags = arith.FastMathFlagsAttr("fast")

        if isinstance(self.flags, str):
            fm_flags = arith.FastMathFlagsAttr(self.flags)
        else:
            if "none" in self.flags or "fast" in self.flags:
                raise ValueError(
                    'f{"none" or "fast" cannot be provided along with other fastmath flags'
                )

            fm_flags = arith.FastMathFlagsAttr(_get_flag_list(self.flags))

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddArithFastMathFlags(arith.Addf, fm_flags),
                    AddArithFastMathFlags(arith.Subf, fm_flags),
                    AddArithFastMathFlags(arith.Mulf, fm_flags),
                    AddArithFastMathFlags(arith.Divf, fm_flags),
                    AddArithFastMathFlags(arith.Minimumf, fm_flags),
                    AddArithFastMathFlags(arith.Maximumf, fm_flags),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
