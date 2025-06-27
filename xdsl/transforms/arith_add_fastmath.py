"""Passes to manipulate fastmath flags in FP arith operations."""

from dataclasses import dataclass
from typing import Literal

from xdsl.dialects import arith, builtin
from xdsl.dialects.utils import FastMathFlag
from xdsl.passes import Context, ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

_FASTMATH_NAMES_TO_ENUM = {str(member.value): member for member in FastMathFlag}


def _get_flag_list(flags: tuple[str, ...]):
    try:
        return tuple(_FASTMATH_NAMES_TO_ENUM[flag] for flag in flags)
    except KeyError as e:
        raise ValueError(f"{e} is not a valid fastmath flag.")


@dataclass
class AddArithFastMathFlags(RewritePattern):
    """Adds fastmath flags to FP binary operations from arith dialect."""

    fastmath_op_attr: arith.FastMathFlagsAttr

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: arith.FloatingPointLikeBinaryOperation | arith.CmpfOp,
        rewriter: PatternRewriter,
    ) -> None:
        op.fastmath = self.fastmath_op_attr


@dataclass(frozen=True)
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

    flags: Literal["fast", "none"] | tuple[str, ...] = "fast"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
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
                    AddArithFastMathFlags(fm_flags),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
