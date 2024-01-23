from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin, llvm
from xdsl.ir import Operation
from xdsl.passes import MLContext, ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
)


def _match_string_to_enum(flag: str, enum_type: type[llvm.FastMathFlag]):
    for member in enum_type:
        if flag == str(member.value):
            return member

    raise ValueError(f"{flag} is not a valid fastmath flag.")


def _get_flag_list(flags: list[str], enum_type: type[llvm.FastMathFlag]):
    list_flags: list[enum_type] = []
    for flag in flags:
        list_flags.append(_match_string_to_enum(flag, llvm.FastMathFlag))
    return list_flags


@dataclass
class AddArithFastMathFlags(RewritePattern):
    arith_op_cls: type[arith.FloatingPointLikeBinaryOp]
    fastmath_op_attr: arith.FastMathFlagsAttr

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.arith_op_cls):
            return

        op.fastmath = self.fastmath_op_attr


@dataclass
class AddArithFastMathFlagsPass(ModulePass):
    name = "arith-add-fastmath"

    flags: list[str] = field(default_factory=lambda: ["fast"])

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        fm_flags = arith.FastMathFlagsAttr("fast")

        if len(self.flags) == 1:
            if "none" in self.flags:
                fm_flags = arith.FastMathFlagsAttr("none")
            elif "fast" in self.flags:
                fm_flags = arith.FastMathFlagsAttr("fast")
            else:
                fm_flags = arith.FastMathFlagsAttr(
                    _get_flag_list(self.flags, llvm.FastMathFlag)
                )
        else:
            if "none" in self.flags or "fast" in self.flags:
                raise ValueError(
                    'f{"none" or "fast" cannot be provided along with other fastmath flags'
                )

            fm_flags = arith.FastMathFlagsAttr(
                _get_flag_list(self.flags, llvm.FastMathFlag)
            )

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
