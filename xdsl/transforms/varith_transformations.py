import collections
from dataclasses import dataclass
from typing import Literal, cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, varith
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa
from xdsl.utils.type import get_element_type_or_self

# map the arith operation to the right varith op:
ARITH_TO_VARITH_TYPE_MAP: dict[
    type[arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation],
    type[varith.VarithOp],
] = {
    arith.AddiOp: varith.VarithAddOp,
    arith.AddfOp: varith.VarithAddOp,
    arith.MuliOp: varith.VarithMulOp,
    arith.MulfOp: varith.VarithMulOp,
}


class ArithToVarithPattern(RewritePattern):
    """
    Merges two arith operations into a varith operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: arith.AddiOp | arith.AddfOp | arith.MuliOp | arith.MulfOp,
        rewriter: PatternRewriter,
        /,
    ):
        dest_type = ARITH_TO_VARITH_TYPE_MAP[type(op)]

        if type(use_op := op.result.get_user_of_unique_use()) not in (
            type(op),
            dest_type,
        ):
            return
        # pyright does not understand that `use_op` cannot be None here
        use_op = cast(Operation, use_op)

        other_operands = [o for o in use_op.operands if o != op.result]
        rewriter.replace_op(
            use_op,
            dest_type(op.lhs, op.rhs, *other_operands),
        )
        rewriter.erase_op(op)


class VarithToArithPattern(RewritePattern):
    """
    Splits a varith operation into a sequence of arith operations.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: varith.VarithOp, rewriter: PatternRewriter, /):
        # get the type kind of the target arith ops (float|int)
        type_name: Literal["float", "int"] = (
            "int" if is_integer_like_type(op.res.type) else "float"
        )
        # get the opeation kind of the target arith ops (add|mul)
        kind: Literal["add", "mul"] = (
            "add" if isinstance(op, varith.VarithAddOp) else "mul"
        )

        # get the corresponding arith type (e.g. addi/mulf)
        target_arith_type = ARITH_TYPES[(type_name, kind)]

        arith_ops: list[Operation] = []

        # Break the varith op down into a sequence of arith ops
        first_arg = op.operands[0]

        if len(op.operands) == 1:
            rewriter.replace_op(op, [], new_results=[first_arg])
            return

        for i in range(1, len(op.operands)):
            newop = target_arith_type(first_arg, op.operands[i])
            arith_ops.append(newop)
            first_arg = newop.result

        rewriter.replace_op(op, arith_ops)


# map (int|float)(add|mul) to an arith op type
ARITH_TYPES: dict[
    tuple[Literal["float", "int"], Literal["add", "mul"]],
    type[arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation],
] = {
    ("int", "add"): arith.AddiOp,
    ("int", "mul"): arith.MuliOp,
    ("float", "add"): arith.AddfOp,
    ("float", "mul"): arith.MulfOp,
}


class MergeVarithOpsPattern(RewritePattern):
    """
    Looks at every operand to a varith op, and merges them into it if possible.

    e.g. a
        varith.add(arith.addi(1,2), varith.add(3, 4, 5), 6)
    becomes a
        varith.add(1,2,3,4,5,6)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: varith.VarithOp, rewriter: PatternRewriter, /):
        # get the type kind (float|int)
        type_name: Literal["float", "int"] = (
            "int" if is_integer_like_type(op.res.type) else "float"
        )
        # get the opeation kind (add|mul)
        kind: Literal["add", "mul"] = (
            "add" if isinstance(op, varith.VarithAddOp) else "mul"
        )

        # grab the corresponding arith type (e.g. addi/mulf)
        target_arith_type = ARITH_TYPES[(type_name, kind)]

        # create a list of new operands
        new_operands: list[SSAValue] = []
        # create a list of ops that could be erased if their results become unused
        possibly_erased_ops: list[Operation] = []

        # iterate over operands of the varith op:
        for inp in op.operands:
            # if the input happens to be the right arith op:
            if isinstance(inp.owner, target_arith_type):
                # fuse the operands of the arith op into the new varith op
                new_operands.append(inp.owner.lhs)
                new_operands.append(inp.owner.rhs)
                # check if the old arith op can be erased
                possibly_erased_ops.append(inp.owner)
            # if the parent op is a varith op of the same type as us
            elif isinstance(inp.owner, type(op)):
                # include all operands of that
                new_operands.extend(inp.owner.operands)
                # check the old varith op for usages
                possibly_erased_ops.append(inp.owner)
            else:
                # otherwise don't change the input
                new_operands.append(inp)

        # nothing to do if no new operands are added
        if len(possibly_erased_ops) == 0:
            return

        # instantiate a new varith op of the same type as the old op:
        # we can ignore the type error as we know that all VarithOps are instantiated
        # with an *arg of their operands
        rewriter.replace_op(op, type(op)(*new_operands))

        # check all ops that may be erased later:
        for old_op in possibly_erased_ops:
            if not old_op.results[-1].uses:
                rewriter.erase_op(old_op)


def is_integer_like_type(t: Attribute) -> bool:
    """
    Returns true if t is an integer/container of integers/container of
    container of integers ...
    """
    t = get_element_type_or_self(t)
    return isinstance(t, builtin.IntegerType | builtin.IndexType)


@dataclass
class FuseRepeatedAddArgsPattern(RewritePattern):
    """
    Prefer `operand * count(operand)` over repeated addition of `operand`.

    The minimum count to trigger this rewrite can be specified in `min_reps`.
    """

    min_reps: int
    """Minimum repetitions of operand to trigger fusion."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: varith.VarithAddOp, rewriter: PatternRewriter, /):
        elem_t = get_element_type_or_self(op.res.type)

        assert isa(elem_t, builtin.IntegerType | builtin.IndexType | builtin.AnyFloat)

        consts: list[arith.ConstantOp] = []
        fusions: list[Operation] = []
        new_args: list[Operation | SSAValue] = []
        for arg, count in collections.Counter(op.args).items():
            if count >= self.min_reps:
                c, f = self.fuse(arg, count, elem_t)
                consts.append(c)
                fusions.append(f)
                new_args.append(f)
            else:
                new_args.append(arg)
        if fusions:
            rewriter.insert_op([*consts, *fusions], InsertPoint.before(op))
            rewriter.replace_op(op, varith.VarithAddOp(*new_args))

    @staticmethod
    def fuse(
        arg: SSAValue,
        count: int,
        t: builtin.IntegerType | builtin.IndexType | builtin.AnyFloat,
    ):
        if isinstance(t, builtin.IntegerType | builtin.IndexType):
            c = arith.ConstantOp(builtin.IntegerAttr(count, t))
            f = arith.MuliOp
        else:
            c = arith.ConstantOp(builtin.FloatAttr(count, t))
            f = arith.MulfOp
        return c, f(c, arg)


class ConvertArithToVarithPass(ModulePass):
    """
    Convert chains of arith.{add|mul}{i,f} operations into a single long variadic add or mul operation.

    Used for simplifying arithmetic operations for rewrites that need to either change the order or
    completely "cut an equation in half".
    """

    name = "convert-arith-to-varith"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ArithToVarithPattern(),
                    MergeVarithOpsPattern(),
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)


class ConvertVarithToArithPass(ModulePass):
    """
    Convert a single long variadic add or mul operation into a chain of arith.{add|mul}{i,f} operations.
    Reverses ConvertArithToVarithPass.

    """

    name = "convert-varith-to-arith"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            VarithToArithPattern(),
            apply_recursively=False,
        ).rewrite_module(op)


class VarithFuseRepeatedOperandsPass(ModulePass):
    """
    Fuses several occurrences of the same operand into one.
    """

    name = "varith-fuse-repeated-operands"

    min_reps: int = 2
    """The minimum number of times an operand needs to be repeated before being fused."""

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            FuseRepeatedAddArgsPattern(self.min_reps),
            apply_recursively=False,
        ).rewrite_module(op)
