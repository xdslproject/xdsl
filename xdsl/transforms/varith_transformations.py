from typing import Literal, cast

from xdsl.context import MLContext
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
from xdsl.utils.hints import isa

# map the arith operation to the right varith op:
ARITH_TO_VARITH_TYPE_MAP: dict[
    type[arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation],
    type[varith.VarithOp],
] = {
    arith.Addi: varith.VarithAddOp,
    arith.Addf: varith.VarithAddOp,
    arith.Muli: varith.VarithMulOp,
    arith.Mulf: varith.VarithMulOp,
}


class ArithToVarithPattern(RewritePattern):
    """
    Merges two arith operations into a varith operation.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        # check that the op is of a type that we can convert to varith
        if type(op) not in ARITH_TO_VARITH_TYPE_MAP:
            return

        # this must be true, as all keys of ARITH_TO_VARITH_TYPE_MAP are binary ops
        op = cast(
            arith.SignlessIntegerBinaryOperation
            | arith.FloatingPointLikeBinaryOperation,
            op,
        )

        dest_type = ARITH_TO_VARITH_TYPE_MAP[type(op)]

        # check LHS and the RHS to see if they can be folded
        # but abort after one is merged
        for other in (op.rhs.owner, op.lhs.owner):
            # if me and the other op are the same op
            # (they must necessarily operate on the same data type)
            if type(op) is type(other):
                other_op = cast(
                    arith.SignlessIntegerBinaryOperation
                    | arith.FloatingPointLikeBinaryOperation,
                    other,
                )
                # instantiate a varith op with three operands
                rewriter.replace_matched_op(
                    dest_type(op.rhs, other_op.lhs, other_op.rhs)
                )
                if len(other_op.result.uses) == 0:
                    rewriter.erase_op(other_op)
                return


# map (int|float)(add|mul) to an arith op type
ARITH_TYPES: dict[
    tuple[Literal["float", "int"], Literal["add", "mul"]],
    type[arith.SignlessIntegerBinaryOperation | arith.FloatingPointLikeBinaryOperation],
] = {
    ("int", "add"): arith.Addi,
    ("int", "mul"): arith.Muli,
    ("float", "add"): arith.Addf,
    ("float", "mul"): arith.Mulf,
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
            if isa(inp.owner, target_arith_type):
                # fuse the operands of the arith op into the new varith op
                new_operands.append(inp.owner.lhs)
                new_operands.append(inp.owner.rhs)
                # check if the old arith op can be erased
                possibly_erased_ops.append(inp.owner)
            # if the parent op is a varith op of the same type as us
            elif isa(inp.owner, type(op)):
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
        rewriter.replace_matched_op(type(op)(*new_operands))

        # check all ops that may be erased later:
        for old_op in possibly_erased_ops:
            if len(old_op.results[-1].uses) == 0:
                rewriter.erase_op(old_op)


def is_integer_like_type(t: Attribute) -> bool:
    """
    Returns true if t is an integer/container of integers/container of
    container of integers ...
    """
    if isinstance(t, builtin.IntegerType | builtin.IndexType):
        return True
    if isinstance(t, builtin.ContainerType):
        elm_type = cast(builtin.ContainerType[Attribute], t).get_element_type()
        return is_integer_like_type(elm_type)
    return False


class ConvertArithToVarithPass(ModulePass):
    """
    Convert chains of arith.{add|mul}{i,f} operations into a single long variadic add or mul operation.

    Used for simplifying arithmetic operations for rewrites that need to either change the order or
    completely "cut an equation in half".
    """

    name = "convert-arith-to-varith"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ArithToVarithPattern(),
                    MergeVarithOpsPattern(),
                ]
            ),
        ).rewrite_op(op)
