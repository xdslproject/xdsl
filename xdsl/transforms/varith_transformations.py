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

ARITH_TO_VARITH_TYPE_MAP: dict[
    type[arith.BinaryOperation[Attribute]], type[varith.VarithOp]
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
        if type(op) not in ARITH_TO_VARITH_TYPE_MAP:
            return

        # thus must be true, as all keys of ARITH_TO_VARITH_TYPE_MAP are binary ops
        op = cast(arith.BinaryOperation[Attribute], op)

        dest_type = ARITH_TO_VARITH_TYPE_MAP[type(op)]

        for other in (op.rhs.owner, op.lhs.owner):
            if type(op) is type(other):
                other_op: arith.BinaryOperation[Attribute] = cast(
                    arith.BinaryOperation[Attribute], other
                )
                rewriter.replace_matched_op(
                    dest_type(op.rhs, other_op.lhs, other_op.rhs)
                )
                if len(other_op.result.uses) == 0:
                    rewriter.erase_op(other_op)


ARITH_TYPES: dict[
    tuple[Literal["float", "int"], Literal["add", "mul"]],
    type[arith.BinaryOperation[Attribute]],
] = {
    ("int", "add"): arith.Addi,
    ("int", "mul"): arith.Muli,
    ("float", "add"): arith.Addf,
    ("float", "mul"): arith.Mulf,
}


class MergeVarithOpsPattern(RewritePattern):
    """
    Merges operands that are varith or arith ops into
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: varith.VarithOp, rewriter: PatternRewriter, /):
        type_name: Literal["float", "int"] = (
            "int" if is_integer_like_type(op.res.type) else "float"
        )
        kind: Literal["add", "mul"] = (
            "add" if isinstance(op, varith.VarithAddOp) else "mul"
        )

        target_arith_type = ARITH_TYPES[(type_name, kind)]

        new_operands: list[SSAValue] = []
        possibly_erased_ops: list[Operation] = []
        for inp in op.operands:
            if isa(inp.owner, target_arith_type):
                new_operands.append(inp.owner.lhs)
                new_operands.append(inp.owner.rhs)
                possibly_erased_ops.append(inp.owner)
            elif isa(inp.owner, type(op)):
                new_operands.extend(inp.owner.operands)
                possibly_erased_ops.append(inp.owner)
            else:
                new_operands.append(inp)

        # nothing to do if no new operands are added
        if len(possibly_erased_ops) == 0:
            return

        rewriter.replace_matched_op(type(op)(*new_operands))  # pyright: ignore[reportUnknownArgumentType]

        for old_op in possibly_erased_ops:
            if len(old_op.results[-1].uses) == 0:
                rewriter.erase_op(old_op)


def is_integer_like_type(t: Attribute) -> bool:
    if isinstance(t, builtin.IntegerType | builtin.IndexType):
        return True
    if isinstance(t, builtin.ContainerType):
        elm_type = cast(builtin.ContainerType[Attribute], t).get_element_type()
        return is_integer_like_type(elm_type)
    return False


class ConvertArithToVarithPass(ModulePass):
    name = "convert-arith-to-varith"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ArithToVarithPattern(),
                    MergeVarithOpsPattern(),
                ]
            )
        ).rewrite_op(op)
