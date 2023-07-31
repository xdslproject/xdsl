from math import ceil, log2

from xdsl.dialects import builtin, riscv
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalize import is_canonicalization

_HAS_IMMEDIATE_FORM_OP: dict[type[riscv.RdRsRsIntegerOperation], type[riscv.RdRsImmIntegerOperation]] = {  # type: ignore[reportGeneralTypeIssues]
    riscv.AddOp: riscv.AddiOp,
    riscv.AndOp: riscv.AndiOp,
    riscv.SltOp: riscv.SltiOp,
    riscv.SllOp: riscv.SlliOp,
    riscv.SrlOp: riscv.SrliOp,
    riscv.SraOp: riscv.SraiOp,
    riscv.OrOp: riscv.OriOp,
    riscv.XorOp: riscv.XoriOp,
    riscv.SltuOp: riscv.SltiuOp,
}

# TODO: make interface
_COMMUTATIVE_OPS = (riscv.AddOp, riscv.AndOp, riscv.OrOp, riscv.XorOp)


def twos_complement_bitwidth(num: int):
    """
    Quick approximation of number of bits required to represent a number in two's
    complement.
    """
    is_neg = 1 if num < 0 else 0
    num = abs(num)
    return ceil(log2(num)) + is_neg


def can_be_imm_arg(arg: SSAValue):
    """
    Checks if an SSA Value can be folded into an immediate argument.

    Checks if it is an `li` operation and requires 12 or fewer bits to represent.
    """
    return (
        isinstance(arg.owner, riscv.LiOp)
        and isinstance(arg.owner.immediate, builtin.IntegerAttr)
        and twos_complement_bitwidth(arg.owner.immediate.value.data) <= 12
    )


@is_canonicalization
class FoldRiscvImmediates(RewritePattern):
    """
    Folds a `li` followed by an op that has an immediate form into the immediate
    form of the operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: riscv.AddOp
        | riscv.AndOp
        | riscv.SltOp
        | riscv.SllOp
        | riscv.SrlOp
        | riscv.SraOp
        | riscv.OrOp
        | riscv.XorOp
        | riscv.SltuOp,
        rewriter: PatternRewriter,
        /,
    ):
        args_to_check = op.operands
        if not isinstance(op, _COMMUTATIVE_OPS):
            args_to_check = [op.rs1]
        arg_to_replace = None

        for arg in args_to_check:
            if can_be_imm_arg(arg):
                arg_to_replace = arg
                break

        # do nothing if no suitable arg was found
        if arg_to_replace is None:
            return

        # copy arguments to op
        args = list(op.operands)
        # remove the replaced operand
        args.remove(arg_to_replace)
        assert len(args) == 1
        # construct new op
        assert isinstance(arg_to_replace.owner, riscv.LiOp)
        assert isinstance(arg_to_replace.owner.immediate, builtin.IntegerAttr)
        assert isinstance(op.rd.type, riscv.IntRegisterType)
        new_op = _HAS_IMMEDIATE_FORM_OP[type(op)](
            args[0], arg_to_replace.owner.immediate.value.data, rd=op.rd.type
        )
        # replace old op
        rewriter.replace_matched_op(new_op)
