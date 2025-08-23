from abc import ABC, abstractmethod
from collections.abc import Sequence

from typing_extensions import Self

from xdsl.dialects import riscv
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    RISCVInstruction,
    RISCVRegisterType,
)
from xdsl.interfaces import HasCanonicalizationPatternsInterface
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    Successor,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    successor_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.comparisons import to_signed, to_unsigned
from xdsl.utils.exceptions import VerifyException


def _print_type_pair(printer: Printer, value: SSAValue) -> None:
    printer.print_ssa_value(value)
    printer.print_string(" : ")
    printer.print_attribute(value.type)


def _parse_type_pair(parser: Parser) -> SSAValue:
    unresolved = parser.parse_unresolved_operand()
    parser.parse_punctuation(":")
    type = parser.parse_type()
    return parser.resolve_operand(unresolved, type)


class ConditionalBranchOperation(
    RISCVInstruction, HasCanonicalizationPatternsInterface, ABC
):
    """
    A base class for RISC-V branch operations. Lowers to RsRsOffOperation.
    """

    rs1 = operand_def(IntRegisterType)
    rs2 = operand_def(IntRegisterType)

    then_arguments = var_operand_def(RISCVRegisterType)
    else_arguments = var_operand_def(RISCVRegisterType)

    irdl_options = [AttrSizedOperandSegments()]

    then_block = successor_def()
    else_block = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        rs1: Operation | SSAValue,
        rs2: Operation | SSAValue,
        then_arguments: Sequence[SSAValue],
        else_arguments: Sequence[SSAValue],
        then_block: Successor,
        else_block: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[rs1, rs2, then_arguments, else_arguments],
            attributes={
                "comment": comment,
            },
            successors=(then_block, else_block),
        )

    def verify_(self) -> None:
        # The then block must start with a label op

        then_block_first_op = self.then_block.first_op

        if not isinstance(then_block_first_op, riscv.LabelOp):
            raise VerifyException(
                "riscv_cf branch op then block first op must be a label"
            )

        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.then_arguments, self.then_block.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        for op_arg, block_arg in zip(self.else_arguments, self.else_block.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        # The else block must be the one immediately following this one

        parent_block = self.parent
        if parent_block is None:
            return

        parent_region = parent_block.parent
        if parent_region is None:
            return

        if parent_block.next_block is not self.else_block:
            raise VerifyException(
                "riscv_cf branch op else block must be immediately after op"
            )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        then_label = self.then_block.first_op
        assert isinstance(then_label, riscv.LabelOp)
        return self.rs1, self.rs2, then_label.label

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        _print_type_pair(printer, self.rs1)
        printer.print_string(", ")
        _print_type_pair(printer, self.rs2)
        printer.print_string(", ")
        printer.print_block_name(self.then_block)
        printer.print_string("(")
        printer.print_list(
            self.then_arguments, lambda val: _print_type_pair(printer, val)
        )
        printer.print_string("), ")
        printer.print_block_name(self.else_block)
        printer.print_string("(")
        printer.print_list(
            self.else_arguments, lambda val: _print_type_pair(printer, val)
        )
        printer.print_string(")")
        if self.attributes:
            printer.print_op_attributes(
                self.attributes,
                reserved_attr_names="operandSegmentSizes",
                print_keyword=True,
            )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        rs1 = _parse_type_pair(parser)
        parser.parse_punctuation(",")
        rs2 = _parse_type_pair(parser)
        parser.parse_punctuation(",")
        then_block = parser.parse_successor()
        then_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
        )
        parser.parse_punctuation(",")
        else_block = parser.parse_successor()
        else_args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        op = cls(rs1, rs2, then_args, else_args, then_block, else_block)
        if attrs is not None:
            op.attributes |= attrs.data
        return op

    @abstractmethod
    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        pass

    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.riscv_cf import (
            ElideConstantBranches,
        )

        return (ElideConstantBranches(),)


@irdl_op_definition
class BeqOp(ConditionalBranchOperation):
    """
    Take the branch if registers rs1 and rs2 are equal.

    if (x[rs1] == x[rs2]) pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#beq).
    """

    name = "riscv_cf.beq"

    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        lhs = to_unsigned(rs1, bitwidth)
        rhs = to_unsigned(rs2, bitwidth)
        return lhs == rhs


@irdl_op_definition
class BneOp(ConditionalBranchOperation):
    """
    Take the branch if registers rs1 and rs2 are not equal.

    if (x[rs1] != x[rs2]) pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bne).
    """

    name = "riscv_cf.bne"

    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        lhs = to_unsigned(rs1, bitwidth)
        rhs = to_unsigned(rs2, bitwidth)
        return lhs != rhs


@irdl_op_definition
class BltOp(ConditionalBranchOperation):
    """
    Take the branch if registers rs1 is less than rs2, using signed comparison.

    if (x[rs1] <s x[rs2]) pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#blt).
    """

    name = "riscv_cf.blt"

    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        lhs = to_signed(rs1, bitwidth)
        rhs = to_signed(rs2, bitwidth)
        return lhs < rhs


@irdl_op_definition
class BgeOp(ConditionalBranchOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using signed comparison.

    if (x[rs1] >=s x[rs2]) pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bge).
    """

    name = "riscv_cf.bge"

    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        lhs = to_signed(rs1, bitwidth)
        rhs = to_signed(rs2, bitwidth)
        return lhs >= rhs


@irdl_op_definition
class BltuOp(ConditionalBranchOperation):
    """
    Take the branch if registers rs1 is less than rs2, using unsigned comparison.

    if (x[rs1] <u x[rs2]) pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bltu).
    """

    name = "riscv_cf.bltu"

    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        lhs = to_unsigned(rs1, bitwidth)
        rhs = to_unsigned(rs2, bitwidth)
        return lhs < rhs


@irdl_op_definition
class BgeuOp(ConditionalBranchOperation):
    """
    Take the branch if registers rs1 is greater than or equal to rs2, using unsigned comparison.

    if (x[rs1] >=u x[rs2]) pc += sext(offset)

    See external [documentation](https://msyksphinz-self.github.io/riscv-isadoc/html/rvi.html#bgeu).
    """

    name = "riscv_cf.bgeu"

    def const_evaluate(self, rs1: int, rs2: int, bitwidth: int) -> bool:
        lhs = to_unsigned(rs1, bitwidth)
        rhs = to_unsigned(rs2, bitwidth)
        return lhs >= rhs


@irdl_op_definition
class BranchOp(riscv.RISCVAsmOperation):
    """
    Branches to a different block, which must follow this operation's block in the parent
    region. Is not printed in assembly.
    """

    name = "riscv_cf.branch"

    block_arguments = var_operand_def(RISCVRegisterType)
    successor = successor_def()
    comment = opt_attr_def(StringAttr)
    """
    An optional comment that will be printed along with the instruction.
    """

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        block_arguments: Sequence[SSAValue],
        successor: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[block_arguments],
            attributes={
                "comment": comment,
            },
            successors=(successor,),
        )

    def verify_(self) -> None:
        # Types of arguments must match arg types of blocks
        for op_arg, block_arg in zip(self.block_arguments, self.successor.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        # The successor must immediately follow the parent block.
        if (parent_block := self.parent) is None or (
            parent_region := parent_block.parent
        ) is None:
            return

        parent_index = parent_region.get_block_index(parent_block)
        successor_index = parent_region.get_block_index(self.successor)

        if parent_index + 1 != successor_index:
            raise VerifyException(
                "Successor block must be immediately after parent block in the parent "
                "region."
            )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_block_name(self.successor)
        printer.print_string("(")
        printer.print_list(
            self.block_arguments, lambda val: _print_type_pair(printer, val)
        )
        printer.print_string(")")
        if self.attributes:
            printer.print_op_attributes(self.attributes, print_keyword=True)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        successor = parser.parse_successor()
        block_arguments = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        op = cls(block_arguments, successor)
        if attrs is not None:
            op.attributes |= attrs.data
        return op

    def assembly_line(self) -> str | None:
        if self.comment is None:
            return None

        return f"    # {self.comment.data}"


@irdl_op_definition
class JOp(RISCVInstruction):
    """
    A pseudo-instruction, for unconditional jumps you don't expect to return from.
    Is equivalent to JalOp with `rd` = `x0`.
    Used to be a part of the spec, removed in 2.0.
    """

    name = "riscv_cf.j"

    block_arguments = var_operand_def(RISCVRegisterType)

    successor = successor_def()

    traits = traits_def(IsTerminator())

    def __init__(
        self,
        block_arguments: Sequence[SSAValue],
        successor: Successor,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[block_arguments],
            attributes={
                "comment": comment,
            },
            successors=(successor,),
        )

    def verify_(self) -> None:
        # Types of arguments must match arg types of blocks

        for op_arg, block_arg in zip(self.block_arguments, self.successor.args):
            if op_arg.type != block_arg.type:
                raise VerifyException(
                    f"Block arg types must match {op_arg.type} {block_arg.type}"
                )

        if not isinstance(self.successor.first_op, riscv.LabelOp):
            raise VerifyException(
                "riscv_cf.j operation successor must have a riscv.label operation as a "
                f"first argument, found {self.successor.first_op}"
            )

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_block_name(self.successor)
        printer.print_string("(")
        printer.print_list(
            self.block_arguments, lambda val: _print_type_pair(printer, val)
        )
        printer.print_string(")")
        if self.attributes:
            printer.print_op_attributes(self.attributes, print_keyword=True)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        successor = parser.parse_successor()
        block_arguments = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_type_pair(parser)
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        op = cls(block_arguments, successor)
        if attrs is not None:
            op.attributes |= attrs.data
        return op

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        dest_label = self.successor.first_op
        assert isinstance(dest_label, riscv.LabelOp)
        return (dest_label.label,)


RISCV_Cf = Dialect(
    "riscv_cf",
    [
        JOp,
        BranchOp,
        BeqOp,
        BneOp,
        BltOp,
        BgeOp,
        BltuOp,
        BgeuOp,
    ],
)
