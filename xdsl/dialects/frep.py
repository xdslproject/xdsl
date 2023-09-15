from collections.abc import Sequence

from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntAttr
from xdsl.dialects.riscv import AssemblyInstructionArg
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    region_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import NoTerminator, Pure
from xdsl.utils.exceptions import VerifyException


class FRepOperation(IRDLOperation, riscv.RISCVInstruction):
    """
    From the Snitch paper: https://arxiv.org/abs/2002.10143

    The frep instruction marks the beginning of a floating-point kernel which should be
    repeated. It indicates how many subsequent instructions are stored in the sequence
    buffer, how often and how (operand staggering, repetition mode) each instruction is
    going to be repeated.
    """

    max_rep = operand_def(riscv.IntRegisterType)
    """Number of times to repeat the instructions."""
    body = region_def("single_block")
    """
    Instructions to repeat, containing maximum 15 instructions, with no side effects.
    """
    stagger_mask = attr_def(IntAttr)
    """
    4 bits for each operand (rs1 rs2 rs3 rd). If the bit is set, the corresponding operand
    is staggered.
    """
    stagger_count = attr_def(IntAttr)
    """
    3 bits, indicating for how many iterations the stagger should increment before it
    wraps again (up to 23 = 8).
    """

    traits = frozenset((NoTerminator(),))

    def __init__(
        self,
        max_rep: SSAValue | Operation,
        body: Sequence[Operation] | Sequence[Block] | Region,
        max_inst: IntAttr,
        stagger_mask: IntAttr,
        stagger_count: IntAttr,
    ):
        super().__init__(
            operands=(max_rep,),
            regions=(body,),
            attributes={
                "max_inst": max_inst,
                "stagger_mask": stagger_mask,
                "stagger_count": stagger_count,
            },
        )

    @property
    def max_inst(self) -> int:
        """
        Number of instructions to be repeated.
        """
        return len(
            [op for op in self.body.ops if isinstance(op, riscv.RISCVInstruction)]
        )

    def assembly_instruction_name(self) -> str:
        return self.name

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (
            self.max_rep,
            self.max_inst,
            self.stagger_mask.data,
            self.stagger_count.data,
        )

    def custom_print_attributes(self, printer: Printer):
        printer.print(", ")
        printer.print(self.stagger_mask.data)
        printer.print_string(", ")
        printer.print(self.stagger_count.data)
        return {"stagger_mask", "stagger_count"}

    @classmethod
    def custom_parse_attributes(cls, parser: Parser):
        attributes = dict[str, Attribute]()
        attributes["stagger_mask"] = IntAttr(
            parser.parse_integer(
                allow_boolean=False, context_msg="Expected stagger mask"
            )
        )
        parser.parse_punctuation(",")
        attributes["stagger_count"] = IntAttr(
            parser.parse_integer(
                allow_boolean=False, context_msg="Expected stagger count"
            )
        )
        return attributes

    def verify_(self) -> None:
        if self.stagger_count.data:
            raise VerifyException("Non-zero stagger count currently unsupported")
        if self.stagger_mask.data:
            raise VerifyException("Non-zero stagger mask currently unsupported")
        for instruction in self.body.ops:
            if not instruction.has_trait(Pure):
                raise VerifyException(
                    "Frep operation body may not contain instructions "
                    f"with side-effects, found {instruction.name}"
                )


@irdl_op_definition
class Outer(FRepOperation):
    name = "frep.outer"


@irdl_op_definition
class Inner(FRepOperation):
    name = "frep.inner"


FRep = Dialect(
    [
        Outer,
        Inner,
    ],
    [],
)
