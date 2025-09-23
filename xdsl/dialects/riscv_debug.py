from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from xdsl.dialects import riscv
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import attr_def, irdl_op_definition, var_operand_def
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_op_definition
class PrintfOp(riscv.RISCVCustomFormatOperation, riscv.RISCVInstruction):
    """
    An instruction to print the contents of registers when emulating riscv code.

    Is not a real instruction in the RISC-V instruction set, but supported by riscemu
    and xDSL's interpreter.

    During assembly emission, the results are printed before the operands:

    ``` python
    s0 = riscv.GetRegisterOp(Registers.s0).res
    s1 = riscv.GetRegisterOp(Registers.s1).res
    op = PrintfOp("s0: {}, s1: {}", (s0, s1))

    op.assembly_line()   # 'printf "s0: {}, s1: {}", s0, s1'
    ```
    """

    name = "riscv_debug.printf"
    format_str = attr_def(StringAttr)
    inputs = var_operand_def()

    def __init__(
        self,
        format_str: str | StringAttr,
        inputs: Sequence[SSAValue] = (),
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(format_str, str):
            format_str = StringAttr(format_str)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[inputs],
            attributes={
                "format_str": format_str,
                "comment": comment,
            },
        )

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["format_str"] = StringAttr(
            parser.parse_str_literal("format string.")
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> AbstractSet[str]:
        printer.print_string(" ")
        printer.print_attribute(self.format_str)
        return {"format_str"}

    def assembly_line_args(self) -> tuple[riscv.AssemblyInstructionArg, ...]:
        return f'"{self.format_str.data}"', *self.operands


RISCV_Debug = Dialect(
    "riscv_debug",
    [
        PrintfOp,
    ],
    [],
)
