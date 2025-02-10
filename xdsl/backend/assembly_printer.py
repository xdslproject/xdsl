from __future__ import annotations

from abc import ABC, abstractmethod

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation
from xdsl.utils.base_printer import BasePrinter


class AssemblyPrinter(BasePrinter):
    @staticmethod
    def append_comment(line: str, comment: StringAttr | None) -> str:
        if comment is None:
            return line

        padding = " " * max(0, 48 - len(line))

        return f"{line}{padding} # {comment.data}"

    @staticmethod
    def assembly_line(
        name: str,
        arg_str: str,
        comment: StringAttr | None = None,
        is_indented: bool = True,
    ) -> str:
        code = "    " if is_indented else ""
        code += name
        if arg_str:
            code += f" {arg_str}"
        code = AssemblyPrinter.append_comment(code, comment)
        return code

    def print_module(self, module: ModuleOp) -> None:
        for op in module.body.walk():
            assert isinstance(op, AssemblyPrintable), f"{op}"
            op.print_assembly(self)


class AssemblyPrintable(Operation, ABC):
    """
    Base class for operations that can be a part of assembly printing.
    """

    @abstractmethod
    def print_assembly(self, printer: AssemblyPrinter) -> None:
        raise NotImplementedError()


class OneLineAssemblyPrintable(AssemblyPrintable, ABC):
    """
    Base class for operations that can be printed in one line of assembly, or not
    printed at all.
    """

    @abstractmethod
    def assembly_line(self) -> str | None:
        """
        Returns the line that should be printed in assembly, or `None` to not be
        printed.
        """
        raise NotImplementedError()

    def print_assembly(self, printer: AssemblyPrinter) -> None:
        line = self.assembly_line()
        if line is not None:
            printer.print_string(line)
            printer.print_string("\n")
