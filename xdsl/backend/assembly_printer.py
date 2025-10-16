from __future__ import annotations

from abc import ABC, abstractmethod

from xdsl.backend.register_type import RegisterType
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.utils.base_printer import BasePrinter


class AssemblyPrinter(BasePrinter):
    _current_section: str | None = None

    def emit_section(self, new_section: str):
        if self._current_section == new_section:
            return
        self._current_section = new_section
        self.print_string(new_section, indent=0)
        self.print_string("\n", indent=0)

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


# region: Assembly arg printing utilities


def reg(value: SSAValue) -> str:
    """
    A wrapper around SSAValue to be printed in assembly.
    Only valid if the type of the value is a RegisterType.
    """

    assert isinstance(value.type, RegisterType)
    return value.type.register_name.data


# endregion
