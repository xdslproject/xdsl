from __future__ import annotations

from typing_extensions import Self

from xdsl.dialects.builtin import (
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class Global(IRDLOperation):
    """
    Module level declaration of a global variable

    See https://mlir.llvm.org/docs/Dialects/MLProgramOps/#ml_programglobal-ml_programglobalop
    """

    name = "ml_program.global"

    sym_name = attr_def(StringAttr)
    type = attr_def(TypeAttribute)
    is_mutuable = attr_def(UnitAttr)
    value = attr_def(Attribute)
    sym_visibility = attr_def(StringAttr)

    def __init__(
        self,
        sym_name: Attribute,
        is_mutuable: Attribute,
        value: Attribute,
        sym_visibility: Attribute,
    ):
        super().__init__(
            attributes={
                "sym_name": sym_name,
                "is_mutable": is_mutuable,
                "value": value,
                "sym_visibility": sym_visibility,
            },
        )

    def print(self, printer: Printer):
        pass

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        pass

    def _verify(self) -> None:
        if not (self.is_mutuable and self.value):
            raise VerifyException("Immutable global must have an initial value")


@irdl_op_definition
class GlobalLoadConstant(IRDLOperation):
    """
    Direct load a constant value from a global

    See https://mlir.llvm.org/docs/Dialects/MLProgramOps/#ml_programglobal_load_const-ml_programgloballoadconstop
    """

    name = "ml_program.global_load_const"

    global_attr = attr_def(SymbolRefAttr)
    result = result_def()

    def __init__(
        self,
        global_attr: Attribute,
        result_type: Attribute | None,
    ):
        super().__init__(
            attributes={
                "global_attr": global_attr,
            },
            result_types=[result_type],
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_attribute(self.global_attr)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs = parser.parse_optional_attr_dict()
        global_attr = parser.parse_attribute()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()

        global_const = cls(
            global_attr,
            result_type,
        )
        global_const.attributes |= attrs
        return global_const


MLProgram = Dialect(
    "ml_program",
    [
        Global,
        GlobalLoadConstant,
    ],
)
