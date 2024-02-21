from __future__ import annotations

from xdsl.dialects.builtin import (
    StringAttr,
    TypeAttribute,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
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
    def parse(cls, parser: Parser) -> None:
        pass

    def _verify(self) -> None:
        if not (self.is_mutuable and self.value):
            raise VerifyException("Immutable global must have an initial value")


MLProgram = Dialect(
    "ml_program",
    [
        Global,
    ],
)
