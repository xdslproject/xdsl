from __future__ import annotations
from xdsl.ir import Data, Dialect, TypeAttribute

from xdsl.irdl import (
    VarOpResult,
    VarOperand,
    VarRegion,
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_op_definition
class TestOp(IRDLOperation):
    """
    This operation can produce an arbitrary number of SSAValues with arbitrary
    types. It is used in filecheck testing to reduce to artificial dependencies
    on other dialects (i.e. dependencies that only come from the structure of
    the test rather than the actual dialect).
    """

    name = "test.op"

    res: VarOpResult
    ops: VarOperand
    regs: VarRegion


@irdl_attr_definition
class TestType(Data[str], TypeAttribute):
    """
    This attribute is used for testing in places where any attribute can be
    used. This allows reducing the artificial dependencies on attributes from
    other dialects.
    """

    name = "test.type"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(self.data)


Test = Dialect([TestOp], [TestType])
