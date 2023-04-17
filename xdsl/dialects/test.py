from __future__ import annotations
from xdsl.ir import Attribute, Data, Dialect, OpResult, Operation, TypeAttribute

from xdsl.irdl import VarOpResult, irdl_attr_definition, irdl_op_definition
from xdsl.parser import BaseParser
from xdsl.printer import Printer


@irdl_op_definition
class ProduceValuesOp(Operation):
    """
    This operation can produce an arbitrary number of SSAValues with arbitrary
    types. It is used in filecheck testing to reduce to artificial dependencies
    on other dialects (i.e. dependencies that only come from the structure of
    the test rather than the actual dialect).
    """
    name: str = "test.produce_values"

    res: VarOpResult

    @staticmethod
    def from_result_types(*res: Attribute) -> ProduceValuesOp:
        return ProduceValuesOp.create(result_types=res)

    @staticmethod
    def get_values(
            *res: Attribute) -> tuple[ProduceValuesOp, tuple[OpResult, ...]]:
        op = ProduceValuesOp.from_result_types(*res)
        return op, tuple(op.results)


@irdl_attr_definition
class TestType(Data[str], TypeAttribute):
    """
    This attribute is used for testing in places where any attribute can be
    used. This allows reducing the artificial dependencies on attributes from
    other dialects.
    """
    name: str = "test.type"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> str:
        return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string_literal(self.data)


Test = Dialect([ProduceValuesOp], [TestType])
