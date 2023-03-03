from __future__ import annotations
from xdsl.ir import Attribute, Data, Dialect, MLIRType, OpResult, Operation

from xdsl.irdl import VarOpResult, irdl_attr_definition, irdl_op_definition
from xdsl.parser import BaseParser
from xdsl.printer import Printer


@irdl_op_definition
class ProduceValuesOp(Operation):
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
class StrType(Data[str], MLIRType):
    name: str = "test.str"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> str:
        return parser.parse_str_literal()

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        printer.print_string_literal(data)


Test = Dialect([ProduceValuesOp], [StrType])
