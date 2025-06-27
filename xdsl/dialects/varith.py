from __future__ import annotations

from typing import ClassVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    BFloat16Type,
    ContainerOf,
    DenseIntElementsAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    IndexType,
    IntegerType,
    VectorType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure

integerOrFloatLike = ContainerOf(
    AnyOf(
        [
            IntegerType,
            IndexType,
            BFloat16Type,
            Float16Type,
            Float32Type,
            Float64Type,
            Float80Type,
            Float128Type,
        ]
    )
)


class VarithOp(IRDLOperation):
    """
    Variadic arithmetic operation
    """

    T: ClassVar = VarConstraint("T", integerOrFloatLike)

    args = var_operand_def(T)
    res = result_def(T)

    traits = traits_def(Pure())

    assembly_format = "$args attr-dict `:` type($res)"

    def __init__(self, *args: SSAValue | Operation):
        assert len(args) > 0
        super().__init__(operands=[args], result_types=[SSAValue.get(args[-1]).type])


@irdl_op_definition
class VarithAddOp(VarithOp):
    name = "varith.add"


@irdl_op_definition
class VarithMulOp(VarithOp):
    name = "varith.mul"


@irdl_op_definition
class VarithSwitchOp(IRDLOperation):
    """
    Variadic selection operation

    Similar to `cf.switch`, this operation returns the argument corresponding to
    `flag`, returning the default value otherwise.
    """

    name = "varith.switch"

    T: ClassVar = VarConstraint("T", AnyAttr())

    flag = operand_def(IntegerType | IndexType)
    case_values = prop_def(DenseIntElementsAttr)

    default_arg = operand_def(T)
    args = var_operand_def(T)

    result = result_def(T)

    traits = traits_def(Pure())

    def __init__(
        self,
        flag: SSAValue | Operation,
        case_values: DenseIntElementsAttr,
        default_arg: SSAValue | Operation,
        *args: SSAValue | Operation,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=[
                flag,
                default_arg,
                args,
            ],
            properties={
                "case_values": case_values,
            },
            attributes=attr_dict,
            result_types=(SSAValue.get(default_arg).type,),
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        unresolved_flag = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        flag_type = parser.parse_type()
        assert isinstance(flag_type, IntegerType | IndexType)
        flag = parser.resolve_operand(unresolved_flag, flag_type)
        parser.parse_punctuation("->")
        return_type = parser.parse_type()
        parser.parse_punctuation(",")
        parser.parse_punctuation("[")
        parser.parse_keyword("default")
        parser.parse_punctuation(":")
        default_arg = parser.resolve_operand(
            parser.parse_unresolved_operand(), return_type
        )

        values: list[int] = []
        args: list[SSAValue] = []
        while parser.parse_optional_punctuation(","):
            values.append(parser.parse_integer())
            parser.parse_punctuation(":")
            args.append(
                parser.resolve_operand(parser.parse_unresolved_operand(), return_type)
            )
        parser.parse_punctuation("]")
        attr_dict = parser.parse_optional_attr_dict()

        case_values = DenseIntElementsAttr.from_list(
            VectorType(flag_type, (len(values),)), values
        )

        return cls(
            flag,
            case_values,
            default_arg,
            *args,
            attr_dict=attr_dict,
        )

    @staticmethod
    def _print_case(printer: Printer, case_name: str, arg: SSAValue):
        printer.print_string(case_name)
        printer.print_string(": ")
        printer.print_operand(arg)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_operand(self.flag)
        printer.print_string(" : ")
        printer.print_attribute(self.flag.type)
        printer.print_string(" -> ")
        printer.print_attribute(self.result.type)
        printer.print_string(", [")
        with printer.indented():
            printer.print_string("\n")
            cases = [("default", self.default_arg)] + [
                (str(c), arg)
                for (c, arg) in zip(
                    self.case_values.get_values(),
                    self.args,
                    strict=True,
                )
            ]
            printer.print_list(
                cases, lambda x: self._print_case(printer, x[0], x[1]), ",\n"
            )
        printer.print_string("\n]")
        if self.attributes:
            printer.print_attr_dict(self.attributes)


Varith = Dialect(
    "varith",
    [
        VarithAddOp,
        VarithMulOp,
        VarithSwitchOp,
    ],
)
