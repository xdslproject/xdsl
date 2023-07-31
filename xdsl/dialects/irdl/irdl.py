"""Definition of the IRDL dialect."""

from __future__ import annotations

from typing import Sequence

from xdsl.dialects.builtin import StringAttr, SymbolRefAttr
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    region_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoTerminator,
    SymbolOpInterface,
    SymbolTable,
)

################################################################################
# Dialect, Operation, and Attribute definitions                                #
################################################################################


@irdl_attr_definition
class AttributeType(ParametrizedAttribute, TypeAttribute):
    """Type of a attribute handle."""

    name = "irdl.attribute"


@irdl_op_definition
class DialectOp(IRDLOperation):
    """A dialect definition."""

    name = "irdl.dialect"

    sym_name: StringAttr = attr_def(StringAttr)
    body: Region = region_def("single_block")

    traits = frozenset([NoTerminator(), SymbolOpInterface(), SymbolTable()])

    def __init__(self, name: str | StringAttr, body: Region):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(attributes={"sym_name": name}, regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> DialectOp:
        sym_name = parser.parse_symbol_name()
        region = parser.parse_optional_region()
        if region is None:
            region = Region(Block())
        return DialectOp(sym_name, region)

    def print(self, printer: Printer) -> None:
        printer.print(" @", self.sym_name.data, " ")
        if self.body.block.ops:
            printer.print_region(self.body)


@irdl_op_definition
class TypeOp(IRDLOperation):
    """A type definition."""

    name = "irdl.type"

    sym_name: StringAttr = attr_def(StringAttr)
    body: Region = region_def("single_block")

    traits = frozenset([NoTerminator(), HasParent(DialectOp), SymbolOpInterface()])

    def __init__(self, name: str | StringAttr, body: Region):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(attributes={"sym_name": name}, regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> TypeOp:
        sym_name = parser.parse_symbol_name()
        region = parser.parse_optional_region()
        if region is None:
            region = Region(Block())
        return TypeOp(sym_name, region)

    def print(self, printer: Printer) -> None:
        printer.print(" @", self.sym_name.data, " ")
        if self.body.block.ops:
            printer.print_region(self.body)


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """An attribute definition."""

    name = "irdl.attribute"

    sym_name: StringAttr = attr_def(StringAttr)
    body: Region = region_def("single_block")

    traits = frozenset([NoTerminator(), HasParent(DialectOp), SymbolOpInterface()])

    def __init__(self, name: str | StringAttr, body: Region):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(attributes={"sym_name": name}, regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> AttributeOp:
        sym_name = parser.parse_symbol_name()
        region = parser.parse_optional_region()
        if region is None:
            region = Region(Block())
        return AttributeOp(sym_name, region)

    def print(self, printer: Printer) -> None:
        printer.print(" @", self.sym_name.data, " ")
        if self.body.block.ops:
            printer.print_region(self.body)


@irdl_op_definition
class ParametersOp(IRDLOperation):
    """An attribute or type parameter definition"""

    name = "irdl.parameters"

    args: VarOperand = var_operand_def(AttributeType)

    traits = frozenset([HasParent(TypeOp, AttributeOp)])

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args])

    @classmethod
    def parse(cls, parser: Parser) -> ParametersOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return ParametersOp(args)

    def print(self, printer: Printer) -> None:
        printer.print("(")
        printer.print_list(self.args, printer.print, ", ")
        printer.print(")")


@irdl_op_definition
class OperationOp(IRDLOperation):
    """An operation definition."""

    name = "irdl.operation"

    sym_name: StringAttr = attr_def(StringAttr)
    body: Region = region_def("single_block")

    traits = frozenset([NoTerminator(), HasParent(DialectOp), SymbolOpInterface()])

    def __init__(self, name: str | StringAttr, body: Region):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(attributes={"sym_name": name}, regions=[body])

    @classmethod
    def parse(cls, parser: Parser) -> OperationOp:
        sym_name = parser.parse_symbol_name()
        region = parser.parse_optional_region()
        if region is None:
            region = Region(Block())
        return OperationOp(sym_name, region)

    def print(self, printer: Printer) -> None:
        printer.print(" @", self.sym_name.data, " ")
        if self.body.block.ops:
            printer.print_region(self.body)


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """An operation operand definition."""

    name = "irdl.operands"

    args: VarOperand = var_operand_def(AttributeType)

    traits = frozenset([HasParent(OperationOp)])

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args])

    @classmethod
    def parse(cls, parser: Parser) -> OperandsOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return OperandsOp(args)

    def print(self, printer: Printer) -> None:
        printer.print("(")
        printer.print_list(self.args, printer.print, ", ")
        printer.print(")")


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """An operation result definition."""

    name = "irdl.results"

    args: VarOperand = var_operand_def(AttributeType)

    traits = frozenset([IsTerminator(), HasParent(OperationOp)])

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args])

    @classmethod
    def parse(cls, parser: Parser) -> ResultsOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return ResultsOp(args)

    def print(self, printer: Printer) -> None:
        printer.print("(")
        printer.print_list(self.args, printer.print, ", ")
        printer.print(")")


################################################################################
# Attribute constraints                                                        #
################################################################################


@irdl_op_definition
class IsOp(IRDLOperation):
    """Constraint an attribute/type to be a specific attribute instance."""

    name = "irdl.is"

    expected: Attribute = attr_def(Attribute)
    output: OpResult = result_def(AttributeType)

    def __init__(self, expected: Attribute):
        super().__init__(
            attributes={"expected": expected}, result_types=[AttributeType()]
        )

    @classmethod
    def parse(cls, parser: Parser) -> IsOp:
        expected = parser.parse_attribute()
        return IsOp(expected)

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_attribute(self.expected)


@irdl_op_definition
class ParametricOp(IRDLOperation):
    """Constraint an attribute/type base and its parameters"""

    name = "irdl.parametric"

    base_type: SymbolRefAttr = attr_def(SymbolRefAttr)
    args: VarOperand = var_operand_def(AttributeType)
    output: OpResult = result_def(AttributeType)

    def __init__(
        self, base_type: str | StringAttr | SymbolRefAttr, args: Sequence[SSAValue]
    ):
        if isinstance(base_type, str | StringAttr):
            base_type = SymbolRefAttr(base_type)
        super().__init__(
            attributes={"base_type": base_type},
            operands=[args],
            result_types=[AttributeType()],
        )

    @classmethod
    def parse(cls, parser: Parser) -> ParametricOp:
        base_type = parser.parse_attribute()
        if not isinstance(base_type, SymbolRefAttr):
            parser.raise_error("expected symbol reference")
        args = parser.parse_comma_separated_list(
            parser.Delimiter.ANGLE, parser.parse_operand
        )
        return ParametricOp(base_type, args)

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_attribute(self.base_type)
        printer.print("<")
        printer.print_list(self.args, printer.print, ", ")
        printer.print(">")


@irdl_op_definition
class AnyOp(IRDLOperation):
    """Constraint an attribute/type to be any attribute/type instance."""

    name = "irdl.any"

    output: OpResult = result_def(AttributeType)

    def __init__(self):
        super().__init__(result_types=[AttributeType()])

    @classmethod
    def parse(cls, parser: Parser) -> AnyOp:
        return AnyOp()

    def print(self, printer: Printer) -> None:
        pass


@irdl_op_definition
class AnyOfOp(IRDLOperation):
    """Constraint an attribute/type to the union of the provided constraints."""

    name = "irdl.any_of"

    args: VarOperand = var_operand_def(AttributeType)
    output: OpResult = result_def(AttributeType)

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args], result_types=[AttributeType()])

    @classmethod
    def parse(cls, parser: Parser) -> AnyOfOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return AnyOfOp(args)

    def print(self, printer: Printer) -> None:
        printer.print("(")
        printer.print_list(self.args, printer.print, ", ")
        printer.print(")")


@irdl_op_definition
class AllOfOp(IRDLOperation):
    """Constraint an attribute/type to the intersection of the provided constraints."""

    name = "irdl.all_of"

    args: VarOperand = var_operand_def(AttributeType)
    output: OpResult = result_def(AttributeType)

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args], result_types=[AttributeType()])

    @classmethod
    def parse(cls, parser: Parser) -> AllOfOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return AllOfOp(args)

    def print(self, printer: Printer) -> None:
        printer.print("(")
        printer.print_list(self.args, printer.print, ", ")
        printer.print(")")


IRDL = Dialect(
    [
        DialectOp,
        TypeOp,
        AttributeOp,
        ParametersOp,
        OperationOp,
        OperandsOp,
        ResultsOp,
        IsOp,
        ParametricOp,
        AnyOp,
        AnyOfOp,
        AllOfOp,
    ],
    [AttributeType],
)
