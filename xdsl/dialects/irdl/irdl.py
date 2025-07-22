"""Definition of the IRDL dialect."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import NoneType
from typing import ClassVar

from xdsl.dialects.builtin import (
    I32,
    ArrayAttr,
    IntegerAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    EnumAttribute,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    NoTerminator,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum

################################################################################
# Dialect, Operation, and Attribute definitions                                #
################################################################################


class VariadicityEnum(StrEnum):
    SINGLE = "single"
    OPTIONAL = "optional"
    VARIADIC = "variadic"


@irdl_attr_definition
class VariadicityAttr(EnumAttribute[VariadicityEnum], SpacedOpaqueSyntaxAttribute):
    name = "irdl.variadicity"

    SINGLE: ClassVar[VariadicityAttr]
    OPTIONAL: ClassVar[VariadicityAttr]
    VARIADIC: ClassVar[VariadicityAttr]


setattr(VariadicityAttr, "SINGLE", VariadicityAttr(VariadicityEnum.SINGLE))
setattr(VariadicityAttr, "OPTIONAL", VariadicityAttr(VariadicityEnum.OPTIONAL))
setattr(VariadicityAttr, "VARIADIC", VariadicityAttr(VariadicityEnum.VARIADIC))


@irdl_attr_definition
class VariadicityArrayAttr(ParametrizedAttribute, SpacedOpaqueSyntaxAttribute):
    name = "irdl.variadicity_array"

    value: ArrayAttr[VariadicityAttr]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[ArrayAttr[VariadicityAttr]]:
        data = parser.parse_comma_separated_list(
            AttrParser.Delimiter.SQUARE, lambda: VariadicityAttr.parse_parameter(parser)
        )
        return (ArrayAttr(VariadicityAttr(x) for x in data),)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(self.value, lambda var: var.print_parameter(printer))
        printer.print_string("]")


@irdl_attr_definition
class AttributeType(ParametrizedAttribute, TypeAttribute):
    """Type of a attribute handle."""

    name = "irdl.attribute"


@irdl_attr_definition
class RegionType(ParametrizedAttribute, TypeAttribute):
    """IRDL handle to a region definition"""

    name = "irdl.region"


@irdl_op_definition
class DialectOp(IRDLOperation):
    """A dialect definition."""

    name = "irdl.dialect"

    sym_name = attr_def(SymbolNameConstraint())
    body = region_def("single_block")

    traits = traits_def(NoTerminator(), SymbolOpInterface(), SymbolTable())

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
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        if self.body.block.ops:
            printer.print_string(" ")
            printer.print_region(self.body)


@irdl_op_definition
class TypeOp(IRDLOperation):
    """A type definition."""

    name = "irdl.type"

    sym_name = attr_def(SymbolNameConstraint())
    body = region_def("single_block")

    traits = traits_def(NoTerminator(), HasParent(DialectOp), SymbolOpInterface())

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
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        if self.body.block.ops:
            printer.print_string(" ")
            printer.print_region(self.body)

    @property
    def qualified_name(self):
        dialect_op = self.parent_op()
        if not isinstance(dialect_op, DialectOp):
            raise ValueError("Tried to get qualified name of an unverified TypeOp")
        return f"{dialect_op.sym_name.data}.{self.sym_name.data}"


@irdl_op_definition
class CPredOp(IRDLOperation):
    """Constraints an attribute using a C++ predicate"""

    name = "irdl.c_pred"

    pred = attr_def(StringAttr)

    output = result_def(AttributeType())

    assembly_format = "$pred attr-dict"

    def __init__(self, pred: str | StringAttr):
        if isinstance(pred, str):
            pred = StringAttr(pred)
        super().__init__(attributes={"pred": pred}, result_types=[AttributeType()])


@irdl_op_definition
class AttributeOp(IRDLOperation):
    """An attribute definition."""

    name = "irdl.attribute"

    sym_name = attr_def(SymbolNameConstraint())
    body = region_def("single_block")

    traits = traits_def(NoTerminator(), HasParent(DialectOp), SymbolOpInterface())

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
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        if self.body.block.ops:
            printer.print_string(" ")
            printer.print_region(self.body)

    @property
    def qualified_name(self):
        dialect_op = self.parent_op()
        if not isinstance(dialect_op, DialectOp):
            raise ValueError("Tried to get qualified name of an unverified AttributeOp")
        return f"{dialect_op.sym_name.data}.{self.sym_name.data}"


def _parse_argument(parser: Parser) -> tuple[StringAttr, SSAValue]:
    name = StringAttr(parser.parse_identifier())
    parser.parse_punctuation(":")

    arg = parser.parse_operand()

    return (name, arg)


def _print_argument(printer: Printer, data: tuple[StringAttr, SSAValue]) -> None:
    printer.print_string(data[0].data)
    printer.print_string(": ")
    printer.print_operand(data[1])


@irdl_op_definition
class ParametersOp(IRDLOperation):
    """An attribute or type parameter definition"""

    name = "irdl.parameters"

    args = var_operand_def(AttributeType)

    names = prop_def(ArrayAttr[StringAttr])

    traits = traits_def(HasParent(TypeOp, AttributeOp))

    def __init__(self, args: Sequence[SSAValue], names: ArrayAttr[StringAttr]):
        super().__init__(operands=[args], properties={"names": names})

    @classmethod
    def parse(cls, parser: Parser) -> ParametersOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_argument(parser)
        )
        return ParametersOp(
            tuple(x[1] for x in args),
            ArrayAttr(x[0] for x in args),
        )

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_list(
                zip(self.names, self.args), lambda x: _print_argument(printer, x)
            )


@irdl_op_definition
class OperationOp(IRDLOperation):
    """An operation definition."""

    name = "irdl.operation"

    sym_name = attr_def(SymbolNameConstraint())
    body = region_def("single_block")

    traits = traits_def(NoTerminator(), HasParent(DialectOp), SymbolOpInterface())

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
        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        if self.body.block.ops:
            printer.print_string(" ")
            printer.print_region(self.body)

    @property
    def qualified_name(self):
        dialect_op = self.parent_op()
        if not isinstance(dialect_op, DialectOp):
            raise ValueError("Tried to get qualified name of an unverified OperationOp")
        return f"{dialect_op.sym_name.data}.{self.sym_name.data}"

    def get_py_class_name(self) -> str:
        return (
            "".join(
                y[:1].upper() + y[1:]
                for x in self.sym_name.data.split(".")
                for y in x.split("_")
            )
            + "Op"
        )


def _parse_argument_with_var(
    parser: Parser,
) -> tuple[StringAttr, VariadicityAttr, SSAValue]:
    name = StringAttr(parser.parse_identifier())
    parser.parse_punctuation(":")
    variadicity = parser.parse_optional_str_enum(VariadicityEnum)
    if variadicity is None:
        variadicity = VariadicityEnum.SINGLE

    arg = parser.parse_operand()

    return (name, VariadicityAttr(variadicity), arg)


def _print_argument_with_var(
    printer: Printer, data: tuple[StringAttr, VariadicityAttr, SSAValue]
) -> None:
    printer.print_string(data[0].data)
    printer.print_string(": ")
    variadicity = data[1].data
    if variadicity != VariadicityEnum.SINGLE:
        printer.print_string(variadicity)
        printer.print_string(" ")
    printer.print_ssa_value(data[2])


@irdl_op_definition
class OperandsOp(IRDLOperation):
    """An operation operand definition."""

    name = "irdl.operands"

    args = var_operand_def(AttributeType)

    variadicity = prop_def(VariadicityArrayAttr)

    names = prop_def(ArrayAttr[StringAttr])

    traits = traits_def(HasParent(OperationOp))

    def __init__(
        self,
        operands: Sequence[SSAValue],
        variadicity: VariadicityArrayAttr,
        names: ArrayAttr[StringAttr],
    ):
        properties = {
            "variadicity": variadicity,
            "names": names,
        }
        super().__init__(operands=[operands], properties=properties)

    @classmethod
    def parse(cls, parser: Parser) -> OperandsOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_argument_with_var(parser)
        )
        return OperandsOp(
            tuple(x[2] for x in args),
            VariadicityArrayAttr(ArrayAttr(x[1] for x in args)),
            ArrayAttr(x[0] for x in args),
        )

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_list(
                zip(self.names, self.variadicity.value, self.args),
                lambda x: _print_argument_with_var(printer, x),
                ", ",
            )


@irdl_op_definition
class ResultsOp(IRDLOperation):
    """An operation result definition."""

    name = "irdl.results"

    args = var_operand_def(AttributeType)

    variadicity = prop_def(VariadicityArrayAttr)

    names = prop_def(ArrayAttr[StringAttr])

    traits = traits_def(HasParent(OperationOp))

    def __init__(
        self,
        operands: Sequence[SSAValue],
        variadicity: VariadicityArrayAttr,
        names: ArrayAttr[StringAttr],
    ):
        properties = {
            "variadicity": variadicity,
            "names": names,
        }
        super().__init__(operands=[operands], properties=properties)

    @classmethod
    def parse(cls, parser: Parser) -> ResultsOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_argument_with_var(parser)
        )
        return ResultsOp(
            tuple(x[2] for x in args),
            VariadicityArrayAttr(ArrayAttr(x[1] for x in args)),
            ArrayAttr(x[0] for x in args),
        )

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_list(
                zip(self.names, self.variadicity.value, self.args),
                lambda x: _print_argument_with_var(printer, x),
                ", ",
            )


def _parse_attribute(parser: Parser) -> tuple[str, SSAValue]:
    key = parser.parse_str_literal()
    parser.parse_punctuation("=")
    arg = parser.parse_operand()

    return (key, arg)


def _print_attribute(printer: Printer, item: tuple[StringAttr, SSAValue]):
    printer.print_attribute(item[0])
    printer.print_string(" = ")
    printer.print_operand(item[1])


@irdl_op_definition
class AttributesOp(IRDLOperation):
    """Define the attributes of an operation"""

    name = "irdl.attributes"

    attribute_values = var_operand_def(AttributeType())

    attribute_value_names = attr_def(ArrayAttr[StringAttr])

    def __init__(
        self,
        attribute_values: Sequence[SSAValue],
        attribute_value_names: ArrayAttr[StringAttr],
    ):
        super().__init__(
            operands=(attribute_values,),
            attributes={"attribute_value_names": attribute_value_names},
        )

    @classmethod
    def get(cls, attributes: dict[str, SSAValue]) -> AttributesOp:
        operands = tuple(attributes.values())
        names = ArrayAttr(StringAttr(x) for x in attributes.keys())
        return AttributesOp(operands, names)

    @classmethod
    def parse(cls, parser: Parser) -> AttributesOp:
        tuples = parser.parse_optional_comma_separated_list(
            parser.Delimiter.BRACES, lambda: _parse_attribute(parser)
        )
        if tuples is None:
            return AttributesOp.get(dict())
        return AttributesOp.get(dict(tuples))

    def print(self, printer: Printer) -> None:
        if not self.attribute_values:
            return
        with printer.indented():
            printer.print_string(" {\n")
            printer.print_list(
                zip(self.attribute_value_names, self.attribute_values),
                lambda x: _print_attribute(printer, x),
                ",\n",
            )
        printer.print_string("\n}")

    def verify_(self) -> None:
        if len(self.attribute_values) != len(self.attribute_value_names):
            raise VerifyException(
                (
                    "The number of attribute names and their constraints must be the same",
                    f"but got {len(self.attribute_value_names)} and {len(self.attribute_values)} respectively",
                )
            )


@irdl_op_definition
class RegionsOp(IRDLOperation):
    """Define the regions of an operation"""

    name = "irdl.regions"

    args = var_operand_def(RegionType())

    names = prop_def(ArrayAttr[StringAttr])

    def __init__(self, args: Sequence[SSAValue], names: ArrayAttr[StringAttr]):
        super().__init__(operands=[args], properties={"names": names})

    @classmethod
    def parse(cls, parser: Parser) -> RegionsOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, lambda: _parse_argument(parser)
        )
        return RegionsOp(
            tuple(x[1] for x in args),
            ArrayAttr(x[0] for x in args),
        )

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_list(
                zip(self.names, self.args), lambda x: _print_argument(printer, x)
            )


################################################################################
# Attribute constraints                                                        #
################################################################################


@irdl_op_definition
class IsOp(IRDLOperation):
    """Constraint an attribute/type to be a specific attribute instance."""

    name = "irdl.is"

    expected = attr_def()
    output = result_def(AttributeType)

    def __init__(self, expected: Attribute):
        super().__init__(
            attributes={"expected": expected}, result_types=[AttributeType()]
        )

    @classmethod
    def parse(cls, parser: Parser) -> IsOp:
        expected = parser.parse_attribute()
        return IsOp(expected)

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_attribute(self.expected)


@irdl_op_definition
class BaseOp(IRDLOperation):
    """Constraint an attribute/type base"""

    name = "irdl.base"

    base_ref = opt_attr_def(SymbolRefAttr)
    base_name = opt_attr_def(StringAttr)
    output = result_def(AttributeType)

    def __init__(
        self,
        base: SymbolRefAttr | str | StringAttr,
        attr_dict: Mapping[str, Attribute] | None = None,
    ):
        attr_dict = attr_dict or {}
        if isinstance(base, str):
            base = StringAttr(base)
        if isinstance(base, StringAttr):
            super().__init__(
                attributes={"base_name": base, **attr_dict},
                result_types=[AttributeType()],
            )
        else:
            super().__init__(
                attributes={"base_ref": base, **attr_dict},
                result_types=[AttributeType()],
            )

    @classmethod
    def parse(cls, parser: Parser) -> BaseOp:
        attr = parser.parse_attribute()
        if not isinstance(attr, SymbolRefAttr | StringAttr):
            parser.raise_error("expected symbol reference or string")
        attr_dict = parser.parse_optional_attr_dict()
        return BaseOp(attr, attr_dict)

    def print(self, printer: Printer) -> None:
        if self.base_ref is not None:
            printer.print_string(" ")
            printer.print_attribute(self.base_ref)
        elif self.base_name is not None:
            printer.print_string(" ")
            printer.print_attribute(self.base_name)
        printer.print_op_attributes(self.attributes)

    def verify_(self) -> None:
        if not ((self.base_ref is None) ^ (self.base_name is None)):
            raise VerifyException("expected base as a reference or as a name")


@irdl_op_definition
class ParametricOp(IRDLOperation):
    """Constraint an attribute/type base and its parameters"""

    name = "irdl.parametric"

    base_type = attr_def(SymbolRefAttr)
    args = var_operand_def(AttributeType)
    output = result_def(AttributeType)

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
        printer.print_string(" ")
        printer.print_attribute(self.base_type)
        with printer.in_angle_brackets():
            printer.print_list(self.args, printer.print_ssa_value)


@irdl_op_definition
class RegionOp(IRDLOperation):
    """Define a region of an operation"""

    name = "irdl.region"

    entry_block_args = var_operand_def(AttributeType())

    constrained_arguments = opt_attr_def(UnitAttr)

    number_of_blocks = opt_attr_def(IntegerAttr[I32])

    output = result_def(RegionType())

    assembly_format = (
        "(```(` $entry_block_args $constrained_arguments^ `)`)?"
        "(` ` `with` `size` $number_of_blocks^)? attr-dict"
    )

    def __init__(
        self,
        number_of_blocks: IntegerAttr[I32],
        entry_block_args: Sequence[SSAValue],
        constrained_arguments: UnitAttr | NoneType = None,
    ):
        attributes: dict[str, Attribute] = {
            "number_of_blocks": number_of_blocks,
        }
        if isinstance(constrained_arguments, UnitAttr):
            attributes["constrained_arguments"] = constrained_arguments
        super().__init__(operands=entry_block_args, attributes=attributes)

    def verify_(self) -> None:
        if len(self.entry_block_args) > 0 and not self.constrained_arguments:
            raise VerifyException(
                "constrained_arguments must be set when specifying arguments"
            )


@irdl_op_definition
class AnyOp(IRDLOperation):
    """Constraint an attribute/type to be any attribute/type instance."""

    name = "irdl.any"

    output = result_def(AttributeType)

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

    args = var_operand_def(AttributeType)
    output = result_def(AttributeType)

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args], result_types=[AttributeType()])

    @classmethod
    def parse(cls, parser: Parser) -> AnyOfOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return AnyOfOp(args)

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_list(self.args, printer.print_ssa_value)


@irdl_op_definition
class AllOfOp(IRDLOperation):
    """Constraint an attribute/type to the intersection of the provided constraints."""

    name = "irdl.all_of"

    args = var_operand_def(AttributeType)
    output = result_def(AttributeType)

    def __init__(self, args: Sequence[SSAValue]):
        super().__init__(operands=[args], result_types=[AttributeType()])

    @classmethod
    def parse(cls, parser: Parser) -> AllOfOp:
        args = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_operand
        )
        return AllOfOp(args)

    def print(self, printer: Printer) -> None:
        with printer.in_parens():
            printer.print_list(self.args, printer.print_ssa_value)


IRDL = Dialect(
    "irdl",
    [
        DialectOp,
        TypeOp,
        CPredOp,
        AttributeOp,
        BaseOp,
        ParametersOp,
        OperationOp,
        OperandsOp,
        ResultsOp,
        AttributesOp,
        RegionsOp,
        IsOp,
        ParametricOp,
        RegionOp,
        AnyOp,
        AnyOfOp,
        AllOfOp,
    ],
    [
        AttributeType,
        RegionType,
        VariadicityAttr,
        VariadicityArrayAttr,
    ],
)
