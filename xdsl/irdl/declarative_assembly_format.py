"""
This file contains the data structures necessary for the parsing and printing
of the MLIR declarative assembly format defined at
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format .
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    IRDLOperationInvT,
    OpDef,
    VarIRConstruct,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.lexer import Token

OperandOrResult = Literal[VarIRConstruct.OPERAND, VarIRConstruct.RESULT]


@dataclass
class ParsingState:
    """
    State carried during the parsing of an operation using the declarative assembly
    format.
    It contains the elements that have already been parsed.
    """

    operands: list[UnresolvedOperand | None | list[UnresolvedOperand | None]]
    operand_types: list[Attribute | None | list[Attribute | None]]
    result_types: list[Attribute | None | list[Attribute | None]]
    attributes: dict[str, Attribute]
    properties: dict[str, Attribute]
    constraint_variables: dict[str, Attribute]

    def __init__(self, op_def: OpDef):
        if op_def.regions or op_def.successors:
            raise NotImplementedError(
                "Operation definitions with regions "
                "or successors are not yet supported"
            )
        self.operands = [None] * len(op_def.operands)
        self.operand_types = [None] * len(op_def.operands)
        self.result_types = [None] * len(op_def.results)
        self.attributes = {}
        self.properties = {}
        self.constraint_variables = {}


@dataclass
class PrintingState:
    """
    State carried during the printing of an operation using the declarative assembly
    format.
    It contains information on the last token, to know if a space should be emitted.
    """

    last_was_punctuation: bool = field(default=False)
    """Was the last element parsed a punctuation."""
    should_emit_space: bool = field(default=True)
    """
    Should the printer emit a space before the next element.
    Depending on the directive, the space might not be printed
    (for instance for some punctuations).
    """


@dataclass(frozen=True)
class FormatProgram:
    """
    The toplevel data structure of a declarative assembly format program.
    It is used to parse and print an operation.
    """

    stmts: list[FormatDirective]
    """The list of statements composing the program. They are executed in order."""

    @staticmethod
    def from_str(input: str, op_def: OpDef) -> FormatProgram:
        """
        Create the assembly format data program from its string representation.
        This might raise a ParseError exception if the string is invalid.
        """
        from xdsl.irdl.declarative_assembly_format_parser import FormatParser

        return FormatParser(input, op_def).parse_format()

    def parse(
        self, parser: Parser, op_type: type[IRDLOperationInvT]
    ) -> IRDLOperationInvT:
        """
        Parse the operation with this format.
        The given operation type is expected to be the operation type represented by
        the operation definition passed to the FormatParser that created this
        FormatProgram.
        """
        # Parse elements one by one
        op_def = op_type.get_irdl_definition()
        state = ParsingState(op_def)
        for stmt in self.stmts:
            stmt.parse(parser, state)

        # Get constraint variables from the parsed operand and result types
        self.assign_constraint_variables(parser, state, op_def)

        # Infer operand types that should be inferred
        unresolved_operands = state.operands
        assert isa(
            unresolved_operands, list[UnresolvedOperand | list[UnresolvedOperand]]
        )
        self.resolve_operand_types(state, op_def)
        operand_types = state.operand_types
        assert isa(operand_types, list[Attribute | list[Attribute]])

        # Infer result types that should be inferred
        self.resolve_result_types(state, op_def)
        result_types = state.result_types
        assert isa(result_types, list[Attribute | list[Attribute]])

        # Resolve all operands
        operands: Sequence[SSAValue | Sequence[SSAValue]] = []
        for uo, ot in zip(unresolved_operands, operand_types, strict=True):
            if isinstance(uo, list):
                assert isinstance(
                    ot, list
                ), "Something went wrong with the declarative assembly format parser."
                "Variadic or optional operand has no type or a single type "
                operands.append(parser.resolve_operands(uo, ot, parser.pos))
            else:
                assert isinstance(
                    ot, Attribute
                ), "Something went wrong with the declarative assembly format parser."
                "Single operand has no type or variadic/optional type"
                operands.append(parser.resolve_operand(uo, ot))

        # Get the properties from the attribute dictionary if no properties are
        # defined. This is necessary to be compatible with MLIR format, such as
        # `memref.load`.
        if state.properties:
            properties = state.properties
        else:
            properties = op_def.split_properties(state.attributes)
        return op_type.build(
            result_types=result_types,
            operands=operands,
            attributes=state.attributes,
            properties=properties,
        )

    def assign_constraint_variables(
        self, parser: Parser, state: ParsingState, op_def: OpDef
    ):
        """
        Assign constraint variables with values got from the
        parsed operand and result types.
        """
        if any(type is None for type in (*state.operand_types, *state.result_types)):
            try:
                for (_, operand_def), operand_type in zip(
                    op_def.operands, state.operand_types, strict=True
                ):
                    if operand_type is None:
                        continue
                    if isinstance(operand_type, Attribute):
                        operand_type = [operand_type]
                    for ot in operand_type:
                        if ot is None:
                            continue
                        operand_def.constr.verify(ot, state.constraint_variables)
                for (_, result_def), result_type in zip(
                    op_def.results, state.result_types, strict=True
                ):
                    if result_type is None:
                        continue
                    if isinstance(result_type, Attribute):
                        result_type = [result_type]
                    for rt in result_type:
                        if rt is None:
                            continue
                        result_def.constr.verify(rt, state.constraint_variables)
            except VerifyException as e:
                parser.raise_error(
                    "Verification error while inferring operation type: " + str(e)
                )

    def resolve_operand_types(self, state: ParsingState, op_def: OpDef) -> None:
        """
        Use the inferred type resolutions to fill missing operand types from other parsed
        types.
        """
        for i, (operand_type, (_, operand_def)) in enumerate(
            zip(state.operand_types, op_def.operands, strict=True)
        ):
            if operand_type is None:
                operand_type = operand_def.constr.infer(state.constraint_variables)
                operand = state.operands[i]
                if isinstance(operand, UnresolvedOperand):
                    state.operand_types[i] = operand_type
                elif isinstance(operand, list):
                    state.operand_types[i] = cast(
                        list[Attribute | None], [operand_type]
                    ) * len(operand)

    def resolve_result_types(self, state: ParsingState, op_def: OpDef) -> None:
        """
        Use the inferred type resolutions to fill missing result types from other parsed
        types.
        """
        for i, (result_type, (_, result_def)) in enumerate(
            zip(state.result_types, op_def.results, strict=True)
        ):
            if result_type is None:
                result_type = result_def.constr.infer(state.constraint_variables)
                state.result_types[i] = result_def.constr.infer(
                    state.constraint_variables
                )
                result_type = state.result_types[i]
                if isinstance(result_type, Attribute):
                    state.result_types[i] = result_type
                elif isinstance(result_type, list):
                    state.result_types[i] = cast(
                        list[Attribute | None], [result_type]
                    ) * len(result_type)

    def print(self, printer: Printer, op: IRDLOperation) -> None:
        """
        Print the operation with this format.
        The given operation is expected to be defined using the operation definition
        passed to the FormatParser that created this FormatProgram.
        """
        state = PrintingState()
        for stmt in self.stmts:
            stmt.print(printer, state, op)


@dataclass(frozen=True)
class FormatDirective(ABC):
    """A format directive for operation format."""

    @abstractmethod
    def parse(self, parser: Parser, state: ParsingState) -> None:
        ...

    @abstractmethod
    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        ...


@dataclass(frozen=True)
class AttrDictDirective(FormatDirective):
    """
    An attribute dictionary directive, with the following format:
       attr-dict-directive ::= attr-dict
       attr-dict-with-format-directive ::= `attributes` attr-dict
    The directive (with and without the keyword) will always print a space before, and
    will not request a space to be printed after.
    """

    with_keyword: bool
    """If this is set, the format starts with the `attributes` keyword."""

    reserved_attr_names: set[str]
    """
    The set of attributes that should not be printed.
    These attributes are printed in other places in the format, and thus would be
    printed twice otherwise.
    """

    print_properties: bool
    """
    If this is set, also print properties as part of the attribute dictionary.
    This is used to keep compatibility with MLIR which allows that.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        if self.with_keyword:
            res = parser.parse_optional_attr_dict_with_keyword()
            if res is None:
                res = {}
            else:
                res = res.data
        else:
            res = parser.parse_optional_attr_dict()
        defined_reserved_keys = self.reserved_attr_names & res.keys()
        if defined_reserved_keys:
            parser.raise_error(
                f"attributes {', '.join(defined_reserved_keys)} are defined in other parts of the "
                "assembly format, and thus should not be defined in the attribute "
                "dictionary."
            )
        state.attributes |= res

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if self.print_properties:
            if (
                not (set(op.attributes.keys()) | set(op.properties.keys()))
                - self.reserved_attr_names
            ):
                return
            if any(name in op.attributes for name in op.properties):
                raise ValueError(
                    "Cannot print attributes and properties with the same name "
                    "in a signle dictionary"
                )
            printer.print_op_attributes(
                op.attributes | op.properties,
                reserved_attr_names=self.reserved_attr_names,
                print_keyword=self.with_keyword,
            )
        else:
            if not set(op.attributes.keys()) - self.reserved_attr_names:
                return
            printer.print_op_attributes(
                op.attributes,
                reserved_attr_names=self.reserved_attr_names,
                print_keyword=self.with_keyword,
            )

        # This is changed only if something was printed
        state.last_was_punctuation = False
        state.should_emit_space = False


@dataclass(frozen=True)
class OperandVariable(FormatDirective):
    """
    An operand variable, with the following format:
      operand-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    name: str
    """The operand name. This is only used for error message reporting."""
    index: int
    """Index of the operand definition."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        operand = parser.parse_unresolved_operand()
        state.operands[self.index] = operand

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_ssa_value(op.operands[self.index])
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicOperandVariable(OperandVariable):
    """
    A variadic operand variable, with the following format:
      operand-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        operands = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )
        if operands is None:
            operands = []
        state.operands[self.index] = cast(list[UnresolvedOperand | None], operands)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(getattr(op, self.name), printer.print_ssa_value)
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class OperandTypeDirective(FormatDirective):
    """
    An operand variable type directive, with the following format:
      operand-type-directive ::= type(dollar-ident)
    The directive will request a space to be printed right after.
    """

    name: str
    """The operand name. This is only used for error message reporting."""
    index: int
    """Index of the operand definition."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        type = parser.parse_type()
        state.operand_types[self.index] = type

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_attribute(op.operands[self.index].type)
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicOperandTypeDirective(OperandTypeDirective):
    """
    A variadic operand variable, with the following format:
      operand-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        operand_types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        if operand_types is None:
            operand_types = []
        state.operand_types[self.index] = cast(list[Attribute | None], operand_types)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(
            (o.type for o in getattr(op, self.name)), printer.print_attribute
        )
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class ResultVariable(FormatDirective):
    """
    An result variable, with the following format:
      result-directive ::= dollar-ident
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    name: str
    """The result name. This is only used for error message reporting."""
    index: int
    """Index of the result definition."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )


@dataclass(frozen=True)
class VariadicResultVariable(ResultVariable):
    """
    A variadic result variable, with the following format:
      result-directive ::= percent-ident (( `,` percent-id )* )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )


@dataclass(frozen=True)
class ResultTypeDirective(FormatDirective):
    """
    A result variable type directive, with the following format:
      result-type-directive ::= type(dollar-ident)
    The directive will request a space to be printed right after.
    """

    name: str
    """The result name. This is only used for error message reporting."""
    index: int
    """Index of the result definition."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        type = parser.parse_type()
        state.result_types[self.index] = type

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_attribute(op.results[self.index].type)
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class AttributeVariable(FormatDirective):
    """
    An attribute variable, with the following format:
      result-directive ::= dollar-ident
    The directive will request a space to be printed right after.
    """

    attr_name: str
    """The attribute name as it should be in the attribute or property dictionary."""
    is_property: bool
    """Should this attribute be put in the attribute or property dictionary."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        attribute = parser.parse_attribute()
        if self.is_property:
            state.properties[self.attr_name] = attribute
        else:
            state.attributes[self.attr_name] = attribute

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        state.should_emit_space = True
        state.last_was_punctuation = False
        if self.is_property:
            printer.print_attribute(op.properties[self.attr_name])
        else:
            printer.print_attribute(op.attributes[self.attr_name])


@dataclass(frozen=True)
class VariadicResultTypeDirective(ResultTypeDirective):
    """
    A variadic result variable type directive, with the following format:
      variadic-result-type-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        result_types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        if result_types is None:
            result_types = []
        state.result_types[self.index] = cast(list[Attribute | None], result_types)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(
            (r.type for r in getattr(op, self.name)), printer.print_attribute
        )
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class WhitespaceDirective(FormatDirective):
    """
    A whitespace directive, with the following format:
      whitespace-directive ::= `\n` | ` `
    This directive is only applied during printing, and has no effect during
    parsing.
    The directive will not request any space to be printed after.
    """

    whitespace: Literal[" ", "\n"]
    """The whitespace that should be printed."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        pass

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        printer.print(self.whitespace)
        state.last_was_punctuation = False
        state.should_emit_space = False


@dataclass(frozen=True)
class PunctuationDirective(FormatDirective):
    """
    A punctuation directive, with the following format:
      punctuation-directive ::= punctuation
    The directive will request a space to be printed right after, unless the punctuation
    is `<`, `(`, `{`, or `[`.
    It will also print a space before if a space is requested, and that the punctuation
    is neither `>`, `)`, `}`, `]`, or `,` if the last element was a punctuation, and
    additionally neither `<`, `(`, `}`, `]`, if the last element was not a punctuation.
    """

    punctuation: Token.PunctuationSpelling
    """The punctuation that should be printed/parsed."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        parser.parse_punctuation(self.punctuation)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        emit_space = False
        if state.should_emit_space:
            if state.last_was_punctuation:
                if self.punctuation not in (">", ")", "}", "]", ","):
                    emit_space = True
            elif self.punctuation not in ("<", ">", "(", ")", "{", "}", "[", "]", ","):
                emit_space = True

            if emit_space:
                printer.print(" ")

        printer.print(self.punctuation)

        state.should_emit_space = self.punctuation not in ("<", "(", "{", "[")
        state.last_was_punctuation = True


@dataclass(frozen=True)
class KeywordDirective(FormatDirective):
    """
    A keyword directive, with the following format:
      keyword-directive ::= bare-ident
    The directive expects a specific identifier, and will request a space to be printed
    after.
    """

    keyword: str
    """The identifier that should be printed."""

    def parse(self, parser: Parser, state: ParsingState):
        parser.parse_keyword(self.keyword)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space:
            printer.print(" ")
        printer.print(self.keyword)
        state.should_emit_space = True
        state.last_was_punctuation = False
