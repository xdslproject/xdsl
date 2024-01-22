"""
This file contains the data structures necessary for the parsing and printing
of the MLIR declarative assembly format defined at
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format .
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from xdsl.ir import Attribute
from xdsl.irdl import (
    IRDLOperation,
    IRDLOperationInvT,
    OpDef,
    VariadicDef,
    VarIRConstruct,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
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

    operands: list[UnresolvedOperand | None]
    operand_types: list[Attribute | None]
    result_types: list[Attribute | None]
    attributes: dict[str, Attribute]

    def __init__(self, op_def: OpDef):
        if op_def.attributes or op_def.regions or op_def.successors:
            raise NotImplementedError(
                "Operation definitions with attributes, regions, "
                "or successors are not yet supported"
            )
        for _, operand in (*op_def.operands, *op_def.results):
            if isinstance(operand, VariadicDef):
                raise NotImplementedError(
                    "Operation definition with variadic operand or "
                    "result definitions are not supported."
                )
        self.operands = [None] * len(op_def.operands)
        self.operand_types = [None] * len(op_def.operands)
        self.result_types = [None] * len(op_def.results)
        self.attributes = {}


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

    type_resolutions: dict[
        tuple[OperandOrResult, int],
        tuple[Callable[[Attribute], Attribute], OperandOrResult, int],
    ]
    """A mapping describing how to resolve unparsed operand and result types."""

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
        state = ParsingState(op_type.get_irdl_definition())
        for stmt in self.stmts:
            stmt.parse(parser, state)

        # Ensure that all operands and operand types are parsed
        unresolved_operands = state.operands
        assert isa(unresolved_operands, list[UnresolvedOperand])
        self.resolve_operand_types(state)
        operand_types = state.operand_types
        assert isa(operand_types, list[Attribute])

        # Ensure that all result types are parsed or resolved
        self.resolve_result_types(state)
        result_types = state.result_types
        assert isa(state.result_types, list[Attribute])

        # Resolve all operands
        operands = parser.resolve_operands(
            unresolved_operands, operand_types, parser.pos
        )
        return op_type.build(
            result_types=result_types, operands=operands, attributes=state.attributes
        )

    def resolve_operand_types(self, state: ParsingState) -> None:
        """
        Use the inferred type resolutions to fill missing operand types from other parsed
        types.
        """
        for i, operand_type in enumerate(state.operand_types):
            if operand_type is None:
                state.operand_types[i] = self._resolve_type(
                    state, VarIRConstruct.OPERAND, i
                )

    def resolve_result_types(self, state: ParsingState) -> None:
        """
        Use the inferred type resolutions to fill missing result types from other parsed
        types.
        """
        for i, result_type in enumerate(state.result_types):
            if result_type is None:
                state.result_types[i] = self._resolve_type(
                    state, VarIRConstruct.RESULT, i
                )

    def _resolve_type(
        self, state: ParsingState, construct: OperandOrResult, index: int
    ):
        """
        Helper function resolving a specific operand or result type from the inferred
        resolution map.
        """
        resolve, construct, idx = self.type_resolutions[construct, index]
        match construct:
            case VarIRConstruct.OPERAND:
                input_type = state.operand_types[idx]
            case VarIRConstruct.RESULT:
                input_type = state.result_types[idx]
        assert input_type is not None
        return resolve(input_type)

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

    def parse(self, parser: Parser, state: ParsingState) -> None:
        if self.with_keyword:
            res = parser.parse_optional_attr_dict_with_keyword()
            if res is None:
                res = {}
            else:
                res = res.data
        else:
            res = parser.parse_optional_attr_dict()
        state.attributes = res

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if not op.attributes:
            return
        if self.with_keyword:
            printer.print(" attributes")
        printer.print_op_attributes(op.attributes)
        state.last_was_punctuation = False
        state.should_emit_space = False


@dataclass(frozen=True)
class OperandVariable(FormatDirective):
    """
    An operand variable, with the following format:
      operand-directive ::= percent-ident
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
class OperandTypeDirective(FormatDirective):
    """
    An operand variable type directive, with the following format:
      operand-type-directive ::= type(percent-ident)
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
class ResultVariable(FormatDirective):
    """
    An result variable, with the following format:
      result-directive ::= percent-ident
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
class ResultTypeDirective(FormatDirective):
    """
    A result variable type directive, with the following format:
      result-type-directive ::= type(percent-ident)
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
