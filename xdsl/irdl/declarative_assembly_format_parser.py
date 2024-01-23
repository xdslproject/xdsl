"""
This file contains a lexer and a parser for the MLIR declarative assembly format:
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from xdsl.ir import Attribute
from xdsl.irdl import OpDef, VariadicDef
from xdsl.irdl.declarative_assembly_format import (
    AttrDictDirective,
    FormatDirective,
    FormatProgram,
    KeywordDirective,
    OperandOrResult,
    OperandTypeDirective,
    OperandVariable,
    PunctuationDirective,
    ResultTypeDirective,
    ResultVariable,
    VariadicOperandTypeDirective,
    VariadicOperandVariable,
    VariadicResultTypeDirective,
    VariadicResultVariable,
    WhitespaceDirective,
)
from xdsl.parser import BaseParser, ParserState
from xdsl.utils.lexer import Input, Lexer, Token


@dataclass
class FormatLexer(Lexer):
    """
    A lexer for the declarative assembly format.
    The differences with the MLIR lexer are the following:
    * It can parse '`' or '$' as tokens. The token will have the `BARE_IDENT` kind.
    * Bare identifiers may also may contain `-`.
    """

    def lex(self) -> Token:
        """Lex a token from the input, and returns it."""
        # First, skip whitespaces
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._peek_chars()

        # Handle end of file
        if current_char is None:
            return self._form_token(Token.Kind.EOF, start_pos)

        # We parse '`', `\\` and '$' as a BARE_IDENT.
        # This is a hack to reuse the MLIR lexer.
        if current_char in ("`", "$", "\\"):
            self._consume_chars()
            return self._form_token(Token.Kind.BARE_IDENT, start_pos)
        return super().lex()

    # Authorize `-` in bare identifier
    bare_identifier_suffix_regex = re.compile(r"[a-zA-Z0-9_$.\-]*")


class ParsingContext(Enum):
    """Indicates if the parser is nested in a particular directive."""

    TopLevel = auto()
    TypeDirective = auto()


@dataclass(init=False)
class FormatParser(BaseParser):
    """
    Parser for the declarative assembly format.
    The parser keeps track of the operands, operand types, results, regions, and
    attributes that are already parsed, and checks at the end that the provided format
    is correct, i.e., that it is unambiguous and refer to all elements exactly once.
    """

    op_def: OpDef
    """The operation definition we are parsing the format for."""
    seen_operands: list[bool]
    """The operand variables that are already parsed."""
    seen_operand_types: list[bool]
    """The operand types that are already parsed."""
    seen_result_types: list[bool]
    """The result types that are already parsed."""
    has_attr_dict: bool = field(default=False)
    """True if the attribute dictionary has already been parsed."""
    context: ParsingContext = field(default=ParsingContext.TopLevel)
    """Indicates if the parser is nested in a particular directive."""
    type_resolutions: dict[
        tuple[OperandOrResult, int],
        tuple[Callable[[Attribute], Attribute], OperandOrResult, int],
    ]
    """Map a variable to a way to infer its type"""

    def __init__(self, input: str, op_def: OpDef):
        super().__init__(ParserState(FormatLexer(Input(input, "<input>"))))
        self.op_def = op_def
        self.seen_operands = [False] * len(op_def.operands)
        self.seen_operand_types = [False] * len(op_def.operands)
        self.seen_result_types = [False] * len(op_def.results)
        self.type_resolutions = {}

    def parse_format(self) -> FormatProgram:
        """
        Parse a declarative format, with the following syntax:
          format ::= directive*
        Once the format is parsed, check that it is correct, i.e., that it is
        unambiguous and refer to all elements exactly once.
        """
        elements: list[FormatDirective] = []
        while self._current_token.kind != Token.Kind.EOF:
            elements.append(self.parse_directive())

        seen_variables = self.resolve_types()
        self.verify_attr_dict()
        self.verify_operands(seen_variables)
        self.verify_results(seen_variables)
        return FormatProgram(elements)

    def resolve_types(self) -> set[str]:
        """
        Find out which constraint variables can be inferred from the parsed attributes.
        """
        seen_variables = set[str]()
        for i, (_, operand_def) in enumerate(self.op_def.operands):
            if self.seen_operand_types[i]:
                seen_variables |= operand_def.constr.get_resolved_variables()
        for i, (_, result_def) in enumerate(self.op_def.results):
            if self.seen_result_types[i]:
                seen_variables |= result_def.constr.get_resolved_variables()
        return seen_variables

    def verify_operands(self, seen_variables: set[str]):
        """
        Check that all operands and operand types are refered at least once, or inferred
        from another construct.
        """
        for (
            seen_operand,
            seen_operand_type,
            (operand_name, operand_def),
        ) in zip(
            self.seen_operands,
            self.seen_operand_types,
            self.op_def.operands,
            strict=True,
        ):
            if not seen_operand:
                self.raise_error(
                    f"operand '{operand_name}' "
                    f"not found, consider adding a '${operand_name}' "
                    "directive to the custom assembly format"
                )
            if not seen_operand_type:
                if not operand_def.constr.can_infer(seen_variables):
                    self.raise_error(
                        f"type of operand '{operand_name}' cannot be inferred, "
                        f"consider adding a 'type(${operand_name})' directive to the "
                        "custom assembly format"
                    )

    def verify_results(self, seen_variables: set[str]):
        """Check that all result types are refered at least once, or inferred
        from another construct."""

        for result_type, (result_name, result_def) in zip(
            self.seen_result_types, self.op_def.results, strict=True
        ):
            if not result_type:
                if not result_def.constr.can_infer(seen_variables):
                    self.raise_error(
                        f"type of result '{result_name}' cannot be inferred, "
                        f"consider adding a 'type(${result_name})' directive to the "
                        "custom assembly format"
                    )

    def verify_attr_dict(self):
        """
        Check that the attribute dictionary is present.
        """
        if not self.has_attr_dict:
            self.raise_error("'attr-dict' directive not found")

    def parse_optional_variable(self) -> OperandVariable | ResultVariable | None:
        """
        Parse a variable, if present, with the following format:
          variable ::= `$` bare-ident
        The variable should refer to an operand, attribute, region, result,
        or successor.
        """
        if self._current_token.text[0] != "$":
            return None
        self._consume_token()
        variable_name = self.parse_identifier(" after '$'")

        # Check if the variable is an operand
        for idx, (operand_name, operand_def) in enumerate(self.op_def.operands):
            if variable_name != operand_name:
                continue
            if self.context == ParsingContext.TopLevel:
                if self.seen_operands[idx]:
                    self.raise_error(f"operand '{variable_name}' is already bound")
                self.seen_operands[idx] = True
            if isinstance(operand_def, VariadicDef):
                return VariadicOperandVariable(variable_name, idx)
            else:
                return OperandVariable(variable_name, idx)

        # Check if the variable is a result
        for idx, (result_name, result_def) in enumerate(self.op_def.results):
            if variable_name != result_name:
                continue
            if self.context == ParsingContext.TopLevel:
                self.raise_error(
                    "result variable cannot be in a toplevel directive. "
                    f"Consider using 'type({variable_name})' instead."
                )
            if isinstance(result_def, VariadicDef):
                return VariadicResultVariable(variable_name, idx)
            else:
                return ResultVariable(variable_name, idx)

        self.raise_error(
            "expected variable to refer to an operand, "
            "attribute, region, result, or successor"
        )

    def parse_type_directive(self) -> FormatDirective:
        """
        Parse a type directive with the following format:
          type-directive ::= `type` `(` variable `)`
        `type` is expected to have already been parsed
        """
        self.parse_punctuation("(")

        # Update the current context, since we are now in a type directive
        previous_context = self.context
        self.context = ParsingContext.TypeDirective

        variable = self.parse_optional_variable()
        match variable:
            case None:
                self.raise_error("'type' directive expects a variable argument")
            case VariadicOperandVariable(name, index):
                if self.seen_operand_types[index]:
                    self.raise_error(f"types of '{name}' is already bound")
                self.seen_operand_types[index] = True
                res = VariadicOperandTypeDirective(name, index)
            case OperandVariable(name, index):
                if self.seen_operand_types[index]:
                    self.raise_error(f"type of '{name}' is already bound")
                self.seen_operand_types[index] = True
                res = OperandTypeDirective(name, index)
            case VariadicResultVariable(name, index):
                if self.seen_result_types[index]:
                    self.raise_error(f"types of '{name}' is already bound")
                self.seen_result_types[index] = True
                res = VariadicResultTypeDirective(name, index)
            case ResultVariable(name, index):
                if self.seen_result_types[index]:
                    self.raise_error(f"type of '{name}' is already bound")
                self.seen_result_types[index] = True
                res = ResultTypeDirective(name, index)

        self.parse_punctuation(")")
        self.context = previous_context
        return res

    def parse_keyword_or_punctuation(self) -> FormatDirective:
        """
        Parse a keyword or a punctuation directive, with the following format:
          keyword-or-punctuation-directive ::= `\\`` (bare-ident | punctuation) `\\``
        """
        self.parse_characters("`")
        start_token = self._current_token

        # New line case
        if self.parse_optional_keyword("\\"):
            self.parse_keyword("n")
            self.parse_characters("`")
            return WhitespaceDirective("\n")

        # Space case
        if self.parse_optional_characters("`"):
            end_token = self._current_token
            whitespace = self.lexer.input.content[
                start_token.span.end : end_token.span.start
            ]
            if whitespace != " ":
                self.raise_error(
                    "unexpected whitespace in directive, only ` ` whitespace is allowed"
                )
            return WhitespaceDirective(" ")

        # Punctuation case
        if self._current_token.kind.is_punctuation():
            punctuation = self._consume_token().text
            self.parse_characters("`")
            assert Token.Kind.is_spelling_of_punctuation(punctuation)
            return PunctuationDirective(punctuation)

        # Identifier case
        ident = self.parse_optional_identifier()
        if ident is None or ident == "`":
            self.raise_error("punctuation or identifier expected")

        self.parse_characters("`")
        return KeywordDirective(ident)

    def parse_directive(self) -> FormatDirective:
        """
        Parse a format directive, with the following format:
          directive ::= `attr-dict`
                        | `attr-dict-with-keyword`
                        | type-directive
                        | keyword-or-punctuation-directive
                        | variable
        """
        if self.parse_optional_keyword("attr-dict"):
            return self.create_attr_dict_directive(False)
        if self.parse_optional_keyword("attr-dict-with-keyword"):
            return self.create_attr_dict_directive(True)
        if self.parse_optional_keyword("type"):
            return self.parse_type_directive()
        if self._current_token.text == "`":
            return self.parse_keyword_or_punctuation()
        if variable := self.parse_optional_variable():
            return variable
        self.raise_error(f"unexpected token '{self._current_token.text}'")

    def create_attr_dict_directive(self, with_keyword: bool) -> AttrDictDirective:
        """Create an attribute dictionary directive, and update the parsing state."""
        if self.has_attr_dict:
            self.raise_error(
                "'attr-dict' directive can only occur once "
                "in the assembly format description"
            )
        self.has_attr_dict = True
        return AttrDictDirective(with_keyword=with_keyword)
