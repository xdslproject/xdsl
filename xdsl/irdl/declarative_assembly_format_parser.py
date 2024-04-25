"""
This file contains a lexer and a parser for the MLIR declarative assembly format:
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import pairwise
from typing import cast

from xdsl.dialects.builtin import Builtin
from xdsl.ir import Attribute, TypedAttribute
from xdsl.irdl import (
    AttrOrPropDef,
    AttrSizedOperandSegments,
    OpDef,
    OptionalDef,
    OptOperandDef,
    OptResultDef,
    ParamAttrConstraint,
    ParsePropInAttrDict,
    VariadicDef,
    VarOperandDef,
    VarResultDef,
)
from xdsl.irdl.declarative_assembly_format import (
    AnchorableDirective,
    AttrDictDirective,
    AttributeVariable,
    FormatDirective,
    FormatProgram,
    KeywordDirective,
    OperandOrResult,
    OperandTypeDirective,
    OperandVariable,
    OptionalAttributeVariable,
    OptionalGroupDirective,
    OptionallyParsableDirective,
    OptionalOperandTypeDirective,
    OptionalOperandVariable,
    OptionalResultTypeDirective,
    OptionalResultVariable,
    PunctuationDirective,
    ResultTypeDirective,
    ResultVariable,
    VariableDirective,
    VariadicLikeFormatDirective,
    VariadicLikeTypeDirective,
    VariadicLikeVariable,
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
        if current_char in ("`", "$", "\\", "^"):
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
    seen_attributes: set[str]
    """The attributes that are already parsed."""
    seen_properties: set[str]
    """The properties that are already parsed."""
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
        self.seen_attributes = set[str]()
        self.seen_properties = set[str]()
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

        self.add_reserved_attrs_to_directive(elements)
        seen_variables = self.resolve_types()
        self.verify_directives(elements)
        self.verify_attr_dict()
        self.verify_properties()
        self.verify_operands(seen_variables)
        self.verify_results(seen_variables)
        return FormatProgram(elements)

    def verify_directives(self, elements: list[FormatDirective]):
        """
        Check correctness of the declarative format; e.g, chaining variadiclike operand
        directives leads to ambiguous parsing, and should raise an error here.
        """
        for a, b in pairwise(elements):
            match a, b:
                case VariadicLikeFormatDirective(), PunctuationDirective(","):
                    self.raise_error(
                        "A variadic directive cannot be followed by a comma literal."
                    )
                case VariadicLikeTypeDirective(), VariadicLikeTypeDirective():
                    self.raise_error(
                        "A variadic type directive cannot be followed by another variadic type directive."
                    )
                case VariadicLikeVariable(), VariadicLikeVariable() if not (
                    isinstance(a, VariadicLikeTypeDirective)
                    or isinstance(b, VariadicLikeTypeDirective)
                ):
                    self.raise_error(
                        "A variadic operand variable cannot be followed by another variadic operand variable."
                    )
                case _:
                    pass

    def add_reserved_attrs_to_directive(self, elements: list[FormatDirective]):
        """
        Add reserved attributes to the attr-dict directive.
        These are the attributes that are printed/parsed in other places in the format,
        and thus should not be printed in the attr-dict directive.
        """
        for idx, element in enumerate(elements):
            if isinstance(element, AttrDictDirective):
                elements[idx] = AttrDictDirective(
                    with_keyword=element.with_keyword,
                    reserved_attr_names=self.seen_attributes,
                    print_properties=element.print_properties,
                )
                return

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

    def verify_properties(self):
        """
        Check that all properties are present, unless `ParsePropInAttrDict` option is
        used.
        """
        # This is used for compatibility with MLIR
        if any(
            isinstance(option, ParsePropInAttrDict) for option in self.op_def.options
        ):
            if self.seen_properties:
                self.raise_error(
                    "properties cannot be specified in the declarative format "
                    "when 'ParsePropInAttrDict' IRDL option is used. They are instead "
                    "parsed from the attribute dictionary."
                )
            return
        missing_properties = set(self.op_def.properties.keys()) - self.seen_properties
        if missing_properties:
            self.raise_error(
                f"{', '.join(missing_properties)} properties are missing from "
                "the declarative format. If this is intentional, consider using "
                "'ParsePropInAttrDict' IRDL option."
            )

    def parse_optional_variable(
        self,
    ) -> VariableDirective | AttributeVariable | None:
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
                if isinstance(operand_def, VariadicDef | OptionalDef):
                    self.seen_attributes.add(AttrSizedOperandSegments.attribute_name)
            match operand_def:
                case OptOperandDef():
                    return OptionalOperandVariable(variable_name, idx)
                case VarOperandDef():
                    return VariadicOperandVariable(variable_name, idx)
                case _:
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
            match result_def:
                case OptResultDef():
                    return OptionalResultVariable(variable_name, idx)
                case VarResultDef():
                    return VariadicResultVariable(variable_name, idx)
                case _:
                    return ResultVariable(variable_name, idx)
            if isinstance(result_def, VariadicDef):
                return VariadicResultVariable(variable_name, idx)
            else:
                return ResultVariable(variable_name, idx)

        attr_or_prop_by_name = {
            attr_name: attr_or_prop
            for attr_name, attr_or_prop in self.op_def.accessor_names.values()
        }

        # Check if the variable is an attribute
        if variable_name in attr_or_prop_by_name:
            attr_name = variable_name
            attr_or_prop = attr_or_prop_by_name[attr_name]
            if self.context == ParsingContext.TopLevel:
                if attr_or_prop == "property":
                    if attr_name in self.seen_properties:
                        self.raise_error(f"property '{variable_name}' is already bound")
                    self.seen_properties.add(attr_name)
                else:
                    if attr_name in self.seen_attributes:
                        self.raise_error(
                            f"attribute '{variable_name}' is already bound"
                        )
                    self.seen_attributes.add(attr_name)

            attr_def = (
                self.op_def.properties.get(attr_name)
                if attr_or_prop == "property"
                else self.op_def.attributes.get(attr_name)
            )
            if isinstance(attr_def, AttrOrPropDef):
                unique_base = attr_def.constr.get_unique_base()
                # Always qualify builtin attributes
                # This is technically an approximation, but appears to be good enough
                # for xDSL right now.
                unique_type = None
                if unique_base is not None and issubclass(unique_base, TypedAttribute):
                    constr = attr_def.constr
                    # TODO: generalize.
                    # https://github.com/xdslproject/xdsl/issues/2499
                    if isinstance(constr, ParamAttrConstraint):
                        type_constraint = constr.param_constrs[
                            unique_base.get_type_index()
                        ]
                        if type_constraint.can_infer(set()):
                            unique_type = type_constraint.infer({})
                if (
                    unique_base is not None
                    and unique_base in Builtin.attributes
                    and unique_type is None
                ):
                    unique_base = None

                # Chill pyright with TypedAttribute without parameter
                unique_base = cast(type[Attribute] | None, unique_base)

                variable_type = (
                    OptionalAttributeVariable
                    if isinstance(attr_def, OptionalDef)
                    else AttributeVariable
                )
                is_property = attr_or_prop == "property"
                return variable_type(
                    variable_name, is_property, unique_base, unique_type
                )

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
            case OptionalOperandVariable(name, index):
                if self.seen_operand_types[index]:
                    self.raise_error(f"types of '{name}' is already bound")
                self.seen_operand_types[index] = True
                res = OptionalOperandTypeDirective(name, index)
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
            case OptionalResultVariable(name, index):
                if self.seen_result_types[index]:
                    self.raise_error(f"types of '{name}' is already bound")
                self.seen_result_types[index] = True
                res = OptionalResultTypeDirective(name, index)
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
            case AttributeVariable():
                self.raise_error("can only take the type of an operand or result")
            case _:
                raise ValueError(f"Unexpected variable type {type(variable)}")

        self.parse_punctuation(")")
        self.context = previous_context
        return res

    def parse_optional_group(self) -> FormatDirective:
        """
        Parse an optional group, with the following format:
          group ::= `(` then-elements `)` `?`
        """
        then_elements = tuple[FormatDirective, ...]()
        anchor: FormatDirective | None = None

        while not self.parse_optional_punctuation(")"):
            then_elements += (self.parse_directive(),)
            if self.parse_optional_keyword("^"):
                if anchor is not None:
                    self.raise_error("An optional group can only have one anchor.")
                anchor = then_elements[-1]
        self.parse_punctuation("?")

        if not then_elements:
            self.raise_error("An optional group cannot be empty")
        if anchor is None:
            self.raise_error("Every optional group must have an anchor.")
        # TODO: allow attribute and region variables when implemented.
        if not isinstance(then_elements[0], OptionallyParsableDirective):
            self.raise_error(
                "First element of an optional group must be optionally parsable."
            )
        if not isinstance(anchor, AnchorableDirective):
            self.raise_error(
                "An optional group's anchor must be an achorable directive."
            )

        return OptionalGroupDirective(anchor, then_elements[0], then_elements[1:])

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
        if self.parse_optional_punctuation("("):
            return self.parse_optional_group()
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
        print_properties = any(
            isinstance(option, ParsePropInAttrDict) for option in self.op_def.options
        )
        # reserved_attr_names is populated once the format is parsed, as some attributes
        # might appear after the attr-dict directive
        return AttrDictDirective(
            with_keyword=with_keyword,
            reserved_attr_names=set(),
            print_properties=print_properties,
        )
