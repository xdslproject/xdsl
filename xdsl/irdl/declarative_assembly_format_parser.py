"""
This file contains a lexer and a parser for the MLIR declarative assembly format:
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format
"""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

from xdsl.dialects.builtin import (
    AnyFloat,
    Builtin,
    DenseArrayBase,
    IntegerType,
    SymbolNameConstraint,
    UnitAttr,
)
from xdsl.ir import TypedAttribute
from xdsl.irdl import (
    AttrSizedOperandSegments,
    AttrSizedSegments,
    ConstraintContext,
    OpDef,
    OptionalDef,
    OptOperandDef,
    OptRegionDef,
    OptResultDef,
    OptSingleBlockRegionDef,
    OptSuccessorDef,
    ParamAttrConstraint,
    ParsePropInAttrDict,
    SameVariadicOperandSize,
    SameVariadicResultSize,
    VarConstraint,
    VariadicDef,
    VarOperandDef,
    VarRegionDef,
    VarResultDef,
    VarSingleBlockRegionDef,
    VarSuccessorDef,
)
from xdsl.irdl.declarative_assembly_format import (
    AnchorRegionVariable,
    AttrDictDirective,
    AttributeVariable,
    DenseArrayAttributeVariable,
    Directive,
    FormatDirective,
    FormatProgram,
    FunctionalTypeDirective,
    KeywordDirective,
    OperandDirective,
    OperandsDirective,
    OperandVariable,
    OptionalGroupDirective,
    OptionalOperandVariable,
    OptionalRegionVariable,
    OptionalResultVariable,
    OptionalSuccessorVariable,
    OptionalUnitAttrVariable,
    PunctuationDirective,
    RegionDirective,
    RegionVariable,
    ResultsDirective,
    ResultVariable,
    SuccessorDirective,
    SuccessorVariable,
    SymbolNameAttributeVariable,
    TypeableDirective,
    TypedAttributeVariable,
    TypeDirective,
    UniqueBaseAttributeVariable,
    VariadicOperandVariable,
    VariadicRegionVariable,
    VariadicResultVariable,
    VariadicSuccessorVariable,
    WhitespaceDirective,
)
from xdsl.parser import BaseParser, ParserState
from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRToken, MLIRTokenKind


@dataclass
class FormatLexer(MLIRLexer):
    """
    A lexer for the declarative assembly format.
    The differences with the MLIR lexer are the following:
    * It can parse '`' or '$' as tokens. The token will have the `BARE_IDENT` kind.
    * Bare identifiers may also may contain `-`.
    """

    def lex(self) -> MLIRToken:
        """Lex a token from the input, and returns it."""
        # First, skip whitespaces
        self._consume_whitespace()

        start_pos = self.pos
        current_char = self._peek_chars()

        # Handle end of file
        if current_char is None:
            return self._form_token(MLIRTokenKind.EOF, start_pos)

        # We parse '`', `\\` and '$' as a BARE_IDENT.
        # This is a hack to reuse the MLIR lexer.
        if current_char in ("`", "$", "\\", "^"):
            self._consume_chars()
            return self._form_token(MLIRTokenKind.BARE_IDENT, start_pos)
        return super().lex()

    # Authorize `-` in bare identifier
    bare_identifier_suffix_regex = re.compile(r"[a-zA-Z0-9_$.\-]*")


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
    seen_regions: list[bool]
    """The region variables that are already parsed."""
    seen_successors: list[bool]
    """The successor variables that are already parsed."""

    def __init__(self, input: str, op_def: OpDef):
        super().__init__(ParserState(FormatLexer(Input(input, "<input>"))))
        self.op_def = op_def
        self.seen_operands = [False] * len(op_def.operands)
        self.seen_operand_types = [False] * len(op_def.operands)
        self.seen_result_types = [False] * len(op_def.results)
        self.seen_attributes = set[str]()
        self.seen_properties = set[str]()
        self.seen_regions = [False] * len(op_def.regions)
        self.seen_successors = [False] * len(op_def.successors)

    def parse_format(self) -> FormatProgram:
        """
        Parse a declarative format, with the following syntax:
          format ::= directive*
        Once the format is parsed, check that it is correct, i.e., that it is
        unambiguous and refer to all elements exactly once.
        """
        elements: list[FormatDirective] = []
        while self._current_token.kind != MLIRTokenKind.EOF:
            elements.append(self.parse_format_directive(False))

        attr_dict_idx = self.verify_attr_dict(elements)
        variables = self.get_constraint_variables()
        self.verify_directives(elements)
        self.verify_properties(elements, attr_dict_idx)
        self.verify_operands(variables)
        self.verify_results(variables)
        self.verify_regions()
        self.verify_successors()
        return FormatProgram(tuple(elements))

    def verify_directives(self, elements: list[FormatDirective]):
        """
        Check correctness of the declarative format; e.g, chaining variadiclike operand
        directives leads to ambiguous parsing, and should raise an error here.
        """
        for a, b in pairwise(elements):
            if not a.is_optional_like():
                continue
            match a, b:
                case _, PunctuationDirective(",") if a.is_variadic_like():
                    self.raise_error(
                        "A variadic directive cannot be followed by a comma literal."
                    )
                case TypeDirective(), TypeDirective():
                    self.raise_error(
                        "An optional/variadic type directive cannot be followed by another "
                        "type directive."
                    )
                case OperandDirective(), OperandDirective():
                    self.raise_error(
                        "An optional/variadic operand variable cannot be followed by another "
                        "operand variable."
                    )
                case RegionDirective(), RegionDirective():
                    self.raise_error(
                        "An optional/variadic region variable cannot be followed by another "
                        "region variable."
                    )
                case SuccessorDirective(), SuccessorDirective():
                    self.raise_error(
                        "A variadic successor variable cannot be followed by another "
                        "variadic successor variable."
                    )
                case AttrDictDirective(), RegionDirective() if not (a.with_keyword):
                    self.raise_error(
                        "An `attr-dict' directive without keyword cannot be directly "
                        "followed by a region variable as it is ambiguous."
                    )
                case _:
                    pass

    def get_constraint_variables(self) -> set[str]:
        """
        Find out which constraint variables can be inferred from the parsed attributes.
        """
        vars = set[str]()
        for i, (_, operand_def) in enumerate(self.op_def.operands):
            vars |= operand_def.constr.variables_from_length()
            if self.seen_operand_types[i]:
                vars |= operand_def.constr.variables()
        for i, (_, result_def) in enumerate(self.op_def.results):
            if self.seen_result_types[i]:
                vars |= result_def.constr.variables()
        for prop_def in self.op_def.properties.values():
            if isinstance(prop_def, OptionalDef) and prop_def.default_value is None:
                continue
            vars |= prop_def.constr.variables()
        for attr_def in self.op_def.attributes.values():
            if isinstance(attr_def, OptionalDef) and attr_def.default_value is None:
                continue
            vars |= attr_def.constr.variables()

        return vars

    def verify_operands(self, var_constraint_names: AbstractSet[str]):
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
                if not operand_def.constr.can_infer(
                    var_constraint_names, length_known=True
                ):
                    self.raise_error(
                        f"type of operand '{operand_name}' cannot be inferred from "
                        f"constraint {operand_def.constr}, consider adding a "
                        f"'type(${operand_name})' directive to the custom assembly "
                        "format"
                    )

    def verify_results(self, var_constraint_names: AbstractSet[str]):
        """Check that all result types are refered at least once, or inferred
        from another construct."""

        for result_type, (result_name, result_def) in zip(
            self.seen_result_types, self.op_def.results, strict=True
        ):
            if not result_type:
                if not result_def.constr.can_infer(
                    var_constraint_names, length_known=False
                ):
                    self.raise_error(
                        f"type of result '{result_name}' cannot be inferred, "
                        f"consider adding a 'type(${result_name})' directive to the "
                        "custom assembly format"
                    )

    def verify_attr_dict(self, elements: list[FormatDirective]) -> int:
        """
        Check that the attribute dictionary is present, returning its index
        """
        for i, element in enumerate(elements):
            if isinstance(element, AttrDictDirective):
                if any(isinstance(e, AttrDictDirective) for e in elements[i + 1 :]):
                    self.raise_error(
                        "'attr-dict' directive can only occur once "
                        "in the assembly format description"
                    )
                return i
        self.raise_error("'attr-dict' directive not found")

    def verify_properties(self, elements: list[FormatDirective], attr_dict_idx: int):
        """
        Check that all properties are present, unless `ParsePropInAttrDict` option is
        used.
        """

        missing_properties = set(self.op_def.properties.keys()) - self.seen_properties

        for option in self.op_def.options:
            if isinstance(option, AttrSizedSegments) and option.as_property:
                missing_properties.remove(option.attribute_name)

        parse_prop_in_attr_dict = any(
            isinstance(option, ParsePropInAttrDict) for option in self.op_def.options
        )

        if missing_properties and not parse_prop_in_attr_dict:
            self.raise_error(
                f"{', '.join(missing_properties)} properties are missing from "
                "the declarative format. If this is intentional, consider using "
                "'ParsePropInAttrDict' IRDL option."
            )

        attr_dict = elements[attr_dict_idx]
        assert isinstance(attr_dict, AttrDictDirective)

        elements[attr_dict_idx] = AttrDictDirective(
            with_keyword=attr_dict.with_keyword,
            reserved_attr_names=self.seen_attributes,
            expected_properties=missing_properties,
        )

    def verify_regions(self):
        """
        Check that all regions are present.
        """
        for (
            seen_region,
            (region_name, _),
        ) in zip(
            self.seen_regions,
            self.op_def.regions,
            strict=True,
        ):
            if not seen_region:
                self.raise_error(
                    f"region '{region_name}' "
                    f"not found, consider adding a '${region_name}' "
                    "directive to the custom assembly format."
                )

    def verify_successors(self):
        """
        Check that all successors are present.
        """
        for (
            seen_successor,
            (successor_name, _),
        ) in zip(
            self.seen_successors,
            self.op_def.successors,
            strict=True,
        ):
            if not seen_successor:
                self.raise_error(
                    f"successor '{successor_name}' "
                    f"not found, consider adding a '${successor_name}' "
                    "directive to the custom assembly format."
                )

    def _parse_optional_operand(
        self, variable_name: str, inside_type: bool, inside_ref: bool
    ) -> OptionalOperandVariable | VariadicOperandVariable | OperandVariable | None:
        for idx, (operand_name, operand_def) in enumerate(self.op_def.operands):
            if variable_name != operand_name:
                continue
            if not inside_ref:
                if not inside_type:
                    if self.seen_operands[idx]:
                        self.raise_error(f"operand '{variable_name}' is already bound")
                    self.seen_operands[idx] = True
                    if isinstance(operand_def, VariadicDef | OptionalDef):
                        self.seen_attributes.add(
                            AttrSizedOperandSegments.attribute_name
                        )
                else:
                    if self.seen_operand_types[idx]:
                        self.raise_error(f"type of '{variable_name}' is already bound")
                    self.seen_operand_types[idx] = True
            match operand_def:
                case OptOperandDef():
                    return OptionalOperandVariable(variable_name, idx)
                case VarOperandDef():
                    return VariadicOperandVariable(variable_name, idx)
                case _:
                    return OperandVariable(variable_name, idx)

    def parse_optional_typeable_variable(
        self, inside_ref: bool
    ) -> TypeableDirective | None:
        """
        Parse a variable, if present, with the following format:
          variable ::= `$` bare-ident
        The variable should refer to an operand or result.
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        start_pos = self.pos
        if self._current_token.text[0] != "$":
            return None
        self._consume_token()
        end_pos = self._current_token.span.end
        variable_name = self.parse_identifier(" after '$'")

        # Check if the variable is an operand
        if (
            variable := self._parse_optional_operand(variable_name, True, inside_ref)
        ) is not None:
            return variable

        # Check if the variable is a result
        for idx, (result_name, result_def) in enumerate(self.op_def.results):
            if variable_name != result_name:
                continue
            if not inside_ref:
                if self.seen_result_types[idx]:
                    self.raise_error(f"type of '{variable_name}' is already bound")
                self.seen_result_types[idx] = True
            match result_def:
                case OptResultDef():
                    return OptionalResultVariable(variable_name, idx)
                case VarResultDef():
                    return VariadicResultVariable(variable_name, idx)
                case _:
                    return ResultVariable(variable_name, idx)

        self.raise_error(
            "expected typeable variable to refer to an operand or result",
            at_position=start_pos,
            end_position=end_pos,
        )

    def parse_optional_variable(
        self,
        inside_ref: bool,
        *,
        qualified: bool = False,
    ) -> FormatDirective | None:
        """
        Parse a variable, if present, with the following format:
          variable ::= `$` bare-ident
        The variable should refer to an operand, attribute, region, or successor.
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        if self._current_token.text[0] != "$":
            return None
        start_pos = self.pos
        self._consume_token()
        end_pos = self._current_token.span.end
        variable_name = self.parse_identifier(" after '$'")

        # Check if the variable is an operand
        if (
            variable := self._parse_optional_operand(variable_name, False, inside_ref)
        ) is not None:
            return variable

        # Check if the variable is a region
        for idx, (region_name, region_def) in enumerate(self.op_def.regions):
            if variable_name != region_name:
                continue
            if not inside_ref:
                if self.seen_regions[idx]:
                    self.raise_error(f"region '{region_name}' is already bound")
                self.seen_regions[idx] = True
            match region_def:
                case OptRegionDef() | OptSingleBlockRegionDef():
                    return OptionalRegionVariable(variable_name, idx)
                case VarRegionDef() | VarSingleBlockRegionDef():
                    return VariadicRegionVariable(variable_name, idx)
                case _:
                    return RegionVariable(variable_name, idx)

        # Check if the variable is a successor
        for idx, (successor_name, successor_def) in enumerate(self.op_def.successors):
            if variable_name != successor_name:
                continue
            if not inside_ref:
                if self.seen_successors[idx]:
                    self.raise_error(f"successor '{successor_name}' is already bound")
                self.seen_successors[idx] = True
            match successor_def:
                case OptSuccessorDef():
                    return OptionalSuccessorVariable(variable_name, idx)
                case VarSuccessorDef():
                    return VariadicSuccessorVariable(variable_name, idx)
                case _:
                    return SuccessorVariable(variable_name, idx)

        attr_or_prop_by_name = {
            attr_name: attr_or_prop
            for attr_name, attr_or_prop in self.op_def.accessor_names.values()
        }

        # Check if the variable is an attribute
        if variable_name in attr_or_prop_by_name:
            attr_name = variable_name
            attr_or_prop = attr_or_prop_by_name[attr_name]
            is_property = attr_or_prop == "property"
            if is_property:
                if not inside_ref:
                    if attr_name in self.seen_properties:
                        self.raise_error(f"property '{variable_name}' is already bound")
                    self.seen_properties.add(attr_name)
                attr_def = self.op_def.properties[attr_name]
            else:
                if not inside_ref:
                    if attr_name in self.seen_attributes:
                        self.raise_error(
                            f"attribute '{variable_name}' is already bound"
                        )
                    self.seen_attributes.add(attr_name)
                attr_def = self.op_def.attributes[attr_name]

            is_optional = isinstance(attr_def, OptionalDef)

            bases = attr_def.constr.get_bases()
            unique_base = bases.pop() if bases is not None and len(bases) == 1 else None

            if qualified:
                # Ensure qualified attributes stay qualified
                unique_base = None

            if unique_base is not None:
                if unique_base == UnitAttr:
                    return OptionalUnitAttrVariable(variable_name, is_property)

                # We special case `SymbolNameConstr`, just as MLIR does.
                if isinstance(attr_def.constr, SymbolNameConstraint):
                    return SymbolNameAttributeVariable(
                        variable_name, is_property, is_optional, attr_def.default_value
                    )

                constr = attr_def.constr
                if isinstance(constr, VarConstraint):
                    constr = constr.constraint
                if isinstance(constr, ParamAttrConstraint):
                    if unique_base is DenseArrayBase and (
                        elt_type_constr := constr.param_constrs[0]
                    ).can_infer(set()):
                        elt_type = elt_type_constr.infer(ConstraintContext())
                        return DenseArrayAttributeVariable(
                            variable_name,
                            is_property,
                            is_optional,
                            attr_def.default_value,
                            cast(IntegerType | AnyFloat, elt_type),
                        )

                    if issubclass(unique_base, TypedAttribute):
                        # TODO: generalize.
                        # https://github.com/xdslproject/xdsl/issues/2499
                        type_constraint = constr.param_constrs[
                            unique_base.get_type_index()
                        ]
                        if type_constraint.can_infer(set()):
                            unique_type = type_constraint.infer(ConstraintContext())
                            return TypedAttributeVariable(
                                variable_name,
                                is_property,
                                is_optional,
                                attr_def.default_value,
                                unique_base,
                                unique_type,
                            )

                if unique_base not in Builtin.attributes:
                    # Always qualify builtin attributes
                    # This is technically an approximation, but appears to be good enough
                    # for xDSL right now.
                    return UniqueBaseAttributeVariable(
                        variable_name,
                        is_property,
                        is_optional,
                        attr_def.default_value,
                        unique_base,
                    )

            return AttributeVariable(
                variable_name,
                is_property,
                is_optional,
                attr_def.default_value,
            )

        self.raise_error(
            "expected variable to refer to an operand, attribute, region, or successor",
            at_position=start_pos,
            end_position=end_pos,
        )

    def parse_type_directive(self, inside_ref: bool) -> FormatDirective:
        """
        Parse a type directive with the following format:
          type-directive ::= `type` `(` typeable-directive `)`
        `type` is expected to have already been parsed.
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        with self.in_parens():
            return TypeDirective(self.parse_typeable_directive(inside_ref))

    def parse_functional_type_directive(self, inside_ref: bool) -> FormatDirective:
        """
        Parse a functional-type directive with the following format
          functional-type-directive ::= `functional-type` `(` typeable-directive `,` typeable-directive `)`
        `functional-type` is expected to have already been parsed.
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        with self.in_parens():
            operands = self.parse_typeable_directive(inside_ref)
            self.parse_punctuation(",")
            results = self.parse_typeable_directive(inside_ref)
        return FunctionalTypeDirective(operands, results)

    def parse_qualified_directive(self, inside_ref: bool) -> FormatDirective:
        """
        Parse a qualified attribute or type directive, with the following format:
            qualified-directive ::= `qualified` `(` variable `)`
        `qualified` is expected to have already been parsed.
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        with self.in_parens():
            res = self.parse_optional_variable(inside_ref, qualified=True)
            if res is None:
                self.raise_error(
                    "expected a variable after 'qualified', found "
                    f"'{self._current_token.text}'"
                )
        return res

    def parse_optional_group(self) -> FormatDirective:
        """
        Parse an optional group, with the following format:
          group ::= `(` then-elements `)` (`:` `(` else-elements `)` )? `?`
        """
        then_elements = tuple[FormatDirective, ...]()
        else_elements = tuple[FormatDirective, ...]()
        anchor: Directive | None = None

        while not self.parse_optional_punctuation(")"):
            then_elements += (self.parse_format_directive(False),)
            if self.parse_optional_keyword("^"):
                if anchor is not None:
                    self.raise_error("An optional group can only have one anchor.")
                anchor = then_elements[-1]
                if isinstance(anchor, RegionVariable):
                    anchor = AnchorRegionVariable(anchor.name, anchor.index)
                    then_elements = then_elements[:-1] + (anchor,)

        if self.parse_optional_punctuation(":"):
            self.parse_punctuation("(")
            while not self.parse_optional_punctuation(")"):
                else_elements += (self.parse_format_directive(False),)

        self.parse_punctuation("?")

        # Pull whitespace element of front, as they are not parsed
        first_non_whitespace_index = None
        for i, x in enumerate(then_elements):
            if not isinstance(x, WhitespaceDirective):
                first_non_whitespace_index = i
                break

        if first_non_whitespace_index is None:
            self.raise_error("An optional group must have a non-whitespace directive")
        if anchor is None:
            self.raise_error("Every optional group must have an anchor.")
        if not then_elements[first_non_whitespace_index].is_optional_like():
            self.raise_error(
                "First element of an optional group must be optionally parsable."
            )
        if not anchor.is_anchorable():
            self.raise_error(
                "An optional group's anchor must be an anchorable directive."
            )

        return OptionalGroupDirective(
            anchor,
            cast(
                tuple[WhitespaceDirective, ...],
                then_elements[:first_non_whitespace_index],
            ),
            then_elements[first_non_whitespace_index],
            then_elements[first_non_whitespace_index + 1 :],
            else_elements,
        )

    def parse_keyword_or_punctuation(self) -> FormatDirective:
        """
        Parse a keyword or a punctuation directive, with the following format:
          keyword-or-punctuation-directive ::= `\\`` (bare-ident | punctuation) `\\``
        """
        start_token = self._current_token
        self.parse_characters("`")

        # New line case
        if self.parse_optional_keyword("\\"):
            self.parse_keyword("n")
            self.parse_characters("`")
            return WhitespaceDirective("\n")

        # Space case
        end_token = self._current_token
        if self.parse_optional_characters("`"):
            whitespace = self.lexer.input.content[
                start_token.span.end : end_token.span.start
            ]
            if whitespace != " " and whitespace != "":
                self.raise_error(
                    "unexpected whitespace in directive, only ` ` or `` whitespace is allowed"
                )
            return WhitespaceDirective(whitespace)

        # Punctuation case
        if self._current_token.kind.is_punctuation():
            punctuation = self._consume_token().text
            self.parse_characters("`")
            assert MLIRTokenKind.is_spelling_of_punctuation(punctuation)
            return PunctuationDirective(punctuation)

        # Identifier case
        ident = self.parse_optional_identifier()
        if ident is None or ident == "`":
            self.raise_error("punctuation or identifier expected")

        self.parse_characters("`")
        return KeywordDirective(ident)

    def parse_typeable_directive(self, inside_ref: bool) -> TypeableDirective:
        """
        Parse a typeable directive, with the following format:
          typeable-directive ::= variable | `operands` | `results`
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        if self.parse_optional_keyword("operands"):
            return self.create_operands_directive(True, inside_ref)
        if self.parse_optional_keyword("results"):
            return self.create_results_directive(inside_ref)
        if variable := self.parse_optional_typeable_variable(inside_ref):
            return variable
        self.raise_error(f"unexpected token '{self._current_token.text}'")

    def parse_custom_directive(self) -> FormatDirective:
        """
        Parse a custom directive, with the following format:
          custom-directive ::= `custom` `<` bare-ident `>` `(`
            (possibly-ref-directive (`,` possibly-ref-directive)*)?
          `)`
        Assumes the keyword `custom` has already been parsed.
        """

        with self.in_angle_brackets():
            name = self.parse_identifier()
            if name not in self.op_def.custom_directives:
                self.raise_error(f"Custom directive {name} cannot be found.")
        directive = self.op_def.custom_directives[name]
        param_types = directive.parameters
        params = list[FormatDirective]()
        with self.in_parens():
            for i, (field, ty) in enumerate(param_types.items()):
                if i:
                    self.parse_punctuation(",")
                param = self.parse_possible_ref_directive()
                if not isinstance(param, ty):
                    self.raise_error(
                        f"{name}.{field} was expected to be of type {ty}, but got {param}"
                    )
                params.append(param)
        return directive(*params)

    def parse_possible_ref_directive(self) -> FormatDirective:
        """
        Parse a ref directive or other format directive, with format:
          possibly-ref-directive ::= `ref` `(` directive `)` | directive
        """
        if self.parse_optional_keyword("ref"):
            with self.in_parens():
                return self.parse_format_directive(True)
        return self.parse_format_directive(False)

    def parse_format_directive(self, inside_ref: bool) -> FormatDirective:
        """
        Parse a format directive, with the following format:
          directive ::= `attr-dict`
                        | `attr-dict-with-keyword`
                        | type-directive
                        | keyword-or-punctuation-directive
                        | variable
        If `inside_ref` is `True`, then this directive can have appeared in the
        assembly format already.
        """
        if self.parse_optional_keyword("attr-dict"):
            return self.create_attr_dict_directive(False)
        if self.parse_optional_keyword("attr-dict-with-keyword"):
            return self.create_attr_dict_directive(True)
        if self.parse_optional_keyword("type"):
            return self.parse_type_directive(inside_ref)
        if self.parse_optional_keyword("operands"):
            return self.create_operands_directive(False, inside_ref)
        if self.parse_optional_keyword("functional-type"):
            return self.parse_functional_type_directive(inside_ref)
        if self.parse_optional_keyword("qualified"):
            return self.parse_qualified_directive(inside_ref)
        if self.parse_optional_keyword("custom"):
            return self.parse_custom_directive()
        if self._current_token.text == "`":
            return self.parse_keyword_or_punctuation()
        if self.parse_optional_punctuation("("):
            return self.parse_optional_group()
        if variable := self.parse_optional_variable(inside_ref):
            return variable
        self.raise_error(f"unexpected token '{self._current_token.text}'")

    def create_attr_dict_directive(self, with_keyword: bool) -> AttrDictDirective:
        """Create an attribute dictionary directive, and update the parsing state."""
        # reserved_attr_names and expected_properties are populated once the format is parsed,
        # as some attributes might appear after the attr-dict directive.
        return AttrDictDirective(
            with_keyword=with_keyword,
            reserved_attr_names=set(),
            expected_properties=set(),
        )

    def create_operands_directive(
        self, inside_type: bool, inside_ref: bool
    ) -> OperandsDirective:
        """
        Create an operands directive.
        If `inside_type` is true, then we are nested within a `type` directive.
        If `inside_ref` is true, we allow operands to have been previously parsed.
        """
        if not self.op_def.operands:
            self.raise_error("'operands' should not be used when there are no operands")
        if not inside_ref and not inside_type and any(self.seen_operands):
            self.raise_error("'operands' cannot be used with other operand directives")
        if not inside_ref and inside_type and any(self.seen_operand_types):
            self.raise_error(
                "'operands' cannot be used in a type directive with other operand type directives"
            )
        variadics = tuple(
            (isinstance(o, OptionalDef), i)
            for i, (_, o) in enumerate(self.op_def.operands)
            if isinstance(o, VariadicDef)
        )
        if len(variadics) > 1 and SameVariadicOperandSize() not in self.op_def.options:
            self.raise_error("'operands' is ambiguous with multiple variadic operands")
        if not inside_ref:
            if not inside_type:
                self.seen_operands = [True] * len(self.seen_operands)
            else:
                self.seen_operand_types = [True] * len(self.seen_operand_types)
        return OperandsDirective()

    def create_results_directive(self, inside_ref: bool) -> ResultsDirective:
        """
        Create an results directive.
        If `inside_ref` is true, we allow results to have been previously parsed.
        """
        if not self.op_def.results:
            self.raise_error("'results' should not be used when there are no results")
        if not inside_ref and any(self.seen_result_types):
            self.raise_error(
                "'results' cannot be used in a type directive with other result type directives"
            )
        variadics = tuple(
            (isinstance(o, OptResultDef), i)
            for i, (_, o) in enumerate(self.op_def.results)
            if isinstance(o, VarResultDef)
        )
        if len(variadics) > 1 and SameVariadicResultSize() not in self.op_def.options:
            self.raise_error("'results' is ambiguous with multiple variadic results")
        if not inside_ref:
            self.seen_result_types = [True] * len(self.seen_result_types)
        return ResultsDirective()
