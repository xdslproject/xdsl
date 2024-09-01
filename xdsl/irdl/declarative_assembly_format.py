"""
This file contains the data structures necessary for the parsing and printing
of the MLIR declarative assembly format defined at
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format .
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from xdsl.dialects.builtin import UnitAttr
from xdsl.ir import (
    Attribute,
    Data,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypedAttribute,
)
from xdsl.irdl import (
    ConstraintContext,
    IRDLOperation,
    IRDLOperationInvT,
    OpDef,
    OptionalDef,
    Successor,
    VariadicDef,
    VarIRConstruct,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.lexer import PunctuationSpelling

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
    regions: list[Region | None | list[Region]]
    successors: list[Successor | None | list[Successor]]
    attributes: dict[str, Attribute]
    properties: dict[str, Attribute]
    constraint_context: ConstraintContext

    def __init__(self, op_def: OpDef):
        self.operands = [None] * len(op_def.operands)
        self.operand_types = [None] * len(op_def.operands)
        self.result_types = [None] * len(op_def.results)
        self.regions = [None] * len(op_def.regions)
        self.successors = [None] * len(op_def.successors)
        self.attributes = {}
        self.properties = {}
        self.constraint_context = ConstraintContext()


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
        ), unresolved_operands
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
            regions=state.regions,
            successors=state.successors,
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
                    assert isa(operand_type, list[Attribute])
                    operand_def.constr.verify(operand_type, state.constraint_context)
                for (_, result_def), result_type in zip(
                    op_def.results, state.result_types, strict=True
                ):
                    if result_type is None:
                        continue
                    if isinstance(result_type, Attribute):
                        result_type = [result_type]
                    assert isa(result_type, list[Attribute])
                    result_def.constr.verify(result_type, state.constraint_context)
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
                operand = state.operands[i]
                range_length = len(operand) if isinstance(operand, list) else 1
                operand_type = operand_def.constr.infer(
                    range_length, state.constraint_context
                )
                if isinstance(operand_def, OptionalDef):
                    operand_type = (
                        list[Attribute | None]()
                        if len(operand_type) == 0
                        else operand_type[0]
                    )
                elif isinstance(operand_def, VariadicDef):
                    operand_type = cast(list[Attribute | None], operand_type)
                else:
                    operand_type = operand_type[0]
                state.operand_types[i] = operand_type

    def resolve_result_types(self, state: ParsingState, op_def: OpDef) -> None:
        """
        Use the inferred type resolutions to fill missing result types from other parsed
        types.
        """
        for i, (result_type, (_, result_def)) in enumerate(
            zip(state.result_types, op_def.results, strict=True)
        ):
            if result_type is None:
                result_type = state.result_types[i]
                range_length = len(result_type) if isinstance(result_type, list) else 1
                result_type = result_def.constr.infer(
                    range_length, state.constraint_context
                )
                if isinstance(result_def, OptionalDef):
                    result_type = (
                        list[Attribute | None]()
                        if len(result_type) == 0
                        else result_type[0]
                    )
                elif isinstance(result_def, VariadicDef):
                    result_type = cast(list[Attribute | None], result_type)
                else:
                    result_type = result_type[0]
                state.result_types[i] = result_type

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
    def parse(self, parser: Parser, state: ParsingState) -> None: ...

    @abstractmethod
    def print(
        self, printer: Printer, state: PrintingState, op: IRDLOperation
    ) -> None: ...


class AnchorableDirective(FormatDirective, ABC):
    """
    Base class for Directive usable as anchors to optional groups.
    """

    @abstractmethod
    def is_present(self, op: IRDLOperation) -> bool:
        """
        Check if the directive is present in the input.
        """
        ...


class OptionallyParsableDirective(FormatDirective, ABC):
    """
    Base class for Directive that can be optionally parsed.
    Those are the ones usable as first element of an optional group.
    """

    @abstractmethod
    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        """
        Try parsing the directive and return if it was present.
        """
        ...

    def parse(self, parser: Parser, state: ParsingState) -> None:
        self.parse_optional(parser, state)


class VariadicLikeFormatDirective(AnchorableDirective, ABC):
    """
    Baseclass to help keep typechecking simple.
    VariadicLike is mostly Variadic or Optional: Whatever directive that can accept
    having nothing to parse.
    """

    pass


@dataclass(frozen=True)
class VariableDirective(FormatDirective, ABC):
    """
    A variable directive, with the following format:
      variable-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    name: str
    """The variable name. This is only used for error message reporting."""
    index: int
    """Index of the variable(operand or result) definition."""


class TypeDirective(VariableDirective, ABC):
    """
    Base class for Directive meant to parse types.
    """

    pass


class RegionDirective(OptionallyParsableDirective, ABC):
    """
    Baseclass to help keep typechecking simple.
    RegionDirective is for any RegionVariable, which are all OptionallyParsable.
    """

    pass


class VariadicLikeVariable(VariadicLikeFormatDirective, VariableDirective, ABC):
    pass


class VariadicVariable(VariadicLikeVariable, ABC):
    def is_present(self, op: IRDLOperation) -> bool:
        return len(getattr(op, self.name)) > 0


class OptionalVariable(VariadicLikeVariable, ABC):
    def is_present(self, op: IRDLOperation) -> bool:
        return getattr(op, self.name) is not None


class VariadicLikeTypeDirective(VariadicLikeFormatDirective, VariableDirective, ABC):
    pass


class VariadicTypeDirective(VariadicLikeTypeDirective, VariadicVariable, ABC):
    pass


class OptionalTypeDirective(VariadicLikeTypeDirective, OptionalVariable, ABC):
    pass


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
        state.should_emit_space = True


@dataclass(frozen=True)
class OperandVariable(VariableDirective):
    """
    An operand variable, with the following format:
      operand-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        operand = parser.parse_unresolved_operand()
        state.operands[self.index] = operand

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_ssa_value(getattr(op, self.name))
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicOperandVariable(
    VariadicVariable, VariableDirective, OptionallyParsableDirective
):
    """
    A variadic operand variable, with the following format:
      operand-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        operands = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )
        if operands is None:
            operands = []
        state.operands[self.index] = cast(list[UnresolvedOperand | None], operands)
        return bool(operands)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        operand = getattr(op, self.name)
        if operand:
            printer.print_list(operand, printer.print_ssa_value)
            state.last_was_punctuation = False
            state.should_emit_space = True


class OptionalOperandVariable(OptionalVariable, OptionallyParsableDirective):
    """
    An optional operand variable, with the following format:
      operand-directive ::= ( percent-ident )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        operand = parser.parse_optional_unresolved_operand()
        if operand is None:
            operand = list[UnresolvedOperand | None]()
        state.operands[self.index] = operand
        return bool(operand)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        operand = getattr(op, self.name)
        if operand:
            printer.print_ssa_value(operand)
            state.last_was_punctuation = False
            state.should_emit_space = True


@dataclass(frozen=True)
class OperandTypeDirective(TypeDirective):
    """
    An operand variable type directive, with the following format:
      operand-type-directive ::= type(dollar-ident)
    The directive will request a space to be printed right after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        type = parser.parse_type()
        state.operand_types[self.index] = type

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_attribute(getattr(op, self.name).type)
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicOperandTypeDirective(
    TypeDirective, VariadicTypeDirective, OptionallyParsableDirective
):
    """
    A variadic operand variable, with the following format:
      operand-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        operand_types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        if operand_types is None:
            operand_types = []
        state.operand_types[self.index] = cast(list[Attribute | None], operand_types)
        return bool(operand_types)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(getattr(op, self.name).types, printer.print_attribute)
        state.last_was_punctuation = False
        state.should_emit_space = True


class OptionalOperandTypeDirective(OptionalTypeDirective, OptionallyParsableDirective):
    """
    An optional operand variable type directive, with the following format:
      operand-type-directive ::= ( type(dollar-ident) )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        type = parser.parse_optional_type()
        if type is None:
            type = list[Attribute | None]()
        state.operand_types[self.index] = type
        return bool(type)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        operand = getattr(op, self.name)
        if operand:
            printer.print_attribute(operand.type)
            state.last_was_punctuation = False
            state.should_emit_space = True


@dataclass(frozen=True)
class ResultVariable(VariableDirective):
    """
    An result variable, with the following format:
      result-directive ::= dollar-ident
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
class VariadicResultVariable(
    ResultVariable, VariadicVariable, OptionallyParsableDirective
):
    """
    A variadic result variable, with the following format:
      result-directive ::= percent-ident (( `,` percent-id )* )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )
        return False

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )


class OptionalResultVariable(OptionalVariable, OptionallyParsableDirective):
    """
    An optional result variable, with the following format:
      result-directive ::= ( percent-ident )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )
        return False

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        assert (
            "Result variables cannot be used directly to parse/print in "
            "declarative formats."
        )


@dataclass(frozen=True)
class ResultTypeDirective(TypeDirective):
    """
    A result variable type directive, with the following format:
      result-type-directive ::= type(dollar-ident)
    The directive will request a space to be printed right after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        type = parser.parse_type()
        state.result_types[self.index] = type

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_attribute(getattr(op, self.name).type)
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicResultTypeDirective(
    TypeDirective, VariadicTypeDirective, OptionallyParsableDirective
):
    """
    A variadic result variable type directive, with the following format:
      variadic-result-type-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        result_types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        if result_types is None:
            result_types = []
        state.result_types[self.index] = cast(list[Attribute | None], result_types)
        return bool(result_types)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(getattr(op, self.name).types, printer.print_attribute)
        state.last_was_punctuation = False
        state.should_emit_space = True


class OptionalResultTypeDirective(
    TypeDirective, OptionalTypeDirective, OptionallyParsableDirective
):
    """
    An optional result variable type directive, with the following format:
      result-type-directive ::= ( type(dollar-ident) )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        type = parser.parse_optional_type()
        if type is None:
            type = list[Attribute | None]()
        state.result_types[self.index] = type
        return bool(type)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        result = getattr(op, self.name)
        if result:
            printer.print_attribute(result.type)
            state.last_was_punctuation = False
            state.should_emit_space = True


@dataclass(frozen=True)
class RegionVariable(RegionDirective, VariableDirective):
    """
    A region variable, with the following format:
      region-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        region = parser.parse_optional_region()
        state.regions[self.index] = region
        return region is not None

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_region(getattr(op, self.name))
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicRegionVariable(RegionDirective, VariadicVariable):
    """
    A variadic region variable, with the following format:
      region-directive ::= dollar-ident

    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        regions: list[Region] = []
        current_region = parser.parse_optional_region()
        while current_region is not None:
            regions.append(current_region)
            current_region = parser.parse_optional_region()

        state.regions[self.index] = regions
        return bool(regions)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        region = getattr(op, self.name)
        if region:
            printer.print_list(region, printer.print_region, delimiter=" ")
            state.last_was_punctuation = False
            state.should_emit_space = True


class OptionalRegionVariable(RegionDirective, OptionalVariable):
    """
    An optional region variable, with the following format:
      region-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        region = parser.parse_optional_region()
        if region is None:
            region = list[Region]()
        state.regions[self.index] = region
        return bool(region)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        region = getattr(op, self.name)
        if region:
            printer.print_region(region)
            state.last_was_punctuation = False
            state.should_emit_space = True


class SuccessorVariable(VariableDirective, OptionallyParsableDirective):
    """
    A successor variable, with the following format:
      successor-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        successor = parser.parse_optional_successor()

        state.successors[self.index] = successor

        return successor is not None

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_block_name(getattr(op, self.name))
        state.last_was_punctuation = False
        state.should_emit_space = True


class VariadicSuccessorVariable(VariadicVariable, OptionallyParsableDirective):
    """
    A variadic successor variable, with the following format:
      successor-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        successors: list[Successor] = []
        current_successor = parser.parse_optional_successor()
        while current_successor is not None:
            successors.append(current_successor)
            current_successor = parser.parse_optional_successor()

        state.successors[self.index] = successors

        return bool(successors)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        successor = getattr(op, self.name)
        if successor:
            printer.print_list(successor, printer.print_block_name, delimiter=" ")
            state.last_was_punctuation = False
            state.should_emit_space = True


class OptionalSuccessorVariable(OptionalVariable, OptionallyParsableDirective):
    """
    An optional successor variable, with the following format:
      successor-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        successor = parser.parse_optional_successor()
        if successor is None:
            successor = list[Successor]()
        state.successors[self.index] = successor
        return bool(successor)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        successor = getattr(op, self.name)
        if successor:
            printer.print_block_name(successor)
            state.last_was_punctuation = False
            state.should_emit_space = True


@dataclass(frozen=True)
class AttributeVariable(FormatDirective):
    """
    An attribute variable, with the following format:
      result-directive ::= dollar-ident
    The directive will request a space to be printed right after.
    """

    name: str
    """The attribute name as it should be in the attribute or property dictionary."""
    is_property: bool
    """Should this attribute be put in the attribute or property dictionary."""
    unique_base: type[Attribute] | None
    """The known base class of the Attribute, if any."""
    unique_type: Attribute | None
    """The known type of the Attribute, if any."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        unique_base = self.unique_base
        if unique_base is None:
            attr = parser.parse_attribute()
        elif self.unique_type is not None:
            unique_base = cast(
                type[TypedAttribute[Attribute]],
                unique_base,
            )
            attr = unique_base.parse_with_type(parser, self.unique_type)
        elif issubclass(
            unique_base,
            ParametrizedAttribute,
        ):
            attr = unique_base.new(unique_base.parse_parameters(parser))
        elif issubclass(unique_base, Data):
            unique_base = cast(
                type[Data[Any]],
                unique_base,
            )
            attr = unique_base.new(unique_base.parse_parameter(parser))
        else:
            raise ValueError("Attributes must be Data or ParameterizedAttribute.")
        if self.is_property:
            state.properties[self.name] = attr
        else:
            state.attributes[self.name] = attr

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        state.should_emit_space = True
        state.last_was_punctuation = False

        if self.is_property:
            attr = op.properties[self.name]
        else:
            attr = op.attributes[self.name]

        if self.unique_type is not None:
            return cast(TypedAttribute[Attribute], attr).print_without_type(printer)
        if self.unique_base is None:
            return printer.print_attribute(attr)
        if isinstance(attr, ParametrizedAttribute):
            return attr.print_parameters(printer)
        if isinstance(attr, Data):
            return attr.print_parameter(printer)
        raise ValueError("Attributes must be Data or ParameterizedAttribute!")


class OptionalAttributeVariable(AttributeVariable, OptionalVariable):
    """
    An optional attribute variable, with the following format:
      operand-directive ::= ( percent-ident )?
    The directive will request a space to be printed after.
    """


class OptionalUnitAttrVariable(OptionalAttributeVariable):
    """
    An optional UnitAttr variable that holds no value and derives its meaning from its existence. Holds a parse
    and print method to reflect this.

      operand-directive ::= (`unit_attr` unit_attr^)?

    Also see: https://mlir.llvm.org/docs/DefiningDialects/Operations/#unit-attributes
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        if self.is_property:
            state.properties[self.name] = UnitAttr()
        else:
            state.attributes[self.name] = UnitAttr()

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        return


@dataclass(frozen=True)
class WhitespaceDirective(FormatDirective):
    """
    A whitespace directive, with the following format:
      whitespace-directive ::= `\n` | ` ` | ``
    This directive is only applied during printing, and has no effect during
    parsing.
    The directive will not request any space to be printed after.
    """

    whitespace: Literal[" ", "\n", ""]
    """The whitespace that should be printed."""

    def parse(self, parser: Parser, state: ParsingState) -> None:
        pass

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        printer.print(self.whitespace)
        state.last_was_punctuation = self.whitespace == ""
        state.should_emit_space = False


@dataclass(frozen=True)
class PunctuationDirective(OptionallyParsableDirective):
    """
    A punctuation directive, with the following format:
      punctuation-directive ::= punctuation
    The directive will request a space to be printed right after, unless the punctuation
    is `<`, `(`, `{`, or `[`.
    It will also print a space before if a space is requested, and that the punctuation
    is neither `>`, `)`, `}`, `]`, or `,` if the last element was a punctuation, and
    additionally neither `<`, `(`, `}`, `]`, if the last element was not a punctuation.
    """

    punctuation: PunctuationSpelling
    """The punctuation that should be printed/parsed."""

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        return parser.parse_optional_punctuation(self.punctuation) is not None

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
class KeywordDirective(OptionallyParsableDirective):
    """
    A keyword directive, with the following format:
      keyword-directive ::= bare-ident
    The directive expects a specific identifier, and will request a space to be printed
    after.
    """

    keyword: str
    """The identifier that should be printed."""

    def parse_optional(self, parser: Parser, state: ParsingState):
        return parser.parse_optional_keyword(self.keyword) is not None

    def parse(self, parser: Parser, state: ParsingState):
        parser.parse_keyword(self.keyword)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space:
            printer.print(" ")
        printer.print(self.keyword)
        state.should_emit_space = True
        state.last_was_punctuation = False


@dataclass(frozen=True)
class OptionalGroupDirective(FormatDirective):
    anchor: AnchorableDirective
    then_whitespace: tuple[WhitespaceDirective, ...]
    then_first: OptionallyParsableDirective
    then_elements: tuple[FormatDirective, ...]

    def parse(self, parser: Parser, state: ParsingState) -> None:
        # If the first element was parsed, parse the then-elements as usual
        if self.then_first.parse_optional(parser, state):
            for element in self.then_elements:
                element.parse(parser, state)
        # Otherwise, just explicitly set the variadic/optional variables and
        # type to empty
        else:
            for element in self.then_elements:
                match element:
                    case (
                        OperandVariable(_, index)
                        | VariadicOperandVariable(_, index)
                        | OptionalOperandVariable(_, index)
                    ):
                        state.operands[index] = list[UnresolvedOperand | None]()
                    case (
                        OperandTypeDirective(_, index)
                        | VariadicOperandTypeDirective(_, index)
                        | OptionalOperandTypeDirective(_, index)
                    ):
                        state.operand_types[index] = list[Attribute | None]()
                    case (
                        RegionVariable(_, index)
                        | VariadicRegionVariable(_, index)
                        | OptionalRegionVariable(_, index)
                    ):
                        state.regions[index] = list[Region]()
                    case (
                        ResultTypeDirective(_, index)
                        | VariadicResultTypeDirective(_, index)
                        | OptionalResultTypeDirective(_, index)
                    ):
                        state.result_types[index] = list[Attribute | None]()
                    case _:
                        pass

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if self.anchor.is_present(op):
            for element in (
                *self.then_whitespace,
                self.then_first,
                *self.then_elements,
            ):
                element.print(printer, state, op)
