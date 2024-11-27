"""
This file contains the data structures necessary for the parsing and printing
of the MLIR declarative assembly format defined at
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format .
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar

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
    ConstraintVariableType,
    InferenceContext,
    IRDLOperation,
    IRDLOperationInvT,
    OpDef,
    OptionalDef,
    Successor,
    VarExtractor,
    VariadicDef,
    VarIRConstruct,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.lexer import PunctuationSpelling

OperandOrResult = Literal[VarIRConstruct.OPERAND, VarIRConstruct.RESULT]


@dataclass
class ParsingState:
    """
    State carried during the parsing of an operation using the declarative assembly
    format.
    It contains the elements that have already been parsed.
    """

    operands: list[UnresolvedOperand | None | Sequence[UnresolvedOperand]]
    operand_types: list[Attribute | None | Sequence[Attribute]]
    result_types: list[Attribute | None | Sequence[Attribute]]
    regions: list[Region | None | Sequence[Region]]
    successors: list[Successor | None | Sequence[Successor]]
    attributes: dict[str, Attribute]
    properties: dict[str, Attribute]
    variables: dict[str, ConstraintVariableType]

    def __init__(self, op_def: OpDef):
        self.operands = [None] * len(op_def.operands)
        self.operand_types = [None] * len(op_def.operands)
        self.result_types = [None] * len(op_def.results)
        self.regions = [None] * len(op_def.regions)
        self.successors = [None] * len(op_def.successors)
        self.attributes = {}
        self.properties = {}
        self.variables = {}


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

    stmts: tuple[FormatDirective, ...]
    """The statements composing the program. They are executed in order."""

    extractors: dict[str, VarExtractor[ParsingState]]
    """Extractors for all type variables from the parsing state."""

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
        self.resolve_constraint_variables(state)

        # Infer operand types that should be inferred
        unresolved_operands = state.operands
        self.resolve_operand_types(state, op_def)
        operand_types = state.operand_types
        assert None not in operand_types

        # Infer result types that should be inferred
        self.resolve_result_types(state, op_def)
        result_types = state.result_types
        assert None not in result_types

        # Resolve all operands
        operands: Sequence[SSAValue | Sequence[SSAValue]] = []
        for uo, ot in zip(unresolved_operands, operand_types, strict=True):
            assert uo is not None
            if isinstance(uo, UnresolvedOperand):
                assert isinstance(
                    ot, Attribute
                ), "Something went wrong with the declarative assembly format parser."
                "Single operand has no type or variadic/optional type"
                operands.append(parser.resolve_operand(uo, ot))
            else:
                assert isinstance(
                    ot, Sequence
                ), f"Something went wrong with the declarative assembly format parser. {type(ot)} {ot}"
                "Variadic or optional operand has no type or a single type "
                operands.append(parser.resolve_operands(uo, ot, parser.pos))

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

    def resolve_constraint_variables(self, state: ParsingState):
        state.variables = {v: r.extract_var(state) for v, r in self.extractors.items()}

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
                range_length = len(operand) if isinstance(operand, Sequence) else 1
                operand_type = operand_def.constr.infer(
                    range_length,
                    InferenceContext(state.variables),
                )
                resolved_operand_type: Attribute | Sequence[Attribute]
                if isinstance(operand_def, OptionalDef):
                    resolved_operand_type = operand_type[0] if operand_type else ()
                elif isinstance(operand_def, VariadicDef):
                    resolved_operand_type = operand_type
                else:
                    resolved_operand_type = operand_type[0]
                state.operand_types[i] = resolved_operand_type

    def resolve_result_types(self, state: ParsingState, op_def: OpDef) -> None:
        """
        Use the inferred type resolutions to fill missing result types from other parsed
        types.
        """
        for i, (result_type, (result_name, result_def)) in enumerate(
            zip(state.result_types, op_def.results, strict=True)
        ):
            if result_type is None:
                # The number of results is not passed in when parsing operations.
                # In the generic format, the type of the operation always specifies the
                # types of the results, and `resultSegmentSizes` specifies the ranges of
                # of the results if multiple are variadic.
                # In order to support variadic results, the types an length of all
                # variadic results must be present in the custom syntax.
                if isinstance(result_def, OptionalDef | VariadicDef):
                    raise NotImplementedError(
                        f"Inference of length of variadic result '{result_name}' not "
                        "implemented"
                    )
                range_length = 1
                inferred_result_types = result_def.constr.infer(
                    range_length,
                    InferenceContext(state.variables),
                )
                resolved_result_type = inferred_result_types[0]
                state.result_types[i] = resolved_result_type

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
class Directive(ABC):
    """An assembly format directive"""


class AnchorableDirective(Directive, ABC):
    """
    Base class for Directive usable as anchors to optional groups.
    """

    @abstractmethod
    def is_present(self, op: IRDLOperation) -> bool:
        """
        Check if the directive is present in the input.
        """
        ...


class FormatDirective(Directive, ABC):
    """A format directive for operation format."""

    @abstractmethod
    def parse(self, parser: Parser, state: ParsingState) -> None: ...

    @abstractmethod
    def print(
        self, printer: Printer, state: PrintingState, op: IRDLOperation
    ) -> None: ...


class OptionallyParsableDirective(FormatDirective, ABC):
    """
    Base class for Directive that can be optionally parsed.
    Those are the ones usable as first element of an optional group.
    """

    @abstractmethod
    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        """
        Try parsing the directive and return True if it was present.
        """
        ...

    def parse(self, parser: Parser, state: ParsingState) -> None:
        self.parse_optional(parser, state)


class VariadicLikeFormatDirective(
    OptionallyParsableDirective, AnchorableDirective, ABC
):
    """
    A directive which parses/prints multiple objects separated by commas.
    Such directives can not be followed by comma literals.
    """

    def set_empty(self, state: ParsingState):
        """
        Set the appropriate field of the parsing state to be empty.
        Used when a variable appears in an optional group which is not parsed.
        """
        return


class TypeableDirective(Directive, ABC):
    """
    Directives which can be used to set or get types.
    """

    @abstractmethod
    def parse_single_type(self, parser: Parser, state: ParsingState) -> None: ...

    @abstractmethod
    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]: ...


class VariadicTypeableDirective(TypeableDirective, AnchorableDirective, ABC):
    """
    Directives which can set or get multiple types.
    """

    @abstractmethod
    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool: ...

    @abstractmethod
    def set_types_empty(self, state: ParsingState) -> None: ...


@dataclass(frozen=True)
class TypeDirective(FormatDirective):
    """
    A directive which parses the type of a typeable directive, with format:
      type-directive ::= type(typeable-directive)
    """

    inner: TypeableDirective

    def parse(self, parser: Parser, state: ParsingState) -> None:
        self.inner.parse_single_type(parser, state)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(self.inner.get_types(op), printer.print_attribute)
        state.last_was_punctuation = False
        state.should_emit_space = True


@dataclass(frozen=True)
class VariadicTypeDirective(VariadicLikeFormatDirective):
    """
    A directive which parses the type of a variadic typeable directive, with format:
      type-directive ::= type(typeable-directive)
    """

    inner: VariadicTypeableDirective

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        return self.inner.parse_many_types(parser, state)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_list(self.inner.get_types(op), printer.print_attribute)
        state.last_was_punctuation = False
        state.should_emit_space = True

    def is_present(self, op: IRDLOperation) -> bool:
        return self.inner.is_present(op)

    def set_empty(self, state: ParsingState):
        self.inner.set_types_empty(state)


@dataclass(frozen=True)
class VariableDirective(Directive, ABC):
    """
    A variable directive, with the following format:
      variable-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    name: str
    """The variable name. This is only used for error message reporting."""
    index: int
    """Index of the variable(operand or result) definition."""


class VariadicVariable(VariableDirective, AnchorableDirective, ABC):
    def is_present(self, op: IRDLOperation) -> bool:
        return bool(getattr(op, self.name))


class OptionalVariable(VariableDirective, AnchorableDirective, ABC):
    def is_present(self, op: IRDLOperation) -> bool:
        return getattr(op, self.name) is not None


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
class OperandVariable(VariableDirective, FormatDirective, TypeableDirective):
    """
    An operand variable, with the following format:
      operand-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def parse(self, parser: Parser, state: ParsingState) -> None:
        operand = parser.parse_unresolved_operand()
        state.operands[self.index] = operand

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.operand_types[self.index] = parser.parse_type()

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        printer.print_ssa_value(getattr(op, self.name))
        state.last_was_punctuation = False
        state.should_emit_space = True

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return (getattr(op, self.name).type,)


class VariadicOperandDirective(
    VariadicLikeFormatDirective, VariadicTypeableDirective, ABC
):
    """
    Base class for typechecking.
    A variadic operand directive cannot follow another variadic operand directive.
    """


@dataclass(frozen=True)
class VariadicOperandVariable(VariadicVariable, VariadicOperandDirective):
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
        state.operands[self.index] = operands
        return bool(operands)

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.operand_types[self.index] = (parser.parse_type(),)

    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool:
        types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        ret = types is None
        if ret:
            types = ()
        state.operand_types[self.index] = types
        return ret

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        operand = getattr(op, self.name)
        if operand:
            printer.print_list(operand, printer.print_ssa_value)
            state.last_was_punctuation = False
            state.should_emit_space = True

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return getattr(op, self.name).types

    def set_empty(self, state: ParsingState):
        state.operands[self.index] = ()

    def set_types_empty(self, state: ParsingState) -> None:
        state.operand_types[self.index] = ()


class OptionalOperandVariable(OptionalVariable, VariadicOperandDirective):
    """
    An optional operand variable, with the following format:
      operand-directive ::= ( percent-ident )?
    The directive will request a space to be printed after.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        operand = parser.parse_optional_unresolved_operand()
        if operand is None:
            operand = ()
        state.operands[self.index] = operand
        return bool(operand)

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.operand_types[self.index] = (parser.parse_type(),)

    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool:
        type = parser.parse_optional_type()
        ret = type is None
        if ret:
            type = ()
        state.operand_types[self.index] = type
        return ret

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space or not state.last_was_punctuation:
            printer.print(" ")
        operand = getattr(op, self.name)
        if operand:
            printer.print_ssa_value(operand)
            state.last_was_punctuation = False
            state.should_emit_space = True

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        operand = getattr(op, self.name)
        if operand:
            return (operand.type,)
        return ()

    def set_empty(self, state: ParsingState):
        state.operands[self.index] = ()

    def set_types_empty(self, state: ParsingState) -> None:
        state.operand_types[self.index] = ()


_T = TypeVar("_T")


@dataclass(frozen=True)
class OperandsOrResultDirective(VariadicTypeableDirective, ABC):
    """
    Base class for the 'operands' and 'results' directives.
    """

    variadic_index: tuple[bool, int] | None
    """
    Represents the position of a (single) variadic variable, with the boolean
    representing whether it is optional
    """

    def _set_using_variadic_index(
        self,
        field: list[_T | None | Sequence[_T]],
        field_name: str,
        set_to: Sequence[_T],
    ) -> str | None:
        if self.variadic_index is None:
            if len(set_to) != len(field):
                return f"Expected {len(field)} {field_name} but found {len(set_to)}"
            field = [o for o in set_to]  # Copy needed as list is not covariant
            return

        is_optional, var_position = self.variadic_index
        var_length = len(set_to) - len(field) + 1
        if var_length < 0:
            return f"Expected at least {len(field) - 1} {field_name} but found {len(set_to)}"
        if var_length > 1 and is_optional:
            return f"Expected at most {len(field)} {field_name} but found {len(set_to)}"
        field[:var_position] = set_to[:var_position]
        field[var_position] = set_to[var_position : var_position + var_length]
        field[var_position + 1 :] = set_to[var_position + var_length :]


class OperandsDirective(VariadicOperandDirective, OperandsOrResultDirective):
    """
    An operands directive, with the following format:
      operands-directive ::= operands
    Prints each operand of the operation, inserting a comma between each.
    """

    def parse_optional(self, parser: Parser, state: ParsingState) -> bool:
        pos_start = parser.pos
        operands = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_unresolved_operand,
                parser.parse_unresolved_operand,
            )
            or []
        )

        if s := self._set_using_variadic_index(state.operands, "operands", operands):
            parser.raise_error(s, at_position=pos_start, end_position=parser.pos)
        return bool(operands)

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        if len(state.operand_types) > 1:
            parser.raise_error("Expected multiple types but received one.")
        state.operand_types[0] = parser.parse_type()

    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool:
        pos_start = parser.pos
        types = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_type, parser.parse_type
            )
            or []
        )

        if s := self._set_using_variadic_index(
            state.operand_types, "operand types", types
        ):
            parser.raise_error(s, at_position=pos_start, end_position=parser.pos)
        return bool(types)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if op.operands:
            if state.should_emit_space or not state.last_was_punctuation:
                printer.print(" ")
            printer.print_list(op.operands, printer.print_ssa_value)
            state.last_was_punctuation = False
            state.should_emit_space = True

    def set_types_empty(self, state: ParsingState) -> None:
        state.operand_types = [() for _ in state.operand_types]

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return op.operand_types

    def set_empty(self, state: ParsingState):
        state.operands = [() for _ in state.operands]

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(op.operands)


@dataclass(frozen=True)
class ResultVariable(VariableDirective, TypeableDirective):
    """
    An result variable, with the following format:
      result-directive ::= dollar-ident
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.result_types[self.index] = parser.parse_type()

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return (getattr(op, self.name).type,)


@dataclass(frozen=True)
class VariadicResultVariable(VariadicVariable, VariadicTypeableDirective):
    """
    A variadic result variable, with the following format:
      result-directive ::= percent-ident (( `,` percent-id )* )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.result_types[self.index] = (parser.parse_type(),)

    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool:
        types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        ret = types is None
        if ret:
            types = ()
        state.result_types[self.index] = types
        return ret

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return getattr(op, self.name).types

    def set_types_empty(self, state: ParsingState) -> None:
        state.result_types[self.index] = ()


class OptionalResultVariable(OptionalVariable, VariadicTypeableDirective):
    """
    An optional result variable, with the following format:
      result-directive ::= ( percent-ident )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.result_types[self.index] = (parser.parse_type(),)

    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool:
        type = parser.parse_optional_type()
        ret = type is None
        if ret:
            type = ()
        state.result_types[self.index] = type
        return ret

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        res = getattr(op, self.name)
        if res:
            return (res.type,)
        return ()

    def set_types_empty(self, state: ParsingState) -> None:
        state.result_types[self.index] = ()


class ResultsDirective(OperandsOrResultDirective):
    """
    A results directive, with the following format:
      results-directive ::= results
    A typeable directive which processes the result types of the operation.
    """

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        if len(state.result_types) > 1:
            parser.raise_error("Expected multiple types but received one.")
        state.result_types[0] = parser.parse_type()

    def parse_many_types(self, parser: Parser, state: ParsingState) -> bool:
        pos_start = parser.pos
        types = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_type, parser.parse_type
            )
            or []
        )

        if s := self._set_using_variadic_index(
            state.result_types, "result types", types
        ):
            parser.raise_error(s, at_position=pos_start, end_position=parser.pos)
        return bool(types)

    def set_types_empty(self, state: ParsingState) -> None:
        state.result_types = [() for _ in state.operand_types]

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return op.result_types

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(op.results)


class RegionDirective(OptionallyParsableDirective, ABC):
    """
    Baseclass to help keep typechecking simple.
    RegionDirective is for any RegionVariable, which are all OptionallyParsable.
    """


class VariadicRegionDirective(RegionDirective, VariadicLikeFormatDirective, ABC):
    """
    Base class for typechecking.
    A variadic region directive cannot follow another variadic region directive.
    """


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
class VariadicRegionVariable(VariadicRegionDirective, VariadicVariable):
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

    def set_empty(self, state: ParsingState):
        state.regions[self.index] = ()


class OptionalRegionVariable(VariadicRegionDirective, OptionalVariable):
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

    def set_empty(self, state: ParsingState):
        state.regions[self.index] = ()


class VariadicSuccessorDirective(VariadicLikeFormatDirective, ABC):
    """
    Base class for type checking.
    A variadic successor directive cannot follow another variadic successor directive.
    """


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


class VariadicSuccessorVariable(VariadicSuccessorDirective, VariadicVariable):
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

    def set_empty(self, state: ParsingState):
        state.successors[self.index] = ()


class OptionalSuccessorVariable(VariadicSuccessorDirective, OptionalVariable):
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

    def set_empty(self, state: ParsingState):
        state.successors[self.index] = ()


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
            assert issubclass(unique_base, TypedAttribute)
            attr = unique_base.parse_with_type(parser, self.unique_type)
        elif issubclass(
            unique_base,
            ParametrizedAttribute,
        ):
            attr = unique_base.new(unique_base.parse_parameters(parser))
        elif issubclass(unique_base, Data):
            attr = unique_base.new(  # pyright: ignore[reportUnknownVariableType]
                unique_base.parse_parameter(parser)
            )
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
            assert isinstance(attr, TypedAttribute)
            return attr.print_without_type(printer)
        if self.unique_base is None:
            return printer.print_attribute(attr)
        if isinstance(attr, ParametrizedAttribute):
            return attr.print_parameters(printer)
        if isinstance(attr, Data):
            return attr.print_parameter(printer)
        raise ValueError("Attributes must be Data or ParameterizedAttribute!")


@dataclass(frozen=True)
class DefaultValuedAttributeVariable(AttributeVariable, AnchorableDirective):
    """
    An attribute variable with default value, with the following format:
      result-directive ::= dollar-ident
    The directive will request a space to be printed right after.
    """

    default_value: Attribute

    def is_present(self, op: IRDLOperation) -> bool:
        if self.is_property:
            attr = op.properties.get(self.name)
        else:
            attr = op.attributes.get(self.name)
        return attr is not None and attr != self.default_value


class OptionalAttributeVariable(AttributeVariable, AnchorableDirective):
    """
    An optional attribute variable, with the following format:
      operand-directive ::= ( percent-ident )?
    The directive will request a space to be printed after.
    """

    def is_present(self, op: IRDLOperation) -> bool:
        if self.is_property:
            attr = op.properties.get(self.name)
        else:
            attr = op.attributes.get(self.name)
        return attr is not None


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
                if isinstance(element, VariadicLikeFormatDirective):
                    element.set_empty(state)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if self.anchor.is_present(op):
            for element in (
                *self.then_whitespace,
                self.then_first,
                *self.then_elements,
            ):
                element.print(printer, state, op)
