"""
This file contains the data structures necessary for the parsing and printing
of the MLIR declarative assembly format defined at
https://mlir.llvm.org/docs/DefiningDialects/Operations/#declarative-assembly-format .
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    AnyFloat,
    BytesAttr,
    DenseArrayBase,
    IntegerType,
    StringAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypedAttribute,
)
from xdsl.irdl import (
    BaseAccessor,
    ConstraintContext,
    IRDLOperation,
    IRDLOperationInvT,
    OpDef,
    OptionalDef,
    Successor,
    VarIRConstruct,
    VarOperand,
    get_construct_defs,
    is_const_classvar,
    verify_variadic_same_size,
)
from xdsl.parser import Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import PyRDLError, VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.mlir_lexer import PunctuationSpelling


@dataclass
class ParsingState:
    """
    State carried during the parsing of an operation using the declarative assembly
    format.
    It contains the elements that have already been parsed.
    """

    operands: list[None | Sequence[UnresolvedOperand]]
    operand_types: list[None | Sequence[Attribute]]
    result_types: list[None | Sequence[Attribute]]
    regions: list[None | Sequence[Region]]
    successors: list[None | Sequence[Successor]]
    attributes: dict[str, Attribute]
    properties: dict[str, Attribute]
    op_type: type[IRDLOperation]
    context: ConstraintContext

    def __init__(self, op_type: type[IRDLOperation]):
        op_def = op_type.get_irdl_definition()
        self.op_type = op_type
        self.operands = [None] * len(op_def.operands)
        self.operand_types = [None] * len(op_def.operands)
        self.result_types = [None] * len(op_def.results)
        self.regions = [None] * len(op_def.regions)
        self.successors = [None] * len(op_def.successors)
        self.attributes = {}
        self.properties = {}
        self.context = ConstraintContext()


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

    def print_whitespace(self, printer: Printer):
        """
        Handles whitespace printing for the majority of format directives.
        """
        if self.should_emit_space or not self.last_was_punctuation:
            printer.print_string(" ")
        self.should_emit_space = True
        self.last_was_punctuation = False


@dataclass(frozen=True)
class FormatProgram:
    """
    The toplevel data structure of a declarative assembly format program.
    It is used to parse and print an operation.
    """

    stmts: tuple[FormatDirective, ...]
    """The statements composing the program. They are executed in order."""

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
        state = ParsingState(op_type)
        for stmt in self.stmts:
            stmt.parse(parser, state)

        # Get constraint variables from the parsed operand and result types
        self.resolve_constraint_variables(state, op_def)

        # Infer operand types that should be inferred
        assert None not in state.operands
        unresolved_operands = cast(list[Sequence[UnresolvedOperand]], state.operands)
        operand_types = self.resolve_operand_types(state, op_def)

        # Infer result types that should be inferred
        result_types = self.resolve_result_types(state, op_def)

        # Resolve all operands
        operands = tuple(
            parser.resolve_operands(uo, ot, parser.pos)
            for uo, ot in zip(unresolved_operands, operand_types, strict=True)
        )

        return op_type.build(
            result_types=result_types,
            operands=operands,
            attributes=state.attributes,
            properties=state.properties,
            regions=state.regions,
            successors=state.successors,
        )

    def resolve_constraint_variables(self, state: ParsingState, op_def: OpDef):
        """
        Runs verification on the parsed parts of the operation, adding the resolved value
        of each constraint variable to the `ConstraintContext` `state.context`.
        """
        ctx = state.context

        for operand, operand_type, (_, operand_def) in zip(
            state.operands, state.operand_types, op_def.operands, strict=True
        ):
            length = len(operand) if isinstance(operand, Sequence) else 1
            operand_def.constr.verify_length(length, ctx)
            if operand_type is None:
                continue
            if isinstance(operand_type, Attribute):
                operand_type = (operand_type,)
            operand_def.constr.verify(operand_type, ctx)

        for result_type, (_, result_def) in zip(
            state.result_types, op_def.results, strict=True
        ):
            if result_type is None:
                continue
            if isinstance(result_type, Attribute):
                result_type = (result_type,)
            result_def.constr.verify(result_type, ctx)

        for prop_name, prop_def in op_def.properties.items():
            if isinstance(prop_def, OptionalDef) and prop_def.default_value is None:
                continue
            attr = state.properties.get(prop_name, prop_def.default_value)
            if attr is None:
                continue
            prop_def.constr.verify(attr, ctx)

        for attr_name, attr_def in op_def.attributes.items():
            if isinstance(attr_def, OptionalDef) and attr_def.default_value is None:
                continue
            attr = state.attributes.get(attr_name, attr_def.default_value)
            if attr is None:
                continue
            attr_def.constr.verify(attr, ctx)

    def resolve_operand_types(
        self, state: ParsingState, op_def: OpDef
    ) -> Sequence[Sequence[Attribute]]:
        """
        Use the inferred type resolutions to fill missing operand types from other parsed
        types.
        """
        return tuple(
            operand_def.constr.infer(
                state.context,
                length=len(cast(Sequence[Any], operand)),
            )
            if operand_type is None
            else operand_type
            for operand_type, operand, (_, operand_def) in zip(
                state.operand_types, state.operands, op_def.operands, strict=True
            )
        )

    def resolve_result_types(
        self, state: ParsingState, op_def: OpDef
    ) -> Sequence[Sequence[Attribute]]:
        """
        Use the inferred type resolutions to fill missing result types from other parsed
        types.
        """
        return tuple(
            result_def.constr.infer(
                state.context,
                length=None,
            )
            if result_type is None
            else result_type
            for result_type, (_, result_def) in zip(
                state.result_types, op_def.results, strict=True
            )
        )

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

    def is_present(self, op: IRDLOperation) -> bool:
        """
        Check if the directive is present in the input.
        """
        return True

    def is_anchorable(self) -> bool:
        """
        Can appear as an anchor in an optional group.
        """
        return False

    def is_variadic_like(self) -> bool:
        """
        Variadic-like format directives parse a comma separated list, and cannot be
        followed by `,` directive.
        """
        return False

    def is_optional_like(self) -> bool:
        """
        Directives that successfully parse the empty string.
        The first element in an optional group must be optional-like.
        """
        return self.is_variadic_like()


class FormatDirective(Directive, ABC):
    """A format directive for operation format."""

    @abstractmethod
    def parse(self, parser: Parser, state: ParsingState) -> bool:
        """
        Parses the directive, returning True if input was consumed.
        """
        ...

    @abstractmethod
    def print(
        self, printer: Printer, state: PrintingState, op: IRDLOperation
    ) -> None: ...

    def set_empty(self, state: ParsingState):
        """
        Set the appropriate field of the parsing state to be empty.
        Used when a variable appears in an optional group which is not parsed.
        """
        return


class CustomDirective(FormatDirective, ABC):
    """
    A user defined assembly format directive.
    A custom directive can have multiple parameters, whose types should be
    declared in the `parameters` field.
    """

    parameters: ClassVar[dict[str, type[FormatDirective]]]


CustomDirectiveInvT = TypeVar("CustomDirectiveInvT", bound=CustomDirective)


def irdl_custom_directive(cls: type[CustomDirectiveInvT]) -> type[CustomDirectiveInvT]:
    """Decorator used on custom directives to define the `parameters` class variable."""

    cls.parameters = {}
    param_types = inspect.get_annotations(cls, eval_str=True)
    for field_name, ty in param_types.items():
        if is_const_classvar(field_name, ty, PyRDLError):
            continue
        if not issubclass(ty, FormatDirective):
            raise PyRDLError(
                f"Custom directive {cls.__name__} has parameter {field_name} which is not a format directive."
            )
        cls.parameters[field_name] = ty
    return dataclass(frozen=True)(cls)


class TypeableDirective(Directive, ABC):
    """
    Directives which can be used to set or get types.
    """

    @abstractmethod
    def set_types(self, state: ParsingState, types: Sequence[Attribute]) -> None:
        """
        Set the types for this directive to the given sequence.
        """
        ...

    @abstractmethod
    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        """
        Parses types for the directive, returning True if input was consumed.
        """
        ...

    @abstractmethod
    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        """
        Parse exactly one type for the directive.
        """

    @abstractmethod
    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]: ...


@dataclass(frozen=True)
class TypeDirective(FormatDirective):
    """
    A directive which parses the type of a typeable directive, with format:
      type-directive ::= type(typeable-directive)
    """

    inner: TypeableDirective

    def set(self, state: ParsingState, types: Sequence[Attribute]):
        self.inner.set_types(state, types)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        return self.inner.parse_types(parser, state)

    def get(self, op: IRDLOperation) -> Sequence[Attribute]:
        return self.inner.get_types(op)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        types = self.inner.get_types(op)
        if not types:
            return
        state.print_whitespace(printer)
        printer.print_list(types, printer.print_attribute)

    def is_present(self, op: IRDLOperation) -> bool:
        return self.inner.is_present(op)

    def is_anchorable(self) -> bool:
        return self.inner.is_anchorable()

    def is_variadic_like(self) -> bool:
        return self.inner.is_variadic_like()

    def is_optional_like(self) -> bool:
        return self.inner.is_optional_like()

    def set_empty(self, state: ParsingState):
        self.set(state, ())


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


class VariadicVariable(VariableDirective, ABC):
    def is_present(self, op: IRDLOperation) -> bool:
        return bool(getattr(op, self.name))

    def is_anchorable(self) -> bool:
        return True

    def is_variadic_like(self) -> bool:
        return True


class OptionalVariable(VariableDirective, ABC):
    def is_present(self, op: IRDLOperation) -> bool:
        return getattr(op, self.name) is not None

    def is_anchorable(self) -> bool:
        return True

    def is_optional_like(self) -> bool:
        return True


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

    expected_properties: set[str]
    """
    Properties that should be printed and parsed as part of this attr-dict.
    This is used to keep compatibility with MLIR which allows that.
    """

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if self.with_keyword:
            res = parser.parse_optional_attr_dict_with_keyword()
            if res is None:
                res = {}
            else:
                res = dict(res.data)
        else:
            res = parser.parse_optional_attr_dict()
        defined_reserved_keys = self.reserved_attr_names & res.keys()
        if defined_reserved_keys:
            parser.raise_error(
                f"attributes {', '.join(defined_reserved_keys)} are defined in other parts of the "
                "assembly format, and thus should not be defined in the attribute "
                "dictionary."
            )

        props = tuple(k for k in res.keys() if k in self.expected_properties)
        for name in props:
            state.properties[name] = res.pop(name)
        state.attributes |= res
        return bool(res) or bool(props)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if not op.attributes.keys().isdisjoint(self.expected_properties):
            raise ValueError(
                "Cannot print attributes and properties with the same name "
                "in a single dictionary"
            )
        op_def = op.get_irdl_definition()
        dictionary = op.attributes | {
            k: v for k, v in op.properties.items() if k in self.expected_properties
        }
        defs = {
            x: op_def.properties[x] for x in self.expected_properties
        } | op_def.attributes

        reserved_or_default = self.reserved_attr_names.union(
            name
            for name, d in defs.items()
            if d.default_value is not None and dictionary.get(name) == d.default_value
        )

        printed = printer.print_op_attributes(
            dictionary,
            reserved_attr_names=reserved_or_default,
            print_keyword=self.with_keyword,
        )

        if printed:
            state.last_was_punctuation = False
            state.should_emit_space = True

    def is_optional_like(self) -> bool:
        return True


class OperandDirective(FormatDirective, TypeableDirective, ABC):
    """
    Base class for operand directives to aid typechecking.
    """

    pass


@dataclass(frozen=True)
class OperandVariable(VariableDirective, OperandDirective):
    """
    An operand variable, with the following format:
      operand-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, operand: UnresolvedOperand):
        state.operands[self.index] = (operand,)

    def set_types(self, state: ParsingState, types: Sequence[Attribute]):
        state.operand_types[self.index] = types

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        operand = parser.parse_unresolved_operand()
        self.set(state, operand)
        return True

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        self.set_types(state, (parser.parse_type(),))
        return True

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        self.parse_types(parser, state)

    def get(self, op: IRDLOperation) -> SSAValue:
        return getattr(op, self.name)

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return (self.get(op).type,)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        printer.print_ssa_value(self.get(op))


@dataclass(frozen=True)
class VariadicOperandVariable(VariadicVariable, OperandDirective):
    """
    A variadic operand variable, with the following format:
      operand-directive ::= ( percent-ident ( `,` percent-id )* )?
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, operands: Sequence[UnresolvedOperand]):
        state.operands[self.index] = operands

    def set_types(self, state: ParsingState, types: Sequence[Attribute]):
        state.operand_types[self.index] = types

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        operands = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_unresolved_operand, parser.parse_unresolved_operand
        )
        if operands is None:
            operands = []
        self.set(state, operands)
        return bool(operands)

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        ret = types is None
        if ret:
            types = ()
        self.set_types(state, types)
        return ret

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.operand_types[self.index] = (parser.parse_type(),)

    def get(self, op: IRDLOperation) -> VarOperand:
        return getattr(op, self.name)

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return self.get(op).types

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand = self.get(op)
        if not operand:
            return
        state.print_whitespace(printer)
        printer.print_list(operand, printer.print_ssa_value)

    def set_empty(self, state: ParsingState):
        self.set(state, ())


class OptionalOperandVariable(OptionalVariable, OperandDirective):
    """
    An optional operand variable, with the following format:
      operand-directive ::= ( percent-ident )?
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, operand: UnresolvedOperand | None):
        state.operands[self.index] = () if operand is None else (operand,)

    def set_types(self, state: ParsingState, types: Sequence[Attribute]):
        state.operand_types[self.index] = types

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        operand = parser.parse_optional_unresolved_operand()
        self.set(state, operand)
        return bool(operand)

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        type = parser.parse_optional_type()
        self.set_types(state, () if type is None else (type,))
        return type is None

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        self.set_types(state, (parser.parse_type(),))

    def get(self, op: IRDLOperation) -> SSAValue | None:
        return getattr(op, self.name)

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        operand = self.get(op)
        if operand:
            return (operand.type,)
        return ()

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        operand = self.get(op)
        if not operand:
            return
        state.print_whitespace(printer)
        printer.print_ssa_value(operand)

    def set_empty(self, state: ParsingState):
        self.set(state, None)


_T = TypeVar("_T")


@dataclass(frozen=True)
class OperandsOrResultDirective(TypeableDirective, ABC):
    """
    Base class for the 'operands' and 'results' directives.
    """

    def is_variadic_like(self) -> bool:
        return True

    def is_anchorable(self) -> bool:
        return True

    def _set_using_variadic_index(
        self,
        op_type: type[IRDLOperation],
        field: list[None | Sequence[_T]],
        construct: VarIRConstruct,
        field_name: str,
        set_to: Sequence[_T],
    ) -> None:
        op_def = op_type.get_irdl_definition()
        verify_variadic_same_size(len(set_to), op_def, construct, field_name)

        for i, (name, _) in enumerate(get_construct_defs(op_def, construct)):
            accessor = op_type.__dict__[name]
            assert isinstance(accessor, BaseAccessor)
            res = accessor.index(set_to)
            if res is None:
                res = ()
            if not isinstance(res, Sequence):
                res = (res,)
            field[i] = res


class OperandsDirective(OperandsOrResultDirective, FormatDirective):
    """
    An operands directive, with the following format:
      operands-directive ::= operands
    Prints each operand of the operation, inserting a comma between each.
    """

    def set_types(self, state: ParsingState, types: Sequence[Attribute]) -> None:
        self._set_using_variadic_index(
            state.op_type,
            state.operand_types,
            VarIRConstruct.OPERAND,
            "operand type",
            types,
        )

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        pos_start = parser.pos
        operands = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_unresolved_operand,
                parser.parse_unresolved_operand,
            )
            or []
        )

        try:
            self._set_using_variadic_index(
                state.op_type,
                state.operands,
                VarIRConstruct.OPERAND,
                "operand",
                operands,
            )
        except VerifyException as e:
            parser.raise_error(str(e), at_position=pos_start, end_position=parser.pos)
        return bool(operands)

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        pos_start = parser.pos
        types = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_type, parser.parse_type
            )
            or []
        )

        try:
            self.set_types(state, types)
        except VerifyException as e:
            parser.raise_error(str(e), at_position=pos_start, end_position=parser.pos)
        return bool(types)

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        pos_start = parser.pos
        if s := self.set_types(state, (parser.parse_type(),)):
            parser.raise_error(s, at_position=pos_start, end_position=parser.pos)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if op.operands:
            state.print_whitespace(printer)
        printer.print_list(op.operands, printer.print_ssa_value)

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

    def set_types(self, state: ParsingState, types: Sequence[Attribute]):
        state.result_types[self.index] = types

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        self.set_types(state, (parser.parse_type(),))
        return True

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        self.parse_types(parser, state)

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return (getattr(op, self.name).type,)


@dataclass(frozen=True)
class VariadicResultVariable(VariadicVariable, TypeableDirective):
    """
    A variadic result variable, with the following format:
      result-directive ::= percent-ident (( `,` percent-id )* )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def set_types(self, state: ParsingState, types: Sequence[Attribute]):
        state.result_types[self.index] = types

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        types = parser.parse_optional_undelimited_comma_separated_list(
            parser.parse_optional_type, parser.parse_type
        )
        self.set_types(state, () if types is None else types)
        return types is None

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        state.result_types[self.index] = (parser.parse_type(),)

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return getattr(op, self.name).types


class OptionalResultVariable(OptionalVariable, TypeableDirective):
    """
    An optional result variable, with the following format:
      result-directive ::= ( percent-ident )?
    This directive can not be used for parsing and printing directly, as result
    parsing is not handled by the custom operation parser.
    """

    def set_types(self, state: ParsingState, types: Sequence[Attribute]):
        state.result_types[self.index] = types

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        type = parser.parse_optional_type()
        self.set_types(state, () if type is None else (type,))
        return type is not None

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        self.set_types(state, (parser.parse_type(),))

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        res = getattr(op, self.name)
        if res:
            return (res.type,)
        return ()


class ResultsDirective(OperandsOrResultDirective):
    """
    A results directive, with the following format:
      results-directive ::= results
    A typeable directive which processes the result types of the operation.
    """

    def set_types(self, state: ParsingState, types: Sequence[Attribute]) -> None:
        self._set_using_variadic_index(
            state.op_type,
            state.result_types,
            VarIRConstruct.RESULT,
            "result type",
            types,
        )

    def parse_types(self, parser: Parser, state: ParsingState) -> bool:
        pos_start = parser.pos
        types = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_type, parser.parse_type
            )
            or []
        )

        try:
            self.set_types(state, types)
        except VerifyException as e:
            parser.raise_error(str(e), at_position=pos_start, end_position=parser.pos)
        return bool(types)

    def parse_single_type(self, parser: Parser, state: ParsingState) -> None:
        pos_start = parser.pos
        if s := self.set_types(state, (parser.parse_type(),)):
            parser.raise_error(s, at_position=pos_start, end_position=parser.pos)

    def get_types(self, op: IRDLOperation) -> Sequence[Attribute]:
        return op.result_types

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(op.results)


@dataclass(frozen=True)
class FunctionalTypeDirective(FormatDirective):
    """
    A directive which parses a functional type, with format:
      functional-type-directive ::= functional-type(typeable-directive, typeable-directive)
    A functional type is either of the form
      `(` type-list `)` `->` `(` type-list `)`
    or
      `(` type-list `)` `->` type
    where type-list is a comma separated list of types (or the empty string to signify the empty list).
    The second format is preferred for printing when possible.
    """

    operand_typeable_directive: TypeableDirective
    result_typeable_directive: TypeableDirective

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        if not parser.parse_optional_punctuation("("):
            return False
        self.operand_typeable_directive.parse_types(parser, state)
        parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        if parser.parse_optional_punctuation("("):
            self.result_typeable_directive.parse_types(parser, state)
            parser.parse_punctuation(")")
        else:
            self.result_typeable_directive.parse_single_type(parser, state)
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        with printer.in_parens():
            printer.print_list(
                self.operand_typeable_directive.get_types(op), printer.print_attribute
            )
        printer.print_string(" -> ")
        result_types = self.result_typeable_directive.get_types(op)
        if len(result_types) == 1:
            printer.print_attribute(result_types[0])
        else:
            with printer.in_parens():
                printer.print_list(result_types, printer.print_attribute)


class RegionDirective(FormatDirective, ABC):
    """
    Baseclass to help keep typechecking simple.
    """


@dataclass(frozen=True)
class RegionVariable(RegionDirective, VariableDirective):
    """
    A region variable, with the following format:
      region-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, region: Region):
        state.regions[self.index] = (region,)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        self.set(state, parser.parse_region())
        return True

    def get(self, op: IRDLOperation) -> Region:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        printer.print_region(self.get(op))

    def is_anchorable(self) -> bool:
        return True

    def set_empty(self, state: ParsingState):
        self.set(state, Region())

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(self.get(op).blocks)


@dataclass(frozen=True)
class VariadicRegionVariable(RegionDirective, VariadicVariable):
    """
    A variadic region variable, with the following format:
      region-directive ::= dollar-ident

    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, region: Sequence[Region]):
        state.regions[self.index] = region

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        regions: list[Region] = []
        current_region = parser.parse_optional_region()
        while current_region is not None:
            regions.append(current_region)
            current_region = parser.parse_optional_region()

        self.set(state, regions)
        return bool(regions)

    def get(self, op: IRDLOperation) -> Sequence[Region]:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        region = self.get(op)
        if not region:
            return
        state.print_whitespace(printer)
        printer.print_list(region, printer.print_region, delimiter=" ")

    def set_empty(self, state: ParsingState):
        self.set(state, ())


class OptionalRegionVariable(RegionDirective, OptionalVariable):
    """
    An optional region variable, with the following format:
      region-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, region: Region | None):
        state.regions[self.index] = () if region is None else (region,)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        region = parser.parse_optional_region()
        self.set(state, region)
        return region is not None

    def get(self, op: IRDLOperation) -> Region | None:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        region = self.get(op)
        if not region:
            return
        state.print_whitespace(printer)
        printer.print_region(region)

    def set_empty(self, state: ParsingState):
        self.set(state, None)


@dataclass(frozen=True)
class AnchorRegionVariable(RegionDirective, VariableDirective):
    """
    A special case for non-optional anchor regions, which treat the empty region as a special case.
    """

    def set(self, state: ParsingState, region: Region):
        state.regions[self.index] = (region,)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        region = parser.parse_optional_region()
        if region is None:
            region = Region()
        self.set(state, region)
        return True

    def get(self, op: IRDLOperation) -> Region:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        printer.print_region(self.get(op))

    def is_anchorable(self) -> bool:
        return True

    def set_empty(self, state: ParsingState):
        self.set(state, Region())

    def is_present(self, op: IRDLOperation) -> bool:
        return bool(self.get(op).blocks)

    def is_optional_like(self) -> bool:
        return True


class SuccessorDirective(FormatDirective, ABC):
    """
    Base class for type checking.
    A variadic successor directive cannot follow another variadic successor directive.
    """


class SuccessorVariable(VariableDirective, SuccessorDirective):
    """
    A successor variable, with the following format:
      successor-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, successor: Successor):
        state.successors[self.index] = (successor,)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        successor = parser.parse_successor()

        self.set(state, successor)

        return True

    def get(self, op: IRDLOperation) -> Successor:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        state.print_whitespace(printer)
        printer.print_block_name(self.get(op))


class VariadicSuccessorVariable(VariadicVariable, SuccessorDirective):
    """
    A variadic successor variable, with the following format:
      successor-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, successors: Sequence[Successor]):
        state.successors[self.index] = successors

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        successors = (
            parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_successor, parser.parse_successor
            )
            or []
        )

        self.set(state, successors)

        return bool(successors)

    def get(self, op: IRDLOperation) -> Sequence[Successor]:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        successor = self.get(op)
        if not successor:
            return
        state.print_whitespace(printer)
        printer.print_list(successor, printer.print_block_name)

    def set_empty(self, state: ParsingState):
        self.set(state, ())


class OptionalSuccessorVariable(OptionalVariable, SuccessorDirective):
    """
    An optional successor variable, with the following format:
      successor-directive ::= dollar-ident
    The directive will request a space to be printed after.
    """

    def set(self, state: ParsingState, successor: Successor | None):
        state.successors[self.index] = () if successor is None else (successor,)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        successor = parser.parse_optional_successor()
        self.set(state, successor)
        return successor is not None

    def get(self, op: IRDLOperation) -> Successor | None:
        return getattr(op, self.name)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        successor = self.get(op)
        if not successor:
            return
        state.print_whitespace(printer)
        printer.print_block_name(successor)

    def set_empty(self, state: ParsingState):
        self.set(state, None)


@dataclass(frozen=True)
class AttributeVariable(FormatDirective):
    """
    An attribute variable, with the following format:
      attribute-variable ::= dollar-ident
    The directive will request a space to be printed right after.
    """

    name: str
    """The attribute name as it should be in the attribute or property dictionary."""
    is_property: bool
    """Should this attribute be put in the attribute or property dictionary."""
    is_optional: bool
    """Is this attribute optional in the operation definition."""
    default_value: Attribute | None

    def set(self, state: ParsingState, attr: Attribute):
        if self.is_property:
            state.properties[self.name] = attr
        else:
            state.attributes[self.name] = attr

    def parse_attr(self, parser: Parser) -> Attribute | None:
        if self.is_optional:
            return parser.parse_optional_attribute()
        else:
            return parser.parse_attribute()

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        attr = self.parse_attr(parser)
        if attr is None:
            return False
        self.set(state, attr)
        return True

    def get(self, op: IRDLOperation) -> Attribute | None:
        if self.is_property:
            return op.properties.get(self.name)
        else:
            return op.attributes.get(self.name)

    def print_attr(self, printer: Printer, attr: Attribute) -> None:
        return printer.print_attribute(attr)

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        attr = self.get(op)

        if attr is None or attr == self.default_value:
            return

        state.print_whitespace(printer)

        self.print_attr(printer, attr)

    def is_present(self, op: IRDLOperation) -> bool:
        attr = self.get(op)
        return attr is not None and attr != self.default_value

    def is_anchorable(self) -> bool:
        return self.is_optional or self.default_value is not None

    def is_optional_like(self) -> bool:
        return self.is_optional


@dataclass(frozen=True)
class UniqueBaseAttributeVariable(AttributeVariable):
    unique_base: type[Attribute]
    """The known base class of the Attribute, if any."""

    def parse_attr(self, parser: Parser) -> Attribute | None:
        unique_base = self.unique_base

        if issubclass(unique_base, ParametrizedAttribute):
            return unique_base.new(unique_base.parse_parameters(parser))
        elif issubclass(unique_base, Data):
            unique_base = cast(type[Data[Any]], unique_base)
            return unique_base.new(unique_base.parse_parameter(parser))
        else:
            raise ValueError("Attributes must be Data or ParameterizedAttribute.")

    def print_attr(self, printer: Printer, attr: Attribute) -> None:
        if isinstance(attr, ParametrizedAttribute):
            return attr.print_parameters(printer)
        if isinstance(attr, Data):
            return attr.print_parameter(printer)
        raise ValueError("Attributes must be Data or ParameterizedAttribute!")


@dataclass(frozen=True)
class TypedAttributeVariable(UniqueBaseAttributeVariable):
    unique_type: Attribute
    """The known type of the Attribute, if any."""

    def parse_attr(self, parser: Parser) -> Attribute | None:
        unique_base = self.unique_base
        assert issubclass(unique_base, TypedAttribute)
        return unique_base.parse_with_type(parser, self.unique_type)

    def print_attr(self, printer: Printer, attr: Attribute) -> None:
        assert isinstance(attr, TypedAttribute)
        return attr.print_without_type(printer)


@dataclass(frozen=True)
class DenseArrayAttributeVariable(AttributeVariable):
    elt_type: IntegerType | AnyFloat

    def parse_attr(self, parser: Parser) -> Attribute | None:
        if isinstance(self.elt_type, IntegerType):
            if self.is_optional:
                elements = parser.parse_optional_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
                if elements is None:
                    return None
            else:
                elements = parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            return DenseArrayBase(
                self.elt_type, BytesAttr(self.elt_type.pack(elements))
            )
        else:
            if self.is_optional:
                elements = parser.parse_optional_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_float
                )
                if elements is None:
                    return None
            else:
                elements = parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_float
                )
            return DenseArrayBase(
                self.elt_type, BytesAttr(self.elt_type.pack(elements))
            )

    def print_attr(self, printer: Printer, attr: Attribute) -> None:
        with printer.in_square_brackets():
            if isa(attr, DenseArrayBase[IntegerType]):
                printer.print_list(attr.iter_values(), printer.print_int)
            elif isa(attr, DenseArrayBase[AnyFloat]):
                printer.print_list(
                    attr.iter_values(),
                    lambda value: printer.print_float(value, attr.elt_type),
                )


class SymbolNameAttributeVariable(AttributeVariable):
    def parse_attr(self, parser: Parser) -> Attribute | None:
        if self.is_optional:
            return parser.parse_optional_symbol_name()
        else:
            return parser.parse_symbol_name()

    def print_attr(self, printer: Printer, attr: Attribute) -> None:
        assert isinstance(attr, StringAttr)
        return printer.print_symbol_name(attr.data)


class OptionalUnitAttrVariable(AttributeVariable):
    """
    An optional UnitAttr variable that holds no value and derives its meaning from its existence. Holds a parse
    and print method to reflect this.

      operand-directive ::= (`unit_attr` unit_attr^)?

    Also see: https://mlir.llvm.org/docs/DefiningDialects/Operations/#unit-attributes
    """

    def __init__(self, name: str, is_property: bool):
        super().__init__(name, is_property, True, None)

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        self.set(state, UnitAttr())
        return True

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        return

    def is_optional_like(self) -> bool:
        return False


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

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        return False

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        printer.print_string(self.whitespace)
        state.last_was_punctuation = self.whitespace == ""
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

    punctuation: PunctuationSpelling
    """The punctuation that should be printed/parsed."""

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        return parser.parse_optional_punctuation(self.punctuation) is not None

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        emit_space = False
        if state.should_emit_space:
            if state.last_was_punctuation:
                if self.punctuation not in (">", ")", "}", "]", ","):
                    emit_space = True
            elif self.punctuation not in ("<", ">", "(", ")", "{", "}", "[", "]", ","):
                emit_space = True

            if emit_space:
                printer.print_string(" ")

        printer.print_string(self.punctuation)

        state.should_emit_space = self.punctuation not in ("<", "(", "{", "[")
        state.last_was_punctuation = True

    def is_optional_like(self) -> bool:
        return True


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

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        return parser.parse_optional_keyword(self.keyword) is not None

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if state.should_emit_space:
            printer.print_string(" ")
        state.should_emit_space = True
        state.last_was_punctuation = False

        printer.print_string(self.keyword)

    def is_optional_like(self) -> bool:
        return True


@dataclass(frozen=True)
class OptionalGroupDirective(FormatDirective):
    anchor: Directive
    then_whitespace: tuple[WhitespaceDirective, ...]
    then_first: FormatDirective
    then_elements: tuple[FormatDirective, ...]
    else_elements: tuple[FormatDirective, ...]

    def parse(self, parser: Parser, state: ParsingState) -> bool:
        # If the first element was parsed, parse the then-elements as usual
        if ret := self.then_first.parse(parser, state):
            for element in self.then_elements:
                element.parse(parser, state)
            for element in self.else_elements:
                element.set_empty(state)
        else:
            for element in self.then_elements:
                element.set_empty(state)
            for element in self.else_elements:
                element.parse(parser, state)
        return ret

    def print(self, printer: Printer, state: PrintingState, op: IRDLOperation) -> None:
        if self.anchor.is_present(op):
            for element in (
                *self.then_whitespace,
                self.then_first,
                *self.then_elements,
            ):
                element.print(printer, state, op)
        else:
            for element in self.else_elements:
                element.print(printer, state, op)

    def set_empty(self, state: ParsingState) -> None:
        self.then_first.set_empty(state)
        for element in self.then_elements:
            element.set_empty(state)
        for element in self.else_elements:
            element.set_empty(state)
