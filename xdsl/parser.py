from __future__ import annotations

import functools
import itertools
import math
import re
import sys
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    NoReturn,
    TypeVar,
    Iterable,
    IO,
    cast,
    Literal,
    Callable,
    overload,
    Sequence,
)

from xdsl.utils.exceptions import ParseError, MultipleSpansParseError
from xdsl.utils.lexer import Input, Lexer, Position, Span, StringLiteral, Token
from xdsl.dialects.memref import AnyIntegerAttr, MemRefType, UnrankedMemrefType
from xdsl.dialects.builtin import (
    AnyArrayAttr,
    AnyFloat,
    AnyFloatAttr,
    AnyTensorType,
    AnyUnrankedTensorType,
    AnyVectorType,
    BFloat16Type,
    DenseResourceAttr,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerType,
    Signedness,
    StringAttr,
    IntegerAttr,
    ArrayAttr,
    TensorType,
    UnrankedTensorType,
    UnregisteredAttr,
    RankedVectorOrTensorOf,
    VectorType,
    SymbolRefAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    OpaqueAttr,
    NoneAttr,
    ModuleOp,
    UnitAttr,
    i64,
    StridedLayoutAttr,
    ComplexType,
)
from xdsl.ir import (
    SSAValue,
    Block,
    Attribute,
    Operation,
    Region,
    MLContext,
    ParametrizedAttribute,
    Data,
)
from xdsl.utils.hints import isa


@dataclass
class BacktrackingHistory:
    """
    This class holds on to past errors encountered during parsing.

    Given the following error message:
       <unknown>:2:12
         %0 : !invalid = arith.constant() ["value" = 1 : !i32]
               ^^^^^^^
               'invalid' is not a known attribute

       <unknown>:2:7
         %0 : !invalid = arith.constant() ["value" = 1 : !i32]
              ^
              Expected type of value-id here!

    The BacktrackingHistory will contain the outermost error "
    "(expected type of value-id here)
    It's parent will be the next error message (not a known attribute).
    Some errors happen in named regions (e.g. "parsing of operation")
    """

    error: ParseError
    parent: BacktrackingHistory | None
    region_name: str | None
    pos: Position

    def print_unroll(self, file: IO[str] = sys.stderr):
        if self.parent:
            if self.parent.get_farthest_point() > self.pos:
                self.parent.print_unroll(file)
                self.print(file)
            else:
                self.print(file)
                self.parent.print_unroll(file)

    def print(self, file: IO[str] = sys.stderr):
        print(
            "Parsing of {} failed:".format(self.region_name or "<unknown>"), file=file
        )
        self.error.print_pretty(file=file)

    @functools.cache
    def get_farthest_point(self) -> Position:
        """
        Find the farthest this history managed to parse
        """
        if self.parent:
            return max(self.pos, self.parent.get_farthest_point())
        return self.pos

    def iterate(self) -> Iterable[BacktrackingHistory]:
        yield self
        if self.parent:
            yield from self.parent.iterate()

    def __hash__(self):
        return id(self)


@dataclass
class ForwardDeclaredValue(SSAValue):
    """
    An SSA value that is used before it is defined.
    It will be replaced to an operation result or a block argument when it is defined.
    """

    @property
    def owner(self) -> Operation | Block:
        assert False, "Forward declared values do not have an owner"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:  # type: ignore
        return id(self)


@dataclass
class UnresolvedOperand:
    """
    An operand that is not yet resolved in an operation parser.
    It will either be resolved to an SSA value, or to a forward reference of
    an SSA value.
    To resolve it, you need to provide its type.
    """

    span: Span
    """
    The parsing location of the operand name, including the `%`,
    but excluding the optional tuple index.
    """

    index: int
    """The value tuple index, if it is a tuple value."""

    @property
    def operand_name(self) -> str:
        return self.span.text[1:]


class Parser(ABC):
    """
    Basic recursive descent parser.

    methods marked try_... will attempt to parse, and return None if they failed.
    If they return None they must make sure to restore all state.

    methods marked parse_... will do "greedy" parsing, meaning they consume
    as much as they can. They will also throw an error if the think they should
    still be parsing. e.g. when parsing a list of numbers separated by '::',
    the following input will trigger an exception:
        1::2::
    Due to the '::' present after the last element. This is useful for parsing lists,
    as a trailing separator is usually considered a syntax error there.

    must_ type parsers are preferred because they are explicit about their failure modes.
    """

    ctx: MLContext
    """xDSL context."""

    ssa_values: dict[str, tuple[SSAValue]]
    blocks: dict[str, tuple[Block, Span | None]]
    forward_block_references: dict[str, list[Span]]
    """
    Blocks we encountered references to before the definition (must be empty after
    parsing of region completes)
    """
    forward_ssa_references: dict[str, dict[int, ForwardDeclaredValue]]
    """
    SSA values that are referenced, but are not yet defined.
    This field map a name and a tuple index to the forward declared SSA value.
    """

    lexer: Lexer

    _current_token: Token
    """Token at the current location"""

    T_ = TypeVar("T_")
    """
    Type var used for handling function that return single or multiple Spans.
    Basically the output type of all try_parse functions is `T_ | None`
    """

    allow_unregistered_dialect: bool

    def __init__(
        self,
        ctx: MLContext,
        input: str,
        name: str = "<unknown>",
        allow_unregistered_dialect: bool = False,
    ):
        self.lexer = Lexer(Input(input, name))
        self._current_token = self.lexer.lex()
        self.ctx = ctx
        self.ssa_values = dict()
        self.blocks = dict()
        self.forward_block_references = dict()
        self.forward_ssa_references = dict()
        self.allow_unregistered_dialect = allow_unregistered_dialect

    def resume_from(self, pos: Position):
        """
        Resume parsing from a given position.
        """
        self.lexer.pos = pos
        self._current_token = self.lexer.lex()

    @property
    def pos(self) -> Position:
        """Get the position of the next token."""
        return self._current_token.span.start

    def _consume_token(self, expected_kind: Token.Kind | None = None) -> Token:
        """
        Advance the lexer to the next token.
        Additionally check that the current token was of a specific kind,
        and assert if it was not.
        For reporting errors if the token was not of the expected kind,
        use `_parse_token` instead.
        """
        consumed_token = self._current_token
        if expected_kind is not None:
            assert consumed_token.kind == expected_kind, "Consumed an unexpected token!"
        self._current_token = self.lexer.lex()
        return consumed_token

    def _parse_optional_token(self, expected_kind: Token.Kind) -> Token | None:
        """
        If the current token is of the expected kind, consume it and return it.
        Otherwise, return None.
        """
        if self._current_token.kind == expected_kind:
            current_token = self._current_token
            self._consume_token(expected_kind)
            return current_token
        return None

    def _parse_token(self, expected_kind: Token.Kind, error_msg: str) -> Token:
        """
        Parse a specific token, and raise an error if it is not present.
        Returns the token that was parsed.
        """
        if self._current_token.kind != expected_kind:
            self.raise_error(error_msg, self._current_token.span)
        current_token = self._current_token
        self._consume_token(expected_kind)
        return current_token

    def _parse_optional_token_in(
        self, expected_kinds: Iterable[Token.Kind]
    ) -> Token | None:
        """Parse one of the expected tokens if present, and returns it."""
        if self._current_token.kind not in expected_kinds:
            return None
        return self._consume_token()

    def parse_module(self) -> ModuleOp:
        op = self.parse_optional_operation()

        if op is None:
            self.raise_error("Could not parse entire input!")

        if not isinstance(op, ModuleOp):
            self.resume_from(0)
            self.raise_error("builtin.module operation expected", 0)

        if self.forward_ssa_references:
            value_names = ", ".join(
                "%" + name for name in self.forward_ssa_references.keys()
            )
            if len(self.forward_block_references.keys()) > 1:
                self.raise_error(f"values {value_names} were used but not defined")
            else:
                self.raise_error(f"value {value_names} was used but not defined")

        return op

    def _get_block_from_name(self, block_name: Span) -> Block:
        """
        This function takes a span containing a block id (like `^42`) and returns a block.

        If the block definition was not seen yet, we create a forward declaration.
        """
        name = block_name.text[1:]
        if name not in self.blocks:
            self.forward_block_references[name].append(block_name)
            self.blocks[name] = (Block(), None)
        return self.blocks[name][0]

    def _parse_optional_block_arg_list(self, block: Block):
        """
        Parse a block argument list, if present, and add them to the block.

            value-id-and-type-list ::= value-id-and-type (`,` ssa-id-and-type)*
            block-arg-list ::= `(` value-id-and-type-list? `)`
        """
        if self._current_token.kind != Token.Kind.L_PAREN:
            return None

        def parse_argument() -> None:
            """Parse a single block argument with its type."""
            arg_name = self._parse_token(
                Token.Kind.PERCENT_IDENT, "block argument expected"
            ).span
            self.parse_punctuation(":")
            arg_type = self.parse_attribute()

            # Insert the block argument in the block, and register it in the parser
            block_arg = block.insert_arg(arg_type, len(block.args))
            self._register_ssa_definition(arg_name.text[1:], (block_arg,), arg_name)

        self.parse_comma_separated_list(self.Delimiter.PAREN, parse_argument)
        return block

    def _parse_block_body(self, block: Block):
        """
        Parse a block body, which consist of a list of operations.
        The operations are added at the end of the block.
        """
        while (op := self.parse_optional_operation()) is not None:
            block.add_op(op)

    def _parse_block(self) -> Block:
        """
        Parse a block with the following format:
          block ::= block-label operation*
          block-label    ::= block-id block-arg-list? `:`
          block-id       ::= caret-id
          block-arg-list ::= `(` ssa-id-and-type-list? `)`
        """
        name_token = self._parse_token(Token.Kind.CARET_IDENT, " in block definition")

        name = name_token.text[1:]
        if name not in self.blocks:
            block = Block()
            self.blocks[name] = (block, name_token.span)
        else:
            block, original_definition = self.blocks[name]
            if original_definition is not None:
                raise MultipleSpansParseError(
                    name_token.span,
                    f"re-declaration of block '{name}'",
                    "originally declared here:",
                    [(original_definition, None)],
                )
            self.forward_block_references.pop(name)

        self._parse_optional_block_arg_list(block)
        self.parse_punctuation(":")
        self._parse_block_body(block)
        return block

    def parse_optional_symbol_name(self) -> StringAttr | None:
        """
        Parse an @-identifier if present, and return its name (without the '@') in a
        string attribute.
        """
        if (token := self._parse_optional_token(Token.Kind.AT_IDENT)) is None:
            return None

        assert len(token.text) > 1, "token should be at least 2 characters long"

        # In the case where the symbol name is quoted, remove the quotes and escape
        # sequences.
        if token.text[1] == '"':
            literal_span = StringLiteral(
                token.span.start + 1, token.span.end, token.span.input
            )
            return StringAttr(literal_span.string_contents)
        return StringAttr(token.text[1:])

    def parse_symbol_name(self) -> StringAttr:
        """
        Parse an @-identifier and return its name (without the '@') in a string
        attribute.
        """
        return self.expect(self.parse_optional_symbol_name, "expect symbol name")

    class Delimiter(Enum):
        """
        Supported delimiters when parsing lists.
        """

        PAREN = auto()
        ANGLE = auto()
        SQUARE = auto()
        BRACES = auto()
        NONE = auto()

    def parse_comma_separated_list(
        self, delimiter: Delimiter, parse: Callable[[], T_], context_msg: str = ""
    ) -> list[T_]:
        """
        Parses greedily a list of elements separated by commas, and delimited
        by the specified delimiter. The parsing stops when the delimiter is
        closed, or when an error is produced. If no delimiter is specified, at
        least one element is expected to be parsed.
        """
        if delimiter == self.Delimiter.NONE:
            pass
        elif delimiter == self.Delimiter.PAREN:
            self._parse_token(Token.Kind.L_PAREN, "Expected '('" + context_msg)
            if self._parse_optional_token(Token.Kind.R_PAREN) is not None:
                return []
        elif delimiter == self.Delimiter.ANGLE:
            self._parse_token(Token.Kind.LESS, "Expected '<'" + context_msg)
            if self._parse_optional_token(Token.Kind.GREATER) is not None:
                return []
        elif delimiter == self.Delimiter.SQUARE:
            self._parse_token(Token.Kind.L_SQUARE, "Expected '['" + context_msg)
            if self._parse_optional_token(Token.Kind.R_SQUARE) is not None:
                return []
        elif delimiter == self.Delimiter.BRACES:
            self._parse_token(Token.Kind.L_BRACE, "Expected '{'" + context_msg)
            if self._parse_optional_token(Token.Kind.R_BRACE) is not None:
                return []
        else:
            assert False, "Unknown delimiter"

        elems = [parse()]
        while self._parse_optional_token(Token.Kind.COMMA) is not None:
            elems.append(parse())

        if delimiter == self.Delimiter.NONE:
            pass
        elif delimiter == self.Delimiter.PAREN:
            self._parse_token(Token.Kind.R_PAREN, "Expected ')'" + context_msg)
        elif delimiter == self.Delimiter.ANGLE:
            self._parse_token(Token.Kind.GREATER, "Expected '>'" + context_msg)
        elif delimiter == self.Delimiter.SQUARE:
            self._parse_token(Token.Kind.R_SQUARE, "Expected ']'" + context_msg)
        elif delimiter == self.Delimiter.BRACES:
            self._parse_token(Token.Kind.R_BRACE, "Expected '}'" + context_msg)
        else:
            assert False, "Unknown delimiter"

        return elems

    def parse_optional_boolean(self) -> bool | None:
        """
        Parse a boolean, if present, with the format `true` or `false`.
        """
        if self._current_token.kind == Token.Kind.BARE_IDENT:
            if self._current_token.text == "true":
                self._consume_token(Token.Kind.BARE_IDENT)
                return True
            elif self._current_token.text == "false":
                self._consume_token(Token.Kind.BARE_IDENT)
                return False
        return None

    def parse_boolean(self, context_msg: str = "") -> bool:
        """
        Parse a boolean with the format `true` or `false`.
        """
        return self.expect(
            lambda: self.parse_optional_boolean(),
            "Expected boolean literal" + context_msg,
        )

    def parse_optional_integer(
        self, allow_boolean: bool = True, allow_negative: bool = True
    ) -> int | None:
        """
        Parse an (possible negative) integer. The integer can either be
        decimal or hexadecimal.
        Optionally allow parsing of 'true' or 'false' into 1 and 0.
        """
        # Parse true and false if needed
        if allow_boolean:
            if (boolean := self.parse_optional_boolean()) is not None:
                return 1 if boolean else 0

        # Parse negative numbers if required
        is_negative = False
        if allow_negative:
            is_negative = self._parse_optional_token(Token.Kind.MINUS) is not None

        # Parse the actual number
        if (int_token := self._parse_optional_token(Token.Kind.INTEGER_LIT)) is None:
            if is_negative:
                self.raise_error("Expected integer literal after '-'")
            return None

        # Get the value and optionally negate it
        value = int_token.get_int_value()
        if is_negative:
            value = -value
        return value

    def parse_optional_number(self) -> int | float | None:
        """
        Parse an integer or float literal, if present.
        """

        is_negative = self._parse_optional_token(Token.Kind.MINUS) is not None

        if (
            value := self.parse_optional_integer(
                allow_boolean=False, allow_negative=False
            )
        ) is not None:
            return -value if is_negative else value

        if (value := self._parse_optional_token(Token.Kind.FLOAT_LIT)) is not None:
            value = value.get_float_value()
            return -value if is_negative else value

        if is_negative:
            self.raise_error("Expected integer or float literal after '-'")
        return None

    def parse_number(self, context_msg: str = "") -> int | float:
        """
        Parse an integer or float literal.
        """
        return self.expect(
            lambda: self.parse_optional_number(),
            "integer or float literal expected" + context_msg,
        )

    def parse_integer(
        self,
        allow_boolean: bool = True,
        allow_negative: bool = True,
        context_msg: str = "",
    ) -> int:
        """
        Parse an (possible negative) integer. The integer can
        either be decimal or hexadecimal.
        Optionally allow parsing of 'true' or 'false' into 1 and 0.
        """

        return self.expect(
            lambda: self.parse_optional_integer(allow_boolean, allow_negative),
            "Expected integer literal" + context_msg,
        )

    def parse_optional_str_literal(self) -> str | None:
        """
        Parse a string literal with the format `"..."`, if present.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """

        if (token := self._parse_optional_token(Token.Kind.STRING_LIT)) is None:
            return None
        return token.get_string_literal_value()

    def parse_str_literal(self, context_msg: str = "") -> str:
        """
        Parse a string literal with the format `"..."`.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """
        return self.expect(
            self.parse_optional_str_literal,
            "string literal expected" + context_msg,
        )

    def parse_optional_identifier(self) -> str | None:
        """
        Parse an identifier, if present, with syntax:
            ident ::= (letter|[_]) (letter|digit|[_$.])*
        """
        if (token := self._parse_optional_token(Token.Kind.BARE_IDENT)) is not None:
            return token.text
        return None

    def parse_identifier(self, context_msg: str = "") -> str:
        """
        Parse an identifier, if present, with syntax:
            ident ::= (letter|[_]) (letter|digit|[_$.])*
        """
        return self.expect(
            self.parse_optional_identifier, "identifier expected" + context_msg
        )

    _decimal_integer_regex = re.compile(r"[0-9]+")

    def parse_optional_unresolved_operand(self) -> UnresolvedOperand | None:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`, if present.
        The operand may be forward declared.
        """
        name_token = self._parse_optional_token(Token.Kind.PERCENT_IDENT)
        if name_token is None:
            return None

        index = 0
        index_token = self._parse_optional_token(Token.Kind.HASH_IDENT)
        if index_token is not None:
            if re.fullmatch(self._decimal_integer_regex, index_token.text[1:]) is None:
                self.raise_error(
                    "Expected integer as SSA value tuple index", index_token.span
                )
            index = int(index_token.text[1:], 10)

        return UnresolvedOperand(name_token.span, index)

    def parse_unresolved_operand(
        self, msg: str = "operand expected"
    ) -> UnresolvedOperand:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`.
        The operand may be forward declared.
        """
        return self.expect(self.parse_optional_unresolved_operand, msg)

    def resolve_operand(self, operand: UnresolvedOperand, type: Attribute) -> SSAValue:
        """
        Resolve an unresolved operand.
        If the operand is not yet defined, it creates a forward reference.
        If the operand is already defined, it returns the corresponding SSA value,
        and checks that the type is consistent.
        """
        name = operand.operand_name

        # If the indexed operand is already used as a forward reference, return it
        if (
            name in self.forward_ssa_references
            and operand.index in self.forward_ssa_references[name]
        ):
            return self.forward_ssa_references[name][operand.index]

        # If the operand is not yet defined, create a forward reference
        if name not in self.ssa_values:
            forward_value = ForwardDeclaredValue(type)
            reference_tuple = self.forward_ssa_references.setdefault(name, {})
            reference_tuple[operand.index] = forward_value
            return forward_value

        # If the operand is already defined, check that the tuple index is in range
        tuple_size = len(self.ssa_values[name])
        if operand.index >= tuple_size:
            self.raise_error(
                "SSA value tuple index out of bounds. "
                f"Tuple is of size {tuple_size} but tried to access element {operand.index}.",
                operand.span,
            )

        # Check that the type is consistent
        resolved = self.ssa_values[name][operand.index]
        if resolved.typ != type:
            self.raise_error(
                f"operand is used with type {type}, but has been "
                f"previously used or defined with type {resolved.typ}",
                operand.span,
            )

        return resolved

    def parse_optional_operand(self) -> SSAValue | None:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`, if present.
        """
        unresolved_operand = self.parse_optional_unresolved_operand()
        if unresolved_operand is None:
            return None

        name = unresolved_operand.operand_name
        index = unresolved_operand.index

        if name not in self.ssa_values.keys():
            self.raise_error(
                "SSA value used before assignment", unresolved_operand.span
            )

        tuple_size = len(self.ssa_values[name])
        if index >= tuple_size:
            self.raise_error(
                "SSA value tuple index out of bounds. "
                f"Tuple is of size {tuple_size} but tried to access element {index}.",
                unresolved_operand.span,
            )

        return self.ssa_values[name][index]

    def parse_operand(self, msg: str = "Expected an operand.") -> SSAValue:
        """Parse an operand with format `%<value-id>`."""
        return self.expect(self.parse_optional_operand, msg)

    def parse_type(self) -> Attribute:
        """
        Parse an xDSL type.
        An xDSL type is either a builtin type, which can have various format,
        or a dialect type, with the following format:
            dialect-type  ::= `!` type-name (`<` dialect-type-contents+ `>`)?
            type-name     ::= bare-id
            dialect-type-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        return self.expect(self.parse_optional_type, "type expected")

    def parse_optional_type(self) -> Attribute | None:
        """
        Parse an xDSL type, if present.
        An xDSL type is either a builtin type, which can have various format,
        or a dialect type, with the following format:
            dialect-type  ::= `!` type-name (`<` dialect-type-contents+ `>`)?
            type-name     ::= bare-id
            dialect-type-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        if (
            token := self._parse_optional_token(Token.Kind.EXCLAMATION_IDENT)
        ) is not None:
            return self._parse_dialect_type_or_attribute_inner(token.text[1:], True)
        return self._parse_optional_builtin_type()

    def parse_attribute(self) -> Attribute:
        """
        Parse an xDSL attribute.
        An attribute is either a builtin attribute, which can have various format,
        or a dialect attribute, with the following format:
            dialect-attr  ::= `!` attr-name (`<` dialect-attr-contents+ `>`)?
            attr-name     ::= bare-id
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        return self.expect(self.parse_optional_attribute, "attribute expected")

    def parse_optional_attribute(self) -> Attribute | None:
        """
        Parse an xDSL attribute, if present.
        An attribute is either a builtin attribute, which can have various format,
        or a dialect attribute, with the following format:
            dialect-attr  ::= `!` attr-name (`<` dialect-attr-contents+ `>`)?
            attr-name     ::= bare-id
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                            | `(` dialect-attribute-contents+ `)`
                            | `[` dialect-attribute-contents+ `]`
                            | `{` dialect-attribute-contents+ `}`
                            | [^[]<>(){}\0]+
        """
        if (token := self._parse_optional_token(Token.Kind.HASH_IDENT)) is not None:
            return self._parse_dialect_type_or_attribute_inner(token.text[1:], False)
        return self._parse_optional_builtin_attr()

    def _parse_dialect_type_or_attribute_inner(
        self, attr_name: str, is_type: bool = True
    ) -> Attribute:
        """
        Parse the contents of a dialect type or attribute, with format:
            dialect-attr-contents ::= `<` dialect-attribute-contents+ `>`
                                    | `(` dialect-attribute-contents+ `)`
                                    | `[` dialect-attribute-contents+ `]`
                                    | `{` dialect-attribute-contents+ `}`
                                    | [^[]<>(){}\0]+
        The contents will be parsed by a user-defined parser, or by a generic parser
        if the dialect attribute/type is not registered.
        """
        attr_def = self.ctx.get_optional_attr(
            attr_name,
            self.allow_unregistered_dialect,
            create_unregistered_as_type=is_type,
        )
        if attr_def is None:
            self.raise_error(f"'{attr_name}' is not registered")

        # Pass the task of parsing parameters on to the attribute/type definition
        if issubclass(attr_def, UnregisteredAttr):
            body = self._parse_unregistered_attr_body()
            return attr_def(attr_name, is_type, body)
        if issubclass(attr_def, ParametrizedAttribute):
            param_list = attr_def.parse_parameters(self)
            return attr_def.new(param_list)
        if issubclass(attr_def, Data):
            self.parse_punctuation("<")
            param: Any = attr_def.parse_parameter(self)
            self.parse_punctuation(">")
            return cast(Data[Any], attr_def(param))
        assert False, "Attributes are either ParametrizedAttribute or Data."

    def _parse_unregistered_attr_body(self) -> str:
        """
        Parse the body of an unregistered attribute, which is a balanced
        string for `<`, `(`, `[`, `{`, and may contain string literals.
        """
        start_token = self._parse_optional_token(Token.Kind.LESS)
        if start_token is None:
            return ""

        start_pos = start_token.span.start
        end_pos: Position = start_pos

        symbols_stack = [Token.Kind.LESS]
        parentheses = {
            Token.Kind.GREATER: Token.Kind.LESS,
            Token.Kind.R_PAREN: Token.Kind.L_PAREN,
            Token.Kind.R_SQUARE: Token.Kind.L_SQUARE,
            Token.Kind.R_BRACE: Token.Kind.L_BRACE,
        }
        parentheses_names = {
            Token.Kind.GREATER: "`>`",
            Token.Kind.R_PAREN: "`)`",
            Token.Kind.R_SQUARE: "`]`",
            Token.Kind.R_BRACE: "`}`",
        }
        while True:
            # Opening a new parenthesis
            if (
                token := self._parse_optional_token_in(parentheses.values())
            ) is not None:
                symbols_stack.append(token.kind)
                continue

            # Closing a parenthesis
            if (token := self._parse_optional_token_in(parentheses.keys())) is not None:
                closing = parentheses[token.kind]
                if symbols_stack[-1] != closing:
                    self.raise_error(
                        "Mismatched {} in attribute body!".format(
                            parentheses_names[token.kind]
                        ),
                        self._current_token.span,
                    )
                symbols_stack.pop()
                if len(symbols_stack) == 0:
                    end_pos = token.span.end
                    break
                continue

            # Checking for unexpected EOF
            if self._parse_optional_token(Token.Kind.EOF) is not None:
                self.raise_error(
                    "Unexpected end of file before closing of attribute body!"
                )

            # Other tokens
            self._consume_token()

        body = self.lexer.input.slice(start_pos, end_pos)
        assert body is not None
        return body

    def _parse_optional_builtin_parametrized_type(self) -> ParametrizedAttribute | None:
        """
        Parse an builtin parametrized type, if present, with format:
            builtin-parametrized-type ::= builtin-name `<` args `>`
            builtin-name ::= vector | memref | tensor | complex | tuple
            args ::= <defined by the builtin name>
        """
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            return None

        name = self._current_token.text

        def unimplemented() -> NoReturn:
            raise ParseError(
                self._current_token.span,
                "Builtin {} is not supported yet!".format(name),
            )

        builtin_parsers: dict[str, Callable[[], ParametrizedAttribute]] = {
            "vector": self._parse_vector_attrs,
            "memref": self._parse_memref_attrs,
            "tensor": self._parse_tensor_attrs,
            "complex": self._parse_complex_attrs,
            "tuple": unimplemented,
        }

        if name not in builtin_parsers:
            return None
        self._consume_token(Token.Kind.BARE_IDENT)

        self.parse_punctuation("<", " after builtin name")
        # Get the parser for the type, falling back to the unimplemented warning
        res = builtin_parsers.get(name, unimplemented)()
        self.parse_punctuation(">", " after builtin parameter list")
        return res

    def parse_shape_dimension(self, allow_dynamic: bool = True) -> int:
        """
        Parse a single shape dimension, which is a decimal literal or `?`.
        `?` is interpreted as -1. Note that if the integer literal is in
        hexadecimal form, it will be split into multiple tokens. For example,
        `0x10` will be split into `0` and `x10`.
        Optionally allows to not parse `?` as -1.
        """
        if self._current_token.kind not in (
            Token.Kind.INTEGER_LIT,
            Token.Kind.QUESTION,
        ):
            if allow_dynamic:
                self.raise_error(
                    "Expected either integer literal or '?' in shape dimension, "
                    f"got {self._current_token.kind.name}!"
                )
            self.raise_error(
                "Expected integer literal in shape dimension, "
                f"got {self._current_token.kind.name}!"
            )

        if self.parse_optional_punctuation("?") is not None:
            if allow_dynamic:
                return -1
            self.raise_error("Unexpected dynamic dimension!")

        # If the integer literal starts with `0x`, this is decomposed into
        # `0` and `x`.
        int_token = self._consume_token(Token.Kind.INTEGER_LIT)
        if int_token.text[:2] == "0x":
            self.resume_from(int_token.span.start + 1)
            return 0

        return int_token.get_int_value()

    def parse_shape_delimiter(self) -> None:
        """
        Parse 'x', a shape delimiter. Note that if 'x' is followed by other
        characters, it will split the token. For instance, 'x1' will be split
        into 'x' and '1'.
        """
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            self.raise_error(
                "Expected 'x' in shape delimiter, got "
                f"{self._current_token.kind.name}"
            )

        if self._current_token.text[0] != "x":
            self.raise_error(
                "Expected 'x' in shape delimiter, got " f"{self._current_token.text}"
            )

        # Move the lexer to the position after 'x'.
        self.resume_from(self._current_token.span.start + 1)

    def parse_ranked_shape(self) -> tuple[list[int], Attribute]:
        """
        Parse a ranked shape with the following format:
          ranked-shape ::= (dimension `x`)* type
          dimension ::= `?` | decimal-literal
        each dimension is also required to be non-negative.
        """
        dims: list[int] = []
        while self._current_token.kind in (Token.Kind.INTEGER_LIT, Token.Kind.QUESTION):
            dim = self.parse_shape_dimension()
            dims.append(dim)
            self.parse_shape_delimiter()

        type = self.expect(self.parse_optional_type, "Expected shape type.")
        return dims, type

    def parse_shape(self) -> tuple[list[int] | None, Attribute]:
        """
        Parse a ranked or unranked shape with the following format:

        shape ::= ranked-shape | unranked-shape
        ranked-shape ::= (dimension `x`)* type
        unranked-shape ::= `*`x type
        dimension ::= `?` | decimal-literal

        each dimension is also required to be non-negative.
        """
        if self.parse_optional_punctuation("*") is not None:
            self.parse_shape_delimiter()
            type = self.expect(self.parse_optional_type, "Expected shape type.")
            return None, type
        return self.parse_ranked_shape()

    def _parse_complex_attrs(self) -> ComplexType:
        element_type = self.parse_attribute()
        if not isa(element_type, IntegerType | AnyFloat):
            self.raise_error(
                "Complex type must be parameterized by an integer or float type!"
            )
        return ComplexType(element_type)

    def _parse_memref_attrs(
        self,
    ) -> MemRefType[Attribute] | UnrankedMemrefType[Attribute]:
        shape, type = self.parse_shape()

        # Unranked case
        if shape is None:
            if self.parse_optional_punctuation(",") is None:
                return UnrankedMemrefType.from_type(type)
            memory_space = self.parse_attribute()
            return UnrankedMemrefType.from_type(type, memory_space)

        if self.parse_optional_punctuation(",") is None:
            return MemRefType.from_element_type_and_shape(type, shape)

        memory_or_layout = self.parse_attribute()

        # If there is both a memory space and a layout, we know that the
        # layout is the second one
        if self.parse_optional_punctuation(",") is not None:
            memory_space = self.parse_attribute()
            return MemRefType.from_element_type_and_shape(
                type, shape, memory_or_layout, memory_space
            )

        # Otherwise, there is a single argument, so we check based on the
        # attribute type. If we don't know, we return an error.
        # MLIR base itself on the `MemRefLayoutAttrInterface`, which we do not
        # support.

        # If the argument is an integer, it is a memory space
        if isa(memory_or_layout, AnyIntegerAttr):
            return MemRefType.from_element_type_and_shape(
                type, shape, memory_space=memory_or_layout
            )

        # We only accept strided layouts and affine_maps
        if isa(memory_or_layout, StridedLayoutAttr) or (
            isinstance(memory_or_layout, UnregisteredAttr)
            and memory_or_layout.attr_name.data == "affine_map"
        ):
            return MemRefType.from_element_type_and_shape(
                type, shape, layout=memory_or_layout
            )
        self.raise_error(
            "Cannot decide if the given attribute " "is a layout or a memory space!"
        )

    def _parse_vector_attrs(self) -> AnyVectorType:
        dims: list[int] = []
        num_scalable_dims = 0
        # First, parse the static dimensions
        while self._current_token.kind == Token.Kind.INTEGER_LIT:
            dims.append(self.parse_shape_dimension(allow_dynamic=False))
            self.parse_shape_delimiter()

        # Then, parse the scalable dimensions, if any
        if self.parse_optional_punctuation("[") is not None:
            # Parse the scalable dimensions
            dims.append(self.parse_shape_dimension(allow_dynamic=False))
            num_scalable_dims += 1

            while self.parse_optional_punctuation("]") is None:
                self.parse_shape_delimiter()
                dims.append(self.parse_shape_dimension(allow_dynamic=False))
                num_scalable_dims += 1

            # Parse the `x` between the scalable dimensions and the type
            self.parse_shape_delimiter()

        type = self.parse_optional_type()
        if type is None:
            self.raise_error("Expected the vector element types!")

        return VectorType.from_element_type_and_shape(type, dims, num_scalable_dims)

    def _parse_tensor_attrs(self) -> AnyTensorType | AnyUnrankedTensorType:
        shape, type = self.parse_shape()

        if shape is None:
            if self.parse_optional_punctuation(",") is not None:
                self.raise_error("Unranked tensors don't have an encoding!")
            return UnrankedTensorType.from_type(type)

        if self.parse_optional_punctuation(",") is not None:
            encoding = self.parse_attribute()
            return TensorType.from_type_and_list(type, shape, encoding)

        return TensorType.from_type_and_list(type, shape)

    def expect(self, try_parse: Callable[[], T_ | None], error_message: str) -> T_:
        """
        Used to force completion of a try_parse function.
        Will throw a parse error if it can't.
        """
        res = try_parse()
        if res is None:
            self.raise_error(error_message)
        return res

    @overload
    def raise_error(
        self,
        msg: str,
        at_position: Position,
        end_position: Position,
    ) -> NoReturn:
        ...

    @overload
    def raise_error(
        self,
        msg: str,
        at_position: Position | Span | None = None,
    ) -> NoReturn:
        ...

    def raise_error(
        self,
        msg: str,
        at_position: Span | Position | None = None,
        end_position: Position | None = None,
    ) -> NoReturn:
        """
        Helper for raising exceptions, provides as much context as possible to them.

        If no position is provided, the error will be displayed at the next token.
        This will, for example, include backtracking errors, if any occurred previously.
        """
        if end_position is not None:
            assert isinstance(at_position, Position)
            at_position = Span(at_position, end_position, self.lexer.input)
        if at_position is None:
            at_position = self._current_token.span
        elif isinstance(at_position, Position):
            at_position = Span(at_position, at_position, self.lexer.input)

        raise ParseError(at_position, msg)

    def parse_optional_characters(self, text: str) -> str | None:
        """
        Parse a given token text, if present.
        If the given text is the beginning of the next token, this will still
        return None.
        """
        if self._current_token.text == text:
            self._consume_token()
            return text
        return None

    def parse_characters(self, text: str, context_msg: str = "") -> str:
        """
        Parse a given token text.
        The context message is appended to the error message if the parsing fails.
        If the given text is the start of the next token, this will still raise
        an error.
        """
        if (res := self.parse_optional_characters(text)) is not None:
            return res
        self.raise_error(f"'{text}' expected" + context_msg)

    def _register_ssa_definition(
        self, name: str, values: Sequence[SSAValue], span: Span
    ) -> None:
        """
        Register an SSA definition in the parsing context.
        In the case the value was already used as a forward reference, the forward
        references are replaced by this value.
        """

        # Check for duplicate SSA value names.
        if name in self.ssa_values:
            self.raise_error(f"SSA value %{name} is already defined", span)

        # Register the SSA values in the context
        self.ssa_values[name] = tuple(values)

        tuple_size = len(values)
        # Check for forward references of this value
        if name in self.forward_ssa_references:
            index_references = self.forward_ssa_references[name]
            del self.forward_ssa_references[name]
            if any(index >= tuple_size for index in index_references):
                self.raise_error(
                    f"SSA value %{name} is referenced with an index "
                    f"larger than its size",
                    span,
                )

            # Replace the forward references with the actual SSA value
            for index, value in index_references.items():
                if index >= tuple_size:
                    self.raise_error(
                        f"SSA value tuple %{name} is referenced with index {index}, but "
                        f"has size {tuple_size}",
                        span,
                    )

                result = values[index]
                if value.typ != result.typ:
                    result_name = f"%{name}"
                    if tuple_size != 1:
                        result_name = f"%{name}#{index}"
                    self.raise_error(
                        f"Result {result_name} is defined with "
                        f"type {result.typ}, but used with type {value.typ}",
                        span,
                    )
                value.replace_by(result)

        if SSAValue.is_valid_name(name):
            for val in values:
                val.name_hint = name

    @dataclass
    class Argument:
        """
        A block argument parsed from the assembly.
        Arguments should be parsed by `parse_argument` or `parse_optional_argument`.
        """

        name: Span
        """The name as displayed in the assembly."""

        type: Attribute | None
        """The type of the argument, if any."""

    def parse_optional_argument(self, expect_type: bool = True) -> Argument | None:
        """
        Parse a block argument, if present, with format:
          arg ::= percent-id `:` type
        if `expect_type` is False, the type is not parsed.
        """

        # The argument name
        name_token = self._parse_optional_token(Token.Kind.PERCENT_IDENT)
        if name_token is None:
            return None

        # The argument type
        type = None
        if expect_type:
            self.parse_punctuation(":", " after block argument name!")
            type = self.parse_type()
        return self.Argument(name_token.span, type)

    def parse_argument(self, expect_type: bool = True) -> Argument:
        """
        Parse a block argument with format:
          arg ::= percent-id `:` type
        if `expect_type` is False, the type is not parsed.
        """

        arg = self.parse_optional_argument(expect_type)
        if arg is None:
            self.raise_error("Expected block argument!")
        return arg

    def parse_optional_region(
        self, arguments: Iterable[Argument] | None = None
    ) -> Region | None:
        """
        Parse a region, if present, with format:
          region ::= `{` entry-block? block* `}`
        If `arguments` is provided, the entry block will use these as block arguments,
        and the entry-block cannot be labeled. It also cannot be empty, unless it is the
        only block in the region.
        """
        # Check if a region is present.
        if self.parse_optional_punctuation("{") is None:
            return None

        region = Region()

        # Create a new scope for values and blocks.
        # Outside blocks cannot be referenced from inside the region, and vice versa.
        # Outside values can be referenced from inside the region, but the region
        # values cannot be referred to from outside the region.
        old_ssa_values = self.ssa_values.copy()
        old_blocks = self.blocks
        old_forward_blocks = self.forward_block_references
        self.blocks = dict()
        self.forward_block_references = defaultdict(list)

        # Parse the entry block without label if arguments are provided.
        # Since the entry block cannot be jumped to, this is fine.
        if arguments is not None:
            # Check that the provided arguments have types.
            if any(arg.type is None for arg in arguments):
                raise ValueError("provided entry block arguments must have a type")
            arg_types = cast(list[Attribute], [arg.type for arg in arguments])

            # Check that the entry block has no label.
            # Since a multi-block region block must have a terminator, there isn't a
            # possibility of having an empty entry block, and thus parsing the label directly.
            if self._current_token.kind == Token.Kind.CARET_IDENT:
                self.raise_error("invalid block name in region with named arguments")

            # Set the block arguments in the context
            entry_block = Block(arg_types=arg_types)
            for block_arg, arg in zip(entry_block.args, arguments):
                self._register_ssa_definition(arg.name.text[1:], (block_arg,), arg.name)

            # Parse the entry block body
            self._parse_block_body(entry_block)
            region.add_block(entry_block)

        # If no arguments was provided, parse the entry block if present.
        elif self._current_token.kind not in (
            Token.Kind.CARET_IDENT,
            Token.Kind.R_BRACE,
        ):
            block = Block()
            self._parse_block_body(block)
            region.add_block(block)

        # Parse the region blocks.
        # In the case where arguments are provided, the entry block is already parsed,
        # and the following blocks will have a label (since the entry block will parse
        # greedily all operations).
        # In the case where no arguments areprovided, the entry block can either have a
        # label or not.
        while self.parse_optional_punctuation("}") is None:
            block = self._parse_block()
            region.add_block(block)

        # Finally, check that all forward block references have been resolved.
        if len(self.forward_block_references) > 0:
            pos = self.lexer.pos
            raise MultipleSpansParseError(
                Span(pos, pos + 1, self.lexer.input),
                "region ends with missing block declarations for block(s) {}".format(
                    ", ".join(self.forward_block_references.keys())
                ),
                "dangling block references:",
                [
                    (
                        span,
                        'reference to block "{}" without implementation'.format(
                            span.text
                        ),
                    )
                    for span in itertools.chain(*self.forward_block_references.values())
                ],
            )

        # Close the value and block scope.
        self.ssa_values = old_ssa_values
        self.blocks = old_blocks
        self.forward_block_references = old_forward_blocks

        return region

    def parse_region(self, arguments: Iterable[Argument] | None = None) -> Region:
        """
        Parse a region with format:
          region ::= `{` entry-block? block* `}`
        If `arguments` is provided, the entry block will use these as block arguments,
        and the entry-block cannot be labeled. It also cannot be empty, unless it is the
        only block in the region.
        """
        region = self.parse_optional_region(arguments)
        if region is None:
            self.raise_error("Expected region!")
        return region

    def _parse_attribute_entry(self) -> tuple[str, Attribute]:
        """
        Parse entry in attribute dict. Of format:

        attribute_entry := (bare-id | string-literal) `=` attribute
        attribute       := dialect-attribute | builtin-attribute
        """
        if (name := self._parse_optional_token(Token.Kind.BARE_IDENT)) is not None:
            name = name.span.text
        else:
            name = self.parse_optional_str_literal()

        if name is None:
            self.raise_error(
                "Expected bare-id or string-literal here as part of attribute entry!"
            )

        if self.parse_optional_punctuation("=") is None:
            return name, UnitAttr()

        return name, self.parse_attribute()

    def _parse_attribute_type(self) -> Attribute:
        """
        Parses `:` type and returns the type
        """
        self.parse_characters(":", " in attribute type")
        return self.parse_type()

    def _parse_optional_builtin_attr(self) -> Attribute | None:
        """
        Tries to parse a builtin attribute, e.g. a string literal, int, array, etc..
        """

        # String literal
        if (str_lit := self.parse_optional_str_literal()) is not None:
            return StringAttr(str_lit)

        attrs = (
            self.parse_optional_builtin_int_or_float_attr,
            self._parse_optional_array_attr,
            self._parse_optional_symref_attr,
            self._parse_optional_builtin_dict_attr,
            self.parse_optional_type,
            self._parse_optional_builtin_parametrized_attr,
        )

        for attr_parser in attrs:
            if (val := attr_parser()) is not None:
                return val

        return None

    def _parse_int_or_question(self, context_msg: str = "") -> int | Literal["?"]:
        """Parse either an integer literal, or a '?'."""
        if self._parse_optional_token(Token.Kind.QUESTION) is not None:
            return "?"
        if (v := self.parse_optional_integer(allow_boolean=False)) is not None:
            return v
        self.raise_error("Expected an integer literal or `?`" + context_msg)

    def parse_keyword(self, keyword: str, context_msg: str = "") -> str:
        """Parse a specific identifier."""

        error_msg = f"Expected '{keyword}'" + context_msg
        if self.parse_optional_keyword(keyword) is not None:
            return keyword
        self.raise_error(error_msg)

    def parse_optional_keyword(self, keyword: str) -> str | None:
        """Parse a specific identifier if it is present"""

        if (
            self._current_token.kind == Token.Kind.BARE_IDENT
            and self._current_token.text == keyword
        ):
            self._consume_token(Token.Kind.BARE_IDENT)
            return keyword
        return None

    def _parse_strided_layout_attr(self, name: Span) -> Attribute:
        """
        Parse a strided layout attribute parameters.
        | `<` `[` comma-separated-int-or-question `]`
          (`,` `offset` `:` integer-literal)? `>`
        """
        # Parse stride list
        self._parse_token(Token.Kind.LESS, "Expected `<` after `strided`")
        strides = self.parse_comma_separated_list(
            self.Delimiter.SQUARE,
            lambda: self._parse_int_or_question(" in stride list"),
            " in stride list",
        )
        # Pyright widen `Literal['?']` to `str` for some reasons
        strides = cast(list[int | Literal["?"]], strides)

        # Convert to the attribute expected input
        strides = [None if stride == "?" else stride for stride in strides]

        # Case without offset
        if self._parse_optional_token(Token.Kind.GREATER) is not None:
            return StridedLayoutAttr(strides)

        # Parse the optional offset
        self._parse_token(
            Token.Kind.COMMA, "Expected end of strided attribute or ',' for offset."
        )
        self.parse_keyword("offset", " after comma")
        self._parse_token(Token.Kind.COLON, "Expected ':' after 'offset'")
        offset = self._parse_int_or_question(" in stride offset")
        self._parse_token(Token.Kind.GREATER, "Expected '>' in end of stride attribute")
        return StridedLayoutAttr(strides, None if offset == "?" else offset)

    def _parse_optional_builtin_parametrized_attr(self) -> Attribute | None:
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            return None
        name = self._current_token.span
        parsers = {
            "dense": self._parse_builtin_dense_attr,
            "opaque": self._parse_builtin_opaque_attr,
            "dense_resource": self._parse_builtin_dense_resource_attr,
            "array": self._parse_builtin_densearray_attr,
            "affine_map": self._parse_builtin_affine_attr,
            "affine_set": self._parse_builtin_affine_attr,
            "strided": self._parse_strided_layout_attr,
        }

        if name.text not in parsers:
            return None
        self._consume_token(Token.Kind.BARE_IDENT)
        return parsers[name.text](name)

    def _parse_builtin_dense_attr(self, _name: Span) -> DenseIntOrFPElementsAttr:
        self.parse_punctuation("<", " in dense attribute")

        # The flatten list of elements
        values: list[Parser._TensorLiteralElement]
        # The dense shape.
        # If it is `None`, then there is no values.
        # If it is `[]`, then this is a splat attribute, meaning it has the same
        # value everywhere.
        shape: list[int] | None
        if self._current_token.text == ">":
            values, shape = [], None
        else:
            values, shape = self._parse_tensor_literal()
        self.parse_punctuation(">", " in dense attribute")

        # Parse the dense type.
        self.parse_punctuation(":", " in dense attribute")
        type = self.expect(self.parse_optional_type, "Dense attribute must be typed!")

        # Check that the type is correct.
        if not isa(
            type,
            RankedVectorOrTensorOf[IntegerType]
            | RankedVectorOrTensorOf[IndexType]
            | RankedVectorOrTensorOf[AnyFloat],
        ):
            self.raise_error(
                "Expected vector or tensor type of " "integer, index, or float type"
            )

        # Check that the shape matches the data when given a shaped data.
        type_shape = [dim.value.data for dim in type.shape.data]
        num_values = math.prod(type_shape)

        if shape is None and num_values != 0:
            self.raise_error(
                "Expected at least one element in the " "dense literal, but got None"
            )
        if shape is not None and shape != [] and type_shape != shape:
            self.raise_error(
                f"Shape mismatch in dense literal. Expected {type_shape} "
                f"shape from the type, but got {shape} shape."
            )
        if any(dim == -1 for dim in type_shape):
            self.raise_error(f"Dense literal attribute should have a static shape.")

        element_type = type.element_type
        # Convert list of elements to a list of values.
        if shape != []:
            data_values = [value.to_type(self, element_type) for value in values]
        else:
            assert len(values) == 1, "Fatal error in parser"
            data_values = [values[0].to_type(self, element_type)] * num_values

        return DenseIntOrFPElementsAttr.from_list(type, data_values)

    def _parse_builtin_opaque_attr(self, _name: Span):
        str_lit_list = self.parse_comma_separated_list(
            self.Delimiter.ANGLE, self.parse_str_literal
        )

        if len(str_lit_list) != 2:
            self.raise_error("Opaque expects 2 string literal parameters!")

        type = NoneAttr()
        if self.parse_optional_punctuation(":") is not None:
            type = self.expect(
                self.parse_optional_type, "opaque attribute must be typed!"
            )

        return OpaqueAttr.from_strings(*str_lit_list, type=type)

    def _parse_builtin_dense_resource_attr(self, _name: Span) -> DenseResourceAttr:
        self.parse_characters("<", " in dense_resource attribute")
        resource_handle = self.parse_identifier(" for resource handle")
        self.parse_characters(">", " in dense_resource attribute")
        self.parse_characters(":", " in dense_resource attribute")
        type = self.parse_type()
        return DenseResourceAttr.from_params(resource_handle, type)

    def _parse_builtin_densearray_attr(self, name: Span) -> DenseArrayBase | None:
        self.parse_characters("<", " in dense array")
        element_type = self.parse_attribute()

        if not isinstance(element_type, IntegerType | AnyFloat):
            raise ParseError(
                name,
                "dense array element type must be an " "integer or floating point type",
            )

        # Empty array
        if self.parse_optional_punctuation(">"):
            return DenseArrayBase.from_list(element_type, [])

        self.parse_characters(":", " in dense array")

        values = self.parse_comma_separated_list(self.Delimiter.NONE, self.parse_number)
        self.parse_characters(">", " in dense array")

        return DenseArrayBase.from_list(element_type, values)

    def _parse_builtin_affine_attr(self, name: Span) -> UnregisteredAttr:
        # First, retrieve the attribute definition.
        # Since we do not define affine attributes, we use an unregistered
        # attribute definition.
        attr_def = self.ctx.get_optional_attr(
            name.text,
            allow_unregistered=self.allow_unregistered_dialect,
            create_unregistered_as_type=False,
        )
        if attr_def is None:
            self.raise_error(f"Unknown {name.text} attribute", at_position=name)
        assert issubclass(
            attr_def, UnregisteredAttr
        ), f"{name.text} was registered, but should be reserved for builtin"

        # We then parse the attribute body. Affine attributes are closed by
        # `>`, so we can wait until we see this token. We just need to make
        # sure that we do not stop at a `>=`.
        start_pos = self._current_token.span.start
        end_pos = start_pos
        self.parse_punctuation("<", f" in {name.text} attribute")

        # Loop until we see the closing `>`.
        while True:
            token = self._consume_token()

            # Check for early EOF.
            if token.kind == Token.Kind.EOF:
                self.raise_error(f"Expected '>' in end of {name.text} attribute")

            # Check for closing `>`.
            if token.kind == Token.Kind.GREATER:
                # Check that there is no `=` after the `>`.
                if self._parse_optional_token(Token.Kind.EQUAL) is None:
                    end_pos = token.span.end
                    break
                self._consume_token()

        contents = self.lexer.input.slice(start_pos, end_pos)
        assert contents is not None, "Fatal error in parser"

        return attr_def(name.text, False, contents)

    @dataclass
    class _TensorLiteralElement:
        """
        The representation of a tensor literal element used during parsing.
        It is either an integer, float, or boolean. It also has a check if
        the element has a negative sign (it is already applied to the value).
        This class is used to parse a tensor literal before the tensor literal
        type is known
        """

        is_negative: bool
        value: int | float | bool
        """
        An integer, float, boolean, integer complex, or float complex value.
        The tuple should be of type `_TensorLiteralElement`, but python does
        not allow classes to self-reference.
        """
        span: Span

        def to_int(
            self,
            parser: Parser,
            allow_negative: bool = True,
            allow_booleans: bool = True,
        ) -> int:
            """
            Convert the element to an int value, possibly disallowing negative
            values. Raises an error if the type is compatible.
            """
            if self.is_negative and not allow_negative:
                parser.raise_error(
                    "Expected non-negative integer values", at_position=self.span
                )
            if isinstance(self.value, bool) and not allow_booleans:
                parser.raise_error(
                    "Boolean values are only allowed for i1 types",
                    at_position=self.span,
                )
            if not isinstance(self.value, bool | int):
                parser.raise_error("Expected integer value", at_position=self.span)
            if self.is_negative:
                return -int(self.value)
            return int(self.value)

        def to_float(self, parser: Parser) -> float:
            """
            Convert the element to a float value. Raises an error if the type
            is compatible.
            """
            if self.is_negative:
                return -float(self.value)
            return float(self.value)

        def to_type(self, parser: Parser, type: AnyFloat | IntegerType | IndexType):
            if isinstance(type, AnyFloat):
                return self.to_float(parser)

            match type:
                case IntegerType():
                    return self.to_int(
                        parser,
                        type.signedness.data != Signedness.UNSIGNED,
                        type.width.data == 1,
                    )
                case IndexType():
                    return self.to_int(
                        parser, allow_negative=True, allow_booleans=False
                    )

    def _parse_tensor_literal_element(self) -> _TensorLiteralElement:
        """
        Parse a tensor literal element, which can be a boolean, an integer
        literal, or a float literal.
        """
        # boolean case
        if self._current_token.text == "true":
            token = self._consume_token(Token.Kind.BARE_IDENT)
            return self._TensorLiteralElement(False, True, token.span)
        if self._current_token.text == "false":
            token = self._consume_token(Token.Kind.BARE_IDENT)
            return self._TensorLiteralElement(False, False, token.span)

        # checking for negation
        is_negative = False
        if self._parse_optional_token(Token.Kind.MINUS) is not None:
            is_negative = True

        # Integer and float case
        if self._current_token.kind == Token.Kind.FLOAT_LIT:
            token = self._consume_token(Token.Kind.FLOAT_LIT)
            value = token.get_float_value()
        elif self._current_token.kind == Token.Kind.INTEGER_LIT:
            token = self._consume_token(Token.Kind.INTEGER_LIT)
            value = token.get_int_value()
        else:
            self.raise_error("Expected either a float, integer, or complex literal")

        if is_negative:
            value = -value
        return self._TensorLiteralElement(is_negative, value, token.span)

    def _parse_tensor_literal(
        self,
    ) -> tuple[list[Parser._TensorLiteralElement], list[int]]:
        """
        Parse a tensor literal, and returns its flatten data and its shape.

        For instance, [[0, 1, 2], [3, 4, 5]] will return [0, 1, 2, 3, 4, 5] for
        the data, and [2, 3] for the shape.
        """
        if self._current_token.kind == Token.Kind.L_SQUARE:
            res = self.parse_comma_separated_list(
                self.Delimiter.SQUARE, self._parse_tensor_literal
            )
            if len(res) == 0:
                return [], [0]
            sub_literal_shape = res[0][1]
            if any(r[1] != sub_literal_shape for r in res):
                self.raise_error(
                    "Tensor literal has inconsistent ranks between elements"
                )
            shape = [len(res)] + sub_literal_shape
            values = [elem for sub_list in res for elem in sub_list[0]]
            return values, shape
        else:
            element = self._parse_tensor_literal_element()
            return [element], []

    def _parse_optional_symref_attr(self) -> SymbolRefAttr | None:
        """
        Parse a symbol reference attribute, if present.
          symbol-attr ::= symbol-ref-id (`::` symbol-ref-id)*
          symbol-ref-id ::= at-ident
        """
        # Parse the root symbol
        sym_root = self.parse_optional_symbol_name()
        if sym_root is None:
            return None

        # Parse nested symbols
        refs: list[StringAttr] = []
        while self._current_token.kind == Token.Kind.COLON:
            # Parse `::`. As in MLIR, this require to backtrack if a single `:` is given.
            pos = self._current_token.span.start
            self._consume_token(Token.Kind.COLON)
            if self._parse_optional_token(Token.Kind.COLON) is None:
                self.resume_from(pos)
                break

            refs.append(self.parse_symbol_name())

        return SymbolRefAttr(sym_root, ArrayAttr(refs))

    def parse_optional_builtin_int_or_float_attr(
        self,
    ) -> AnyIntegerAttr | AnyFloatAttr | None:
        bool = self.try_parse_builtin_boolean_attr()
        if bool is not None:
            return bool

        # Parse the value
        if (value := self.parse_optional_number()) is None:
            return None

        # If no types are given, we take the default ones
        if self._current_token.kind != Token.Kind.COLON:
            if isinstance(value, float):
                return FloatAttr(value, Float64Type())
            return IntegerAttr(value, i64)

        # Otherwise, we parse the attribute type
        type = self._parse_attribute_type()

        if isinstance(type, AnyFloat):
            return FloatAttr(float(value), type)

        if isinstance(type, IntegerType | IndexType):
            if isinstance(value, float):
                self.raise_error("Floating point value is not valid for integer type.")
            return IntegerAttr(value, type)

        self.raise_error("Invalid type given for integer or float attribute.")

    def try_parse_builtin_boolean_attr(
        self,
    ) -> IntegerAttr[IntegerType | IndexType] | None:
        if (value := self.parse_optional_boolean()) is not None:
            return IntegerAttr(1 if value else 0, IntegerType(1))
        return None

    def _parse_optional_string_attr(self) -> StringAttr | None:
        """
        Parse a string attribute, if present.
          string-attr ::= string-literal
        """
        token = self._parse_optional_token(Token.Kind.STRING_LIT)
        return (
            StringAttr(token.get_string_literal_value()) if token is not None else None
        )

    def _parse_optional_array_attr(self) -> AnyArrayAttr | None:
        """
        Parse an array attribute, if present, with format:
            array-attr ::= `[` (attribute (`,` attribute)*)? `]`
        """
        if self._current_token.kind != Token.Kind.L_SQUARE:
            return None
        attrs = self.parse_comma_separated_list(
            self.Delimiter.SQUARE, self.parse_attribute
        )
        return ArrayAttr(attrs)

    def parse_optional_dictionary_attr_dict(self) -> dict[str, Attribute]:
        if self._current_token.kind != Token.Kind.L_BRACE:
            return dict()
        attrs = self.parse_comma_separated_list(
            self.Delimiter.BRACES, self._parse_attribute_entry
        )
        return dict(attrs)

    def _parse_function_type(self) -> FunctionType:
        """
        Parse a function type.
            function-type ::= type-list `->` (type | type-list)
            type-list     ::= `(` `)` | `(` type (`,` type)* `)`
        """
        return self.expect(
            self._parse_optional_function_type,
            "function type expected",
        )

    def _parse_optional_function_type(self) -> FunctionType | None:
        """
        Parse a function type, if present.
            function-type ::= type-list `->` (type | type-list)
            type-list     ::= `(` `)` | `(` type (`,` type)* `)`
        """
        if self._current_token.kind != Token.Kind.L_PAREN:
            return None

        # Parse the arguments
        args = self.parse_comma_separated_list(self.Delimiter.PAREN, self.parse_type)

        self.parse_punctuation("->")

        # Parse the returns
        if self._current_token.kind == Token.Kind.L_PAREN:
            returns = self.parse_comma_separated_list(
                self.Delimiter.PAREN, self.parse_type
            )
        else:
            returns = [self.parse_type()]
        return FunctionType.from_lists(args, returns)

    def parse_paramattr_parameters(
        self, skip_white_space: bool = True
    ) -> list[Attribute]:
        if self._current_token.kind != Token.Kind.LESS:
            return []
        res = self.parse_comma_separated_list(
            self.Delimiter.ANGLE, self.parse_attribute
        )
        return res

    def parse_op(self) -> Operation:
        return self.parse_operation()

    def _parse_optional_builtin_dict_attr(self) -> DictionaryAttr | None:
        """
        Parse a dictionary attribute, if present, with format:
        `dictionary-attr ::= `{` ( attribute-entry (`,` attribute-entry)* )? `}`
        `attribute-entry` := (bare-id | string-literal) `=` attribute
        """
        if self._current_token.kind != Token.Kind.L_BRACE:
            return None
        param = DictionaryAttr.parse_parameter(self)
        return DictionaryAttr(param)

    def _parse_builtin_dict_attr(self) -> DictionaryAttr:
        """
        Parse a dictionary attribute with format:
        `dictionary-attr ::= `{` ( attribute-entry (`,` attribute-entry)* )? `}`
        `attribute-entry` := (bare-id | string-literal) `=` attribute
        """
        param = DictionaryAttr.parse_parameter(self)
        return DictionaryAttr(param)

    def parse_optional_attr_dict_with_keyword(
        self, reserved_attr_names: Iterable[str] = ()
    ) -> DictionaryAttr | None:
        """
        Parse a dictionary attribute, preceeded with `attributes` keyword, if the
        keyword is present.
        This is intended to be used in operation custom assembly format.
        `reserved_attr_names` contains names that should not be present in the attribute
        dictionary, and usually correspond to the names of the attributes that are
        already passed through the operation custom assembly format.
        """
        begin_pos = self.lexer.pos
        if self.parse_optional_keyword("attributes") is None:
            return None
        attr = self._parse_builtin_dict_attr()
        for reserved_name in reserved_attr_names:
            if reserved_name in attr.data:
                self.raise_error(
                    f"Attribute dictionary entry '{reserved_name}' is already passed "
                    "through the operation custom assembly format.",
                    Span(begin_pos, begin_pos, self.lexer.input),
                )
        return attr

    def parse_optional_punctuation(
        self, punctuation: Token.PunctuationSpelling
    ) -> Token.PunctuationSpelling | None:
        """
        Parse a punctuation, if it is present. Otherwise, return None.
        Punctuations are defined by `Token.PunctuationSpelling`.
        """
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert Token.Kind.is_spelling_of_punctuation(punctuation), (
            "'parse_optional_punctuation' must be " "called with a valid punctuation"
        )
        kind = Token.Kind.get_punctuation_kind_from_spelling(punctuation)
        if self._parse_optional_token(kind) is not None:
            return punctuation
        return None

    def parse_punctuation(
        self, punctuation: Token.PunctuationSpelling, context_msg: str = ""
    ) -> Token.PunctuationSpelling:
        """
        Parse a punctuation. Punctuations are defined by
        `Token.PunctuationSpelling`.
        """
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert Token.Kind.is_spelling_of_punctuation(
            punctuation
        ), "'parse_punctuation' must be called with a valid punctuation"
        kind = Token.Kind.get_punctuation_kind_from_spelling(punctuation)
        self._parse_token(kind, f"Expected '{punctuation}'" + context_msg)
        return punctuation

    _builtin_integer_type_regex = re.compile(r"^[su]?i(\d+)$")
    _builtin_float_type_regex = re.compile(r"^f(\d+)$")

    def _parse_optional_integer_or_float_type(self) -> Attribute | None:
        """
        Parse as integer or float type, if present.
          integer-or-float-type ::= index-type | integer-type | float-type
          index-type            ::= `index`
          integer-type          ::= (`i` | `si` | `ui`) decimal-literal
          float-type            ::= `f16` | `f32` | `f64` | `f80` | `f128` | `bf16`
        """
        if self._current_token.kind != Token.Kind.BARE_IDENT:
            return None
        name = self._current_token.text

        # Index type
        if name == "index":
            self._consume_token()
            return IndexType()

        # Integer type
        if (match := self._builtin_integer_type_regex.match(name)) is not None:
            signedness = {
                "s": Signedness.SIGNED,
                "u": Signedness.UNSIGNED,
                "i": Signedness.SIGNLESS,
            }
            self._consume_token()
            return IntegerType(int(match.group(1)), signedness[name[0]])

        # bf16 type
        if name == "bf16":
            self._consume_token()
            return BFloat16Type()

        # Float type
        if (re_match := self._builtin_float_type_regex.match(name)) is not None:
            width = int(re_match.group(1))
            type = {
                16: Float16Type,
                32: Float32Type,
                64: Float64Type,
                80: Float80Type,
                128: Float128Type,
            }.get(width, None)
            if type is None:
                self.raise_error("Unsupported floating point width: {}".format(width))
            self._consume_token()
            return type()

        return None

    def _parse_optional_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type, like i32, index, vector<i32>, if present.
        """

        # Check for a function type
        if (function_type := self._parse_optional_function_type()) is not None:
            return function_type

        # Check for an integer or float type
        if (number_type := self._parse_optional_integer_or_float_type()) is not None:
            return number_type

        return self._parse_optional_builtin_parametrized_type()

    def parse_optional_operation(self) -> Operation | None:
        """
        Parse an operation, if present, with format:
            operation             ::= op-result-list? (generic-operation | custom-operation)
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      region-list? dictionary-attribute? `:` function-type
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        """
        if self._current_token.kind not in (
            Token.Kind.PERCENT_IDENT,
            Token.Kind.BARE_IDENT,
            Token.Kind.STRING_LIT,
        ):
            return None
        return self.parse_operation()

    def parse_operation(self) -> Operation:
        """
        Parse an operation with format:
            operation             ::= op-result-list? (generic-operation | custom-operation)
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      region-list? dictionary-attribute? `:` function-type
            custom-operation      ::= bare-id custom-operation-format
            op-result-list        ::= op-result (`,` op-result)* `=`
            op-result             ::= value-id (`:` integer-literal)
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id (`:` block-arg-list)?
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        """
        # Parse the operation results
        bound_results = self._parse_op_result_list()

        if (op_name := self._parse_optional_token(Token.Kind.BARE_IDENT)) is not None:
            # Custom operation format
            op_type = self._get_op_by_name(op_name.text)
            op = op_type.parse(self)
        else:
            # Generic operation format
            op_name = self.expect(
                self.parse_optional_str_literal, "operation name expected"
            )
            op_type = self._get_op_by_name(op_name)
            op = self._parse_generic_operation(op_type)

        n_bound_results = sum(r[1] for r in bound_results)
        if (n_bound_results != 0) and (len(op.results) != n_bound_results):
            self.raise_error(
                f"Operation has {len(op.results)} results, "
                f"but was given {n_bound_results} to bind."
            )

        # Register the result SSA value names in the parser
        res_idx = 0
        for res_span, res_size in bound_results:
            ssa_val_name = res_span.text[1:]  # Removing the leading '%'
            self._register_ssa_definition(
                ssa_val_name, op.results[res_idx : res_idx + res_size], res_span
            )
            res_idx += res_size

        return op

    def _get_op_by_name(self, name: str) -> type[Operation]:
        """
        Get an operation type by its name.
        Raises an error if the operation is not registered, and if unregistered
        dialects are not allowed.
        """
        op_type = self.ctx.get_optional_op(
            name, allow_unregistered=self.allow_unregistered_dialect
        )

        if op_type is not None:
            return op_type

        self.raise_error(f"unregistered operation {name}!")

    def _parse_op_result(self) -> tuple[Span, int]:
        """
        Parse an operation result.
        Returns the span of the SSA value name (including the `%`), and the size of the
        value tuple (by default 1).
        """
        value_token = self._parse_token(
            Token.Kind.PERCENT_IDENT, "Expected result SSA value!"
        )
        if self._parse_optional_token(Token.Kind.COLON) is None:
            return (value_token.span, 1)

        size_token = self._parse_token(
            Token.Kind.INTEGER_LIT, "Expected SSA value tuple size"
        )
        size = size_token.get_int_value()
        return (value_token.span, size)

    def _parse_op_result_list(self) -> list[tuple[Span, int]]:
        """
        Parse the list of operation results.
        If no results are present, returns an empty list.
        Each result is a tuple of the span of the SSA value name (including the `%`),
        and the size of the value tuple (by default 1).
        """
        if self._current_token.kind == Token.Kind.PERCENT_IDENT:
            res = self.parse_comma_separated_list(
                self.Delimiter.NONE, self._parse_op_result, " in operation result list"
            )
            self.parse_punctuation("=", " after operation result list")
            return res
        return []

    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        return self.parse_optional_dictionary_attr_dict()

    def _parse_generic_operation(self, op_type: type[Operation]) -> Operation:
        """
        Parse an operation with format:
            generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                                      region-list? dictionary-attribute? `:` function-type
            successor-list        ::= `[` successor (`,` successor)* `]`
            successor             ::= caret-id
            region-list           ::= `(` region (`,` region)* `)`
            dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
        """
        # Parse arguments
        args = self._parse_op_args_list()

        # Parse successors
        successors = self.parse_optional_successors()

        # Parse regions
        regions = []
        if self._current_token.kind == Token.Kind.L_PAREN:
            regions = self.parse_comma_separated_list(
                self.Delimiter.PAREN, self.parse_region, " in operation region list"
            )

        # Parse attribute dictionary
        attrs = self.parse_optional_attr_dict()

        self.parse_punctuation(":", "function type signature expected")

        func_type_pos = self._current_token.span.start

        # Parse function type
        func_type = self._parse_function_type()

        if len(args) != len(func_type.inputs):
            self.raise_error(
                f"expected {len(func_type.inputs)} operand types but had {len(args)}",
                func_type_pos,
            )

        operands = [
            self.resolve_operand(operand, type)
            for operand, type in zip(args, func_type.inputs)
        ]

        return op_type.create(
            operands=operands,
            result_types=func_type.outputs.data,
            attributes=attrs,
            successors=successors,
            regions=regions,
        )

    def parse_optional_successor(self) -> Block | None:
        """
        Parse a successor with format:
            successor      ::= caret-id
        """
        block_token = self._parse_optional_token(Token.Kind.CARET_IDENT)
        if block_token is None:
            return None
        name = block_token.text[1:]
        if name not in self.blocks:
            self.forward_block_references[name].append(block_token.span)
            self.blocks[name] = (Block(), None)
        return self.blocks[name][0]

    def parse_successor(self) -> Block:
        """
        Parse a successor with format:
            successor      ::= caret-id
        """
        return self.expect(self.parse_optional_successor, "successor expected")

    def parse_optional_successors(self) -> list[Block] | None:
        """
        Parse a list of successors, if present, with format
            successor-list ::= `[` successor (`,` successor)* `]`
            successor      ::= caret-id
        """
        if self._current_token.kind != Token.Kind.L_SQUARE:
            return None
        return self.parse_successors()

    def parse_successors(self) -> list[Block]:
        """
        Parse a list of successors with format:
            successor-list ::= `[` successor (`,` successor)* `]`
            successor      ::= caret-id
        """
        return self.parse_comma_separated_list(
            self.Delimiter.SQUARE,
            lambda: self.expect(self.parse_successor, "block-id expected"),
        )

    def _parse_op_args_list(self) -> list[UnresolvedOperand]:
        """
        Parse a list of arguments with format:
           args-list ::= `(` value-use-list? `)`
           value-use-list ::= `%` suffix-id (`,` `%` suffix-id)*
        """
        return self.parse_comma_separated_list(
            self.Delimiter.PAREN,
            self.parse_unresolved_operand,
            " in operation argument list",
        )
