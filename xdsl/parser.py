from __future__ import annotations

import contextlib
import functools
import itertools
import math
import re
import sys
import traceback
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO
from typing import (
    Any,
    NoReturn,
    TypeVar,
    Iterable,
    IO,
    cast,
    Literal,
    Sequence,
    Callable,
    overload,
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


save_t = Position


@dataclass
class Tokenizer:
    """
    This class is used to tokenize an Input.

    It provides an interface for backtracking, so you can use:

    with tokenizer.backtracking():
        # Try stuff
        raise ParseError(...)

    and not worry about manually resetting the input position. Backtracking will also
    record errors that happen during backtracking to provide a richer error reporting
    experience.

    It also provides the following methods to inspect the input:

     - next_token(peek) is used to get the next token
        (which just breaks the input as per the rules defined in break_on)
        peek=True doesn't advance the position in the file.
     - next_token_of_pattern(pattern, peek) can be used to get a next token if it
        conforms to a specific pattern. If a literal string is given, it'll check
        if the next characters match. If a regex is given, it will check
        the regex.
     - starts_with(pattern) checks if the input starts with a literal string or
        regex pattern
    """

    input: Input

    pos: Position = field(init=False, default=0)
    """
    The position in the input. Points to the first unconsumed character.
    """

    _break_on: tuple[str, ...] = (
        ".",
        "%",
        " ",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        ":",
        "=",
        "@",
        "?",
        "|",
        "->",
        "-",
        "//",
        "\n",
        "\t",
        "#",
        '"',
        "'",
        ",",
        "!",
        "+",
        "*",
    )
    """
    characters the tokenizer should break on
    """

    history: BacktrackingHistory | None = field(init=False, default=None, repr=False)

    last_token: Span | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.last_token = self.next_token(peek=True)

    def save(self) -> save_t:
        """
        Create a checkpoint in the parsing process, useful for backtracking
        """
        return self.pos

    def resume_from(self, save: save_t):
        """
        Resume from a previously saved position.

        Restores the state of the tokenizer to the exact previous position
        """
        self.pos = save

    def _history_entry_from_exception(
        self, ex: Exception, region: str | None, pos: Position
    ) -> BacktrackingHistory:
        """
        Given an exception generated inside a backtracking attempt,
        generate a BacktrackingHistory object with the relevant information in it.

        If an unexpected exception type is encountered, print a traceback to stderr
        """
        assert self.last_token is not None
        if isinstance(ex, ParseError):
            return BacktrackingHistory(ex, self.history, region, pos)
        elif isinstance(ex, AssertionError):
            reason = [
                "Generic assertion failure",
                *(reason for reason in ex.args if isinstance(reason, str)),
            ]
            # We assume that assertions fail because of the last read-in token
            if len(reason) == 1:
                tb = StringIO()
                traceback.print_exc(file=tb)
                reason[0] += "\n" + tb.getvalue()

            return BacktrackingHistory(
                ParseError(self.last_token, reason[-1], self.history),
                self.history,
                region,
                pos,
            )
        elif isinstance(ex, EOFError):
            return BacktrackingHistory(
                ParseError(self.last_token, "Encountered EOF", self.history),
                self.history,
                region,
                pos,
            )

        print("Warning: Unexpected error in backtracking:", file=sys.stderr)
        traceback.print_exception(ex, file=sys.stderr)

        return BacktrackingHistory(
            ParseError(
                self.last_token, "Unexpected exception: {}".format(ex), self.history
            ),
            self.history,
            region,
            pos,
        )

    def next_token(self, peek: bool = False) -> Span:
        """
        Return a Span of the next token, according to the self.break_on rules.

        Can be modified using:
         - peek: don't advance the position, only "peek" at the input

        This will skip over line comments. Meaning it will skip the entire line if
        it encounters '//'
        """
        i = self.next_pos()
        # Construct the span:
        span = Span(i, self._find_token_end(i), self.input)
        # Advance pointer if not peeking
        if not peek:
            self.pos = span.end

        # Save last token
        self.last_token = span
        return span

    def next_token_of_pattern(
        self, pattern: re.Pattern[str] | str, peek: bool = False
    ) -> Span | None:
        """
        Return a span that matched the pattern, or nothing.
        You can choose not to consume the span.
        """
        try:
            start = self.next_pos()
        except EOFError:
            return None

        # Handle search for string literal
        if isinstance(pattern, str):
            if self.starts_with(pattern):
                if not peek:
                    self.pos = start + len(pattern)
                return Span(start, start + len(pattern), self.input)
            return None

        # Handle regex logic
        match = pattern.match(self.input.content, start)
        if match is None:
            return None

        if not peek:
            self.pos = match.end()

        # Save last token
        self.last_token = Span(start, match.end(), self.input)
        return self.last_token

    def consume_peeked(self, peeked_span: Span):
        if peeked_span.start != self.next_pos():
            raise ParseError(peeked_span, "This is not the peeked span!")
        self.pos = peeked_span.end

    def _find_token_end(self, start: Position | None = None) -> Position:
        """
        Find the point (optionally starting from start) where the token ends
        """
        i = self.next_pos() if start is None else start
        # Search for literal breaks
        for part in self._break_on:
            if self.input.content.startswith(part, i):
                return i + len(part)
        # Otherwise return the start of the next break
        break_pos = list(
            filter(
                lambda x: x >= 0,
                (self.input.content.find(part, i) for part in self._break_on),
            )
        )
        # Make sure that we break at some point
        break_pos.append(self.input.len)
        return min(break_pos)

    def next_pos(self, i: Position | None = None) -> Position:
        """
        Find the next starting position (optionally starting from i)

        This will skip line comments!
        """
        i = self.pos if i is None else i
        # Skip whitespaces
        while (c := self.input.at(i)) is not None and c.isspace():
            i += 1
        if c is None:
            raise EOFError()

        # Skip comments as well
        if self.input.content.startswith("//", i):
            i = self.input.content.find("\n", i) + 1
            return self.next_pos(i)

        return i

    def is_eof(self):
        """
        Check if the end of the input was reached.
        """
        try:
            self.next_pos()
            return False
        except EOFError:
            return True

    def starts_with(self, text: str | re.Pattern[str]) -> bool:
        try:
            start = self.next_pos()
            if isinstance(text, re.Pattern):
                return text.match(self.input.content, start) is None
            return self.input.content.startswith(text, start)
        except EOFError:
            return False


class ParserCommons:
    """
    Collection of common things used in parsing MLIR/IRDL

    """

    integer_literal = re.compile(r"[+-]?([0-9]+|0x[0-9A-Fa-f]+)")
    decimal_literal = re.compile(r"[+-]?([1-9][0-9]*)")
    string_literal = re.compile(r'"(\\[nfvtr"\\]|[^\n\f\v\r"\\])*"')
    float_literal = re.compile(r"[-+]?[0-9]+\.[0-9]*([eE][-+]?[0-9]+)?")
    bare_id = re.compile(r"[A-Za-z_][\w$.]*")
    value_id = re.compile(r"%([0-9]+|([A-Za-z_$.-][\w$.-]*))")
    suffix_id = re.compile(r"([0-9]+|([A-Za-z_$.-][\w$.-]*))")
    """
    suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))
    id-punct  ::= [$._-]
    """
    block_id = re.compile(r"\^([0-9]+|([A-Za-z_$.-][\w$.-]*))")
    type_alias = re.compile(r"![A-Za-z_][\w$.]+")
    attribute_alias = re.compile(r"#[A-Za-z_][\w$.]+")
    boolean_literal = re.compile(r"(true|false)")
    # A list of names that are builtin types
    _builtin_type_names = (
        r"[su]?i\d+",
        "bf16",
        r"f\d+",
        "tensor",
        "vector",
        "memref",
        "complex",
        "opaque",
        "tuple",
        "index",
        "dense"
        # TODO: add all the Float8E4M3FNType, Float8E5M2Type, and BFloat16Type
    )
    builtin_attr_names = (
        "dense",
        "opaque",
        "affine_set",
        "affine_map",
        "array",
        "dense_resource",
        "sparse",
    )
    builtin_type = re.compile("(({}))".format(")|(".join(_builtin_type_names)))
    builtin_type_xdsl = re.compile("!(({}))".format(")|(".join(_builtin_type_names)))
    double_colon = re.compile("::")
    comma = re.compile(",")


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

    lexer: Lexer
    tokenizer: Tokenizer

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
        self.tokenizer = Tokenizer(Input(input, name))
        self.lexer = Lexer(Input(input, name))
        self._current_token = self.lexer.lex()
        self.ctx = ctx
        self.ssa_values = dict()
        self.blocks = dict()
        self.forward_block_references = dict()
        self.allow_unregistered_dialect = allow_unregistered_dialect

    def resume_from(self, pos: Position):
        """
        Resume parsing from a given position.
        """
        self.tokenizer.pos = pos
        self.lexer.pos = pos
        self._current_token = self.lexer.lex()

    @contextlib.contextmanager
    def backtracking(self, region_name: str | None = None):
        """
        This context manager can be used to mark backtracking regions.

        When an error is thrown during backtracking, it is recorded and stored together
        with some meta information in the history attribute.

        The backtracker accepts the following exceptions:
        - ParseError: signifies that the region could not be parsed because of
          (unexpected) syntax errors
        - AssertionError: this error should probably be phased out in favour
          of the two above
        - EOFError: signals that EOF was reached unexpectedly

        Any other error will be printed to stderr, but backtracking will continue
        as normal.
        """
        self._synchronize_lexer_and_tokenizer()
        save = self.tokenizer.save()
        starting_position = self.tokenizer.pos
        try:
            yield
            # Clear error history when something doesn't fail
            # This is because we are only interested in the last "cascade" of failures.
            # If a backtracking() completes without failure,
            # something has been parsed (we assume)
            if (
                self.tokenizer.pos > starting_position
                and self.tokenizer.history is not None
            ):
                self.tokenizer.history = None
        except Exception as ex:
            how_far_we_got = self.tokenizer.pos

            # If we have no error history, start recording!
            if not self.tokenizer.history:
                self.tokenizer.history = (
                    self.tokenizer._history_entry_from_exception(  # type: ignore
                        ex, region_name, how_far_we_got
                    )
                )

            # If we got further than on previous attempts
            elif how_far_we_got > self.tokenizer.history.get_farthest_point():
                # Throw away history
                self.tokenizer.history = None
                # Generate new history entry,
                self.tokenizer.history = (
                    self.tokenizer._history_entry_from_exception(  # type: ignore
                        ex, region_name, how_far_we_got
                    )
                )

            # Otherwise, add to exception, if we are in a named region
            elif region_name is not None and how_far_we_got - starting_position > 0:
                self.tokenizer.history = (
                    self.tokenizer._history_entry_from_exception(  # type: ignore
                        ex, region_name, how_far_we_got
                    )
                )

            self.resume_from(save)

    def _synchronize_lexer_and_tokenizer(self):
        """
        Advance the lexer and the tokenizer to the same position,
        which is the maximum of the two.
        This is used to allow using both the tokenizer and the lexer,
        to deprecate slowly the tokenizer.
        """
        lexer_pos = self._current_token.span.start
        tokenizer_pos = self.tokenizer.save()
        pos = max(lexer_pos, tokenizer_pos)
        self.lexer.pos = pos
        self.tokenizer.pos = pos
        self._current_token = self.lexer.lex()
        # Make sure both point to the same position,
        # to avoid having problems with `backtracking`.
        if self._current_token.span.start > self.tokenizer.pos:
            self.tokenizer.pos = self._current_token.span.start

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
        op = self.try_parse_operation()

        if op is None:
            self.raise_error("Could not parse entire input!")

        if isinstance(op, ModuleOp):
            return op
        else:
            self.tokenizer.pos = 0
            self.raise_error(
                "Expected ModuleOp at top level!", self.tokenizer.next_token()
            )

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

    def _parse_ssa_definition(self) -> str:
        """
        Parse an SSA definition, with the format `%ident`. Returns the value name.
        If a value with the same name is already in scope, return an error.
        """
        name_token = self._parse_token(
            Token.Kind.PERCENT_IDENT, "Expected result SSA value!"
        )
        name = name_token.text[1:]
        if name in self.ssa_values:
            self.raise_error(
                f"a value with name '{name}' is already defined", name_token.span
            )
        return name

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
            arg_name = self._parse_ssa_definition()
            self.parse_punctuation(":")
            self._synchronize_lexer_and_tokenizer()
            arg_type = self.parse_attribute()
            self._synchronize_lexer_and_tokenizer()

            # Insert the block argument in the block, and set its name.
            block_arg = block.insert_arg(arg_type, len(block.args))
            if SSAValue.is_valid_name(arg_name):
                block_arg.name_hint = arg_name

            # Register the value name in the parser
            self.ssa_values[arg_name] = (block_arg,)

        self.parse_comma_separated_list(self.Delimiter.PAREN, parse_argument)
        return block

    def _parse_block_body(self, block: Block):
        """
        Parse a block body, which consist of a list of operations.
        The operations are added at the end of the block.
        """
        self._synchronize_lexer_and_tokenizer()
        while (op := self.try_parse_operation()) is not None:
            self._synchronize_lexer_and_tokenizer()
            block.add_op(op)

    def _parse_block(self) -> Block:
        """
        Parse a block with the following format:
          block ::= block-label operation*
          block-label    ::= block-id block-arg-list? `:`
          block-id       ::= caret-id
          block-arg-list ::= `(` ssa-id-and-type-list? `)`
        """
        self._synchronize_lexer_and_tokenizer()
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
        self._synchronize_lexer_and_tokenizer()
        return block

    def parse_optional_symbol_name(self) -> StringAttr | None:
        """
        Parse an @-identifier if present, and return its name (without the '@') in a
        string attribute.
        """
        self._synchronize_lexer_and_tokenizer()
        if (token := self._parse_optional_token(Token.Kind.AT_IDENT)) is None:
            return None
        self._synchronize_lexer_and_tokenizer()

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
        self._synchronize_lexer_and_tokenizer()
        if delimiter == self.Delimiter.NONE:
            pass
        elif delimiter == self.Delimiter.PAREN:
            self._parse_token(Token.Kind.L_PAREN, "Expected '('" + context_msg)
            if self._parse_optional_token(Token.Kind.R_PAREN) is not None:
                self._synchronize_lexer_and_tokenizer()
                return []
        elif delimiter == self.Delimiter.ANGLE:
            self._parse_token(Token.Kind.LESS, "Expected '<'" + context_msg)
            if self._parse_optional_token(Token.Kind.GREATER) is not None:
                self._synchronize_lexer_and_tokenizer()
                return []
        elif delimiter == self.Delimiter.SQUARE:
            self._parse_token(Token.Kind.L_SQUARE, "Expected '['" + context_msg)
            if self._parse_optional_token(Token.Kind.R_SQUARE) is not None:
                self._synchronize_lexer_and_tokenizer()
                return []
        elif delimiter == self.Delimiter.BRACES:
            self._parse_token(Token.Kind.L_BRACE, "Expected '{'" + context_msg)
            if self._parse_optional_token(Token.Kind.R_BRACE) is not None:
                self._synchronize_lexer_and_tokenizer()
                return []
        else:
            assert False, "Unknown delimiter"

        self._synchronize_lexer_and_tokenizer()
        elems = [parse()]
        self._synchronize_lexer_and_tokenizer()
        while self._parse_optional_token(Token.Kind.COMMA) is not None:
            self._synchronize_lexer_and_tokenizer()
            elems.append(parse())
            self._synchronize_lexer_and_tokenizer()

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

        self._synchronize_lexer_and_tokenizer()
        return elems

    def parse_optional_boolean(self) -> bool | None:
        """
        Parse a boolean, if present, with the format `true` or `false`.
        """
        self._synchronize_lexer_and_tokenizer()
        if self._current_token.kind == Token.Kind.BARE_IDENT:
            if self._current_token.text == "true":
                self._consume_token(Token.Kind.BARE_IDENT)
                self._synchronize_lexer_and_tokenizer()
                return True
            elif self._current_token.text == "false":
                self._consume_token(Token.Kind.BARE_IDENT)
                self._synchronize_lexer_and_tokenizer()
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
        self._synchronize_lexer_and_tokenizer()
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
            self._synchronize_lexer_and_tokenizer()
            return None

        # Get the value and optionally negate it
        value = int_token.get_int_value()
        if is_negative:
            value = -value
        self._synchronize_lexer_and_tokenizer()
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
            "Expected integer or float literal" + context_msg,
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

    def try_parse_integer_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.integer_literal)

    def try_parse_decimal_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.decimal_literal)

    def try_parse_string_literal(self) -> StringLiteral | None:
        return StringLiteral.from_span(
            self.tokenizer.next_token_of_pattern(ParserCommons.string_literal)
        )

    def parse_string_literal(self) -> str:
        """
        Parse a string literal with the format `"..."`.

        Returns the string contents without the quotes and with escape sequences
        resolved.
        """
        return self.expect(
            self.try_parse_string_literal,
            "string literal expected",
        ).string_contents

    def try_parse_float_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.float_literal)

    def try_parse_bare_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.bare_id)

    def try_parse_value_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.value_id)

    _decimal_integer_regex = re.compile(r"[0-9]+")

    def parse_optional_operand(self) -> SSAValue | None:
        """
        Parse an operand with format `%<value-id>(#<int-literal>)?`, if present.
        """
        self._synchronize_lexer_and_tokenizer()
        name_token = self._parse_optional_token(Token.Kind.PERCENT_IDENT)
        if name_token is None:
            return None
        name = name_token.text[1:]

        index = 0
        index_token = self._parse_optional_token(Token.Kind.HASH_IDENT)
        if index_token is not None:
            if re.fullmatch(self._decimal_integer_regex, index_token.text[1:]) is None:
                self.raise_error(
                    "Expected integer as SSA value tuple index", index_token.span
                )
            index = int(index_token.text[1:], 10)

        if name not in self.ssa_values.keys():
            self.raise_error("SSA value used before assignment", name_token.span)

        tuple_size = len(self.ssa_values[name])
        if index >= tuple_size:
            assert index_token is not None, "Fatal error in SSA value parsing"
            self.raise_error(
                "SSA value tuple index out of bounds. "
                f"Tuple is of size {tuple_size} but tried to access element {index}.",
                index_token.span,
            )

        self._synchronize_lexer_and_tokenizer()
        return self.ssa_values[name][index]

    def parse_operand(self, msg: str = "Expected an operand.") -> SSAValue:
        """Parse an operand with format `%<value-id>`."""
        return self.expect(self.parse_optional_operand, msg)

    def try_parse_suffix_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.suffix_id)

    def try_parse_block_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.block_id)

    def try_parse_boolean_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.boolean_literal)

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
        self._synchronize_lexer_and_tokenizer()
        if (
            token := self._parse_optional_token(Token.Kind.EXCLAMATION_IDENT)
        ) is not None:
            return self._parse_dialect_type_or_attribute_inner(token.text[1:], True)
        return self.try_parse_builtin_type()

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
        self._synchronize_lexer_and_tokenizer()
        if (token := self._parse_optional_token(Token.Kind.HASH_IDENT)) is not None:
            return self._parse_dialect_type_or_attribute_inner(token.text[1:], False)
        return self.try_parse_builtin_attr()

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
            self._synchronize_lexer_and_tokenizer()
            return attr_def(attr_name, is_type, body)
        if issubclass(attr_def, ParametrizedAttribute):
            self._synchronize_lexer_and_tokenizer()
            param_list = attr_def.parse_parameters(self)
            self._synchronize_lexer_and_tokenizer()
            return attr_def.new(param_list)
        if issubclass(attr_def, Data):
            self.parse_punctuation("<")
            self._synchronize_lexer_and_tokenizer()
            param: Any = attr_def.parse_parameter(self)
            self.parse_punctuation(">")
            self._synchronize_lexer_and_tokenizer()
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

    def _parse_builtin_parametrized_type(self, name: Span) -> ParametrizedAttribute:
        """
        This function is called after we parse the name of a parameterized type
        such as vector.
        """

        def unimplemented() -> ParametrizedAttribute:
            raise ParseError(name, "Builtin {} not supported yet!".format(name.text))

        builtin_parsers: dict[str, Callable[[], ParametrizedAttribute]] = {
            "vector": self.parse_vector_attrs,
            "memref": self.parse_memref_attrs,
            "tensor": self.parse_tensor_attrs,
            "complex": self.parse_complex_attrs,
            "tuple": unimplemented,
        }

        self.parse_characters("<", "Expected parameter list here!")
        # Get the parser for the type, falling back to the unimplemented warning
        self._synchronize_lexer_and_tokenizer()
        res = builtin_parsers.get(name.text, unimplemented)()
        self._synchronize_lexer_and_tokenizer()
        self.parse_characters(">", "Expected end of parameter list here!")

        return res

    def _parse_shape_dimension(self, allow_dynamic: bool = True) -> int:
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

    def _parse_shape_delimiter(self) -> None:
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
        self._synchronize_lexer_and_tokenizer()
        dims: list[int] = []
        while self._current_token.kind in (Token.Kind.INTEGER_LIT, Token.Kind.QUESTION):
            dim = self._parse_shape_dimension()
            dims.append(dim)
            self._parse_shape_delimiter()

        self._synchronize_lexer_and_tokenizer()
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
        self._synchronize_lexer_and_tokenizer()
        if self.parse_optional_punctuation("*") is not None:
            self._parse_shape_delimiter()
            type = self.expect(self.parse_optional_type, "Expected shape type.")
            self._synchronize_lexer_and_tokenizer()
            return None, type
        res = self.parse_ranked_shape()
        self._synchronize_lexer_and_tokenizer()
        return res

    def parse_complex_attrs(self) -> ComplexType:
        element_type = self.parse_attribute()
        if not isa(element_type, IntegerType | AnyFloat):
            self.raise_error(
                "Complex type must be parameterized by an integer or float type!"
            )
        return ComplexType(element_type)

    def parse_memref_attrs(
        self,
    ) -> MemRefType[Attribute] | UnrankedMemrefType[Attribute]:
        shape, type = self.parse_shape()

        # Unranked case
        if shape is None:
            if self.parse_optional_punctuation(",") is None:
                self._synchronize_lexer_and_tokenizer()
                return UnrankedMemrefType.from_type(type)
            self._synchronize_lexer_and_tokenizer()
            memory_space = self.parse_attribute()
            self._synchronize_lexer_and_tokenizer()
            return UnrankedMemrefType.from_type(type, memory_space)

        if self.parse_optional_punctuation(",") is None:
            return MemRefType.from_element_type_and_shape(type, shape)

        self._synchronize_lexer_and_tokenizer()
        memory_or_layout = self.parse_attribute()
        self._synchronize_lexer_and_tokenizer()

        # If there is both a memory space and a layout, we know that the
        # layout is the second one
        if self.parse_optional_punctuation(",") is not None:
            self._synchronize_lexer_and_tokenizer()
            memory_space = self.parse_attribute()
            self._synchronize_lexer_and_tokenizer()
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

    def parse_vector_attrs(self) -> AnyVectorType:
        self._synchronize_lexer_and_tokenizer()

        dims: list[int] = []
        num_scalable_dims = 0
        # First, parse the static dimensions
        while self._current_token.kind == Token.Kind.INTEGER_LIT:
            dims.append(self._parse_shape_dimension(allow_dynamic=False))
            self._parse_shape_delimiter()

        # Then, parse the scalable dimensions, if any
        if self.parse_optional_punctuation("[") is not None:
            # Parse the scalable dimensions
            dims.append(self._parse_shape_dimension(allow_dynamic=False))
            num_scalable_dims += 1

            while self.parse_optional_punctuation("]") is None:
                self._parse_shape_delimiter()
                dims.append(self._parse_shape_dimension(allow_dynamic=False))
                num_scalable_dims += 1

            # Parse the `x` between the scalable dimensions and the type
            self._parse_shape_delimiter()

        self._synchronize_lexer_and_tokenizer()
        type = self.parse_optional_type()
        if type is None:
            self.raise_error("Expected the vector element types!")

        self._synchronize_lexer_and_tokenizer()
        return VectorType.from_element_type_and_shape(type, dims, num_scalable_dims)

    def parse_tensor_attrs(self) -> AnyTensorType | AnyUnrankedTensorType:
        shape, type = self.parse_shape()

        if shape is None:
            if self.parse_optional_punctuation(",") is not None:
                self.raise_error("Unranked tensors don't have an encoding!")
            return UnrankedTensorType.from_type(type)

        self._synchronize_lexer_and_tokenizer()
        if self.parse_optional_punctuation(",") is not None:
            self._synchronize_lexer_and_tokenizer()
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
        self._synchronize_lexer_and_tokenizer()
        if end_position is not None:
            assert isinstance(at_position, Position)
            at_position = Span(at_position, end_position, self.lexer.input)
        if at_position is None:
            at_position = self._current_token.span
        elif isinstance(at_position, Position):
            at_position = Span(at_position, at_position, self.lexer.input)

        raise ParseError(at_position, msg, self.tokenizer.history)

    def try_parse_characters(self, text: str) -> Span | None:
        return self.tokenizer.next_token_of_pattern(text)

    def parse_characters(self, text: str, msg: str) -> Span:
        if (match := self.try_parse_characters(text)) is None:
            self.raise_error(msg)
        return match

    def try_parse_operation(self) -> Operation | None:
        with self.backtracking("operation"):
            return self.parse_operation()

    def parse_operation(self) -> Operation:
        self._synchronize_lexer_and_tokenizer()
        if self._current_token.kind == Token.Kind.PERCENT_IDENT:
            results = self._parse_op_result_list()
        else:
            results = []
        ret_types = [result[2] for result in results]
        if len(results) > 0:
            self.parse_characters(
                "=", "Operation definitions expect an `=` after op-result-list!"
            )

        # Check for custom op format
        op_name = self.try_parse_bare_id()
        if op_name is not None:
            op_type = self._get_op_by_name(op_name)
            op = op_type.parse(self)
        else:
            # Check for basic op format
            op_name = self.try_parse_string_literal()
            if op_name is None:
                self.raise_error(
                    "Expected an operation name here, either a bare-id, or a string "
                    "literal!"
                )

            args, successors, attrs, regions, func_type = self.parse_operation_details()

            if any(res_type is None for res_type in ret_types):
                assert func_type is not None
                ret_types = func_type.outputs.data
            ret_types = cast(Sequence[Attribute], ret_types)

            op_type = self._get_op_by_name(op_name)

            op = op_type.create(
                operands=args,
                result_types=ret_types,
                attributes=attrs,
                successors=[
                    self._get_block_from_name(block_name) for block_name in successors
                ],
                regions=regions,
            )

        expected_results = sum(r[1] for r in results)
        if len(op.results) != expected_results:
            self.raise_error(
                f"Operation has {len(op.results)} results, "
                f"but were given {expected_results} to bind."
            )

        # Register the result SSA value names in the parser
        res_idx = 0
        for res_span, res_size, _ in results:
            ssa_val_name = res_span.text[1:]  # Removing the leading '%'
            if ssa_val_name in self.ssa_values:
                self.raise_error(
                    f"SSA value %{ssa_val_name} is already defined", res_span
                )
            self.ssa_values[ssa_val_name] = tuple(
                op.results[res_idx : res_idx + res_size]
            )
            res_idx += res_size
            # Carry over `ssa_val_name` for non-numeric names:
            if SSAValue.is_valid_name(ssa_val_name):
                for val in self.ssa_values[ssa_val_name]:
                    val.name_hint = ssa_val_name

        return op

    def _get_op_by_name(self, span: Span) -> type[Operation]:
        if isinstance(span, StringLiteral):
            op_name = span.string_contents
        else:
            op_name = span.text

        op_type = self.ctx.get_optional_op(
            op_name, allow_unregistered=self.allow_unregistered_dialect
        )

        if op_type is not None:
            return op_type

        self.raise_error(f"Unknown operation {op_name}!", span)

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
        self._synchronize_lexer_and_tokenizer()
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
                if arg.name.text[1:] in self.ssa_values:
                    self.raise_error(
                        f"block argument %{arg.name} is already defined", arg.name
                    )
                self.ssa_values[arg.name.text[1:]] = (block_arg,)

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
                self.tokenizer.history,
            )

        # Close the value and block scope.
        self.ssa_values = old_ssa_values
        self.blocks = old_blocks
        self.forward_block_references = old_forward_blocks

        self._synchronize_lexer_and_tokenizer()
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

    def _parse_attribute_entry(self) -> tuple[Span, Attribute]:
        """
        Parse entry in attribute dict. Of format:

        attribute_entry := (bare-id | string-literal) `=` attribute
        attribute       := dialect-attribute | builtin-attribute
        """
        if (name := self.try_parse_bare_id()) is None:
            name = self.try_parse_string_literal()

        if name is None:
            self.raise_error(
                "Expected bare-id or string-literal here as part of attribute entry!"
            )

        if not self.tokenizer.starts_with("="):
            return name, UnitAttr()

        self.parse_characters(
            "=", "Attribute entries must be of format name `=` attribute!"
        )

        return name, self.parse_attribute()

    def _parse_attribute_type(self) -> Attribute:
        """
        Parses `:` type and returns the type
        """
        self.parse_characters(
            ":", "Expected attribute type definition here ( `:` type )"
        )
        return self.expect(
            self.parse_optional_type,
            "Expected attribute type definition here ( `:` type )",
        )

    def try_parse_builtin_attr(self) -> Attribute | None:
        """
        Tries to parse a builtin attribute, e.g. a string literal, int, array, etc..
        """
        next_token = self.tokenizer.next_token(peek=True)
        if next_token.text == '"':
            return self.try_parse_builtin_str_attr()
        elif next_token.text == "[":
            return self.try_parse_builtin_arr_attr()
        elif next_token.text == "@":
            return self.parse_optional_symref_attr()
        elif next_token.text == "{":
            return self.parse_builtin_dict_attr()
        elif next_token.text == "(":
            return self.try_parse_function_type()
        elif next_token.text in ParserCommons.builtin_attr_names:
            return self.try_parse_builtin_named_attr()

        attrs = (
            self.parse_optional_builtin_int_or_float_attr,
            self.parse_optional_type,
        )

        for attr_parser in attrs:
            if (val := attr_parser()) is not None:
                return val

        self._synchronize_lexer_and_tokenizer()
        if self._current_token.text == "strided":
            strided = self.parse_strided_layout_attr()
            self._synchronize_lexer_and_tokenizer()
            return strided

        return None

    def _parse_int_or_question(self, context_msg: str = "") -> int | Literal["?"]:
        """Parse either an integer literal, or a '?'."""
        self._synchronize_lexer_and_tokenizer()
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

        self._synchronize_lexer_and_tokenizer()
        if (
            self._current_token.kind == Token.Kind.BARE_IDENT
            and self._current_token.text == keyword
        ):
            self._consume_token(Token.Kind.BARE_IDENT)
            self._synchronize_lexer_and_tokenizer()
            return keyword
        self._synchronize_lexer_and_tokenizer()
        return None

    def parse_strided_layout_attr(self) -> Attribute:
        """
        Parse a strided layout attribute.
        | `strided` `<` `[` comma-separated-int-or-question `]`
          (`,` `offset` `:` integer-literal)? `>`
        """
        # Parse `strided` keyword
        self.parse_keyword("strided")

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

    def try_parse_builtin_named_attr(self) -> Attribute | None:
        name = self.tokenizer.next_token(peek=True)
        with self.backtracking("Builtin attribute {}".format(name.text)):
            self.tokenizer.consume_peeked(name)
            parsers = {
                "dense": self._parse_builtin_dense_attr,
                "opaque": self._parse_builtin_opaque_attr,
                "dense_resource": self._parse_builtin_dense_resource_attr,
                "array": self._parse_builtin_array_attr,
                "affine_map": self._parse_builtin_affine_attr,
                "affine_set": self._parse_builtin_affine_attr,
            }

            def not_implemented(_name: Span):
                raise NotImplementedError()

            return parsers.get(name.text, not_implemented)(name)

    def _parse_builtin_dense_attr(self, _name: Span) -> DenseIntOrFPElementsAttr:
        self._synchronize_lexer_and_tokenizer()
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
        self._synchronize_lexer_and_tokenizer()
        type = self.expect(self.parse_optional_type, "Dense attribute must be typed!")
        self._synchronize_lexer_and_tokenizer()

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
            self.Delimiter.ANGLE, self.parse_string_literal
        )

        if len(str_lit_list) != 2:
            self.raise_error("Opaque expects 2 string literal parameters!")

        type = NoneAttr()
        if self.tokenizer.starts_with(":"):
            self.parse_characters(":", "opaque attribute must be typed!")
            type = self.expect(
                self.parse_optional_type, "opaque attribute must be typed!"
            )

        return OpaqueAttr.from_strings(*str_lit_list, type=type)

    def _parse_builtin_dense_resource_attr(self, _name: Span) -> DenseResourceAttr:
        err_msg = (
            "Malformed dense_resource attribute, format must be "
            "(`dense_resource` `<` resource-handle `>`)"
        )
        self.parse_characters("<", err_msg)
        resource_handle = self.expect(self.try_parse_bare_id, err_msg)
        self.parse_characters(">", err_msg)
        self.parse_characters(":", err_msg)
        type = self.expect(
            self.parse_optional_type, "Dense resource attribute must be typed!"
        )
        return DenseResourceAttr.from_params(resource_handle.text, type)

    def _parse_builtin_array_attr(self, name: Span) -> DenseArrayBase | None:
        err_msg = (
            "Malformed dense array, format must be "
            "`array` `<` (integer-type | float-type) (`:` tensor-literal)? `>`"
        )
        self.parse_characters("<", err_msg)
        element_type = self.parse_attribute()

        if not isinstance(element_type, IntegerType | AnyFloat):
            raise ParseError(
                name,
                "dense array element type must be an " "integer or floating point type",
            )

        # Empty array
        if self.try_parse_characters(">"):
            return DenseArrayBase.from_list(element_type, [])

        self.parse_characters(":", err_msg)

        def parse_dense_array_value() -> int | float:
            if (v := self.try_parse_float_literal()) is not None:
                return float(v.text)
            if (v := self.try_parse_integer_literal()) is not None:
                return int(v.text)
            self.raise_error("integer or float literal expected")

        values = self.parse_comma_separated_list(
            self.Delimiter.NONE, parse_dense_array_value
        )
        self.parse_characters(">", err_msg)

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
        self._synchronize_lexer_and_tokenizer()
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

        self._synchronize_lexer_and_tokenizer()
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
            if not isinstance(self.value, int | float):
                parser.raise_error("Expected float value", at_position=self.span)
            if self.is_negative:
                return -float(self.value)
            return float(self.value)

        def to_type(self, parser: Parser, type: AnyFloat | IntegerType | IndexType):
            if isinstance(type, AnyFloat):
                return self.to_float(parser)
            elif isinstance(type, IntegerType):
                return self.to_int(
                    parser,
                    type.signedness.data != Signedness.UNSIGNED,
                    type.width.data == 1,
                )
            elif isinstance(type, IndexType):
                return self.to_int(parser, allow_negative=True, allow_booleans=False)
            else:
                assert False, "fatal error in parser"

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

    def parse_optional_symref_attr(self) -> SymbolRefAttr | None:
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

        self._synchronize_lexer_and_tokenizer()

        # Parse the value
        if (value := self.parse_optional_number()) is None:
            return None

        self._synchronize_lexer_and_tokenizer()
        # If no types are given, we take the default ones
        if self._current_token.kind != Token.Kind.COLON:
            if isinstance(value, float):
                return FloatAttr(value, Float64Type())
            return IntegerAttr(value, i64)

        # Otherwise, we parse the attribute type
        type = self._parse_attribute_type()
        self._synchronize_lexer_and_tokenizer()

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
        self._synchronize_lexer_and_tokenizer()
        if (value := self.parse_optional_boolean()) is not None:
            self._synchronize_lexer_and_tokenizer()
            return IntegerAttr(1 if value else 0, IntegerType(1))
        return None

    def try_parse_builtin_str_attr(self):
        if not self.tokenizer.starts_with('"'):
            return None

        with self.backtracking("string literal"):
            literal = self.try_parse_string_literal()
            if literal is None:
                self.raise_error("Invalid string literal")
            return StringAttr(literal.string_contents)

    def try_parse_builtin_arr_attr(self) -> AnyArrayAttr | None:
        if not self.tokenizer.starts_with("["):
            return None
        attrs = self.parse_comma_separated_list(
            self.Delimiter.SQUARE, self.parse_attribute
        )
        return ArrayAttr(attrs)

    def parse_optional_dictionary_attr_dict(self) -> dict[str, Attribute]:
        if not self.tokenizer.starts_with("{"):
            return dict()
        attrs = self.parse_comma_separated_list(
            self.Delimiter.BRACES, self._parse_attribute_entry
        )
        return self._attr_dict_from_tuple_list(attrs)

    def _attr_dict_from_tuple_list(
        self, tuple_list: list[tuple[Span, Attribute]]
    ) -> dict[str, Attribute]:
        """
        Convert a list of tuples (Span, Attribute) to a dictionary.
        This function converts the span to a string, trimming quotes from string literals
        """

        def span_to_str(span: Span) -> str:
            if isinstance(span, StringLiteral):
                return span.string_contents
            return span.text

        return dict((span_to_str(span), attr) for span, attr in tuple_list)

    def parse_function_type(self) -> FunctionType:
        """
        Parses function-type:

        viable function types are:
            (i32)   -> ()
            ()      -> (i32, i32)
            (i32, i32) -> ()
            ()      -> i32
        Non-viable types are:
            i32     -> i32
            i32     -> ()

        Uses type-or-type-list-parens internally
        """
        args = self.parse_comma_separated_list(self.Delimiter.PAREN, self.parse_type)

        self.parse_characters("->", "Malformed function type, expected `->`!")

        return FunctionType.from_lists(args, self._parse_type_or_type_list_parens())

    def _parse_type_or_type_list_parens(self) -> list[Attribute]:
        """
        Parses type-or-type-list-parens, which is used in function-type.

        type-or-type-list-parens ::= type | type-list-parens
        type-list-parens         ::= `(` `)` | `(` type-list-no-parens `)`
        type-list-no-parens      ::=  type (`,` type)*
        """
        self._synchronize_lexer_and_tokenizer()
        if self._current_token.kind == Token.Kind.L_PAREN:
            args = self.parse_comma_separated_list(
                self.Delimiter.PAREN, self.parse_type
            )
        else:
            args = [self.parse_type()]
        return args

    def try_parse_function_type(self) -> FunctionType | None:
        if not self.tokenizer.starts_with("("):
            return None
        with self.backtracking("function type"):
            return self.parse_function_type()

    def _parse_builtin_type_with_name(self, name: Span):
        """
        Parses one of the builtin types like i42, vector, etc...
        """
        if name.text == "index":
            return IndexType()
        if (re_match := re.match(r"^[su]?i(\d+)$", name.text)) is not None:
            signedness = {
                "s": Signedness.SIGNED,
                "u": Signedness.UNSIGNED,
                "i": Signedness.SIGNLESS,
            }
            return IntegerType(int(re_match.group(1)), signedness[name.text[0]])

        if name.text == "bf16":
            return BFloat16Type()

        if (re_match := re.match(r"^f(\d+)$", name.text)) is not None:
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
            return type()

        return self._parse_builtin_parametrized_type(name)

    def parse_paramattr_parameters(
        self, skip_white_space: bool = True
    ) -> list[Attribute]:
        self._synchronize_lexer_and_tokenizer()
        if self._current_token.kind != Token.Kind.LESS:
            return []
        res = self.parse_comma_separated_list(
            self.Delimiter.ANGLE, self.parse_attribute
        )
        return res

    def parse_char(self, text: str):
        self.parse_characters(text, "Expected '{}' here!".format(text))

    def parse_str_literal(self) -> str:
        return self.expect(
            self.try_parse_string_literal, "Malformed string literal!"
        ).string_contents

    def parse_op(self) -> Operation:
        return self.parse_operation()

    def parse_builtin_dict_attr(self) -> DictionaryAttr:
        """
        Parse a dictionary attribute, with the following syntax:
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
        self._synchronize_lexer_and_tokenizer()
        begin_pos = self.lexer.pos
        if self.parse_optional_keyword("attributes") is None:
            return None
        self._synchronize_lexer_and_tokenizer()
        attr = self.parse_builtin_dict_attr()
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
        self._synchronize_lexer_and_tokenizer()
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert Token.Kind.is_spelling_of_punctuation(punctuation), (
            "'parse_optional_punctuation' must be " "called with a valid punctuation"
        )
        kind = Token.Kind.get_punctuation_kind_from_spelling(punctuation)
        if self._parse_optional_token(kind) is not None:
            self._synchronize_lexer_and_tokenizer()
            return punctuation
        return None

    def parse_punctuation(
        self, punctuation: Token.PunctuationSpelling, context_msg: str = ""
    ) -> Token.PunctuationSpelling:
        """
        Parse a punctuation. Punctuations are defined by
        `Token.PunctuationSpelling`.
        """
        self._synchronize_lexer_and_tokenizer()
        # This check is only necessary to catch errors made by users that
        # are not using pyright.
        assert Token.Kind.is_spelling_of_punctuation(
            punctuation
        ), "'parse_punctuation' must be called with a valid punctuation"
        kind = Token.Kind.get_punctuation_kind_from_spelling(punctuation)
        self._parse_token(kind, f"Expected '{punctuation}'" + context_msg)
        self._synchronize_lexer_and_tokenizer()
        return punctuation

    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        with self.backtracking("builtin type"):
            # Check the function type separately, it is the only
            # case of a type starting with a symbol
            next_token = self.tokenizer.next_token(peek=True)
            if next_token.text == "(":
                return self.try_parse_function_type()

            name = self.tokenizer.next_token_of_pattern(ParserCommons.builtin_type)
            if name is None:
                self.raise_error("Expected builtin name!")

            return self._parse_builtin_type_with_name(name)

    def _parse_op_result(self) -> tuple[Span, int, Attribute | None]:
        value_token = self._parse_token(
            Token.Kind.PERCENT_IDENT, "Expected result SSA value!"
        )
        if self._parse_optional_token(Token.Kind.COLON) is None:
            return (value_token.span, 1, None)

        size_token = self._parse_token(
            Token.Kind.INTEGER_LIT, "Expected SSA value tuple size"
        )
        size = size_token.get_int_value()
        return (value_token.span, size, None)

    def _parse_op_result_list(self) -> list[tuple[Span, int, Attribute | None]]:
        self._synchronize_lexer_and_tokenizer()
        res = self.parse_comma_separated_list(
            self.Delimiter.NONE, self._parse_op_result, " in operation result list"
        )
        self._synchronize_lexer_and_tokenizer()
        return res

    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        return self.parse_optional_dictionary_attr_dict()

    def parse_operation_details(
        self,
    ) -> tuple[
        list[SSAValue],
        list[Span],
        dict[str, Attribute],
        list[Region],
        FunctionType | None,
    ]:
        """
        Must return a tuple consisting of:
            - a list of arguments to the operation
            - a list of successor names
            - the attributes attached to the OP
            - the regions of the op
            - An optional function type. If not supplied, `parse_op_result_list`
              must return a second value containing the types of the returned SSAValues

        """
        args = self._parse_op_args_list()
        succ = self._parse_optional_successor_list()

        regions = []
        if self.tokenizer.starts_with("("):
            self.parse_characters("(", "Expected brackets enclosing regions!")
            regions = self.parse_region_list()
            self.parse_characters(")", "Expected brackets enclosing regions!")

        attrs = self.parse_optional_attr_dict()

        self.parse_characters(
            ":", "MLIR Operation definitions must end in a function type signature!"
        )
        func_type = self.parse_function_type()

        return args, succ, attrs, regions, func_type

    def _parse_optional_successor_list(self) -> list[Span]:
        self._synchronize_lexer_and_tokenizer()
        if self._current_token.kind != Token.Kind.L_SQUARE:
            return []
        successors = self.parse_comma_separated_list(
            self.Delimiter.SQUARE,
            lambda: self.expect(self.try_parse_block_id, "block-id expected"),
        )
        return successors

    def _parse_op_args_list(self) -> list[SSAValue]:
        return self.parse_comma_separated_list(
            self.Delimiter.PAREN, self.parse_operand, " in operation argument list"
        )

    def parse_region_list(self) -> list[Region]:
        """
        Parses a sequence of regions for as long as there is a `{` in the input.
        """
        regions: list[Region] = []
        while not self.tokenizer.is_eof() and self.tokenizer.starts_with("{"):
            regions.append(self.parse_region())
            if self.tokenizer.starts_with(","):
                self.parse_characters(
                    ",",
                    msg="This error should never be printed, please open "
                    "an issue at github.com/xdslproject/xdsl",
                )
                if not self.tokenizer.starts_with("{"):
                    self.raise_error(
                        "Expected next region (because of `,` after region end)!"
                    )
        return regions
