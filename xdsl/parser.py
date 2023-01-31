from __future__ import annotations

import ast
import contextlib
import functools
import itertools
import re
import sys
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import TypeVar, Iterable

from xdsl.utils.exceptions import ParseError, MultipleSpansParseError
from xdsl.dialects.memref import MemRefType, UnrankedMemrefType
from xdsl.dialects.builtin import (
    AnyTensorType, AnyVectorType, Float16Type, Float32Type, Float64Type,
    FloatAttr, FunctionType, IndexType, IntegerType, Signedness, StringAttr,
    IntegerAttr, ArrayAttr, TensorType, UnrankedTensorType, VectorType,
    FlatSymbolRefAttr, DenseIntOrFPElementsAttr, UnregisteredOp, OpaqueAttr,
    NoneAttr, ModuleOp, UnitAttr, i64)
from xdsl.ir import (SSAValue, Block, Callable, Attribute, Operation, Region,
                     BlockArgument, MLContext, ParametrizedAttribute, Data)


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
    pos: int

    def print_unroll(self, file=sys.stderr):
        if self.parent:
            if self.parent.get_farthest_point() > self.pos:
                self.parent.print_unroll(file)
                self.print(file)
            else:
                self.print(file)
                self.parent.print_unroll(file)

    def print(self, file=sys.stderr):
        print("Parsing of {} failed:".format(self.region_name or "<unknown>"),
              file=file)
        self.error.print_pretty(file=file)

    @functools.cache
    def get_farthest_point(self) -> int:
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


@dataclass(frozen=True)
class Span:
    """
    Parts of the input are always passed around as spans, so we know where "
    "they originated.
    """

    start: int
    """
    Start of tokens location in source file, global byte offset in file
    """
    end: int
    """
    End of tokens location in source file, global byte offset in file
    """
    input: Input
    """
    The input being operated on
    """

    def __len__(self):
        return self.len

    @property
    def len(self):
        return self.end - self.start

    @property
    def text(self):
        return self.input.content[self.start:self.end]

    def get_line_col(self) -> tuple[int, int]:
        info = self.input.get_lines_containing(self)
        if info is None:
            return -1, -1
        lines, offset_of_first_line, line_no = info
        return line_no, self.start - offset_of_first_line

    def print_with_context(self, msg: str | None = None) -> str:
        """
        returns a string containing lines relevant to the span. The Span's contents
        are highlighted by up-carets beneath them (`^`). The message msg is printed
        along these.
        """
        info = self.input.get_lines_containing(self)
        if info is None:
            return "Unknown location of span {}. Error: ".format(msg)
        lines, offset_of_first_line, line_no = info
        # Offset relative to the first line:
        offset = self.start - offset_of_first_line
        remaining_len = max(self.len, 1)
        capture = StringIO()
        print("{}:{}:{}".format(self.input.name, line_no, offset),
              file=capture)
        for line in lines:
            print(line, file=capture)
            if remaining_len < 0:
                continue
            len_on_this_line = min(remaining_len, len(line) - offset)
            remaining_len -= len_on_this_line
            print("{}{}".format(" " * offset, "^" * max(len_on_this_line, 1)),
                  file=capture)
            if msg is not None:
                print("{}{}".format(" " * offset, msg), file=capture)
                msg = None
            offset = 0
        if msg is not None:
            print(msg, file=capture)
        return capture.getvalue()

    def __repr__(self):
        return "{}[{}:{}](text='{}')".format(self.__class__.__name__,
                                             self.start, self.end, self.text)


@dataclass(frozen=True, repr=False)
class StringLiteral(Span):

    def __post_init__(self):
        if len(self) < 2 or self.text[0] != '"' or self.text[-1] != '"':
            raise ParseError(self, "Invalid string literal!")

    @classmethod
    def from_span(cls, span: Span | None) -> StringLiteral | None:
        """
        Convert a normal span into a StringLiteral, to facilitate parsing.

        If argument is None, returns None.
        """
        if span is None:
            return None
        return cls(span.start, span.end, span.input)

    @property
    def string_contents(self):
        # TODO: is this a hack-job?
        return ast.literal_eval(self.text)


@dataclass(frozen=True)
class Input:
    """
    This is a very simple class that is used to keep track of the input.
    """
    content: str = field(repr=False)
    name: str

    @property
    def len(self):
        return len(self.content)

    def __len__(self):
        return self.len

    def get_lines_containing(self,
                             span: Span) -> tuple[list[str], int, int] | None:
        # A pointer to the start of the first line
        start = 0
        line_no = 0
        source = self.content
        while True:
            next_start = source.find('\n', start)
            line_no += 1
            # Handle eof
            if next_start == -1:
                if span.start > len(source):
                    return None
                return [source[start:]], start, line_no
            # As long as the next newline comes before the spans start we can continue
            if next_start < span.start:
                start = next_start + 1
                continue
            # If the whole span is on one line, we are good as well
            if next_start >= span.end:
                return [source[start:next_start]], start, line_no
            while next_start < span.end:
                next_start = source.find('\n', next_start + 1)
            return source[start:next_start].split('\n'), start, line_no

    def at(self, i: int):
        if i >= self.len:
            raise EOFError()
        return self.content[i]


save_t = tuple[int, tuple[str, ...]]


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

    pos: int = field(init=False, default=0)
    """
    The position in the input. Points to the first unconsumed character.
    """

    break_on: tuple[str, ...] = ('.', '%', ' ', '(', ')', '[', ']', '{', '}',
                                 '<', '>', ':', '=', '@', '?', '|', '->', '-',
                                 '//', '\n', '\t', '#', '"', "'", ',', '!')
    """
    characters the tokenizer should break on
    """

    history: BacktrackingHistory | None = field(init=False,
                                                default=None,
                                                repr=False)

    last_token: Span | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.last_token = self.next_token(peek=True)

    def save(self) -> save_t:
        """
        Create a checkpoint in the parsing process, useful for backtracking
        """
        return self.pos, self.break_on

    def resume_from(self, save: save_t):
        """
        Resume from a previously saved position.

        Restores the state of the tokenizer to the exact previous position
        """
        self.pos, self.break_on = save

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
        save = self.save()
        starting_position = self.pos
        try:
            yield
            # Clear error history when something doesn't fail
            # This is because we are only interested in the last "cascade" of failures.
            # If a backtracking() completes without failure,
            # something has been parsed (we assume)
            if self.pos > starting_position and self.history is not None:
                self.history = None
        except Exception as ex:
            how_far_we_got = self.pos

            # If we have no error history, start recording!
            if not self.history:
                self.history = self._history_entry_from_exception(
                    ex, region_name, how_far_we_got)

            # If we got further than on previous attempts
            elif how_far_we_got > self.history.get_farthest_point():
                # Throw away history
                self.history = None
                # Generate new history entry,
                self.history = self._history_entry_from_exception(
                    ex, region_name, how_far_we_got)

            # Otherwise, add to exception, if we are in a named region
            elif region_name is not None and how_far_we_got - starting_position > 0:
                self.history = self._history_entry_from_exception(
                    ex, region_name, how_far_we_got)

            self.resume_from(save)

    def _history_entry_from_exception(self, ex: Exception, region: str | None,
                                      pos: int) -> BacktrackingHistory:
        """
        Given an exception generated inside a backtracking attempt,
        generate a BacktrackingHistory object with the relevant information in it.

        If an unexpected exception type is encountered, print a traceback to stderr
        """
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
            ParseError(self.last_token, "Unexpected exception: {}".format(ex),
                       self.history),
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

    def next_token_of_pattern(self,
                              pattern: re.Pattern | str,
                              peek: bool = False) -> Span | None:
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

    def _find_token_end(self, start: int | None = None) -> int:
        """
        Find the point (optionally starting from start) where the token ends
        """
        i = self.next_pos() if start is None else start
        # Search for literal breaks
        for part in self.break_on:
            if self.input.content.startswith(part, i):
                return i + len(part)
        # Otherwise return the start of the next break
        return min(
            filter(
                lambda x: x >= 0,
                (self.input.content.find(part, i) for part in self.break_on),
            ))

    def next_pos(self, i: int | None = None) -> int:
        """
        Find the next starting position (optionally starting from i)

        This will skip line comments!
        """
        i = self.pos if i is None else i
        # Skip whitespaces
        while self.input.at(i).isspace():
            i += 1

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

    @contextlib.contextmanager
    def configured(self, break_on: tuple[str, ...]):
        """
        This is a helper class to allow expressing a temporary change in config,
        allowing you to write:

        # Parsing double-quoted string now
        string_content = ""
        with tokenizer.configured(break_on=('"', '\\'),):
            # Use tokenizer

        # Now old config is restored automatically

        """
        save = self.save()

        if break_on is not None:
            self.break_on = break_on

        try:
            yield self
        finally:
            self.break_on = save[1]

    def starts_with(self, text: str | re.Pattern) -> bool:
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
    bare_id = re.compile(r"[A-Za-z_][\w$.]+")
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
        r"[su]?i\d+", r"f\d+", "tensor", "vector", "memref", "complex",
        "opaque", "tuple", "index", "dense"
        # TODO: add all the Float8E4M3FNType, Float8E5M2Type, and BFloat16Type
    )
    builtin_attr_names = ('dense', 'opaque', 'affine_map', 'array',
                          'dense_resource', 'sparse')
    builtin_type = re.compile("(({}))".format(")|(".join(_builtin_type_names)))
    builtin_type_xdsl = re.compile("!(({}))".format(
        ")|(".join(_builtin_type_names)))
    double_colon = re.compile("::")
    comma = re.compile(",")


class BaseParser(ABC):
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

    ssaValues: dict[str, SSAValue]
    blocks: dict[str, Block]
    forward_block_references: dict[str, list[Span]]
    """
    Blocks we encountered references to before the definition (must be empty after
    parsing of region completes)
    """

    T_ = TypeVar("T_")
    """
    Type var used for handling function that return single or multiple Spans.
    Basically the output type of all try_parse functions is `T_ | None`
    """

    allow_unregistered_ops: bool

    def __init__(self,
                 ctx: MLContext,
                 input: str,
                 name: str = '<unknown>',
                 allow_unregistered_ops=False):
        self.tokenizer = Tokenizer(Input(input, name))
        self.ctx = ctx
        self.ssaValues = dict()
        self.blocks = dict()
        self.forward_block_references = dict()
        self.allow_unregistered_ops = allow_unregistered_ops

    def parse_module(self) -> ModuleOp:
        op = self.try_parse_operation()

        if op is None:
            self.raise_error("Could not parse entire input!")

        if isinstance(op, ModuleOp):
            return op
        else:
            self.tokenizer.pos = 0
            self.raise_error("Expected ModuleOp at top level!",
                             self.tokenizer.next_token())

    def get_ssa_val(self, name: Span) -> SSAValue:
        if name.text not in self.ssaValues:
            self.raise_error('SSA Value used before assignment', name)
        return self.ssaValues[name.text]

    def _get_block_from_name(self, block_name: Span) -> Block:
        """
        This function takes a span containing a block id (like `^42`) and returns a block.

        If the block definition was not seen yet, we create a forward declaration.
        """
        name = block_name.text
        if name not in self.blocks:
            self.forward_block_references[name].append(block_name)
            self.blocks[name] = Block()
        return self.blocks[name]

    def parse_block(self) -> Block:
        block_id, args = self._parse_optional_block_label()

        if block_id is None:
            block = Block(self.tokenizer.last_token)
        elif self.forward_block_references.pop(block_id.text,
                                               None) is not None:
            block = self.blocks[block_id.text]
            block.declared_at = block_id
        else:
            if block_id.text in self.blocks:
                raise MultipleSpansParseError(
                    block_id,
                    "Re-declaration of block {}".format(block_id.text),
                    "Originally declared here:",
                    [(self.blocks[block_id.text].declared_at, None)],
                    self.tokenizer.history,
                )
            block = Block(block_id)
            self.blocks[block_id.text] = block

        for i, (name, type) in enumerate(args):
            arg = BlockArgument(type, block, i)
            self.ssaValues[name.text] = arg
            block.args.append(arg)

        while (next_op := self.try_parse_operation()) is not None:
            block.add_op(next_op)

        return block

    def _parse_optional_block_label(
            self) -> tuple[Span | None, list[tuple[Span, Attribute]]]:
        """
        A block label consists of block-id ( `(` block-arg `,` ... `)` )?
        """
        block_id = self.try_parse_block_id()
        arg_list = list()

        if block_id is not None:
            if self.tokenizer.starts_with('('):
                arg_list = self._parse_block_arg_list()

            self.parse_characters(':', 'Block label must end in a `:`!')

        return block_id, arg_list

    def _parse_block_arg_list(self) -> list[tuple[Span, Attribute]]:
        self.parse_characters('(', 'Block arguments must start with `(`')

        args = self.parse_list_of(self.try_parse_value_id_and_type,
                                  "Expected value-id and type here!")

        self.parse_characters(')', 'Expected closing of block arguments!')

        return args

    def try_parse_single_reference(self) -> Span | None:
        with self.tokenizer.backtracking('part of a reference'):
            self.parse_characters('@', "references must start with `@`")
            if (reference := self.try_parse_string_literal()) is not None:
                return reference
            if (reference := self.try_parse_suffix_id()) is not None:
                return reference
            self.raise_error(
                "References must conform to `@` (string-literal | suffix-id)")

    def parse_reference(self) -> list[Span]:
        return self.parse_list_of(
            self.try_parse_single_reference,
            'Expected reference here in the format of `@` (suffix-id | string-literal)',
            ParserCommons.double_colon,
            allow_empty=False)

    def parse_list_of(self,
                      try_parse: Callable[[], T_ | None],
                      error_msg: str,
                      separator_pattern: re.Pattern = ParserCommons.comma,
                      allow_empty: bool = True) -> list[T_]:
        """
        This is a greedy list-parser. It accepts input only in these cases:

         - If the separator isn't encountered, which signals the end of the list
         - If an empty list is allowed, it accepts when the first try_parse fails
         - If an empty separator is given, it instead sees a failed try_parse as the
           end of the list.

        This means, that the setup will not accept the input and instead raise an error:

            try_parse = parse_integer_literal
            separator = 'x'
            input = 3x4x4xi32

        as it will read [3,4,4], then see another separator, and expects the next
        `try_parse` call to succeed (which won't as i32 is not a valid integer literal)
        """
        items = list()
        first_item = try_parse()
        if first_item is None:
            if allow_empty:
                return items
            self.raise_error(error_msg)

        items.append(first_item)

        while (match := self.tokenizer.next_token_of_pattern(separator_pattern)
               ) is not None:
            next_item = try_parse()
            if next_item is None:
                # If the separator is emtpy, we are good here
                if separator_pattern.pattern == '':
                    return items
                self.raise_error(error_msg +
                                 ' because was able to match next separator {}'
                                 .format(match.text))
            items.append(next_item)

        return items

    def try_parse_integer_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(
            ParserCommons.integer_literal)

    def try_parse_decimal_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(
            ParserCommons.decimal_literal)

    def try_parse_string_literal(self) -> StringLiteral | None:
        return StringLiteral.from_span(
            self.tokenizer.next_token_of_pattern(ParserCommons.string_literal))

    def try_parse_float_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(
            ParserCommons.float_literal)

    def try_parse_bare_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.bare_id)

    def try_parse_value_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.value_id)

    def try_parse_suffix_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.suffix_id)

    def try_parse_block_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.block_id)

    def try_parse_boolean_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(
            ParserCommons.boolean_literal)

    def try_parse_value_id_and_type(self) -> tuple[Span, Attribute] | None:
        with self.tokenizer.backtracking("value id and type"):
            value_id = self.try_parse_value_id()

            if value_id is None:
                self.raise_error("Invalid value-id format!")

            self.parse_characters(':',
                                  'Expected expression (value-id `:` type)')

            type = self.try_parse_type()

            if type is None:
                self.raise_error("Expected type of value-id here!")
            return value_id, type

    def try_parse_type(self) -> Attribute | None:
        if (builtin_type := self.try_parse_builtin_type()) is not None:
            return builtin_type
        if (dialect_type := self.try_parse_dialect_type()) is not None:
            return dialect_type
        return None

    def try_parse_dialect_type_or_attribute(self) -> Attribute | None:
        """
        Parse a type or an attribute.
        """
        kind = self.tokenizer.next_token_of_pattern(re.compile('[!#]'),
                                                    peek=True)

        if kind is None:
            return None

        with self.tokenizer.backtracking("dialect attribute or type"):
            self.tokenizer.consume_peeked(kind)
            if kind.text == '!':
                return self._parse_dialect_type_or_attribute_inner('type')
            else:
                return self._parse_dialect_type_or_attribute_inner('attribute')

    def try_parse_dialect_type(self):
        """
        Parse a dialect type (something prefixed by `!`, defined by a dialect)
        """
        if not self.tokenizer.starts_with('!'):
            return None
        with self.tokenizer.backtracking("dialect type"):
            self.parse_characters('!', "Dialect type must start with a `!`")
            return self._parse_dialect_type_or_attribute_inner('type')

    def try_parse_dialect_attr(self):
        """
        Parse a dialect attribute (something prefixed by `#`, defined by a dialect)
        """
        if not self.tokenizer.starts_with('#'):
            return None
        with self.tokenizer.backtracking("dialect attribute"):
            self.parse_characters('#',
                                  "Dialect attribute must start with a `#`")
            return self._parse_dialect_type_or_attribute_inner('attribute')

    def _parse_dialect_type_or_attribute_inner(self, kind: str):
        type_name = self.tokenizer.next_token_of_pattern(ParserCommons.bare_id)

        if type_name is None:
            self.raise_error("Expected dialect {} name here!".format(kind))

        type_def = self.ctx.get_optional_attr(type_name.text)
        if type_def is None:
            self.raise_error(
                "'{}' is not a know attribute!".format(type_name.text),
                type_name)

        # Pass the task of parsing parameters on to the attribute/type definition
        if issubclass(type_def, ParametrizedAttribute):
            param_list = type_def.parse_parameters(self)
        elif issubclass(type_def, Data):
            self.parse_characters("<", "This attribute must be parametrized!")
            param_list = type_def.parse_parameter(self)
            self.parse_characters(
                ">", "Invalid attribute parametrization, expected `>`!")
        else:
            assert False, "Mathieu said this cannot be."
        return type_def(param_list)

    @abstractmethod
    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        raise NotImplementedError("Subclasses must implement this method!")

    def _parse_builtin_parametrized_type(self,
                                         name: Span) -> ParametrizedAttribute:
        """
        This function is called after we parse the name of a parameterized type
        such as vector.
        """

        def unimplemented() -> ParametrizedAttribute:
            raise ParseError(name,
                             "Builtin {} not supported yet!".format(name.text))

        builtin_parsers: dict[str, Callable[[], ParametrizedAttribute]] = {
            "vector": self.parse_vector_attrs,
            "memref": self.parse_memref_attrs,
            "tensor": self.parse_tensor_attrs,
            "complex": self.parse_complex_attrs,
            "tuple": unimplemented,
        }

        self.parse_characters("<", "Expected parameter list here!")
        # Get the parser for the type, falling back to the unimplemented warning
        res = builtin_parsers.get(name.text, unimplemented)()
        self.parse_characters(">", "Expected end of parameter list here!")

        return res

    def parse_complex_attrs(self):
        self.raise_error("ComplexType is unimplemented!")

    def parse_memref_attrs(self) -> MemRefType | UnrankedMemrefType:
        dims = self._parse_tensor_or_memref_dims()
        type = self.try_parse_type()
        if dims is None:
            return UnrankedMemrefType.from_type(type)
        return MemRefType.from_element_type_and_shape(type, dims)

    def try_parse_numerical_dims(self,
                                 accept_closing_bracket: bool = False,
                                 lower_bound: int = 1) -> Iterable[int]:
        while (shape_arg :=
               self._try_parse_shape_element(lower_bound)) is not None:
            yield shape_arg
            # Look out for the closing bracket for scalable vector dims
            if accept_closing_bracket and self.tokenizer.starts_with("]"):
                break
            self.parse_characters("x",
                                  "Unexpected end of dimension parameters!")

    def parse_vector_attrs(self) -> AnyVectorType:
        # Also break on 'x' characters as they are separators in dimension parameters
        with self.tokenizer.configured(break_on=self.tokenizer.break_on +
                                       ("x", )):
            shape = list[int](self.try_parse_numerical_dims())
            scaling_shape: list[int] | None = None

            if self.tokenizer.next_token_of_pattern("[") is not None:
                # We now need to parse the scalable dimensions
                scaling_shape = list(self.try_parse_numerical_dims())
                self.parse_characters(
                    "]", "Expected end of scalable vector dimensions here!")
                self.parse_characters(
                    "x", "Expected end of scalable vector dimensions here!")

            if scaling_shape is not None:
                # TODO: handle scaling vectors!
                self.raise_error("Warning: scaling vectors not supported!")
                pass

            type = self.try_parse_type()
            if type is None:
                self.raise_error(
                    "Expected a type at the end of the vector parameters!")

            return VectorType.from_element_type_and_shape(type, shape)

    def _parse_tensor_or_memref_dims(self) -> list[int] | None:
        with self.tokenizer.configured(break_on=self.tokenizer.break_on +
                                       ('x', )):
            # Check for unranked-ness
            if self.tokenizer.next_token_of_pattern('*') is not None:
                # Consume `x`
                self.parse_characters(
                    'x',
                    'Unranked tensors must follow format (`<*x` type `>`)')
            else:
                # Parse rank:
                return list(self.try_parse_numerical_dims(lower_bound=0))

    def parse_tensor_attrs(self) -> AnyTensorType:
        shape = self._parse_tensor_or_memref_dims()
        type = self.try_parse_type()

        if type is None:
            self.raise_error("Expected tensor type here!")

        if self.tokenizer.starts_with(','):
            # TODO: add tensor encoding!
            raise self.raise_error("Parsing tensor encoding is not supported!")

        if shape is None and self.tokenizer.starts_with(','):
            raise self.raise_error("Unranked tensors don't have an encoding!")

        if shape is not None:
            return TensorType.from_type_and_list(type, shape)

        return UnrankedTensorType.from_type(type)

    def _try_parse_shape_element(self, lower_bound: int = 1) -> int | None:
        """
        Parse a shape element, either a decimal integer immediate or a `?`,
        which evaluates to -1 immediate cannot be smaller than lower_bound
        (defaults to 1, is 0 for tensors and memrefs)
        """
        int_lit = self.try_parse_decimal_literal()

        if int_lit is not None:
            value = int(int_lit.text)
            if value < lower_bound:
                # TODO: this is ugly, it's a raise inside a try_ type function, which
                # should instead just give up
                raise ParseError(
                    int_lit,
                    "Shape element literal cannot be negative or zero!")
            return value

        if self.tokenizer.next_token_of_pattern('?') is not None:
            return -1
        return None

    def _parse_type_params(self) -> list[Attribute]:
        # Consume opening bracket
        self.parse_characters('<', 'Type must be parameterized!')

        params = self.parse_list_of(self.try_parse_type,
                                    'Expected a type here!')

        self.parse_characters('>',
                              'Expected end of type parameterization here!')

        return params

    def expect(self, try_parse: Callable[[], T_ | None],
               error_message: str) -> T_:
        """
        Used to force completion of a try_parse function.
        Will throw a parse error if it can't.
        """
        res = try_parse()
        if res is None:
            self.raise_error(error_message)
        return res

    def raise_error(self, msg: str, at_position: Span | None = None):
        """
        Helper for raising exceptions, provides as much context as possible to them.

        This will, for example, include backtracking errors, if any occurred previously.
        """
        if at_position is None:
            at_position = self.tokenizer.next_token(peek=True)

        raise ParseError(at_position, msg, self.tokenizer.history)

    def parse_characters(self, text: str, msg: str) -> Span:
        if (match := self.tokenizer.next_token_of_pattern(text)) is None:
            self.raise_error(msg)
        return match

    @abstractmethod
    def _parse_op_result_list(
            self) -> tuple[list[Span], list[Attribute] | None]:
        raise NotImplementedError()

    def try_parse_operation(self) -> Operation | None:
        with self.tokenizer.backtracking("operation"):
            return self.parse_operation()

    def parse_operation(self) -> Operation:
        result_list, ret_types = self._parse_op_result_list()
        if len(result_list) > 0:
            self.parse_characters(
                '=',
                'Operation definitions expect an `=` after op-result-list!')

        # Check for custom op format
        op_name = self.try_parse_bare_id()
        if op_name is not None:
            op_type = self._get_op_by_name(op_name)
            op = op_type.parse(ret_types, self)
        else:
            # Check for basic op format
            op_name = self.try_parse_string_literal()
            if op_name is None:
                self.raise_error(
                    "Expected an operation name here, either a bare-id, or a string "
                    "literal!")

            args, successors, attrs, regions, func_type = self._parse_operation_details(
            )

            if ret_types is None:
                assert func_type is not None
                ret_types = func_type.outputs.data

            op_type = self._get_op_by_name(op_name)

            op = op_type.create(
                operands=[self.ssaValues[span.text] for span in args],
                result_types=ret_types,
                attributes=attrs,
                successors=[
                    self.blocks[block_name.text] for block_name in successors
                ],
                regions=regions)

        # Register the result SSA value names in the parser
        for idx, res in enumerate(result_list):
            ssa_val_name = res.text
            if ssa_val_name in self.ssaValues:
                self.raise_error(
                    f"SSA value {ssa_val_name} is already defined", res)
            self.ssaValues[ssa_val_name] = op.results[idx]
            self.ssaValues[ssa_val_name].name = ssa_val_name.lstrip('%')

        return op

    def _get_op_by_name(self, span: Span) -> type[Operation]:
        if isinstance(span, StringLiteral):
            op_name = span.string_contents
        else:
            op_name = span.text

        op_type = self.ctx.get_optional_op(op_name)

        if op_type is not None:
            return op_type

        if self.allow_unregistered_ops:
            return UnregisteredOp.with_name(op_name, self.ctx)

        self.raise_error(f'Unknown operation {op_name}!', span)

    def parse_region(self) -> Region:
        oldSSAVals = self.ssaValues.copy()
        oldBBNames = self.blocks
        oldForwardRefs = self.forward_block_references
        self.blocks = dict()
        self.forward_block_references = defaultdict(list)

        region = Region()

        try:
            self.parse_characters("{", "Regions begin with `{`")
            if self.tokenizer.starts_with("}"):
                region.add_block(Block())
            else:
                # Parse first block
                block = self.parse_block()
                region.add_block(block)

                while self.tokenizer.starts_with("^"):
                    region.add_block(self.parse_block())

            end = self.parse_characters(
                "}", "Reached end of region, expected `}`!")

            if len(self.forward_block_references) > 0:
                raise MultipleSpansParseError(
                    end,
                    "Region ends with missing block declarations for block(s) {}!"
                    .format(', '.join(self.forward_block_references.keys())),
                    'The following block references are dangling:',
                    [(span, "Reference to block \"{}\" without implementation!"
                      .format(span.text)) for span in itertools.chain(
                          *self.forward_block_references.values())],
                    self.tokenizer.history)

            return region
        finally:
            self.ssaValues = oldSSAVals
            self.blocks = oldBBNames
            self.forward_block_references = oldForwardRefs

    def _try_parse_op_name(self) -> Span | None:
        if (str_lit := self.try_parse_string_literal()) is not None:
            return str_lit
        return self.try_parse_bare_id()

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

        if not self.tokenizer.starts_with('='):
            return name, UnitAttr()

        self.parse_characters(
            "=", "Attribute entries must be of format name `=` attribute!")

        return name, self.parse_attribute()

    @abstractmethod
    def parse_attribute(self) -> Attribute:
        """
        Parse attribute (either builtin or dialect)

        This is different in xDSL and MLIR, so the actuall implementation is provided by the subclass
        """
        raise NotImplementedError()

    def try_parse_attribute(self) -> Attribute | None:
        with self.tokenizer.backtracking("attribute"):
            return self.parse_attribute()

    def _parse_attribute_type(self) -> Attribute:
        """
        Parses `:` type and returns the type
        """
        self.parse_characters(
            ":", "Expected attribute type definition here ( `:` type )")
        return self.expect(
            self.try_parse_type,
            "Expected attribute type definition here ( `:` type )")

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
            return self.try_parse_ref_attr()
        elif next_token.text == '{':
            return self.try_parse_builtin_dict_attr()
        elif next_token.text == '(':
            return self.try_parse_function_type()
        elif next_token.text in ParserCommons.builtin_attr_names:
            return self.try_parse_builtin_named_attr()
        # Order here is important!
        attrs = (self.try_parse_builtin_float_attr,
                 self.try_parse_builtin_int_attr, self.try_parse_builtin_type)

        for attr_parser in attrs:
            if (val := attr_parser()) is not None:
                return val

    def try_parse_builtin_named_attr(self) -> Attribute | None:
        name = self.tokenizer.next_token(peek=True)
        with self.tokenizer.backtracking("Builtin attribute {}".format(
                name.text)):
            self.tokenizer.consume_peeked(name)
            parsers = {
                'dense': self._parse_builtin_dense_attr,
                'opaque': self._parse_builtin_opaque_attr,
            }

            def not_implemented():
                raise NotImplementedError()

            return parsers.get(name.text, not_implemented)()

    def _parse_builtin_dense_attr(self) -> Attribute | None:
        err_msg = "Malformed dense attribute, format must be (`dense<` array-attr `>:` type)"  # noqa
        self.parse_characters("<", err_msg)
        info = list(self._parse_builtin_dense_attr_args())
        self.parse_characters(">", err_msg)
        self.parse_characters(":", err_msg)
        type = self.expect(self.try_parse_type,
                           "Dense attribute must be typed!")
        return DenseIntOrFPElementsAttr.from_list(type, info)

    def _parse_builtin_opaque_attr(self):
        self.parse_characters("<", "Opaque attribute must be parametrized")
        str_lit_list = self.parse_list_of(self.try_parse_string_literal,
                                          'Expected opaque attr here!')

        if len(str_lit_list) != 2:
            self.raise_error('Opaque expects 2 string literal parameters!')

        self.parse_characters(
            ">", "Unexpected parameters for opaque attr, expected `>`!")

        type = NoneAttr()
        if self.tokenizer.starts_with(':'):
            self.parse_characters(":", "opaque attribute must be typed!")
            type = self.expect(self.try_parse_type,
                               "opaque attribute must be typed!")

        return OpaqueAttr.from_strings(*(span.string_contents
                                         for span in str_lit_list),
                                       type=type)

    def _parse_builtin_dense_attr_args(self) -> Iterable[int | float]:
        """
        Dense attribute params must be:

        dense-attr-params           := float-literal | int-literal | list-of-dense-attrs-params
        list-of-dense-attrs-params  := `[` dense-attr-params (`,` dense-attr-params)* `]`
        """

        def try_parse_int_or_float():
            if (literal := self.try_parse_float_literal()) is not None:
                return float(literal.text)
            if (literal := self.try_parse_integer_literal()) is not None:
                return int(literal.text)
            self.raise_error('Expected int or float literal here!')

        if not self.tokenizer.starts_with('['):
            yield try_parse_int_or_float()
            return

        self.parse_characters('[', '')
        while not self.tokenizer.starts_with(']'):
            yield from self._parse_builtin_dense_attr_args()
            if self.tokenizer.next_token_of_pattern(',') is None:
                break
        self.parse_characters(']', '')

    def try_parse_ref_attr(self) -> FlatSymbolRefAttr | None:
        if not self.tokenizer.starts_with("@"):
            return None

        ref = self.parse_reference()

        if len(ref) > 1:
            self.raise_error("Nested refs are not supported yet!", ref[1])

        return FlatSymbolRefAttr.from_str(ref[0].text)

    def try_parse_builtin_int_attr(self) -> IntegerAttr | None:
        bool = self.try_parse_builtin_boolean_attr()
        if bool is not None:
            return bool

        with self.tokenizer.backtracking("built in int attribute"):
            value = self.expect(
                self.try_parse_integer_literal,
                'Integer attribute must start with an integer literal!')
            if self.tokenizer.next_token(peek=True).text != ':':
                return IntegerAttr.from_params(int(value.text), i64)
            type = self._parse_attribute_type()
            return IntegerAttr.from_params(int(value.text), type)

    def try_parse_builtin_float_attr(self) -> FloatAttr | None:
        with self.tokenizer.backtracking("float literal"):
            value = self.expect(
                self.try_parse_float_literal,
                "Float attribute must start with a float literal!",
            )
            # If we don't see a ':' indicating a type signature
            if not self.tokenizer.starts_with(":"):
                return FloatAttr.from_value(float(value.text))

            type = self._parse_attribute_type()
            return FloatAttr.from_value(float(value.text), type)

    def try_parse_builtin_boolean_attr(self) -> IntegerAttr | None:
        span = self.try_parse_boolean_literal()

        if span is None:
            return None

        int_val = ["false", "true"].index(span.text)
        return IntegerAttr.from_params(int_val, IntegerType.from_width(1))

    def try_parse_builtin_str_attr(self):
        if not self.tokenizer.starts_with('"'):
            return None

        with self.tokenizer.backtracking("string literal"):
            literal = self.try_parse_string_literal()
            if literal is None:
                self.raise_error("Invalid string literal")
            return StringAttr.from_str(literal.string_contents)

    def try_parse_builtin_arr_attr(self) -> ArrayAttr | None:
        if not self.tokenizer.starts_with("["):
            return None
        with self.tokenizer.backtracking("array literal"):
            self.parse_characters("[", "Array literals must start with `[`")
            attrs = self.parse_list_of(self.try_parse_attribute,
                                       "Expected array entry!")
            self.parse_characters(
                "]", "Malformed array contents (expected end of array here!")
            return ArrayAttr.from_list(attrs)

    @abstractmethod
    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        raise NotImplementedError()

    def _attr_dict_from_tuple_list(
            self, tuple_list: list[tuple[Span,
                                         Attribute]]) -> dict[str, Attribute]:
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
        self.parse_characters(
            "(", "First group of function args must start with a `(`")

        args: list[Attribute] = self.parse_list_of(self.try_parse_type,
                                                   "Expected type here!")

        self.parse_characters(
            ")",
            "Malformed function type, expected closing brackets of argument types!"
        )

        self.parse_characters("->", "Malformed function type, expected `->`!")

        return FunctionType.from_lists(args,
                                       self._parse_type_or_type_list_parens())

    def _parse_type_or_type_list_parens(self) -> list[Attribute | None]:
        """
        Parses type-or-type-list-parens, which is used in function-type.

        type-or-type-list-parens ::= type | type-list-parens
        type-list-parens         ::= `(` `)` | `(` type-list-no-parens `)`
        type-list-no-parens      ::=  type (`,` type)*
        """
        if self.tokenizer.next_token_of_pattern("(") is not None:
            args: list[Attribute] = self.parse_list_of(self.try_parse_type,
                                                       "Expected type here!")
            self.parse_characters(")", "Unclosed function type argument list!")
        else:
            args = [self.try_parse_type()]
            if args[0] is None:
                self.raise_error(
                    "Function type must either be single type or list of types in"
                    " parenthesis!")
        return args

    def try_parse_function_type(self) -> FunctionType | None:
        if not self.tokenizer.starts_with("("):
            return None
        with self.tokenizer.backtracking("function type"):
            return self.parse_function_type()

    def parse_region_list(self) -> list[Region]:
        """
        Parses a sequence of regions for as long as there is a `{` in the input.
        """
        regions = []
        while not self.tokenizer.is_eof() and self.tokenizer.starts_with("{"):
            regions.append(self.parse_region())
        return regions

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
            return IntegerType.from_width(int(re_match.group(1)),
                                          signedness[name.text[0]])

        if (re_match := re.match(r"^f(\d+)$", name.text)) is not None:
            width = int(re_match.group(1))
            type = {
                16: Float16Type,
                32: Float32Type,
                64: Float64Type
            }.get(width, None)
            if type is None:
                self.raise_error(
                    "Unsupported floating point width: {}".format(width))
            return type()

        return self._parse_builtin_parametrized_type(name)

    @abstractmethod
    def _parse_operation_details(
        self,
    ) -> tuple[list[Span], list[Span], dict[str, Attribute], list[Region],
               FunctionType | None]:
        """
        Must return a tuple consisting of:
            - a list of arguments to the operation
            - a list of successor names
            - the attributes attached to the OP
            - the regions of the op
            - An optional function type. If not supplied, parse_op_result_list must return a second value
              containing the types of the returned SSAValues

        """
        raise NotImplementedError()

    @abstractmethod
    def _parse_op_args_list(self) -> list[Span]:
        raise NotImplementedError()

    # HERE STARTS A SOMEWHAT CURSED COMPATIBILITY LAYER:
    # Since we don't want to rewrite all dialects currently, the new parser needs to expose the same
    # Interface to the dialect definitions (to some extent). Here we implement that interface.

    _OperationType = TypeVar("_OperationType", bound=Operation)

    def parse_op_with_default_format(
        self,
        op_type: type[_OperationType],
        result_types: list[Attribute],
    ) -> _OperationType:
        """
        Compatibility wrapper so the new parser can be passed instead of the old one. Parses everything after the
        operation name.

        This implicitly assumes XDSL format, and will fail on MLIR style operations
        """
        # TODO: remove this function and restructure custom op / irdl parsing
        assert isinstance(self, XDSLParser)
        args, successors, attributes, regions, _ = self._parse_operation_details(
        )

        for x in args:
            if x.text not in self.ssaValues:
                self.raise_error(
                    "Unknown SSAValue name, known SSA Values are: {}".format(
                        ", ".join(self.ssaValues.keys())), x)

        return op_type.create(
            operands=[self.ssaValues[span.text] for span in args],
            result_types=result_types,
            attributes=attributes,
            successors=[
                self._get_block_from_name(span) for span in successors
            ],
            regions=regions)

    def parse_paramattr_parameters(
            self,
            expect_brackets: bool = False,
            skip_white_space: bool = True) -> list[Attribute]:
        opening_brackets = self.tokenizer.next_token_of_pattern('<')
        if expect_brackets and opening_brackets is None:
            self.raise_error("Expected start attribute parameters here (`<`)!")

        res = self.parse_list_of(self.try_parse_attribute,
                                 'Expected another attribute here!')

        if opening_brackets is not None and self.tokenizer.next_token_of_pattern(
                '>') is None:
            self.raise_error(
                "Malformed parameter list, expected either another parameter or `>`!"
            )

        return res

    def parse_char(self, text: str):
        self.parse_characters(text, "Expected '{}' here!".format(text))

    def parse_str_literal(self) -> str:
        return self.expect(self.try_parse_string_literal,
                           'Malformed string literal!').string_contents

    def parse_op(self) -> Operation:
        return self.parse_operation()

    def parse_int_literal(self) -> int:
        return int(
            self.expect(self.try_parse_integer_literal,
                        'Expected integer literal here').text)

    def try_parse_builtin_dict_attr(self):
        attr_def = self.ctx.get_optional_attr('dictionary')
        if attr_def is None:
            self.raise_error(
                "An attribute named `dictionary` must be available in the "
                "context in order to parse dictionary attributes! Please make "
                "sure the builtin dialect is available, or provide your own "
                "replacement!")
        param = attr_def.parse_parameter(self)
        return attr_def(param)


class MLIRParser(BaseParser):

    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        with self.tokenizer.backtracking("builtin type"):
            name = self.tokenizer.next_token_of_pattern(
                ParserCommons.builtin_type)
            if name is None:
                raise self.raise_error("Expected builtin name!")

            return self._parse_builtin_type_with_name(name)

    def parse_attribute(self) -> Attribute:
        """
        Parse attribute (either builtin or dialect)
        """
        # All dialect attrs must start with '#', so we check for that first
        # (as it's easier)
        if self.tokenizer.starts_with("#"):
            value = self.try_parse_dialect_attr()

            # No value => error
            if value is None:
                self.raise_error(
                    "`#` must be followed by a valid dialect attribute or type!"
                )

            return value

        # If it isn't a dialect attr, parse builtin
        builtin_val = self.try_parse_builtin_attr()

        if builtin_val is None:
            self.raise_error(
                "Unknown attribute (neither builtin nor dialect could be parsed)!"
            )

        return builtin_val

    def _parse_op_result_list(
            self) -> tuple[list[Span], list[Attribute] | None]:
        return (
            self.parse_list_of(self.try_parse_value_id,
                               "Expected op-result here!",
                               allow_empty=True),
            None,
        )

    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        if not self.tokenizer.starts_with("{"):
            return dict()

        self.parse_characters(
            "{",
            "MLIR Attribute dictionary must be enclosed in curly brackets")

        attrs = []
        if not self.tokenizer.starts_with('}'):
            attrs = self.parse_list_of(self._parse_attribute_entry,
                                       "Expected attribute entry")

        self.parse_characters(
            "}",
            "MLIR Attribute dictionary must be enclosed in curly brackets")

        return self._attr_dict_from_tuple_list(attrs)

    def _parse_operation_details(
        self,
    ) -> tuple[list[Span], list[Span], dict[str, Attribute], list[Region],
               FunctionType | None]:
        args = self._parse_op_args_list()
        succ = self._parse_optional_successor_list()

        regions = []
        if self.tokenizer.starts_with("("):
            self.parse_characters("(", "Expected brackets enclosing regions!")
            regions = self.parse_region_list()
            self.parse_characters(")", "Expected brackets enclosing regions!")

        attrs = self.parse_optional_attr_dict()

        self.parse_characters(
            ":",
            "MLIR Operation definitions must end in a function type signature!"
        )
        func_type = self.parse_function_type()

        return args, succ, attrs, regions, func_type

    def _parse_optional_successor_list(self) -> list[Span]:
        if not self.tokenizer.starts_with("["):
            return []
        self.parse_characters("[",
                              "Successor list is enclosed in square brackets")
        successors = self.parse_list_of(self.try_parse_block_id,
                                        "Expected a block-id",
                                        allow_empty=False)
        self.parse_characters("]",
                              "Successor list is enclosed in square brackets")
        return successors

    def _parse_op_args_list(self) -> list[Span]:
        self.parse_characters(
            "(", "Operation args list must be enclosed by brackets!")
        args = self.parse_list_of(self.try_parse_value_id,
                                  "Expected another bare-id here")
        self.parse_characters(
            ")", "Operation args list must be closed by a closing bracket")
        # TODO: check if type is correct here!
        return args


class XDSLParser(BaseParser):

    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        with self.tokenizer.backtracking("builtin type"):
            name = self.tokenizer.next_token_of_pattern(
                ParserCommons.builtin_type_xdsl)
            if name is None:
                self.raise_error("Expected builtin name!")
            # xDSL builtin types have a '!' prefix, we strip that out here
            name = Span(start=name.start + 1, end=name.end, input=name.input)

            return self._parse_builtin_type_with_name(name)

    def parse_attribute(self) -> Attribute:
        """
        Parse attribute (either builtin or dialect)

        xDSL allows types in places of attributes! That's why we parse types here as well
        """
        value = self.try_parse_builtin_attr()

        # xDSL: Allow both # and ! prefixes, as we allow both types and attrs
        # TODO: phase out use of next_token(peek=True) in favour of starts_with
        if value is None and self.tokenizer.next_token(peek=True).text in "#!":
            # In MLIR, `#` and `!` are prefixes for dialect attrs/types, but in xDSL,
            # `!` is also used for builtin types
            value = self.try_parse_dialect_type_or_attribute()

        if value is None:
            self.raise_error(
                "Unknown attribute (neither builtin nor dialect could be parsed)!"
            )

        return value

    def _parse_op_result_list(
            self) -> tuple[list[Span], list[Attribute] | None]:
        if not self.tokenizer.starts_with("%"):
            return list(), list()
        results = self.parse_list_of(
            self.try_parse_value_id_and_type,
            "Expected (value-id `:` type) here!",
            allow_empty=False,
        )
        # TODO: this is hideous, make it cleaner
        # zip(*results) works, but is barely readable :/
        return [name for name, _ in results], [type for _, type in results]

    def try_parse_builtin_attr(self) -> Attribute | None:
        """
        Tries to parse a builtin attribute, e.g. a string literal, int, array, etc..

        If the mode is xDSL, it also allows parsing of builtin types
        """
        # In xdsl, two things are different here:
        #  1. types are considered valid attributes
        #  2. all types, builtins included, are prefixed with !
        if self.tokenizer.starts_with("!"):
            return self.try_parse_builtin_type()

        return super().try_parse_builtin_attr()

    def parse_optional_attr_dict(self) -> dict[str, Attribute]:
        if not self.tokenizer.starts_with("["):
            return dict()

        self.parse_characters(
            "[",
            "xDSL Attribute dictionary must be enclosed in square brackets")

        attrs = self.parse_list_of(self._parse_attribute_entry,
                                   "Expected attribute entry")

        self.parse_characters(
            "]",
            "xDSL Attribute dictionary must be enclosed in square brackets")

        return self._attr_dict_from_tuple_list(attrs)

    def _parse_operation_details(
        self,
    ) -> tuple[list[Span], list[Span], dict[str, Attribute], list[Region],
               FunctionType | None]:
        """
        Must return a tuple consisting of:
            - a list of arguments to the operation
            - a list of successor names
            - the attributes attached to the OP
            - the regions of the op
            - An optional function type. If not supplied, parse_op_result_list must
              return a second value containing the types of the returned SSAValues

        """
        args = self._parse_op_args_list()
        succ = self._parse_optional_successor_list()
        attrs = self.parse_optional_attr_dict()
        regions = self.parse_region_list()

        return args, succ, attrs, regions, None

    def _parse_optional_successor_list(self) -> list[Span]:
        if not self.tokenizer.starts_with("("):
            return []
        self.parse_characters("(",
                              "Successor list is enclosed in round brackets")
        successors = self.parse_list_of(self.try_parse_block_id,
                                        "Expected a block-id",
                                        allow_empty=False)
        self.parse_characters(")",
                              "Successor list is enclosed in round brackets")
        return successors

    def _parse_dialect_type_or_attribute_inner(self, kind: str):
        if self.tokenizer.starts_with('"'):
            name = self.try_parse_string_literal()
            if name is None:
                self.raise_error(
                    "Expected string literal for an attribute in generic format here!"
                )
            return self._parse_generic_attribute_args(name)
        return super()._parse_dialect_type_or_attribute_inner(kind)

    def _parse_generic_attribute_args(self, name: StringLiteral):
        attr = self.ctx.get_optional_attr(name.string_contents)
        if attr is None:
            self.raise_error("Unknown attribute name!", name)
        if not issubclass(attr, ParametrizedAttribute):
            self.raise_error("Expected ParametrizedAttribute name here!", name)
        self.parse_characters('<',
                              'Expected generic attribute arguments here!')
        args = self.parse_list_of(self.try_parse_attribute,
                                  'Unexpected end of attribute list!')
        self.parse_characters(
            '>', 'Malformed attribute arguments, reached end of args list!')
        return attr(args)

    def _parse_op_args_list(self) -> list[Span]:
        self.parse_characters(
            "(", "Operation args list must be enclosed by brackets!")
        args = self.parse_list_of(self.try_parse_value_id_and_type,
                                  "Expected another bare-id here")
        self.parse_characters(
            ")", "Operation args list must be closed by a closing bracket")
        # TODO: check if type is correct here!
        return [name for name, _ in args]

    def try_parse_type(self) -> Attribute | None:
        return self.try_parse_attribute()


# COMPAT layer so parser_ng is a drop-in replacement for parser:


class Source(Enum):
    XDSL = 1
    MLIR = 2


def Parser(ctx: MLContext,
           prog: str,
           source: Source = Source.XDSL,
           filename: str = '<unknown>',
           allow_unregistered_ops=False) -> BaseParser:
    selected_parser = {
        Source.XDSL: XDSLParser,
        Source.MLIR: MLIRParser
    }[source]
    return selected_parser(ctx, prog, filename)


setattr(Parser, 'Source', Source)
