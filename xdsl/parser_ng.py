from __future__ import annotations

import contextlib
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
import ast
from io import StringIO
from typing import Any, TypeVar, Iterable, Literal, Optional
from enum import Enum

from .printer import Printer
from xdsl.ir import (SSAValue, Block, Callable, Attribute, Operation, Region,
                     BlockArgument, MLContext, ParametrizedAttribute)

import xdsl.utils.bnf as BNF

from xdsl.dialects.builtin import (
    AnyFloat, AnyTensorType, AnyUnrankedTensorType, AnyVectorType,
    DenseIntOrFPElementsAttr, Float16Type, Float32Type, Float64Type, FloatAttr,
    FunctionType, IndexType, IntegerType, OpaqueAttr, Signedness, StringAttr,
    FlatSymbolRefAttr, IntegerAttr, ArrayAttr, TensorType, UnitAttr,
    UnrankedTensorType, UnregisteredOp, VectorType, DefaultIntegerAttrType)

from xdsl.irdl import Data


class ParseError(Exception):
    span: Span
    msg: str
    history: BacktrackingHistory | None

    def __init__(self,
                 span: Span,
                 msg: str,
                 history: BacktrackingHistory | None = None):
        super().__init__(span.print_with_context(msg))
        self.span = span
        self.msg = msg
        self.history = history

    def print_pretty(self, file=sys.stderr):
        print(self.span.print_with_context(self.msg), file=file)

    def print_with_history(self):
        if self.history is not None:
            self.history.print_unroll()


@dataclass
class BacktrackingHistory:
    error: ParseError
    parent: BacktrackingHistory | None
    region_name: str | None
    pos: int

    def print_unroll(self, file=sys.stderr):
        if self.parent:
            self.parent.print_unroll(file)

        print("Parsing of {} failed:".format(self.region_name or '<unknown>'),
              file=file)
        self.error.print_pretty(file=file)

    def get_farthest_point(self) -> int:
        """
        Find the farthest this history managed to parse
        """
        if self.parent:
            return max(self.pos, self.parent.get_farthest_point())
        return self.pos


class BacktrackingAbort(Exception):
    reason: str | None

    def __init__(self, reason: str | None = None):
        super().__init__(
            "This message should never escape the parser, it's intended to signal a failed parsing "
            "attempt\n "
            "It should never be used outside of a tokenizer.backtracking() block!\n"
            "The reason for this abort was {}".format(
                'not specified' if reason is None else reason))
        self.reason = reason


@dataclass(frozen=True)
class Span:
    """
    Parts of the input are always passed around as spans, so we know where they originated.
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

    def print_with_context(self, msg: str | None = None) -> str:
        """
        returns a string containing lines relevant to the span. The Span's contents
        are highlighted by up-carets beneath them (`^`). The message msg is printed
        along these.
        """
        info = self.input.get_lines_containing(self)
        if info is None:
            return "Unknown location of span {}. Error: ".format(self, msg)
        lines, offset_of_first_line, line_no = info
        # offset relative to the first line:
        offset = self.start - offset_of_first_line
        remaining_len = max(self.len, 1)
        capture = StringIO()
        print("{}:{}:{}".format(self.input.name, line_no, offset,
                                remaining_len),
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
        return "Span[{}:{}](text='{}')".format(self.start, self.end, self.text)


@dataclass(frozen=True)
class StringLiteral(Span):

    def __post_init__(self):
        if len(self) < 2 or self.text[0] != '"' or self.text[-1] != '"':
            raise ParseError(self, "Invalid string literal!")

    T_ = TypeVar('T_', Span, None)

    @classmethod
    def from_span(cls, span: T_) -> T_:
        if span is None:
            return None
        return cls(span.start, span.end, span.input)

    @property
    def string_contents(self):
        # TODO: is this a hack-job?
        return ast.literal_eval(self.text)

    def __repr__(self):
        return "StringLiteral[{}:{}](text='{}')".format(
            self.start, self.end, self.text)


@dataclass(frozen=True)
class Input:
    """
    This is a very simple class that is used to keep track of the input.
    """
    name: str
    content: str = field(repr=False)

    @property
    def len(self):
        return len(self.content)

    def __len__(self):
        return self.len

    def get_nth_line_bounds(self, n: int):
        start = 0
        for i in range(n):
            next_start = self.content.find('\n', start)
            if next_start == -1:
                return None
            start = next_start + 1
        return start, self.content.find('\n', start)

    def get_lines_containing(self,
                             span: Span) -> tuple[list[str], int, int] | None:
        # A pointer to the start of the first line
        start = 0
        line_no = -1
        source = self.content
        while True:
            next_start = source.find('\n', start)
            line_no += 1
            # handle eof
            if next_start == -1:
                if span.start > len(source):
                    return None
                return source[start:], start, line_no
            # as long as the next newline comes before the spans start we can continue
            if next_start < span.start:
                start = next_start + 1
                continue
            # if the whole span is on one line, we are good as well
            if next_start >= span.end:
                return [source[start:next_start]], start, line_no
            while next_start < span.end:
                next_start = source.find('\n', next_start + 1)
            return source[start:next_start].split('\n'), start, line_no

    def at(self, i: int):
        if i >= self.len:
            raise EOFError()
        return self.content[i]


save_t = tuple[int, tuple[str, ...], bool]
parsed_type_t = tuple[Span, tuple[Span]]


@dataclass
class Tokenizer:
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

    ignore_whitespace: bool = True

    history: BacktrackingHistory | None = field(init=False, default=None)

    last_token: Span | None = field(init=False, default=None)

    def save(self) -> save_t:
        """
        Create a checkpoint in the parsing process, useful for backtracking
        """
        return self.pos, self.break_on, self.ignore_whitespace

    def resume_from(self, save: save_t):
        """
        Resume from a previously saved position.

        Restores the state of the tokenizer to the exact previous position
        """
        self.pos, self.break_on, self.ignore_whitespace = save

    @contextlib.contextmanager
    def backtracking(self, region_name: str | None = None):
        """
        This context manager can be used to mark backtracking regions.

        When an error is thrown during backtracking, it is recorded and stored together
        with some meta information in the history attribute.

        The backtracker accepts the following exceptions:
         - ParseError: signifies that the region could not be parsed because of (unexpected) syntax errors
         - BacktrackingAbort: signifies that backtracking was aborted, not necessarily indicating a syntax error
         - AssertionError: this error should probably be phased out in favour of the two above
         - EOFError: signals that EOF was reached unexpectedly

        Any other error will be printed to stderr, but backtracking will continue as normal.
        """
        save = self.save()
        starting_position = self.pos
        try:
            yield
            # clear error history when something doesn't fail
            # this is because we are only interested in the last "cascade" of failures.
            # if a backtracking() completes without failre, something has been parsed (we assume)
            if self.pos > starting_position and self.history is not None:
                self.history = None
        except Exception as ex:
            how_far_we_got = self.pos

            # AssertionErrors act upon the consumed token, this means we only go to the start of the token
            if isinstance(ex, BacktrackingAbort):
                # TODO: skip space as well
                how_far_we_got -= self.last_token.len

            # if we have no error history, start recording!
            if not self.history:
                self.history = self.history_entry_from_exception(
                    ex, region_name, how_far_we_got)

            # if we got further than on previous attempts
            elif how_far_we_got > self.history.get_farthest_point():
                # throw away history
                self.history = None
                # generate new history entry,
                self.history = self.history_entry_from_exception(
                    ex, region_name, how_far_we_got)

            # otherwise, add to exception, if we are in a named region
            elif region_name is not None and how_far_we_got - starting_position > 0:
                self.history = self.history_entry_from_exception(
                    ex, region_name, how_far_we_got)

            self.resume_from(save)

    def history_entry_from_exception(self, ex: Exception, region: str,
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
                'Generic assertion failure',
                *(reason for reason in ex.args if isinstance(reason, str))
            ]
            # we assume that assertions fail because of the last read-in token
            if len(reason) == 1:
                tb = StringIO()
                traceback.print_exc(file=tb)
                reason[0] += '\n' + tb.getvalue()

            return BacktrackingHistory(ParseError(self.last_token, reason[-1], self.history),
                                       self.history, region, pos)
        elif isinstance(ex, BacktrackingAbort):
            return BacktrackingHistory(
                ParseError(
                    self.next_token(peek=True),
                    'Backtracking aborted: {}'.format(ex.reason
                                                      or 'unknown reason'), self.history),
                self.history, region, pos)
        elif isinstance(ex, EOFError):
            return BacktrackingHistory(
                ParseError(self.last_token, "Encountered EOF", self.history), self.history,
                region, pos)

        print("Warning: Unexpected error in backtracking:", file=sys.stderr)
        traceback.print_exception(ex, file=sys.stderr)

        return BacktrackingHistory(
            ParseError(self.last_token, "Unexpected exception: {}".format(ex), self.history),
            self.history, region, pos)

    def next_token(self, start: int | None = None, peek: bool = False) -> Span:
        """
        Return a Span of the next token, according to the self.break_on rules.

        Can be modified using:

         - start: don't start at the current tokenizer position, instead start here (useful for skipping comments, etc)
         - peek: don't advance the position, only "peek" at the input

        This will skip over line comments. Meaning it will skip the entire line if it encounters '//'
        """
        i = self.next_pos(start)
        # construct the span:
        span = Span(i, self._find_token_end(i), self.input)
        # advance pointer if not peeking
        if not peek:
            self.pos = span.end

        # save last token
        self.last_token = span
        return span

    def next_token_of_pattern(self,
                              pattern: re.Pattern | str,
                              peek: bool = False) -> Span | None:
        """
        Return a span that matched the pattern, or nothing. You can choose not to consume the span.
        """
        start = self.next_pos()

        # handle search for string literal
        if isinstance(pattern, str):
            if self.starts_with(pattern):
                if not peek:
                    self.pos = start + len(pattern)
                return Span(start, start + len(pattern), self.input)
            return None

        # handle regex logic
        match = pattern.match(self.input.content, start)
        if match is None:
            return None

        if not peek:
            self.pos = match.end()

        # save last token
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
        # search for literal breaks
        for part in self.break_on:
            if self.input.content.startswith(part, i):
                return i + len(part)
        # otherwise return the start of the next break
        return min(
            filter(lambda x: x >= 0, (self.input.content.find(part, i)
                                      for part in self.break_on)))

    def next_pos(self, i: int | None = None) -> int:
        """
        Find the next starting position (optionally starting from i), considering ignore_whitespaces

        This will skip line comments!
        """
        i = self.pos if i is None else i
        # skip whitespaces
        if self.ignore_whitespace:
            while self.input.at(i).isspace():
                i += 1
        # skip comments as well
        if self.input.content.startswith('//', i):
            i = self.input.content.find('\n', i) + 1
            return self.next_pos(i)
        return i

    def is_eof(self):
        """
        Check if the end of the input was reached.
        """
        try:
            self.next_pos()
        except EOFError:
            return True

    def consume_opt_whitespace(self) -> Span:
        start = self.pos
        while self.input.at(self.pos).isspace():
            self.pos += 1
        return Span(start, self.pos, self.input)

    @contextlib.contextmanager
    def configured(self,
                   break_on: tuple[str, ...] | None = None,
                   ignore_whitespace: bool | None = None):
        """
        This is a helper class to allow expressing a temporary change in config, allowing you to write:

        # parsing double-quoted string now
        string_content = ""
        with tokenizer.configured(break_on=('"', '\\'), ignore_whitespace=False):
            # use tokenizer

        # now old config is restored automatically

        """
        save = self.save()

        if break_on is not None:
            self.break_on = break_on
        if ignore_whitespace is not None:
            self.ignore_whitespace = ignore_whitespace

        try:
            yield self
        finally:
            self.break_on = save[1]
            self.ignore_whitespace = save[2]

    def starts_with(self, text: str | re.Pattern) -> bool:
        start = self.next_pos()
        if isinstance(text, re.Pattern):
            return text.match(self.input.content, start) is None
        return self.input.content.startswith(text, start)


class ParserCommons:
    """
    Colelction of common things used in parsing MLIR/IRDL

    """
    integer_literal = re.compile(r'[+-]?([0-9]+|0x[0-9A-Fa-f]+)')
    decimal_literal = re.compile(r'[+-]?([1-9][0-9]*)')
    string_literal = re.compile(r'"([^\n\f\v\r"]|\\[nfvr"])+"')
    float_literal = re.compile(r'[-+]?[0-9]+\.[0-9]*([eE][-+]?[0-9]+)?')
    bare_id = re.compile(r'[A-Za-z_][\w$.]+')
    value_id = re.compile(r'%([0-9]+|([A-Za-z_$.-][\w$.-]*))')
    suffix_id = re.compile(r'([0-9]+|([A-Za-z_$.-][\w$.-]*))')
    block_id = re.compile(r'\^([0-9]+|([A-Za-z_$.-][\w$.-]*))')
    type_alias = re.compile(r'![A-Za-z_][\w$.]+')
    attribute_alias = re.compile(r'#[A-Za-z_][\w$.]+')
    boolean_literal = re.compile(r'(true|false)')
    builtin_type = re.compile('(({}))'.format(')|('.join((
        r'[su]?i\d+',
        r'f\d+',
        'tensor',
        'vector',
        'memref',
        'complex',
        'opaque',
        'tuple',
        'index',
        # TODO: add all the Float8E4M3FNType, Float8E5M2Type, and BFloat16Type
    ))))
    builtin_type_xdsl = re.compile('!(({}))'.format(')|('.join((
        r'[su]?i\d+',
        r'f\d+',
        'tensor',
        'vector',
        'memref',
        'complex',
        'opaque',
        'tuple',
        'index',
        # TODO: add all the Float8E4M3FNType, Float8E5M2Type, and BFloat16Type
    ))))
    double_colon = re.compile('::')
    comma = re.compile(',')

    class BNF:
        """
        Collection of BNF trees.
        """
        generic_operation_body = BNF.Group(
            [
                BNF.Nonterminal('string-literal', bind="name"),
                BNF.Literal('('),
                BNF.ListOf(BNF.Nonterminal('value-id'), bind='args'),
                BNF.Literal(')'),
                BNF.OptionalGroup(
                    [
                        BNF.Literal('['),
                        BNF.ListOf(BNF.Nonterminal('block-id'),
                                   allow_empty=False,
                                   bind='blocks'),
                        # TODD: allow for block args here?! (according to spec)
                        BNF.Literal(']')
                    ],
                    debug_name="operations optional block id group"),
                BNF.OptionalGroup([
                    BNF.Literal('('),
                    BNF.ListOf(BNF.Nonterminal('region'),
                               bind='regions',
                               debug_name="regions",
                               allow_empty=False),
                    BNF.Literal(')')
                ],
                    debug_name="operation regions"),
                BNF.Nonterminal('optional-attr-dict',
                                bind='attributes',
                                debug_name="attrbiute dictionary"),
                BNF.Literal(':'),
                BNF.Nonterminal('function-type', bind='type_signature')
            ],
            debug_name="generic operation body")
        attr_dict_mlir = BNF.Group([
            BNF.Literal('{'),
            BNF.ListOf(BNF.Nonterminal('attribute-entry',
                                       debug_name="attribute entry"),
                       bind='attributes'),
            BNF.Literal('}')
        ],
            debug_name="attrbute dictionary")

        attr_dict_xdsl = BNF.Group([
            BNF.Literal('['),
            BNF.ListOf(BNF.Nonterminal('attribute-entry',
                                       debug_name="attribute entry"),
                       bind='attributes'),
            BNF.Literal(']')
        ],
            debug_name="attrbute dictionary")


class BaseParser(ABC):
    """
    Basic recursive descent parser.

    methods marked try_... will attempt to parse, and return None if they failed. If they return None
    they must make sure to restore all state.

    methods marked must_... will do greedy parsing, meaning they consume as much as they can. They will
    also throw an error if the think they should still be parsing. e.g. when parsing a list of numbers
    separated by '::', the following input will trigger an exception:
        1::2::
    Due to the '::' present after the last element. This is useful for parsing lists, as a trailing
    separator is usually considered a syntax error there.

    You can turn a try_ into a must_ by using expect(try_parse_..., error_msg)

    You can turn a must_ into a try_ by wrapping it in tokenizer.backtracking()

    must_ type parsers are preferred because they are explicit about their failure modes.
    """

    ctx: MLContext
    """xDSL context."""

    ssaValues: dict[str, SSAValue]
    blocks: dict[str, Block]

    T_ = TypeVar('T_')
    """
    Type var used for handling function that return single or multiple Spans. Basically the output type
    of all try_parse functions is T_ | None
    """

    def __init__(self,
                 input: str,
                 name: str,
                 ctx: MLContext, ):
        self.tokenizer = Tokenizer(Input(input, name))
        self.ctx = ctx
        self.ssaValues = dict()
        self.blocks = dict()

    def begin_parse(self):
        ops = []
        while (op := self.try_parse_operation()) is not None:
            ops.append(op)
        if not self.tokenizer.is_eof():
            self.raise_error("Could not parse entire input!")
        return ops

    def must_parse_block(self) -> Block:
        block_id, args = self.must_parse_optional_block_label()

        block = Block()
        if block_id is not None:
            assert block_id.text not in self.blocks
            self.blocks[block_id.text] = block

        for i, (name, type) in enumerate(args):
            arg = BlockArgument(type, block, i)
            self.ssaValues[name.text] = arg
            block.args.append(arg)

        while (next_op := self.try_parse_operation()) is not None:
            block.ops.append(next_op)

        return block

    def must_parse_optional_block_label(
            self) -> tuple[Span | None, list[tuple[Span, Attribute]]]:
        block_id = self.try_parse_block_id()
        arg_list = list()

        if block_id is not None:
            assert block_id.text not in self.blocks, "Blocks cannot have the same ID!"

            if self.tokenizer.next_token(peek=True).text == '(':
                arg_list = self.must_parse_block_arg_list()

            self.must_parse_characters(':', 'Block label must end in a `:`!')

        return block_id, arg_list

    def must_parse_block_arg_list(self) -> list[tuple[Span, Attribute]]:
        self.must_parse_characters('(', 'Block arguments must start with `(`')

        args = self.must_parse_list_of(self.try_parse_value_id_and_type,
                                       "Expected value-id and type here!")

        self.must_parse_characters(')',
                                   'Expected closing of block arguments!',
                                   is_parse_error=True)

        return args

    def try_parse_single_reference(self) -> Span | None:
        with self.tokenizer.backtracking('part of a reference'):
            self.must_parse_characters('@', "references must start with `@`")
            if (reference := self.try_parse_string_literal()) is not None:
                return reference
            if (reference := self.try_parse_suffix_id()) is not None:
                return reference
            self.raise_error(
                "References must conform to `@` (string-literal | suffix-id)")

    def must_parse_reference(self) -> list[Span]:
        return self.must_parse_list_of(
            self.try_parse_single_reference,
            'Expected reference here in the format of `@` (suffix-id | string-literal)',
            ParserCommons.double_colon,
            allow_empty=False)

    def must_parse_list_of(self,
                           try_parse: Callable[[], T_ | None],
                           error_msg: str,
                           separator_pattern: re.Pattern = ParserCommons.comma,
                           allow_empty: bool = True) -> list[T_]:
        """
        This is a greedy list-parser. It accepts input only in these cases:

         - If the separator isn't encountered, which signals the end of the list
         - If an empty list is allowed, it accepts when the first try_parse fails
         - If an empty separator is given, it instead sees a failed try_parse as the end of the list.

        This means, that the setup will not accept the input and instead raise an error:
            try_parse = parse_integer_literal
            separator = 'x'
            input = 3x4x4xi32
        as it will read [3,4,4], then see another separator, and expects the next try_parse call to succeed
        (which won't as i32 is not a valid integer literal)
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
                # if the separator is emtpy, we are good here
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

            self.must_parse_characters(
                ':', 'Expected expression (value-id `:` type)')

            type = self.try_parse_type()

            if type is None:
                self.raise_error("Expected type of value-id here!")
            return value_id, type

    def try_parse_type(self) -> Attribute | None:
        if (builtin_type := self.try_parse_builtin_type()) is not None:
            return builtin_type
        if (dialect_type :=
        self.try_parse_dialect_type()) is not None:
            return dialect_type
        return None

    def try_parse_dialect_type_or_attribute(self) -> Attribute | None:
        """
        Parse a type or an attribute.
        """
        kind = self.tokenizer.next_token_of_pattern(re.compile('[!#]'), peek=True)

        if kind is None:
            return None

        with self.tokenizer.backtracking("dialect attribute or type"):
            self.tokenizer.consume_peeked(kind)
            if kind.text == '!':
                return self.must_parse_dialect_type_or_attribute_inner('type')
            else:
                return self.must_parse_dialect_type_or_attribute_inner('attribute')

    def try_parse_dialect_type(self):
        """
        Parse a dialect type (something prefixed by `!`, defined by a dialect)
        """
        if self.tokenizer.next_token_of_pattern('!', peek=True) is None:
            return None
        with self.tokenizer.backtracking("dialect type"):
            self.tokenizer.next_token_of_pattern('!')
            return self.must_parse_dialect_type_or_attribute_inner('type')

    def try_parse_dialect_attr(self):
        """
        Parse a dialect attribute (something prefixed by `#`, defined by a dialect)
        """
        if self.tokenizer.next_token_of_pattern('#', peek=True) is None:
            return None
        with self.tokenizer.backtracking("dialect attribute"):
            self.tokenizer.next_token_of_pattern('#')
            return self.must_parse_dialect_type_or_attribute_inner('attribute')

    def must_parse_dialect_type_or_attribute_inner(self, kind: str):
        type_name = self.tokenizer.next_token_of_pattern(
            ParserCommons.bare_id)

        if type_name is None:
            self.raise_error(
                "Expected dialect {} name here!".format(kind))

        type_def = self.ctx.get_optional_attr(type_name.text)
        if type_def is None:
            self.raise_error("'{}' is not a know attribute!".format(type_name.text), type_name)

        # pass the task of parsing parameters on to the attribute/type definition
        param_list = type_def.parse_parameters(self)
        return type_def(param_list)

    @abstractmethod
    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        raise NotImplemented("Subclasses must implement this method!")

    def must_parse_builtin_parametrized_type(
            self, name: Span) -> ParametrizedAttribute:

        def unimplemented() -> ParametrizedAttribute:
            raise ParseError(name,
                             "Builtin {} not supported yet!".format(name.text))

        builtin_parsers: dict[str, Callable[[], ParametrizedAttribute]] = {
            'vector': self.must_parse_vector_attrs,
            'memref': unimplemented,
            'tensor': self.must_parse_tensor_attrs,
            'complex': self.must_parse_complex_attrs,
            'opaque': unimplemented,
            'tuple': unimplemented,
        }

        self.must_parse_characters('<', 'Expected parameter list here!')
        # get the parser for the type, falling back to the unimplemented warning
        res = builtin_parsers.get(name.text, unimplemented)()
        self.must_parse_characters('>',
                                   'Expected end of parameter list here!',
                                   is_parse_error=True)
        return res

    def must_parse_complex_attrs(self):
        self.raise_error("ComplexType is unimplemented!")

    def try_parse_numerical_dims(self,
                                 accept_closing_bracket: bool = False,
                                 lower_bound: int = 1) -> Iterable[int]:
        while (shape_arg :=
        self.try_parse_shape_element(lower_bound)) is not None:
            yield shape_arg
            # look out for the closing bracket for scalable vector dims
            if accept_closing_bracket and self.tokenizer.next_token(
                    peek=True).text == ']':
                break
            self.must_parse_characters(
                'x',
                'Unexpected end of dimension parameters!',
                is_parse_error=True)

    def must_parse_vector_attrs(self) -> AnyVectorType:
        # also break on 'x' characters as they are separators in dimension parameters
        with self.tokenizer.configured(break_on=self.tokenizer.break_on +
                                                ('x',)):
            shape = list[int](self.try_parse_numerical_dims())
            scaling_shape: list[int] | None = None

            if self.tokenizer.next_token_of_pattern('[') is not None:
                # we now need to parse the scalable dimensions
                scaling_shape = list(self.try_parse_numerical_dims())
                self.must_parse_characters(
                    ']',
                    'Expected end of scalable vector dimensions here!',
                    is_parse_error=True)
                self.must_parse_characters(
                    'x',
                    'Expected end of scalable vector dimensions here!',
                    is_parse_error=True)

            if scaling_shape is not None:
                # TODO: handle scaling vectors!
                print("Warning: scaling vectors not supported!")
                pass

            type = self.try_parse_type()
            if type is None:
                self.raise_error(
                    "Expected a type at the end of the vector parameters!")

            return VectorType.from_type_and_list(type, shape)

    def must_parse_tensor_or_memref_dims(self) -> list[int] | None:
        with self.tokenizer.configured(break_on=self.tokenizer.break_on +
                                                ('x',)):
            # check for unranked-ness
            if self.tokenizer.next_token_of_pattern('*') is not None:
                # consume `x`
                self.must_parse_characters(
                    'x',
                    'Unranked tensors must follow format (`<*x` type `>`)',
                    is_parse_error=True)
            else:
                # parse rank:
                return list(self.try_parse_numerical_dims(lower_bound=0))

    def must_parse_tensor_attrs(self) -> AnyTensorType:
        shape = self.must_parse_tensor_or_memref_dims()
        type = self.try_parse_type()

        if type is None:
            self.raise_error("Expected tensor type here!")

        if self.tokenizer.next_token(peek=True).text == ',':
            # TODO: add tensor encoding!
            raise self.raise_error("Parsing tensor encoding is not supported!")

        if shape is None and self.tokenizer.next_token(peek=True).text == ',':
            raise self.raise_error("Unranked tensors don't have an encoding!")

        if shape is not None:
            return TensorType.from_type_and_list(type, shape)

        return UnrankedTensorType.from_type(type)

    def try_parse_shape_element(self, lower_bound: int = 1) -> int | None:
        """
        Parse a shape element, either a decimal integer immediate or a `?`, which evaluates to -1

        immediate cannot be smaller than lower_bound (defaults to 1) (is 0 for tensors and memrefs)
        """
        int_lit = self.try_parse_decimal_literal()

        if int_lit is not None:
            value = int(int_lit.text)
            if value < lower_bound:
                # TODO: this is ugly, it's a raise inside a try_ type function, which should instead just give up
                raise ParseError(
                    int_lit,
                    "Shape element literal cannot be negative or zero!")
            return value

        next_token = self.tokenizer.next_token(peek=True)

        if next_token.text == '?':
            self.tokenizer.consume_peeked(next_token)
            return -1
        return None

    def must_parse_type_params(self) -> list[Attribute]:
        # consume opening bracket
        assert self.tokenizer.next_token(
        ).text == '<', 'Type must be parameterized!'

        params = self.must_parse_list_of(self.try_parse_type,
                                         'Expected a type here!')

        assert self.tokenizer.next_token(
        ).text == '>', 'Expected end of type parameterization here!'

        return params

    def expect(self, try_parse: Callable[[], T_ | None],
               error_message: str) -> T_:
        """
        Used to force completion of a try_parse function. Will throw a parse error if it can't
        """
        res = try_parse()
        if res is None:
            self.raise_error(error_message)
        return res

    def raise_error(self, msg: str, at_position: Span | None = None):
        """
        Helper for raising exceptions, provides as much context as possible to them.

        This will, for example, include backtracking errors, if any occured previously
        """
        if at_position is None:
            at_position = self.tokenizer.next_token(peek=True)

        raise ParseError(at_position, msg, self.tokenizer.history)

    def must_parse_characters(self,
                              text: str,
                              msg: str,
                              is_parse_error: bool = False) -> Span:
        if (match := self.tokenizer.next_token_of_pattern(text)) is None:
            if is_parse_error:
                self.raise_error(msg)
            raise AssertionError("Unexpected input: {}".format(msg))
        return match

    @abstractmethod
    def must_parse_op_result_list(
            self) -> tuple[list[Span], list[Attribute] | None]:
        raise NotImplemented()

    def try_parse_operation(self) -> Operation | None:
        with self.tokenizer.backtracking("operation"):

            result_list, ret_types = self.must_parse_op_result_list()
            if len(result_list) > 0:
                self.must_parse_characters(
                    '=',
                    'Operation definitions expect an `=` after op-result-list!'
                )

            # check for custom op format
            op_name = self.try_parse_bare_id()
            if op_name is not None:
                op_type = self.ctx.get_op(op_name.text)
                op = op_type.parse(ret_types, self)
            else:
                # check for basic op format
                op_name = self.try_parse_string_literal()
                if op_name is None:
                    self.raise_error(
                        "Expected an operation name here, either a bare-id, or a string literal!"
                    )

                args, successors, attrs, regions, func_type = self.must_parse_operation_details()

                if ret_types is None:
                    assert func_type is not None
                    ret_types = func_type.outputs.data

                op_type = self.ctx.get_op(op_name.string_contents)

                op = op_type.create(
                    operands=[self.ssaValues[span.text] for span in args],
                    result_types=ret_types,
                    attributes=attrs,
                    successors=[
                        self.blocks[block_name.text]
                        for block_name in successors
                    ],
                    regions=regions)

            # Register the result SSA value names in the parser
            for (idx, res) in enumerate(result_list):
                ssa_val_name = res.text
                if ssa_val_name in self.ssaValues:
                    self.raise_error(f"SSA value {ssa_val_name} is already defined", res)
                self.ssaValues[ssa_val_name] = op.results[idx]
                # TODO: check name?
                self.ssaValues[ssa_val_name].name = ssa_val_name

            return op

    def must_parse_region(self) -> Region:
        oldSSAVals = self.ssaValues.copy()
        oldBBNames = self.blocks.copy()
        self.blocks = dict[str, Block]()

        region = Region()

        try:
            self.must_parse_characters('{', 'Regions begin with `{`')
            if self.tokenizer.next_token(peek=True).text != '}':
                # parse first block
                block = self.must_parse_block()
                region.add_block(block)

                while self.tokenizer.next_token(peek=True).text == '^':
                    region.add_block(self.must_parse_block())

            self.must_parse_characters('}',
                                       'Reached end of region, expected `}`!')

            return region
        finally:
            self.ssaValues = oldSSAVals
            self.blocks = oldBBNames

    def try_parse_op_name(self) -> Span | None:
        if (str_lit := self.try_parse_string_literal()) is not None:
            return str_lit
        return self.try_parse_bare_id()

    def must_parse_attribute_entry(self) -> tuple[Span, Attribute]:
        """
        Parse entry in attribute dict. Of format:

        attrbiute_entry := (bare-id | string-literal) `=` attribute
        attrbiute       := dialect-attribute | builtin-attribute
        """
        if (name := self.try_parse_bare_id()) is None:
            name = self.try_parse_string_literal()

        if name is None:
            self.raise_error(
                'Expected bare-id or string-literal here as part of attribute entry!'
            )

        self.must_parse_characters(
            '=', 'Attribute entries must be of format name `=` attribute!')

        return name, self.must_parse_attribute()

    @abstractmethod
    def must_parse_attribute(self) -> Attribute:
        """
        Parse attribute (either builtin or dialect)
        """
        raise NotImplemented()

    def must_parse_attribute_type(self) -> Attribute:
        """
        Parses `:` type and returns the type
        """
        self.must_parse_characters(
            ':', 'Expected attribute type definition here ( `:` type )')
        return self.expect(
            self.try_parse_type,
            'Expected attribute type definition here ( `:` type )')

    def try_parse_builtin_attr(self) -> Attribute:
        """
        Tries to parse a bultin attribute, e.g. a string literal, int, array, etc..
        """
        # order here is important!
        attrs = (self.try_parse_builtin_float_attr,
                 self.try_parse_builtin_int_attr,
                 self.try_parse_builtin_str_attr,
                 self.try_parse_builtin_arr_attr, self.try_parse_function_type)

        for attr_parser in attrs:
            if (val := attr_parser()) is not None:
                return val

    def try_parse_builtin_int_attr(self) -> IntegerAttr | None:
        bool = self.try_parse_builtin_boolean_attr()
        if bool is not None:
            return bool

        with self.tokenizer.backtracking("built in int attribute"):
            value = self.expect(
                self.try_parse_integer_literal,
                'Integer attribute must start with an integer literal!')
            if self.tokenizer.next_token(peek=True).text != ':':
                print(self.tokenizer.next_token(peek=True))
                return IntegerAttr.from_params(int(value.text),
                                               DefaultIntegerAttrType)
            type = self.must_parse_attribute_type()
            return IntegerAttr.from_params(int(value.text), type)

    def try_parse_builtin_float_attr(self) -> FloatAttr | None:
        with self.tokenizer.backtracking("float literal"):
            value = self.expect(
                self.try_parse_float_literal,
                'Float attribute must start with a float literal!')
            # if we don't see a ':' indicating a type signature
            if self.tokenizer.next_token(peek=True).text != ':':
                return FloatAttr.from_value(float(value.text))

            type = self.must_parse_attribute_type()
            return FloatAttr.from_value(float(value.text), type)

    def try_parse_builtin_boolean_attr(self) -> IntegerAttr | None:
        span = self.try_parse_boolean_literal()

        if span is None:
            return None

        int_val = ['false', 'true'].index(span.text)
        return IntegerAttr.from_params(int_val, IntegerType.from_width(1))

    def try_parse_builtin_str_attr(self):
        if self.tokenizer.next_token(peek=True).text != '"':
            return None

        with self.tokenizer.backtracking("string literal"):
            literal = self.try_parse_string_literal()
            if self.tokenizer.next_token(peek=True).text != ':':
                return StringAttr.from_str(literal.string_contents)
            self.raise_error("Typed string literals are not supported!")

    def try_parse_builtin_arr_attr(self) -> list[Attribute] | None:
        if self.tokenizer.next_token(peek=True).text != '[':
            return None
        with self.tokenizer.backtracking("array literal"):
            self.must_parse_characters('[',
                                       'Array literals must start with `[`')
            attrs = self.must_parse_list_of(self.must_parse_attribute,
                                            'Expected array entry!')
            self.must_parse_characters(
                ']', 'Array literals must be enclosed by square brackets!')
            return ArrayAttr.from_list(attrs)

    @abstractmethod
    def must_parse_optional_attr_dict(self) -> dict[str, Attribute]:
        raise NotImplementedError()

    def attr_dict_from_tuple_list(
            self, tuple_list: list[tuple[Span,
                                         Attribute]]) -> dict[str, Attribute]:
        return dict(
            ((span.string_contents if isinstance(span, StringLiteral
                                                 ) else span.text), attr)
            for span, attr in tuple_list)

    def must_parse_function_type(self) -> FunctionType:
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
        self.must_parse_characters(
            '(', 'First group of function args must start with a `(`')
        args: list[Attribute] = self.must_parse_list_of(
            self.try_parse_type, 'Expected type here!')
        self.must_parse_characters(')',
                                   "Malformed function type!",
                                   is_parse_error=True)

        self.must_parse_characters('->',
                                   'Malformed function type!',
                                   is_parse_error=True)

        return FunctionType.from_lists(
            args, self.must_parse_type_or_type_list_parens())

    def must_parse_type_or_type_list_parens(self) -> list[Attribute]:
        """
        Parses type-or-type-list-parens, which is used in function-type.

        type-or-type-list-parens ::= type | type-list-parens
        type-list-parens         ::= `(` `)` | `(` type-list-no-parens `)`
        type-list-no-parens      ::=  type (`,` type)*
        """
        if self.tokenizer.next_token_of_pattern('(') is not None:
            args: list[Attribute] = self.must_parse_list_of(
                self.try_parse_type, 'Expected type here!')
            self.must_parse_characters(')',
                                       "Unclosed function type argument list!",
                                       is_parse_error=True)
        else:
            args = [self.try_parse_type()]
            if args[0] is None:
                self.raise_error(
                    "Function type must either be single type or list of types in parenthesis!"
                )
        return args

    def try_parse_function_type(self) -> FunctionType | None:
        if self.tokenizer.next_token(peek=True).text != '(':
            return None
        with self.tokenizer.backtracking('function type'):
            return self.must_parse_function_type()

    def must_parse_region_list(self) -> list[Region]:
        """
        Parses a sequence of regions for as long as there is a `{` in the input.
        """
        regions = []
        while self.tokenizer.next_token(peek=True).text == '{':
            regions.append(self.must_parse_region())
        return regions

    # HERE STARTS A SOMEWHAT CURSED COMPATIBILITY LAYER:
    # since we don't want to rewrite all dialects currently, the new emulator needs to expose the same
    # interface to the dialect definitions. Here we implement that interface.

    _OperationType = TypeVar('_OperationType', bound=Operation)

    def parse_op_with_default_format(
            self,
            op_type: type[_OperationType],
            result_types: list[Attribute],
            skip_white_space: bool = True) -> _OperationType:
        """
        Compatibility wrapper so the new parser can be passed instead of the old one. Parses everything after the
        operation name.

        This implicitly assumes XDSL format, and will fail on MLIR style operations
        """
        # TODO: remove this function and restructure custom op / irdl parsing

        args = self.must_parse_op_args_list()
        successors: list[Span] = []
        if self.tokenizer.next_token_of_pattern('(') is not None:
            successors = self.must_parse_list_of(self.try_parse_block_id,
                                                 'Malformed block-id!')
            self.must_parse_characters(
                ')',
                'Expected either a block id or the end of the successor list here'
            )

        attributes = self.must_parse_optional_attr_dict()

        regions = self.must_parse_region_list()

        for x in args:
            if x.text not in self.ssaValues:
                self.raise_error(
                    "Unknown SSAValue name, known SSA Values are: {}".format(", ".join(self.ssaValues.keys())),
                    x
                )

        return op_type.create(
            operands=[self.ssaValues[span.text] for span in args],
            result_types=result_types,
            attributes=attributes,
            successors=[self.blocks[span.text] for span in successors],
            regions=regions)

    def parse_paramattr_parameters(
            self,
            expect_brackets: bool = False,
            skip_white_space: bool = True) -> list[Attribute]:
        if self.tokenizer.next_token_of_pattern(
                '<') is None and expect_brackets:
            self.raise_error("Expected start attribute parameters here (`<`)!")

        res = self.must_parse_list_of(self.must_parse_attribute,
                                      'Expected another attribute here!')

        if self.tokenizer.next_token_of_pattern(
                '>') is None and expect_brackets:
            self.raise_error(
                "Malformed parameter list, expected either another parameter or `>`!"
            )

        return res

    # COMMON xDSL/MLIR code:
    def must_parse_builtin_type_with_name(self, name: Span):
        if name.text == 'index':
            return IndexType()
        if (re_match := re.match(r'^[su]?i(\d+)$', name.text)) is not None:
            signedness = {
                's': Signedness.SIGNED,
                'u': Signedness.UNSIGNED,
                'i': Signedness.SIGNLESS
            }
            return IntegerType.from_width(int(re_match.group(1)),
                                          signedness[name.text[0]])

        if (re_match := re.match(r'^f(\d+)$', name.text)) is not None:
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

        return self.must_parse_builtin_parametrized_type(name)

    @abstractmethod
    def must_parse_operation_details(self) -> tuple[
        list[Span], list[Span], dict[str, Attribute], list[Region], FunctionType | None]:
        """
        Must return a tuple consisting of:
            - a list of arguments to the operation
            - a list of successor names
            - the attributes attached to the OP
            - the regions of the op
            - An optional function type. If not supplied, must_parse_op_result_list must return a second value
              containing the types of the returned SSAValues

        Your implementation should make use of the following functions:
            - must_parse_op_args_list
            - must_parse_optional_attr_dict
            - must_parse_
        """
        raise NotImplementedError()


    def must_parse_op_args_list(self) -> list[Span]:
        self.must_parse_characters('(', 'Operation args list must be enclosed by brackets!')
        args = self.must_parse_list_of(self.try_parse_value_id_and_type, 'Expected another bare-id here')
        self.must_parse_characters(')', 'Operation args list must be closed by a closing bracket')
        # TODO: check if type is correct here!
        return [name for name, _ in args]

    @abstractmethod
    def must_parse_optional_successor_list(self) -> list[Span]:
        pass

class MLIRParser(BaseParser):

    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        with self.tokenizer.backtracking("builtin type"):
            name = self.tokenizer.next_token_of_pattern(ParserCommons.builtin_type)
            if name is None:
                raise BacktrackingAbort("Expected builtin name!")

            return self.must_parse_builtin_type_with_name(name)

    def must_parse_attribute(self) -> Attribute:
        """
        Parse attribute (either builtin or dialect)
        """
        # all dialect attrs must start with '#', so we check for that first (as it's easier)
        if self.tokenizer.next_token(peek=True).text == '#':
            value = self.try_parse_dialect_attr()

            # no value => error
            if value is None:
                self.raise_error(
                    '`#` must be followed by a valid dialect attribute or type!'
                )

            return value

        # if it isn't a dialect attr, parse builtin
        builtin_val = self.try_parse_builtin_attr()

        if builtin_val is None:
            self.raise_error(
                "Unknown attribute (neither builtin nor dialect could be parsed)!"
            )

        return builtin_val

    def must_parse_op_result_list(
            self) -> tuple[list[Span], list[Attribute] | None]:
        return self.must_parse_list_of(self.try_parse_value_id,
                                       'Expected op-result here!',
                                       allow_empty=True), None

    def must_parse_optional_attr_dict(self) -> dict[str, Attribute]:
        if self.tokenizer.next_token_of_pattern('{', peek=True) is None:
            return dict()

        res = ParserCommons.BNF.attr_dict_mlir.must_parse(self)

        return self.attr_dict_from_tuple_list(
            ParserCommons.BNF.attr_dict_mlir.collect(res, dict()).get(
                'attributes', list()))

    def must_parse_operation_details(self) -> tuple[
        list[Span], list[Span], dict[str, Attribute], list[Region], FunctionType | None]:

        args = self.must_parse_op_args_list()
        succ = self.must_parse_optional_successor_list()

        regions = []
        if self.tokenizer.starts_with('('):
            self.must_parse_characters('(', 'Expected brackets enclosing regions!')
            regions = self.must_parse_region_list()
            self.must_parse_characters(')', 'Expected brackets enclosing regions!')

        attrs = self.must_parse_optional_attr_dict()

        self.must_parse_characters(':', 'MLIR Operation defintions must end in a function type signature!')
        func_type = self.must_parse_function_type()

        return args, succ, attrs, regions, func_type

    def must_parse_optional_successor_list(self) -> list[Span]:
        if not self.tokenizer.starts_with('['):
            return []
        self.must_parse_characters('[', 'Successor list is enclosed in square brackets')
        successors = self.must_parse_list_of(self.try_parse_block_id, 'Expected a block-id', allow_empty=False)
        self.must_parse_characters(']', 'Successor list is enclosed in square brackets')
        return successors


class XDSLParser(BaseParser):

    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        with self.tokenizer.backtracking("builtin type"):
            name = self.tokenizer.next_token_of_pattern(ParserCommons.builtin_type_xdsl)
            if name is None:
                raise BacktrackingAbort("Expected builtin name!")
            # xdsl builtin types have a '!' prefix, we strip that out here
            name = Span(start=name.start + 1,
                        end=name.end,
                        input=name.input)

            return self.must_parse_builtin_type_with_name(name)

    def must_parse_attribute(self) -> Attribute:
        """
        Parse attribute (either builtin or dialect)

        xDSL allows types in places of attributes! That's why we parse types here as well
        """
        value = self.try_parse_builtin_attr()

        # xDSL: Allow both # and ! prefixes, as we allow both types and attrs
        if value is None and self.tokenizer.next_token(peek=True).text in '#!':
            # in MLIR # and ! are prefixes for dialect attrs/types, but in xDSL ! is also used for builtin types
            value = self.try_parse_dialect_type_or_attribute()

        if value is None:
            self.raise_error(
                "Unknown attribute (neither builtin nor dialect could be parsed)!"
            )

        return value

    def must_parse_op_result_list(
            self) -> tuple[list[Span], list[Attribute] | None]:
        results = self.must_parse_list_of(self.try_parse_value_id_and_type,
                                          'Expected (value-id `:` type) here!',
                                          allow_empty=True)
        # TODO: this is hideous, make it cleaner
        # zip(*results) works, but is barely readable :/
        return [name for name, _ in results], [type for _, type in results]

    def try_parse_builtin_attr(self) -> Attribute:
        """
        Tries to parse a bultin attribute, e.g. a string literal, int, array, etc..

        If the mode is xDSL, it also allows parsing of builtin types
        """
        # in xdsl, two things are different here:
        #  1. types are considered valid attributes
        #  2. all types, builtins included, are prefixed with !
        if self.tokenizer.starts_with('!'):
            return self.try_parse_builtin_type()

        return super().try_parse_builtin_attr()

    def must_parse_optional_attr_dict(self) -> dict[str, Attribute]:
        if self.tokenizer.next_token_of_pattern('[', peek=True) is None:
            return dict()

        res = ParserCommons.BNF.attr_dict_xdsl.must_parse(self)

        return self.attr_dict_from_tuple_list(
            ParserCommons.BNF.attr_dict_mlir.collect(res, dict()).get(
                'attributes', list()))

    def must_parse_operation_details(self) -> tuple[
        list[Span], list[Span], dict[str, Attribute], list[Region], FunctionType | None]:
        """
        Must return a tuple consisting of:
            - a list of arguments to the operation
            - a list of successor names
            - the attributes attached to the OP
            - the regions of the op
            - An optional function type. If not supplied, must_parse_op_result_list must return a second value
              containing the types of the returned SSAValues

        """
        args = self.must_parse_op_args_list()
        succ = self.must_parse_optional_successor_list()
        attrs = self.must_parse_optional_attr_dict()
        regions = self.must_parse_region_list()

        return args, succ, attrs, regions, None

    def must_parse_optional_successor_list(self) -> list[Span]:
        if not self.tokenizer.starts_with('['):
            return []
        self.must_parse_characters('[', 'Successor list is enclosed in square brackets')
        successors = self.must_parse_list_of(self.try_parse_block_id, 'Expected a block-id', allow_empty=False)
        self.must_parse_characters(']', 'Successor list is enclosed in square brackets')
        return successors



"""
digit     ::= [0-9]
hex_digit ::= [0-9a-fA-F]
letter    ::= [a-zA-Z]
id-punct  ::= [$._-]

integer-literal ::= decimal-literal | hexadecimal-literal
decimal-literal ::= digit+
hexadecimal-literal ::= `0x` hex_digit+
float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
string-literal  ::= `"` [^"\n\f\v\r]* `"`   TODO: define escaping rules

bare-id ::= (letter|[_]) (letter|digit|[_$.])*
bare-id-list ::= bare-id (`,` bare-id)*
value-id ::= `%` suffix-id
alias-name :: = bare-id
suffix-id ::= (digit+ | ((letter|id-punct) (letter|id-punct|digit)*))


symbol-ref-id ::= `@` (suffix-id | string-literal) (`::` symbol-ref-id)?
value-id-list ::= value-id (`,` value-id)*

// Uses of value, e.g. in an operand list to an operation.
value-use ::= value-id
value-use-list ::= value-use (`,` value-use)*

operation            ::= op-result-list? (generic-operation | custom-operation)
                         trailing-location?
generic-operation    ::= string-literal `(` value-use-list? `)`  successor-list?
                         region-list? dictionary-attribute? `:` function-type
custom-operation     ::= bare-id custom-operation-format
op-result-list       ::= op-result (`,` op-result)* `=`
op-result            ::= value-id (`:` integer-literal)
successor-list       ::= `[` successor (`,` successor)* `]`
successor            ::= caret-id (`:` block-arg-list)?
region-list          ::= `(` region (`,` region)* `)`
dictionary-attribute ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location    ::= (`loc` `(` location `)`)?

block           ::= block-label operation+
block-label     ::= block-id block-arg-list? `:`
block-id        ::= caret-id
caret-id        ::= `^` suffix-id
value-id-and-type ::= value-id `:` type

// Non-empty list of names and types.
value-id-and-type-list ::= value-id-and-type (`,` value-id-and-type)*

block-arg-list ::= `(` value-id-and-type-list? `)`

type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

function-type ::= (type | type-list-parens) `->` (type | type-list-parens)

"""

if __name__ == '__main__':
    infile = sys.argv[-1]
    from xdsl.dialects.affine import Affine
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.cf import Cf
    from xdsl.dialects.cmath import CMath
    from xdsl.dialects.func import Func
    from xdsl.dialects.irdl import IRDL
    from xdsl.dialects.llvm import LLVM
    from xdsl.dialects.memref import MemRef
    from xdsl.dialects.scf import Scf
    import os

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(MemRef)
    ctx.register_dialect(Affine)
    ctx.register_dialect(Scf)
    ctx.register_dialect(Cf)
    ctx.register_dialect(CMath)
    ctx.register_dialect(IRDL)
    ctx.register_dialect(LLVM)

    parses_by_file_name = {'xdsl': XDSLParser, 'mlir': MLIRParser}

    parser = parses_by_file_name[infile.split('.')[-1]]

    p = parser(infile,
               open(infile, 'r').read(),
               ctx)

    printer = Printer()
    try:
        for op in p.begin_parse():
            printer.print_op(op)
    except ParseError as pe:
        pe.print_with_history()
