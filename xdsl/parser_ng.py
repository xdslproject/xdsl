from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import re
import ast
from io import StringIO
from typing import Any, TypeVar, Iterable, Literal
from enum import Enum

from xdsl.ir import (SSAValue, Block, Callable, Attribute, Operation, Region,
                     BlockArgument, MLContext, ParametrizedAttribute)

import xdsl.utils.bnf as BNF

from xdsl.dialects.builtin import (
    AnyFloat, AnyTensorType, AnyUnrankedTensorType, AnyVectorType,
    DenseIntOrFPElementsAttr, Float16Type, Float32Type, Float64Type, FloatAttr,
    FunctionType, IndexType, IntegerType, OpaqueAttr, Signedness, StringAttr,
    FlatSymbolRefAttr, IntegerAttr, ArrayAttr, TensorType, UnitAttr,
    UnrankedTensorType, UnregisteredOp, VectorType)

from xdsl.irdl import Data


class ParseError(Exception):
    span: Span
    msg: str

    def __init__(self, span: Span, msg: str):
        super().__init__(span.print_with_context(msg))
        self.span = span
        self.msg = msg

    def print_pretty(self):
        print(
            self.span.print_with_context(self.msg)
        )


class BacktrackingAbort(Exception):
    reason: str | None

    def __init__(self, reason: str | None = None):
        super("This message should never escape the parser, it's intended to signal a failed parsing attempt\n"
              "It should never be used outside of a tokenizer.backtracking() block!\n"
              "The reason for this abort was {}".format('not specified' if reason is None else reason))
        self.reason = reason


@dataclass(frozen=True)
class Span:
    """
    Parts of the input are always passed around as spans so we know where they originated.
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

    def print_with_context(self, msg: str | None = None):
        info = self.input.get_lines_containing(self)
        assert info is not None
        lines, offset_of_first_line, line_no = info
        # offset relative to the first line:
        offset = self.start - offset_of_first_line
        remaining_len = self.len
        capture = StringIO()
        print("file: {}:{}".format(self.input.name, line_no), file=capture)
        for line in lines:
            print(line, file=capture)
            if offset > len(line):
                offset -= len(line)
                continue
            if remaining_len <= 0:
                continue
            len_on_this_line = min(remaining_len, len(line) - offset)
            remaining_len -= len_on_this_line
            print("{}{}".format(" " * offset, "^" * max(len_on_this_line, 1)), file=capture)
            if msg is not None:
                print("{}{}".format(" " * offset, msg), file=capture)
                msg = None
            offset = 0
        return capture.getvalue()

    def __repr__(self):
        return "Span[{}:{}](text='{}', input={})".format(self.start, self.end, self.text, self.input)


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

    def get_lines_containing(self, span: Span) -> tuple[list[str], int, int] | None:
        # A pointer to the start of the first line
        start = 0
        line_no = 0
        source = self.content
        while True:
            next_start = source.find('\n', start)
            line_no += 1
            # handle eof
            if next_start == -1:
                return None
            # as long as the next newline comes before the spans start we are good
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

    break_on: tuple[str, ...] = (
        '.', '%', ' ', '(', ')', '[', ']', '{', '}', '<', '>', ':', '=', '@', '?', '|', '->', '-', '//', '\n', '\t', '#'
    )
    """
    characters the tokenizer should break on
    """

    ignore_whitespace: bool = True

    last_error: ParseError | None = field(init=False, default=None)
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
    def backtracking(self):
        """
        Used to create backtracking parsers. You can wrap you parse code into

        with tokenizer.backtracking():
            # do some stuff
            assert x == 'array'

        All exceptions triggered in the body will abort the parsing attempt, but not escape further.

        The tokenizer state will not change.

        When backtracking occurred, the backtracker will save the last exception in last_error
        """
        save = self.save()
        try:
            self.last_error = None
            yield
        except Exception as ex:
            if isinstance(ex, BacktrackingAbort):
                self.last_error = ParseError(
                    self.next_token(peek=True),
                    'Backtracking aborted: {}'.format(ex.reason or 'unknown reason')
                )
            elif isinstance(ex, AssertionError):
                reason = ['Generic assertion failure', *(reason for reason in ex.args if isinstance(reason, str))]
                # we assume that assertions fail because of the last read-in token
                self.last_error = ParseError(self.last_token, reason[-1])
            elif isinstance(ex, ParseError):
                self.last_error = ex
                print("Warning: ParseError in backtracking:\n{}".format(ex))
            else:
                print("Warning: Unexpected error in backtracking:\n{}".format(ex))
            self.resume_from(save)

    def next_token(self, start: int | None = None, skip: int = 0, peek: bool = False,
                   include_comments: bool = False) -> Span:
        """
        Best effort guess at what the next token could be
        """
        i = self.next_pos(start)
        while skip > 0:
            # skip whitespace if able
            i = self.next_pos(self._find_token_end(i))
            skip -= 1
        # advance to the next position
        if not peek:
            self.pos = self._find_token_end(i)

        span = self.span_of(i, self.pos)
        if not include_comments and span.text == '//':
            while self.input.at(i) != '\n':
                i += 1
            return self.next_token(i, 0, peek, include_comments)

        # save last token
        self.last_token = span
        return span

    def next_token_of_pattern(self, pattern: re.Pattern, peek: bool = False) -> Span | None:
        """
        Return a span that matched the pattern, or nothing. You can choose not to consume the span.
        """
        start = self.next_pos()
        match = pattern.match(self.input.content, start)
        if match is None:
            return None
        if not peek:
            self.pos = match.end()
        # save last token
        self.last_token = self.span_of(start, match.end())
        return self.last_token

    def jump_back_to(self, span: Span):
        """
        This can be used to "rewind" the tokenizer back to the point right before you consumed the token.

        This leaves everything except the position untouched
        """
        self.pos = span.start

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
        return min(filter(lambda x: x >= 0, (self.input.content.find(part, i) for part in self.break_on)))

    def next_pos(self, i: int | None = None) -> int:
        """
        Find the next starting position (optionally starting from i), considering ignore_whitespaces
        """
        i = self.pos if i is None else i
        # skip whitespaces
        if self.ignore_whitespace:
            while self.input.at(i).isspace():
                i += 1
        return i

    def is_eof(self):
        try:
            i = self.pos
            while self.input.at(i).isspace():
                i += 1
            return False
        except EOFError:
            return True

    def span_of(self, start: int, end: int) -> Span:
        return Span(start, end, self.input)

    def consume_opt_whitespace(self) -> Span:
        start = self.pos
        while self.input.at(self.pos).isspace():
            self.pos += 1
        return self.span_of(start, self.pos)

    @contextlib.contextmanager
    def configured(self, break_on: tuple[str, ...] | None = None, ignore_whitespace: bool | None = None):
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


class ParserCommons:
    """
    Colelction of common things used in parsing MLIR/IRDL

    """
    integer_literal = re.compile(r'[+-]?([0-9]+|0x[0-9A-f]+)')
    decimal_literal = re.compile(r'[+-]?([1-9][0-9]*)')
    string_literal = re.compile(r'"([^\n\f\v\r"]|\\[nfvr"])+"')
    float_literal = re.compile(r'[-+]?[0-9]+\.[0-9]*([eE][-+]?[0-9]+)?')
    bare_id = re.compile(r'[A-z_][A-z0-9_$.]+')
    value_id = re.compile(r'%[A-z_][A-z0-9_$.]+')
    suffix_id = re.compile(r'([0-9]+|([A-z_$.-][0-9A-z_$.-]*))')
    block_id = re.compile(r'\^([0-9]+|([A-z_$.-][0-9A-z_$.-]*))')
    type_alias = re.compile(r'![A-z_][A-z0-9_$.]+')
    attribute_alias = re.compile(r'#[A-z_][A-z0-9_$.]+')
    builtin_type = re.compile('({})'.format(
        '|'.join((
            r'[su]?i\d+', 'tensor', 'vector',
            'memref', 'complex', 'opaque',
            'tuple', 'index',
            # TODO: add all the FloatNtype, Float8E4M3FNType, Float8E5M2Type, and BFloat16Type
        ))
    ))
    double_colon = re.compile('::')
    comma = re.compile(',')

    class BNF:
        """
        Collection of BNF trees.
        """
        generic_operation = BNF.Group([
            BNF.Nonterminal('string-literal', bind="name"),
            BNF.Literal('('),
            BNF.ListOf(BNF.Nonterminal('value-id'), bind='args'),
            BNF.Literal(')'),
            BNF.OptionalGroup([
                BNF.Literal('['),
                BNF.ListOf(BNF.Nonterminal('block-id'), allow_empty=False, bind='blocks'),
                # TODD: allow for block args here?! (accordin to spec)
                BNF.Literal(']')
            ], bind='blocks_group'),
            BNF.OptionalGroup([
                BNF.Literal('('),
                BNF.ListOf(BNF.Nonterminal('region'), bind='regions', allow_empty=False),
                BNF.Literal(')')
            ], bind='region_group'),
            BNF.Nonterminal('attr-dict', bind='attributes'),
            BNF.Literal(':'),
            BNF.Nonterminal('function-type', bind='type_signature')
        ])
        region = BNF.Group([
            BNF.Literal('{'),
            BNF.ListOf(BNF.Nonterminal('operation'), separator=re.compile('')),
            BNF.Literal('}'),
        ])


class MlirParser:
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

    You can turn a must_ into a try_ by wrapping it inside of a tokenizer.backtracking()

    must_ type parsers are preferred because they are explicit about their failure modes.
    """

    class Accent(Enum):
        XDSL = 'xDSL'
        MLIR = 'MLIR'

    accent: Accent

    ctx: MLContext
    """xDSL context."""

    _ssaValues: dict[str, SSAValue] = field(init=False, default_factory=dict)
    _blocks: dict[str, Block] = field(init=False, default_factory=dict)

    T_ = TypeVar('T_')
    """
    Type var used for handling function that return single or multiple Spans. Basically the output type
    of all try_parse functions is T_ | None
    """

    def __init__(self, input: str, name: str, ctx: MLContext, accent: str | Accent = Accent.XDSL):
        self.tokenizer = Tokenizer(Input(input, name))
        self.ctx = ctx
        if isinstance(accent, str):
            accent = MlirParser.Accent[accent]
        self.accent = accent

    def begin_parse(self):
        pass

    def must_parse_block(self) -> Block | None:
        next_id = self.expect(self.try_parse_block_id, 'Blocks must start with a block id!')

        assert next_id.text not in self._blocks

        block = Block()
        self._blocks[next_id.text] = block

        if self.tokenizer.next_token(peek=True).text == '(':
            for i, (name, type) in enumerate(self.must_parse_block_arg_list()):
                arg = BlockArgument(type, block, i)
                self._ssaValues[name.text] = arg
                block.args.append(arg)

        while (next_op := self.try_parse_op()) is not None:
            block.ops.append(next_op)

        return block

    def get_or_create_block_arg(self, name: Span, type: Attribute):
        if name.text in self._ssaValues:
            val = self._ssaValues.get(name.text)
            assert val.typ == type
            return val
        self._ssaValues[name.text] = BlockArgument(type, )

    def must_parse_block_arg_list(self) -> list[tuple[Span, Attribute]]:
        self.assert_eq(self.tokenizer.next_token(), '(', 'Block arguments must start with `(`')

        args = self.must_parse_list_of(self.try_parse_value_id_and_type, "Expected ")

        self.assert_eq(self.tokenizer.next_token(), ')', 'Expected closing of block arguments!')

        return args

    def try_parse_single_reference(self) -> Span | None:
        with self.tokenizer.backtracking():
            self.must_parse_characters('@', "references must start with `@`")
            if (reference := self.try_parse_string_literal()) is not None:
                return reference
            if (reference := self.try_parse_suffix_id()) is not None:
                return reference
            raise BacktrackingAbort("References must conform to `@` (string-literal | suffix-id)")

    def must_parse_reference(self) -> list[Span]:
        return self.must_parse_list_of(
            self.try_parse_single_reference,
            'Expected reference here in the format of `@` (suffix-id | string-literal)',
            ParserCommons.double_colon,
            allow_empty=False
        )

    def must_parse_list_of(self, try_parse: Callable[[], T_ | None], error_msg: str,
                           separator_pattern: re.Pattern = ParserCommons.comma, allow_empty: bool = True) -> list[T_]:
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

        while self.tokenizer.next_token_of_pattern(separator_pattern) is not None:
            next_item = try_parse()
            if next_item is None:
                # if the separator is emtpy, we are good here
                if separator_pattern.pattern == '':
                    return items
                self.raise_error(error_msg)
            items.append(next_item)

        return items

    def try_parse_integer_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.integer_literal)

    def try_parse_decimal_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.decimal_literal)

    def try_parse_string_literal(self) -> StringLiteral | None:
        return StringLiteral.from_span(self.tokenizer.next_token_of_pattern(ParserCommons.string_literal))

    def try_parse_float_literal(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.float_literal)

    def try_parse_bare_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.bare_id)

    def try_parse_value_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.value_id)

    def try_parse_suffix_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.suffix_id)

    def try_parse_block_id(self) -> Span | None:
        return self.tokenizer.next_token_of_pattern(ParserCommons.block_id)

    def try_parse_value_id_and_type(self) -> tuple[Span, Attribute] | None:
        with self.tokenizer.backtracking():
            value_id = self.try_parse_value_id()

            if value_id is None:
                raise BacktrackingAbort("Expected value id here!")

            self.must_parse_characters(':', 'Expected expression (value-id `:` type)')

            type = self.try_parse_type()

            if type is None:
                raise BacktrackingAbort("Expected type of value-id here!")
            return value_id, type

    def try_parse_type(self) -> Attribute | None:
        if (builtin_type := self.try_parse_builtin_type()) is not None:
            return builtin_type
        if (dialect_type := self.try_parse_dialect_type_or_attribute('type')) is not None:
            return dialect_type
        return None

    def try_parse_dialect_type_or_attribute(self, kind: Literal['type', 'attr']) -> Attribute | None:
        with self.tokenizer.backtracking():
            if kind == 'type':
                self.must_parse_characters('!', "Dialect types must start with a `!`")
            else:
                self.must_parse_characters('#', "Dialect attributes must start with a `#`")

            type_name = self.tokenizer.next_token_of_pattern(ParserCommons.bare_id)

            if type_name is None:
                raise BacktrackingAbort("Expected a type name")

            type_def = self.ctx.get_attr(type_name.text)

            # pass the task of parsing parameters on to the attribute/type definition
            param_list = type_def.parse_parameters(self)
            return type_def(param_list)

    def try_parse_builtin_type(self) -> Attribute | None:
        """
        parse a builtin-type like i32, index, vector<i32> etc.
        """
        with self.tokenizer.backtracking():
            name = self.tokenizer.next_token_of_pattern(ParserCommons.builtin_type)
            if name is None:
                raise BacktrackingAbort("Expected builtin name!")
            if name.text == 'index':
                return IndexType.build()
            if (re_match := re.match(r'^([su]?i(\d)+)$', name.text)) is not None:
                signedness = {
                    's': Signedness.SIGNED,
                    'u': Signedness.UNSIGNED,
                    'i': Signedness.SIGNLESS
                }
                return IntegerType.from_width(int(re_match.group(1)), signedness[name.text[0]])

            return self.must_parse_builtin_parametrized_type(name)

    def must_parse_builtin_parametrized_type(self, name: Span) -> ParametrizedAttribute:
        def unimplemented() -> ParametrizedAttribute:
            raise ParseError(self.tokenizer.next_token(), "Type not supported yet!")

        builtin_parsers: dict[str, Callable[[], ParametrizedAttribute]] = {
            'vector': self.must_parse_vector_attrs,
            'memref': unimplemented,
            'tensor': self.must_parse_tensor_attrs,
            'complex': self.must_parse_complex_attrs,
            'opaque': unimplemented,
            'tuple': unimplemented,
        }
        if name.text not in builtin_parsers:
            raise ParseError(name, "Unknown builtin {}".format(name.text))

        self.assert_eq(self.tokenizer.next_token(), '<', 'Expected parameter list here!')
        res = builtin_parsers[name.text]()
        self.assert_eq(self.tokenizer.next_token(), '>', 'Expected end of parameter list here!')
        return res

    def must_parse_complex_attrs(self):
        type = self.try_parse_type()
        self.raise_error("ComplexType is unimplemented!")

    def try_parse_numerical_dims(self, accept_closing_bracket: bool = False, lower_bound: int = 1) -> Iterable[int]:
        while (shape_arg := self.try_parse_shape_element(lower_bound)) is not None:
            yield shape_arg
            # look out for the closing bracket for scalable vector dims
            if accept_closing_bracket and self.tokenizer.next_token(peek=True).text == ']':
                break
            self.assert_eq(self.tokenizer.next_token(), 'x', 'Unexpected end of dimension parameters!')

    def must_parse_vector_attrs(self) -> AnyVectorType:
        # also break on 'x' characters as they are separators in dimension parameters
        with self.tokenizer.configured(break_on=self.tokenizer.break_on + ('x',)):
            shape = list[int](self.try_parse_numerical_dims())
            scaling_shape: list[int] | None = None

            if self.tokenizer.next_token(peek=True).text == '[':
                self.tokenizer.next_token()
                # we now need to parse the scalable dimensions
                scaling_shape = list(self.try_parse_numerical_dims())
                self.assert_eq(self.tokenizer.next_token(), ']', 'Expected end of scalable vector dimensions here!')
                self.assert_eq(self.tokenizer.next_token(), 'x', 'Expected end of scalable vector dimensions here!')

            if scaling_shape is not None:
                # TODO: handle scaling vectors!
                print("Warning: scaling vectors not supported!")
                pass

            type = self.try_parse_type()
            if type is None:
                self.raise_error("Expected a type at the end of the vector parameters!")

            return VectorType.from_type_and_list(type, shape)

    def must_parse_tensor_or_memref_dims(self) -> list[int] | None:
        with self.tokenizer.configured(break_on=self.tokenizer.break_on + ('x',)):
            if self.tokenizer.next_token(peek=True).text == '*':
                # consume `*`
                self.tokenizer.next_token()
                # consume `x`
                self.assert_eq(self.tokenizer.next_token(), 'x', 'Unranked tensors must follow format (`<*x` type `>`)')
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
                raise ParseError(int_lit, "Shape element literal cannot be negative or zero!")
            return value

        next_token = self.tokenizer.next_token(peek=True)

        if next_token.text == '?':
            self.tokenizer.consume_peeked(next_token)
            return -1
        return None

    def must_parse_type_params(self) -> list[parsed_type_t]:
        # consume opening bracket
        assert self.tokenizer.next_token().text == '<', 'Type must be parameterized!'

        params = self.must_parse_list_of(
            self.try_parse_type,
            'Expected a type here!'
        )

        assert self.tokenizer.next_token().text == '>', 'Expected end of type parameterization here!'

        return params

    def expect(self, try_parse: Callable[[], T_ | None], error_message: str) -> T_:
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

        # include backtracking exception if available
        if self.tokenizer.last_error:
            raise ParseError(at_position, msg) from self.tokenizer.last_error

        raise ParseError(at_position, msg)

    def assert_eq(self, got: Span, want: str, msg: str):
        if got.text == want:
            return
        raise AssertionError("Assertion failed (assert `{}` == `{}`): {}".format(got.text, want, msg), got)

    def must_parse_characters(self, text: str, msg: str):
        self.assert_eq(self.tokenizer.next_token(), text, msg)

    def try_parse_op_result_list(self) -> list[tuple[Span, Attribute] | Span] | None:
        inner_parser = (dict((
            (MlirParser.Accent.MLIR, self.try_parse_value_id),
            (MlirParser.Accent.XDSL, self.try_parse_value_id_and_type)
        )))[self.accent]

        return self.must_parse_list_of(inner_parser, 'Expected op-result here!', allow_empty=False)

    def try_parse_op(self):
        with self.tokenizer.backtracking():
            result_list = self.try_parse_op_result_list()
            self.must_parse_characters('=', 'Operation definitions expect an `=` after op-result-list!')
            name = self.try_parse_op_name()

            # handle custom-operation parsing
            if not isinstance(name, StringLiteral):
                op_type = self.ctx.get_op(name.text)
                # TODO: how do we pass result types if we are in xDSL format?
                op_type.parse()

            op_type = self.ctx.get_op(name.string_contents)

    def try_parse_region(self):
        return ParserCommons.BNF.region.try_parse(self)

    def must_parse_region(self):
        return ParserCommons.BNF.region.must_parse(self)

    def try_parse_op_name(self) -> Span | None:
        if (str_lit := self.try_parse_string_literal()) is not None:
            return str_lit
        return self.try_parse_bare_id()


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

type-alias-def ::= '!' alias-name '=' type
type-alias ::= '!' alias-name
"""
