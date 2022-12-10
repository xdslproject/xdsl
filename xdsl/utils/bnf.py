from __future__ import annotations
import functools
import re
import typing
from dataclasses import dataclass, field
from abc import abstractmethod, ABC
from typing import Any

if typing.TYPE_CHECKING:
    from xdsl.parser_ng import MlirParser, ParseError

T = typing.TypeVar('T')


@dataclass(frozen=True)
class BNFToken:
    bind: str | None = field(kw_only=True, init=False)
    debug_name: str | None = field(kw_only=True, init=False)

    @abstractmethod
    def must_parse(self, parser: MlirParser) -> T:
        raise NotImplemented()

    def try_parse(self, parser: MlirParser) -> T | None:
        with parser.tokenizer.backtracking(self.debug_name or repr(self)):
            return self.must_parse(parser)

    def collect(self, value, collection: dict) -> dict:
        if self.bind is None:
            return collection
        collection[self.bind] = value
        return collection


@dataclass(frozen=True)
class Literal(BNFToken):
    """
    Match a fixed input string
    """
    string: str
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser):
        return parser.must_parse_characters(self.string, 'Expected `{}`'.format(self.string))

    def __repr__(self):
        return '`{}`'.format(self.string)


@dataclass(frozen=True)
class Regex(BNFToken):
    pattern: re.Pattern
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def try_parse(self, parser: MlirParser) -> T | None:
        return parser.tokenizer.next_token_of_pattern(self.pattern)

    def must_parse(self, parser: MlirParser) -> T:
        res = self.try_parse(parser)
        if res is None:
            parser.raise_error('Expected token of form {}!'.format(self))
        return res

    def __repr__(self):
        return 're`{}`'.format(self.pattern.pattern)


@dataclass(frozen=True)
class Nonterminal(BNFToken):
    """
    This is used as an "escape hatch" to switch from BNF to the python parsing code.

    It will look for must_parse_<name>, or try_parse_<name> in the parse object. This can
    probably be improved, idk.
    """

    name: str
    """
    The symbol name of the nonterminal, e.g. string-lieral, tensor-attrs, etc...
    """
    bind: str | None = field(kw_only=True, default=None)

    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser):
        if hasattr(parser, 'must_parse_{}'.format(self.name.replace('-', '_'))):
            return getattr(parser, 'must_parse_{}'.format(self.name.replace('-', '_')))()
        elif hasattr(parser, 'try_parse_{}'.format(self.name.replace('-', '_'))):
            return parser.expect(
                getattr(parser, 'try_parse_{}'.format(self.name.replace('-', '_'))),
                'Expected to parse {} here!'.format(self.name)
            )
        else:
            raise NotImplementedError("Parser cannot parse {}".format(self.name))

    def try_parse(self, parser: MlirParser) -> T | None:
        if hasattr(parser, 'try_parse_{}'.format(self.name.replace('-', '_'))):
            return getattr(parser, 'try_parse_{}'.format(self.name.replace('-', '_')))()
        return super().try_parse(parser)

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Group(BNFToken):
    tokens: list[BNFToken]
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser) -> T:
        return [
            token.must_parse(parser) for token in self.tokens
        ]

    def __repr__(self):
        return '( {} )'.format(' '.join(repr(t) for t in self.tokens))

    def collect(self, value, collection: dict) -> dict:
        for child, value in zip(self.tokens, value):
            child.collect(value, collection)
        return super().collect(value, collection)


@dataclass(frozen=True)
class OneOrMoreOf(BNFToken):
    wraps: BNFToken
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser) -> list[T]:
        res = list()
        while True:
            val = self.wraps.try_parse(parser)
            if val is None:
                if len(res) == 0:
                    raise AssertionError("Expected at least one of {}".format(self.wraps))
                return res
            res.append(val)

    def __repr__(self):
        return '{}+'.format(self.wraps)

    def children(self) -> typing.Iterable[BNFToken]:
        return self.wraps,

    def collect(self, value, collection: dict) -> dict:
        for val in value:
            self.wraps.collect(val, collection)
        return super().collect(value, collection)


@dataclass(frozen=True)
class ZeroOrMoreOf(BNFToken):
    wraps: BNFToken
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser) -> list[T]:
        res = list()
        while True:
            val = self.wraps.try_parse(parser)
            if val is None:
                return res
            res.append(val)

    def __repr__(self):
        return '{}*'.format(self.wraps)

    def children(self) -> typing.Iterable[BNFToken]:
        return self.wraps,

    def collect(self, values, collection: dict) -> dict:
        for value in values:
            self.wraps.collect(value, collection)
        return super().collect(values, collection)


@dataclass(frozen=True)
class ListOf(BNFToken):
    element: BNFToken
    separator: re.Pattern = re.compile(',')

    allow_empty: bool = True
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser) -> T | None:
        return parser.must_parse_list_of(
            lambda : self.element.try_parse(parser),
            'Expected {}!'.format(self.element),
            separator_pattern=self.separator,
            allow_empty=self.allow_empty
        )

    def __repr__(self):
        if self.allow_empty:
            return '( {elm} ( re`{sep}` {elm} )* )?'.format(elm=self.element, sep=self.separator.pattern)
        return '{elm} ( re`{sep}` {elm} )*'.format(elm=self.element, sep=self.separator.pattern)

    def collect(self, values, collection: dict) -> dict:
        for value in values:
            self.element.collect(value, collection)
        return super().collect(values, collection)


@dataclass(frozen=True)
class Optional(BNFToken):
    wraps: BNFToken
    bind: str | None = field(kw_only=True, default=None)
    debug_name: str | None = field(kw_only=True, default=None)

    def must_parse(self, parser: MlirParser) -> T | None:
        return self.wraps.try_parse(parser)

    def try_parse(self, parser: MlirParser) -> T | None:
        return self.wraps.try_parse(parser)

    def __repr__(self):
        return '{}?'.format(self.wraps)

    def collect(self, value, collection: dict) -> dict:
        if value is not None:
            self.wraps.collect(value, collection)
        return super().collect(value, collection)


def OptionalGroup(tokens: list[BNFToken], bind: str | None = None, debug_name: str | None = None) -> Optional:
    return Optional(Group(tokens), bind=bind, debug_name=debug_name)
