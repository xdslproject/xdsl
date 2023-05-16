import re
from dataclasses import dataclass
from typing import Iterator
from xdsl.utils.lexer import Input, Span, StringLiteral
from enum import Enum, auto


class Kind(Enum):
    EOF = auto()

    IDENT = auto()
    L_BRACE = auto()
    R_BRACE = auto()
    EQUALS = auto()
    NUMBER = auto()
    SPACE = auto()
    STRING_LIT = auto()
    COMMA = auto()


@dataclass
class Token:
    span: Span
    kind: Kind


_lexer_rules: list[tuple[re.Pattern[str], Kind]] = [
    (re.compile(r"[-+]?[0-9]+(\.[0-9]*([eE][-+]?[0-9]+)?)?"), Kind.NUMBER),
    (re.compile(r"[A-Za-z0-9_-]+"), Kind.IDENT),
    (re.compile(r'"(\\[nfvtr"\\]|[^\n\f\v\r"\\])*"'), Kind.STRING_LIT),
    (re.compile(r"\{"), Kind.L_BRACE),
    (re.compile(r"}"), Kind.R_BRACE),
    (re.compile(r"="), Kind.EQUALS),
    (re.compile(r"\s+"), Kind.SPACE),
    (re.compile(r","), Kind.COMMA),
]
"""
This is a list of lexer rules that should be tried in this specific order to get the next token.
"""


class PipelineLexer:
    """
    This tokenizes a pass declaration string. Pass syntax is a subset
    of MLIRs pass pipeline syntax:

    pipeline          ::= pipeline-element (`,` pipeline-element)*
    pipeline-element  ::= pass-name options?
    options           ::= `{` options-element ( ` ` options-element)* `}`
    options-element   ::= key (`=` value (`,` value)* )?

    key       ::= IDENT
    pass-name ::= IDENT
    value     :== NUMBER / BOOL / IDENT / STRING_LITERAL
    """

    _stream: Iterator[Token]
    _peeked: Token | None

    def __init__(self, input_str: str):
        self._stream = PipelineLexer._generator(input_str)
        self._peeked = None

    @staticmethod
    def _generator(input_str: str) -> Iterator[Token]:
        input = Input(input_str, "pass-pipeline")
        pos = 0
        end = len(input_str)

        while True:
            token: Token | None = None
            for pattern, kind in _lexer_rules:
                if (match := pattern.match(input_str, pos)) is not None:
                    token = Token(Span(match.start(), match.end(), input), kind)
                    pos = match.end()
                    break
            if token is None:
                raise PassPipelineParseError(
                    Token(Span(pos, pos + 1, input), Kind.IDENT), "Unknown token"
                )
            yield token
            if pos >= end:
                yield Token(Span(pos, pos + 1, input), Kind.EOF)
                return

    def lex(self) -> Token:
        token = self.peek()
        self._peeked = None
        return token

    def peek(self) -> Token:
        if self._peeked is None:
            self._peeked = next(self._stream)
        return self._peeked


class PassPipelineParseError(BaseException):
    def __init__(self, token: Token, msg: str):
        super().__init__(
            "Error parsing pass pipeline specification:\n"
            + token.span.print_with_context(msg)
        )


_PassArgTypes = list[str | int | bool | float]


def parse_pipeline(
    pipeline_spec: str,
) -> Iterator[tuple[str, dict[str, _PassArgTypes]]]:
    """
    This takes a pipeline string and gives a representation of
    the specification.

    Each pass is represented by a tuple of:
     - name: the name of the pass as string
     - args: a dictionary, where each value is zero or more
            of (str | bool | float | int)
    """
    lexer = PipelineLexer(pipeline_spec)

    while True:
        name = lexer.lex()
        if name.kind is Kind.EOF:
            return
        if name.kind is not Kind.IDENT:
            raise PassPipelineParseError(name, "Expected pass name here")

        delim = lexer.lex()
        if delim.kind is Kind.EOF:
            yield name.span.text, dict()
            return

        if delim.kind is Kind.COMMA:
            yield name.span.text, dict()
            continue

        if delim.kind is not Kind.L_BRACE:
            raise PassPipelineParseError(
                delim, "Expected a comma or pass arguments here"
            )

        yield name.span.text, _parse_pass_args(lexer)

        delim = lexer.lex()
        if delim.kind is Kind.EOF:
            return
        if delim.kind is not Kind.COMMA:
            raise PassPipelineParseError(
                delim, "Expected a comma after pass argument dict here"
            )


def _parse_pass_args(lexer: PipelineLexer):
    args: dict[str, _PassArgTypes] = dict()

    while True:
        name = lexer.lex()

        # allow for zero-length arg dicts
        if name.kind is Kind.R_BRACE:
            return args

        if name.kind is not Kind.IDENT:
            raise PassPipelineParseError(name, "Expected argument name here")

        # handle zero-length args
        if lexer.peek().kind in (Kind.SPACE, Kind.R_BRACE):
            args[name.span.text] = []
            continue

        # consume the `=` in between
        equals = lexer.lex()
        if equals.kind is not Kind.EQUALS:
            raise PassPipelineParseError(
                lexer.lex(), "Expected equals as part of the pass argument here"
            )

        args[name.span.text] = _parse_arg_value(lexer)

        delim = lexer.lex()
        if delim.kind is Kind.SPACE:
            continue
        if delim.kind is not Kind.R_BRACE:
            raise PassPipelineParseError(
                delim, "Malformed pass arguments, expected either a space or `}` here"
            )

        return args


def _parse_arg_value(lexer: PipelineLexer) -> _PassArgTypes:
    """
    Parse an argument value of the form: value (`,` value)*
    """
    elms = [_parse_arg_value_element(lexer)]
    while lexer.peek().kind is Kind.COMMA:
        lexer.lex()
        elms.append(_parse_arg_value_element(lexer))
    return elms


def _parse_arg_value_element(lexer: PipelineLexer) -> str | int | bool | float:
    """
    parse a singular value element

    """
    match lexer.lex():
        case Token(kind=Kind.STRING_LIT, span=span):
            str_token = StringLiteral.from_span(span)
            assert str_token is not None
            return str_token.string_contents
        case Token(kind=Kind.NUMBER, span=span):
            if "." in span.text:
                return float(span.text)
            return int(span.text)
        case Token(kind=Kind.IDENT, span=span):
            if span.text == "true":
                return True
            elif span.text == "false":
                return False
            return span.text
        case token:
            raise PassPipelineParseError(
                token,
                "Unknown argument value, wrap argument in quotes to pass arbitrary string values",
            )
