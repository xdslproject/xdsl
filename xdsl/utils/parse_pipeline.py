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


def tokenize_pipeline(input_str: str) -> Iterator[Token]:
    """
    This tokenizes a pass declaration string. Pass syntax is a subset
    of MLIRs pass pipeline syntax:

    pipeline          ::= pipeline-element (`,` pipeline-element)*
    pipeline-element  ::= pass-name options?
    options           ::= `{` options-element ( ` ` options-element)* `}`
    options-element   ::= key `=` value

    key       ::= IDENT
    pass-name ::= IDENT
    value     :== NUMBER / IDENT / STRING_LITERAL
    """
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


class PassPipelineParseError(BaseException):
    def __init__(self, token: Token, msg: str):
        super().__init__(
            "Error parsing pass pipeline specification:\n"
            + token.span.print_with_context(msg)
        )


_PassArgTypes = str | int | float


def parse_pipeline(
    pipeline_spec: str,
) -> Iterator[tuple[str, dict[str, _PassArgTypes]]]:
    tokens = tokenize_pipeline(pipeline_spec)

    while True:
        name = next(tokens)
        if name.kind is Kind.EOF:
            return
        if name.kind is not Kind.IDENT:
            raise PassPipelineParseError(name, "Expected pass name here")

        delim = next(tokens)
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

        yield name.span.text, _parse_pass_args(tokens)

        delim = next(tokens)
        if delim.kind is Kind.EOF:
            return
        if delim.kind is not Kind.COMMA:
            raise PassPipelineParseError(
                delim, "Expected a comma after pass argument dict here"
            )


def _parse_pass_args(tokens: Iterator[Token]):
    args: dict[str, _PassArgTypes] = dict()

    while True:
        name = next(tokens)

        # allow for zero-length arg dicts
        if name.kind is Kind.R_BRACE:
            return args

        if name.kind is not Kind.IDENT:
            raise PassPipelineParseError(name, "Expected argument name here")

        # consume the `=` in between
        _expect(
            tokens, Kind.EQUALS, "Expected equals as part of the pass argument here"
        )

        args[name.span.text] = _parse_pass_arg_val(next(tokens))

        delim = next(tokens)
        if delim.kind is Kind.SPACE:
            continue
        if delim.kind is not Kind.R_BRACE:
            raise PassPipelineParseError(
                delim, "Malformed pass arguments, expected either a space or `}` here"
            )

        return args


def _expect(tokens: Iterator[Token], kind: Kind, err: str):
    token = next(tokens)
    if token.kind is not kind:
        raise PassPipelineParseError(token, err)


def _parse_pass_arg_val(token: Token) -> _PassArgTypes:
    match token:
        case Token(kind=Kind.STRING_LIT, span=span):
            str_token = StringLiteral.from_span(span)
            assert str_token is not None
            return str_token.string_contents
        case Token(kind=Kind.NUMBER, span=span):
            if "." in span.text:
                return float(span.text)
            return int(span.text)
        case Token(kind=Kind.IDENT, span=span):
            return span.text
        case token:
            raise PassPipelineParseError(
                token,
                "Unknown argument value, wrap argument in quotes to pass arbitrary string values",
            )
