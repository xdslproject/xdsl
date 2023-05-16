import re
from dataclasses import dataclass
from typing import Iterator
from xdsl.utils.lexer import Input, Span
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


def tokenize_pass(input_str: str) -> Iterator[Token]:
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
