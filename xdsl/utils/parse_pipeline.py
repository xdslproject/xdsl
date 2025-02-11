from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from xdsl.utils.exceptions import PassPipelineParseError
from xdsl.utils.lexer import Input, Span
from xdsl.utils.mlir_lexer import StringLiteral


@dataclass
class Token:
    span: Span
    kind: Kind

    class Kind(Enum):
        EOF = object()

        IDENT = object()
        L_BRACE = "{"
        R_BRACE = "}"
        EQUALS = "="
        NUMBER = object()
        SPACE = object()
        STRING_LIT = object()
        MLIR_PIPELINE = object()
        COMMA = ","


_lexer_rules: list[tuple[re.Pattern[str], Token.Kind]] = [
    # first rule is special to allow 2d-slice to be recognized as an ident
    (re.compile(r"[0-9]+[A-Za-z_-]+[A-Za-z0-9_-]*"), Token.Kind.IDENT),
    (re.compile(r"[-+]?[0-9]+(\.[0-9]*([eE][-+]?[0-9]+)?)?"), Token.Kind.NUMBER),
    (re.compile(r"[A-Za-z0-9_-]+"), Token.Kind.IDENT),
    (re.compile(r'"(\\[nfvtr"\\]|[^\n\f\v\r"\\])*"'), Token.Kind.STRING_LIT),
    (re.compile(r'\[(\\[nfvtr"\\]|[^\n\f\v\r\]\\])*\]'), Token.Kind.MLIR_PIPELINE),
    (re.compile(r"\{"), Token.Kind.L_BRACE),
    (re.compile(r"}"), Token.Kind.R_BRACE),
    (re.compile(r"="), Token.Kind.EQUALS),
    (re.compile(r"\s+"), Token.Kind.SPACE),
    (re.compile(r","), Token.Kind.COMMA),
]
"""
This is a list of lexer rules that should be tried in this specific order to get the
next token.
"""


class PipelineLexer:
    """
    This tokenizes a pass declaration string:
    pipeline          ::= pipeline-element (`,` pipeline-element)*
    pipeline-element  ::= MLIR_PIPELINE
                        | pass-name options?
    options           ::= `{` options-element ( ` ` options-element)* `}`
    options-element   ::= key (`=` value (`,` value)* )?

    key       ::= IDENT
    pass-name ::= IDENT
    value     ::= NUMBER | BOOL | IDENT | STRING_LITERAL
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

        if len(input_str) == 0:
            yield Token(Span(pos, pos + 1, input), Token.Kind.EOF)
            return

        while True:
            token: Token | None = None
            for pattern, kind in _lexer_rules:
                if (match := pattern.match(input_str, pos)) is not None:
                    token = Token(Span(match.start(), match.end(), input), kind)
                    pos = match.end()
                    break
            if token is None:
                raise PassPipelineParseError(
                    Token(Span(pos, pos + 1, input), Token.Kind.IDENT), "Unknown token"
                )
            yield token
            if pos >= end:
                yield Token(Span(pos, pos + 1, input), Token.Kind.EOF)
                return

    def lex(self) -> Token:
        token = self.peek()
        self._peeked = None
        return token

    def peek(self) -> Token:
        if self._peeked is None:
            self._peeked = next(self._stream)
        return self._peeked


PassArgElementType = str | int | bool | float
PassArgListType = tuple[PassArgElementType, ...]


def _pass_arg_element_type_str(arg: PassArgElementType) -> str:
    match arg:
        case bool():
            return str(arg).lower()
        case str():
            return f'"{arg}"'
        case int():
            return str(arg)
        case float():
            return str(arg)


def _pass_arg_list_type_str(name: str, arg: PassArgListType) -> str:
    if arg:
        return f"{name}={','.join(_pass_arg_element_type_str(val) for val in arg)}"
    else:
        return name


@dataclass(eq=True, frozen=True)
class PipelinePassSpec:
    """
    A pass name and its arguments.
    """

    name: str
    args: dict[str, PassArgListType]

    def normalize_arg_names(self) -> PipelinePassSpec:
        """
        This normalized all arg names by replacing `-` with `_`
        """
        new_args: dict[str, PassArgListType] = dict()
        for k, v in self.args.items():
            new_args[k.replace("-", "_")] = v
        return PipelinePassSpec(name=self.name, args=new_args)

    def __str__(self) -> str:
        """
        This function returns a string containing the PipelineSpec name, its arguments
        and respective values for use on the commandline.
        """
        query = f"{self.name}"
        arguments_pipeline = " ".join(
            _pass_arg_list_type_str(arg_name, arg_val)
            for arg_name, arg_val in self.args.items()
        )
        query += f"{{{arguments_pipeline}}}" if self.args else ""

        return query


def parse_pipeline(
    pipeline_spec: str,
) -> Iterator[PipelinePassSpec]:
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
        # get the pass name
        name = lexer.lex()
        if name.kind is Token.Kind.EOF:
            return
        if name.kind is not Token.Kind.IDENT:
            raise PassPipelineParseError(name, "Expected pass name here")

        # valid next tokens are EOF, COMMA or `{`
        match lexer.lex():
            case Token(kind=Token.Kind.EOF):
                # EOF means we have nothing else left to parse, we are done
                yield PipelinePassSpec(name.span.text, dict())
                return
            case Token(kind=Token.Kind.COMMA):
                # comma means we are done parsing this pass, move on to next pass
                yield PipelinePassSpec(name.span.text, dict())
                continue
            case Token(kind=Token.Kind.L_BRACE):
                # `{` indicates start of args dict, so we parse that next
                yield PipelinePassSpec(name.span.text, _parse_pass_args(lexer))
            case Token(span, Token.Kind.MLIR_PIPELINE):
                if name.span.text != "mlir-opt":
                    raise PassPipelineParseError(
                        name,
                        "Expected `mlir-opt` to mark an MLIR pipeline here",
                    )
                yield PipelinePassSpec(
                    "mlir-opt",
                    {
                        "arguments": (
                            "--mlir-print-op-generic",
                            "--allow-unregistered-dialect",
                            "-p",
                            f"builtin.module({span.text[1:-1]})",
                        )
                    },
                )
            case invalid:
                # every other token is invalid
                raise PassPipelineParseError(
                    invalid, "Expected a comma or pass arguments here"
                )

        # check for comma or EOF
        match lexer.lex():
            case Token(kind=Token.Kind.EOF):
                # EOF means we are finished parsing
                return
            case Token(kind=Token.Kind.COMMA):
                # comma means we move on to parse the next pass spec
                continue
            case invalid:
                # every other token is invalid
                raise PassPipelineParseError(
                    invalid, "Expected a comma after pass argument dict here"
                )


def _parse_pass_args(lexer: PipelineLexer) -> dict[str, PassArgListType]:
    """
    This parses pass arguments. They are a dictionary structure
    with whitespace separated, multi-value elements:

    options           ::= `{` options-element ( ` ` options-element)* `}`
    options-element   ::= key (`=` value (`,` value)* )?

    This function assumes that the leading `{` has already been consumed.
    """
    args: dict[str, PassArgListType] = dict()

    while True:
        # get the name of the argument (or a `}` in case of zero-length dicts)
        name = lexer.lex()

        # allow for zero-length arg dicts
        if name.kind is Token.Kind.R_BRACE:
            return args

        # check that it is a valid identifier
        if name.kind is not Token.Kind.IDENT:
            raise PassPipelineParseError(name, "Expected argument name here")

        # next token should be either a space, `}` or `=`
        match lexer.lex():
            case Token(kind=Token.Kind.SPACE):
                # space means zero-length argument, store empty list
                args[name.span.text] = ()
                # then continue parsing args list
                continue
            case Token(kind=Token.Kind.R_BRACE):
                # `}` means zero-length argument with no further arg
                args[name.span.text] = ()
                # stop parsing args
                return args
            case Token(kind=Token.Kind.EQUALS):
                # equals means we have an arg value given, parse it
                args[name.span.text] = _parse_arg_value(lexer)
            case invalid:
                # every other token is invalid
                raise PassPipelineParseError(
                    invalid, "Expected equals, space or end of arguments here"
                )

        # next token must be either space or `}`
        match lexer.lex():
            case Token(kind=Token.Kind.SPACE):
                # space means we get another argument
                continue
            case Token(kind=Token.Kind.R_BRACE):
                # `}` signifies end of args
                return args
            case invalid:
                # every other token is a syntax error
                raise PassPipelineParseError(
                    invalid,
                    "Malformed pass arguments, expected either a space or `}` here",
                )


def _parse_arg_value(lexer: PipelineLexer) -> PassArgListType:
    """
    Parse an argument value of the form: value (`,` value)*
    """
    elms = [_parse_arg_value_element(lexer)]
    while lexer.peek().kind is Token.Kind.COMMA:
        lexer.lex()
        elms.append(_parse_arg_value_element(lexer))
    return tuple(elms)


def _parse_arg_value_element(lexer: PipelineLexer) -> PassArgElementType:
    """
    Parse a singular value element
    """
    # valid value elements are quoted strings, numbers, true|false, and "ident" type
    # strings
    match lexer.lex():
        case Token(kind=Token.Kind.STRING_LIT, span=span):
            # string literals are converted to unescaped strings
            str_token = StringLiteral.from_span(span)
            assert str_token is not None
            return str_token.string_contents
        case Token(kind=Token.Kind.NUMBER, span=span):
            # NUMBER is both float and int
            # if the token contains a `.` it's a float
            if "." in span.text:
                return float(span.text)
            # otherwise an int
            return int(span.text)
        case Token(kind=Token.Kind.IDENT, span=span):
            # identifiers are either true|false or treated as a string
            if span.text == "true":
                return True
            elif span.text == "false":
                return False
            return span.text
        case token:
            # every other token type is invalid as a value
            raise PassPipelineParseError(
                token,
                "Unknown argument value, wrap argument in quotes to pass arbitrary "
                "string values",
            )
