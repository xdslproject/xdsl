"""
When passing command-line arguments to xdsl-opt, it may be useful to parametrize them.
The parametrized argument model object `ArgSpec` holds the name of the argument, and a
mapping from a key to a tuple of parameters, described below.

This is used when building pass pipelines.
"""

from __future__ import annotations

import dataclasses
import re
from abc import ABC
from collections.abc import Iterator
from dataclasses import Field, dataclass
from enum import Enum
from types import NoneType, UnionType
from typing import Any, ClassVar, TypeAlias, Union, get_args, get_origin, get_type_hints

from typing_extensions import Self

from xdsl.utils.exceptions import ArgSpecPipelineParseError
from xdsl.utils.lexer import Input, Span, Token
from xdsl.utils.mlir_lexer import StringLiteral

ArgType = str | int | bool | float
"""
The only types that can be used as `ArgSpec` parameters.
"""
ArgListType = tuple[ArgType, ...]
"""
The `ArgSpec` holds a dictionary from strings to lists of parameters.
"""


@dataclass(eq=True, frozen=True)
class ArgSpec:
    """
    A pass name and its arguments.
    """

    name: str
    args: dict[str, ArgListType]

    def normalize_arg_names(self) -> ArgSpec:
        """
        This normalized all arg names by replacing `-` with `_`
        """
        new_args: dict[str, ArgListType] = dict()
        for k, v in self.args.items():
            new_args[k.replace("-", "_")] = v
        return ArgSpec(name=self.name, args=new_args)

    @staticmethod
    def _spec_parameter_type_str(arg: ArgType) -> str:
        match arg:
            case bool():
                return str(arg).lower()
            case str():
                return f'"{arg}"'
            case int():
                return str(arg)
            case float():
                return str(arg)

    @staticmethod
    def _spec_parameter_list_type_str(name: str, arg: ArgListType) -> str:
        if arg:
            return f"{name}={','.join(ArgSpec._spec_parameter_type_str(val) for val in arg)}"
        else:
            return name

    def __str__(self) -> str:
        """
        This function returns a string containing the PipelineSpec name, its arguments
        and respective values for use on the commandline.
        """
        query = f"{self.name}"
        arguments_pipeline = " ".join(
            ArgSpec._spec_parameter_list_type_str(arg_name, arg_val)
            for arg_name, arg_val in self.args.items()
        )
        query += f"{{{arguments_pipeline}}}" if self.args else ""

        return query


def _convert_arg_to_type(
    value: ArgListType, dest_type: Any
) -> ArgListType | ArgType | None:
    """
    Takes in a list of args and converts them to the required type.

    value,      dest_type,      result
    []          int | None      None
    [1]         int | None      1
    [1]         tuple[int, ...] (1,)
    [1,2]       tuple[int, ...] (1,2)
    [1,2]       int | None      Error
    []          int             Error

    And so on
    """
    from xdsl.utils.hints import isa

    origin = get_origin(dest_type)

    # we need to special case optionals as [] means no option given
    if origin in [Union, UnionType]:
        if len(value) == 0:
            if NoneType in get_args(dest_type):
                return None
            else:
                raise ValueError("Argument must contain a value")

    # first check if an individual value passes the type check
    if len(value) == 1 and isa(value[0], dest_type):
        return value[0]

    # then check if n-tuple value is okay
    if isa(value, dest_type):
        return value

    # at this point we exhausted all possibilities
    raise ValueError(f"Incompatible types: given {value}, expected {dest_type}")


def _is_optional(field: Field[Any]) -> bool:
    """
    Shorthand to check if the given type allows "None" as a value,
    or has a default value or factory.
    """
    can_be_none = get_origin(field.type) in [Union, UnionType] and NoneType in get_args(
        field.type
    )
    has_default_val = field.default is not dataclasses.MISSING
    has_default_factory = field.default_factory is not dataclasses.MISSING

    return can_be_none or has_default_val or has_default_factory


def _get_default(field: Field[Any]) -> Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None


@dataclass(frozen=True)
class ArgSpecConvertible(ABC):
    """
    A base class for frozen dataclasses with a ``name: ClassVar[str]`` that can
    be instantiated from an ``ArgSpec`` and serialized back to one.

    Subclasses must be decorated with ``@dataclass(frozen=True)``.

    Only the following types are supported as argument types:

    Base types:                int | float | bool | string
    N-tuples of base types:
        tuple[int, ...], tuple[int|float, ...], tuple[int, ...] | tuple[float, ...]
    Top-level optional:        ... | None

    Arguments are formatted as follows::

        Spec arg                            Mapped to class field
        -------------------------           ------------------------------
        my-thing{arg-1=1}                   arg_1: int             = 1
        my-thing{arg-1}                     arg_1: int | None      = None
        my-thing{arg-1=1,2,3}              arg_1: tuple[int, ...] = (1, 2, 3)
        my-thing{arg-1=true}               arg_1: bool | None     = True
    """

    name: ClassVar[str]

    @classmethod
    def from_spec(cls, spec: ArgSpec) -> Self:
        """
        Takes an ArgSpec, does type checking on the arguments, and instantiates
        an instance of this class from the spec.
        """
        if spec.name != cls.name:
            raise ValueError(f"Cannot create {cls.name} from spec for {spec.name}")

        spec_arguments_dict: dict[str, ArgListType] = spec.normalize_arg_names().args

        fields: tuple[Field[Any], ...] = dataclasses.fields(cls)

        arg_dict = dict[str, ArgListType | ArgType | None]()

        required = cls.required_fields()

        field_types = get_type_hints(cls)

        for op_field in fields:
            if op_field.name == "name" or not op_field.init:
                continue
            if op_field.name not in spec_arguments_dict:
                if op_field.name not in required:
                    arg_dict[op_field.name] = _get_default(op_field)
                    continue
                raise ValueError(f'{cls.name} requires argument "{op_field.name}"')

            field_type = field_types[op_field.name]
            arg_dict[op_field.name] = _convert_arg_to_type(
                spec_arguments_dict.pop(op_field.name),
                field_type,
            )

        if len(spec_arguments_dict) != 0:
            arguments_str = ", ".join(f'"{arg}"' for arg in spec_arguments_dict)
            fields_str = ", ".join(f'"{field.name}"' for field in fields)
            raise ValueError(
                f"Provided arguments [{arguments_str}] not found in expected "
                f"arguments [{fields_str}]"
            )

        return cls(**arg_dict)

    @classmethod
    def required_fields(cls) -> set[str]:
        """
        Inspects the definition for fields that do not have default values.
        """
        return {
            field.name for field in dataclasses.fields(cls) if not _is_optional(field)
        }

    def spec(self, *, include_default: bool = False) -> ArgSpec:
        """
        Returns an ArgSpec representation of this instance.

        If ``include_default`` is ``True``, then optional arguments with default
        values are also included in the spec.
        """
        fields = dataclasses.fields(self)
        args: dict[str, ArgListType] = {}

        for op_field in fields:
            name = op_field.name
            if name == "name" or not op_field.init:
                continue

            val = getattr(self, name)

            if _is_optional(op_field):
                if val == _get_default(op_field) and not include_default:
                    continue

            if val is None:
                arg_list = ()
            elif isinstance(val, ArgType):
                arg_list = (val,)
            else:
                arg_list = val

            args[name] = arg_list
        return ArgSpec(self.name, args)

    def __str__(self) -> str:
        return str(self.spec())


class SpecTokenKind(Enum):
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


SpecToken: TypeAlias = Token[SpecTokenKind]

_lexer_rules: list[tuple[re.Pattern[str], SpecTokenKind]] = [
    # first rule is special to allow 2d-slice to be recognized as an ident
    (re.compile(r"[0-9]+[A-Za-z_-]+[A-Za-z0-9_-]*"), SpecTokenKind.IDENT),
    (re.compile(r"[-+]?[0-9]+(\.[0-9]*([eE][-+]?[0-9]+)?)?"), SpecTokenKind.NUMBER),
    (re.compile(r"[A-Za-z0-9_-]+"), SpecTokenKind.IDENT),
    (re.compile(r'"(\\[nfvtr"\\]|[^\n\f\v\r"\\])*"'), SpecTokenKind.STRING_LIT),
    (re.compile(r'\[(\\[nfvtr"\\]|[^\n\f\v\r\]\\])*\]'), SpecTokenKind.MLIR_PIPELINE),
    (re.compile(r"\{"), SpecTokenKind.L_BRACE),
    (re.compile(r"}"), SpecTokenKind.R_BRACE),
    (re.compile(r"="), SpecTokenKind.EQUALS),
    (re.compile(r"\s+"), SpecTokenKind.SPACE),
    (re.compile(r","), SpecTokenKind.COMMA),
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

    _stream: Iterator[SpecToken]
    _peeked: SpecToken | None

    def __init__(self, input_str: str):
        self._stream = PipelineLexer._generator(input_str)
        self._peeked = None

    @staticmethod
    def _generator(input_str: str) -> Iterator[SpecToken]:
        input = Input(input_str, "pass-pipeline")
        pos = 0
        end = len(input_str)

        if len(input_str) == 0:
            yield SpecToken(SpecTokenKind.EOF, Span(pos, pos + 1, input))
            return

        while True:
            token: SpecToken | None = None
            for pattern, kind in _lexer_rules:
                if (match := pattern.match(input_str, pos)) is not None:
                    token = SpecToken(kind, Span(match.start(), match.end(), input))
                    pos = match.end()
                    break
            if token is None:
                raise ArgSpecPipelineParseError(
                    SpecToken(SpecTokenKind.IDENT, Span(pos, pos + 1, input)),
                    "Unknown token",
                )
            yield token
            if pos >= end:
                yield SpecToken(SpecTokenKind.EOF, Span(pos, pos + 1, input))
                return

    def lex(self) -> SpecToken:
        token = self.peek()
        self._peeked = None
        return token

    def peek(self) -> SpecToken:
        if self._peeked is None:
            self._peeked = next(self._stream)
        return self._peeked


def parse_pipeline(
    pipeline_spec: str,
) -> Iterator[ArgSpec]:
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
        if lexer.peek().kind is SpecTokenKind.EOF:
            return

        yield _parse_spec(lexer)

        # check for comma or EOF
        match lexer.lex():
            case Token(kind=SpecTokenKind.EOF):
                # EOF means we are finished parsing
                return
            case Token(kind=SpecTokenKind.COMMA):
                # comma means we move on to parse the next pass spec
                continue
            case invalid:
                # every other token is invalid
                raise ArgSpecPipelineParseError(
                    invalid, "Expected a comma after pass argument dict here"
                )


def parse_spec(spec: str) -> ArgSpec:
    """
    Parses a pass, with optional arguments, or raises a `PassPipelineParseError` if one
    cannot be parsed.
    """
    lexer = PipelineLexer(spec)
    return _parse_spec(lexer)


def _parse_spec(lexer: PipelineLexer) -> ArgSpec:
    """
    Parses a pass, with optional arguments, or raises a `PassPipelineParseError` if one
    cannot be parsed.
    """
    # get the pass name
    name = lexer.lex()
    if name.kind is not SpecTokenKind.IDENT:
        raise ArgSpecPipelineParseError(name, "Expected pass name here")

    # valid next tokens are EOF, COMMA or `{`
    match lexer.peek():
        case Token(kind=SpecTokenKind.EOF):
            # EOF means we have nothing else left to parse, we are done
            return ArgSpec(name.span.text, dict())
        case Token(kind=SpecTokenKind.COMMA):
            # comma means we are done parsing this pass, move on to next pass
            return ArgSpec(name.span.text, dict())
        case Token(kind=SpecTokenKind.L_BRACE):
            # `{` indicates start of args dict, so we parse that next
            lexer.lex()
            return ArgSpec(name.span.text, _parse_pass_args(lexer))
        case Token(SpecTokenKind.MLIR_PIPELINE, span):
            if name.span.text != "mlir-opt":
                raise ArgSpecPipelineParseError(
                    name,
                    "Expected `mlir-opt` to mark an MLIR pipeline here",
                )
            lexer.lex()
            return ArgSpec(
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
            raise ArgSpecPipelineParseError(
                invalid, "Expected a comma or pass arguments here"
            )


def _parse_pass_args(lexer: PipelineLexer) -> dict[str, ArgListType]:
    """
    This parses pass arguments. They are a dictionary structure
    with whitespace separated, multi-value elements:

    options           ::= `{` options-element ( ` ` options-element)* `}`
    options-element   ::= key (`=` value (`,` value)* )?

    This function assumes that the leading `{` has already been consumed.
    """
    args: dict[str, ArgListType] = dict()

    while True:
        # get the name of the argument (or a `}` in case of zero-length dicts)
        name = lexer.lex()

        # allow for zero-length arg dicts
        if name.kind is SpecTokenKind.R_BRACE:
            return args

        # check that it is a valid identifier
        if name.kind is not SpecTokenKind.IDENT:
            raise ArgSpecPipelineParseError(name, "Expected argument name here")

        # next token should be either a space, `}` or `=`
        match lexer.lex():
            case Token(kind=SpecTokenKind.SPACE):
                # space means zero-length argument, store empty list
                args[name.span.text] = ()
                # then continue parsing args list
                continue
            case Token(kind=SpecTokenKind.R_BRACE):
                # `}` means zero-length argument with no further arg
                args[name.span.text] = ()
                # stop parsing args
                return args
            case Token(kind=SpecTokenKind.EQUALS):
                # equals means we have an arg value given, parse it
                args[name.span.text] = _parse_arg_value(lexer)
            case invalid:
                # every other token is invalid
                raise ArgSpecPipelineParseError(
                    invalid, "Expected equals, space or end of arguments here"
                )

        # next token must be either space or `}`
        match lexer.lex():
            case Token(kind=SpecTokenKind.SPACE):
                # space means we get another argument
                continue
            case Token(kind=SpecTokenKind.R_BRACE):
                # `}` signifies end of args
                return args
            case invalid:
                # every other token is a syntax error
                raise ArgSpecPipelineParseError(
                    invalid,
                    "Malformed pass arguments, expected either a space or `}` here",
                )


def _parse_arg_value(lexer: PipelineLexer) -> ArgListType:
    """
    Parse an argument value of the form: value (`,` value)*
    """
    elms = [_parse_arg_value_element(lexer)]
    while lexer.peek().kind is SpecTokenKind.COMMA:
        lexer.lex()
        elms.append(_parse_arg_value_element(lexer))
    return tuple(elms)


def _parse_arg_value_element(lexer: PipelineLexer) -> ArgType:
    """
    Parse a singular value element
    """
    # valid value elements are quoted strings, numbers, true|false, and "ident" type
    # strings
    match lexer.lex():
        case Token(kind=SpecTokenKind.STRING_LIT, span=span):
            # string literals are converted to unescaped strings
            str_token = StringLiteral.from_span(span)
            assert str_token is not None
            return str_token.string_contents
        case Token(kind=SpecTokenKind.NUMBER, span=span):
            # NUMBER is both float and int
            # if the token contains a `.` it's a float
            if "." in span.text:
                return float(span.text)
            # otherwise an int
            return int(span.text)
        case Token(kind=SpecTokenKind.IDENT, span=span):
            # identifiers are either true|false or treated as a string
            if span.text == "true":
                return True
            elif span.text == "false":
                return False
            return span.text
        case token:
            # every other token type is invalid as a value
            raise ArgSpecPipelineParseError(
                token,
                "Unknown argument value, wrap argument in quotes to pass arbitrary "
                "string values",
            )
