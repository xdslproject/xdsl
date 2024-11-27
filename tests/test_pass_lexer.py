import pytest

from xdsl.utils.exceptions import PassPipelineParseError
from xdsl.utils.parse_pipeline import (
    PipelineLexer,
    Token,
    parse_pipeline,
)

Kind = Token.Kind

generator = PipelineLexer._generator  # pyright: ignore[reportPrivateUsage]


def test_pass_lexer():
    tokens = list(
        generator(
            'pass-1,pass-2{arg1=1 arg2=test arg3="test-str" arg-4=-34.4e-12 no-val-arg},pass-3'
        )
    )

    assert [t.kind for t in tokens] == [
        Kind.IDENT, Kind.COMMA,  # pass-1,
        Kind.IDENT, Kind.L_BRACE,  # pass-2{
        Kind.IDENT, Kind.EQUALS, Kind.NUMBER, Kind.SPACE,  # arg1=1
        Kind.IDENT, Kind.EQUALS, Kind.IDENT, Kind.SPACE,  # arg2=test
        Kind.IDENT, Kind.EQUALS, Kind.STRING_LIT, Kind.SPACE,  # arg3="test-str"
        Kind.IDENT, Kind.EQUALS, Kind.NUMBER, Kind.SPACE,  # arg-4=-34.4e-12
        Kind.IDENT,  # no-val-arg
        Kind.R_BRACE, Kind.COMMA,  # },
        Kind.IDENT,  # pass-3
        Kind.EOF,
    ]  # fmt: skip

    assert tokens[-2].span.text == "pass-3"
    assert tokens[1].span.text == ","
    assert tokens[3].span.text == "{"
    assert tokens[18].span.text == "-34.4e-12"

    assert len(list(generator(""))) == 1


def test_pass_lex_errors():
    with pytest.raises(PassPipelineParseError, match="Unknown token"):
        list(generator("pass-1["))

    with pytest.raises(PassPipelineParseError, match="Unknown token"):
        list(generator("pass-1{thing$=1}"))


@pytest.mark.parametrize(
    "input_str, pass_name, pass_arg_names",
    [
        ("pass-1,", "pass-1", set[str]()),
        ("pass-1{}", "pass-1", set[str]()),
        ("pass-1{arg1=true arg2}", "pass-1", {"arg1", "arg2"}),
        ("pass-1{arg2 arg1=false}", "pass-1", {"arg1", "arg2"}),
    ],
)
def test_pass_parser_argument_dict_edge_cases(
    input_str: str, pass_name: str, pass_arg_names: set[str]
):
    """
    This test checks edge-cases in the parsing code.
    """
    passes = list(parse_pipeline(input_str))
    assert len(passes) == 1
    assert passes[0].name == pass_name
    assert set(passes[0].args.keys()) == pass_arg_names


@pytest.mark.parametrize(
    "input_str",
    [
        ("pass-1{"),
        ("pass-1{arg1,arg2}"),
        ("pass-1{arg1=arg2=arg3}"),
        ("pass-1{ }"),
    ],
)
def test_pass_parser_cases_fail(input_str: str):
    with pytest.raises(PassPipelineParseError):
        list(parse_pipeline(input_str))


def test_pass_parse_errors():
    """
    This test triggers all parse errors in the parser in the same order they appear
    in the source file.
    """
    with pytest.raises(PassPipelineParseError, match="Expected pass name here"):
        # numbers are not valid pass names!
        list(parse_pipeline("1"))

    with pytest.raises(
        PassPipelineParseError, match="Expected a comma or pass arguments here"
    ):
        list(parse_pipeline("pass-1="))

    with pytest.raises(
        PassPipelineParseError, match="Expected a comma after pass argument dict here"
    ):
        list(parse_pipeline("pass-1{arg1=1}="))

    with pytest.raises(PassPipelineParseError, match="Expected argument name here"):
        # numbers are not valid argument names
        list(parse_pipeline("pass-1{1=1}"))

    with pytest.raises(
        PassPipelineParseError,
        match="Expected equals, space or end of arguments here",
    ):
        # we don't support the case where there is no `=` after the arg name (yet)
        list(parse_pipeline("pass-1{arg1{1}"))

    with pytest.raises(
        PassPipelineParseError,
        match="Malformed pass arguments, expected either a space or `}` here",
    ):
        list(parse_pipeline("pass-1{arg1=1=}"))

    with pytest.raises(PassPipelineParseError, match="Unknown argument value"):
        list(parse_pipeline("pass-1{arg1={}}"))
