import pytest

from xdsl.utils.parse_pipeline import (
    tokenize_pipeline,
    Kind,
    PassPipelineParseError,
    parse_pipeline,
)


def test_pass_lexer():
    tokens = list(
        tokenize_pipeline(
            'pass-1,pass-2{arg1=1 arg2=test arg3="test-str" arg-4=-34.4e-12},pass-3'
        )
    )

    assert [t.kind for t in tokens] == [
        Kind.IDENT, Kind.COMMA,  # pass-1,
        Kind.IDENT, Kind.L_BRACE,  # pass-2{
        Kind.IDENT, Kind.EQUALS, Kind.NUMBER, Kind.SPACE,  # arg1=1
        Kind.IDENT, Kind.EQUALS, Kind.IDENT, Kind.SPACE,  # arg2=test
        Kind.IDENT, Kind.EQUALS, Kind.STRING_LIT, Kind.SPACE,  # arg3="test-str"
        Kind.IDENT, Kind.EQUALS, Kind.NUMBER,  # arg-4=-34.4e-12
        Kind.R_BRACE, Kind.COMMA,  # },
        Kind.IDENT,  # pass-3
        Kind.EOF,
    ]  # fmt: skip

    assert tokens[-2].span.text == "pass-3"
    assert tokens[1].span.text == ","
    assert tokens[3].span.text == "{"
    assert tokens[18].span.text == "-34.4e-12"


def test_pass_lex_errors():
    with pytest.raises(PassPipelineParseError, match="Unknown token"):
        list(tokenize_pipeline("pass-1["))

    with pytest.raises(PassPipelineParseError, match="Unknown token"):
        list(tokenize_pipeline("pass-1{thing$=1}"))


def test_pass_parser():
    passes = list(
        parse_pipeline(
            'pass-1,pass-2{arg1=1 arg2=test arg3="test-str" arg-4=-34.4e-12},pass-3'
        )
    )

    assert passes == [
        ("pass-1", {}),
        (
            "pass-2",
            {
                "arg1": 1,
                "arg2": "test",
                "arg3": "test-str",
                "arg-4": -3.44e-11,
            },
        ),
        ("pass-3", {}),
    ]


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
        # numbers are not valud argument names
        list(parse_pipeline("pass-1{1=1}"))

    with pytest.raises(
        PassPipelineParseError,
        match="Expected equals as part of the pass argument here",
    ):
        # we don't support the case where there is no `=` after the arg name (yet)
        list(parse_pipeline("pass-1{arg1 1}"))

    with pytest.raises(
        PassPipelineParseError,
        match="Malformed pass arguments, expected either a space or `}` here",
    ):
        list(parse_pipeline("pass-1{arg1=1=}"))

    with pytest.raises(PassPipelineParseError, match="Unknown argument value"):
        list(parse_pipeline("pass-1{arg1={}}"))
