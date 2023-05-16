import pytest

from xdsl.utils.parse_pipeline import (
    PipelineLexer,
    Kind,
    PassPipelineParseError,
)


def test_pass_lexer():
    tokens = list(
        PipelineLexer._generator(  # type: ignore[reportPrivateUsage]
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
        Kind.IDENT, # no-val-arg
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
        list(PipelineLexer._generator("pass-1["))  # type: ignore[reportPrivateUsage]

    with pytest.raises(PassPipelineParseError, match="Unknown token"):
        list(PipelineLexer._generator("pass-1{thing$=1}"))  # type: ignore[reportPrivateUsage]
