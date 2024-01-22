import pytest

from xdsl.utils.parse_pipeline import (
    PipelinePassSpec,
    parse_pipeline,
)


def test_pass_parser():
    passes = list(
        parse_pipeline(
            'pass-1,pass-2{arg1=1 arg2=test,test2,3 arg3="test-str,2,3" '
            "arg-4=-34.4e-12 no-val-arg},pass-3{thing=2d-grid}"
        )
    )

    assert passes == [
        PipelinePassSpec("pass-1", {}),
        PipelinePassSpec(
            "pass-2",
            {
                "arg1": [1],
                "arg2": ["test", "test2", 3],
                "arg3": ["test-str,2,3"],
                "arg-4": [-3.44e-11],
                "no-val-arg": [],
            },
        ),
        PipelinePassSpec(
            "pass-3",
            {
                "thing": ["2d-grid"],
            },
        ),
    ]


@pytest.mark.parametrize(
    "spec",
    [
        PipelinePassSpec("empty", {}),
        PipelinePassSpec(
            "pass-2",
            {
                "arg1": [1],
                "arg2": ["test", "test2", 3],
                "arg3": ["test-str,2,3"],
                "arg-4": [-3.44e-11],
                "no-val-arg": [],
            },
        ),
        PipelinePassSpec(
            "pass-3",
            {
                "thing": ["2d-grid"],
            },
        ),
    ],
)
def test_spec_printer(spec: PipelinePassSpec):
    text = str(spec)
    passes = list(parse_pipeline(text))
    assert len(passes) == 1
    assert passes[0] == spec
