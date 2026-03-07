import pytest

from xdsl.utils.arg_spec import (
    ArgSpec,
    parse_pipeline,
    parse_spec,
)
from xdsl.utils.exceptions import ArgSpecParseError


def test_pass_parser():
    passes = list(
        parse_pipeline(
            'mlir-opt[cse],pass-1,pass-2{arg1=1 arg2=test,test2,3 arg3="test-str,2,3" '
            "arg-4=-34.4e-12 no-val-arg},mlir-opt[cse],pass-3{thing=2d-grid},mlir-opt[cse]"
        )
    )

    assert passes == [
        ArgSpec(
            "mlir-opt",
            {
                "arguments": (
                    "--mlir-print-op-generic",
                    "--allow-unregistered-dialect",
                    "-p",
                    "builtin.module(cse)",
                )
            },
        ),
        ArgSpec("pass-1", {}),
        ArgSpec(
            "pass-2",
            {
                "arg1": (1,),
                "arg2": ("test", "test2", 3),
                "arg3": ("test-str,2,3",),
                "arg-4": (-3.44e-11,),
                "no-val-arg": (),
            },
        ),
        ArgSpec(
            "mlir-opt",
            {
                "arguments": (
                    "--mlir-print-op-generic",
                    "--allow-unregistered-dialect",
                    "-p",
                    "builtin.module(cse)",
                )
            },
        ),
        ArgSpec(
            "pass-3",
            {
                "thing": ("2d-grid",),
            },
        ),
        ArgSpec(
            "mlir-opt",
            {
                "arguments": (
                    "--mlir-print-op-generic",
                    "--allow-unregistered-dialect",
                    "-p",
                    "builtin.module(cse)",
                )
            },
        ),
    ]


@pytest.mark.parametrize(
    "spec",
    [
        ArgSpec("empty", {}),
        ArgSpec(
            "pass-2",
            {
                "arg1": (1,),
                "arg2": (
                    "test",
                    "test2",
                    3,
                ),
                "arg3": ("test-str,2,3",),
                "arg-4": (-3.44e-11,),
                "no-val-arg": (),
            },
        ),
        ArgSpec(
            "pass-3",
            {
                "thing": ("2d-grid",),
            },
        ),
    ],
)
def test_spec_printer(spec: ArgSpec):
    text = str(spec)
    # test pipeline parser
    passes = list(parse_pipeline(text))
    assert len(passes) == 1
    assert passes[0] == spec
    # test individual parser
    p = parse_spec(text)
    assert p == spec


def test_invalid_mlir_pipeline():
    with pytest.raises(
        ArgSpecParseError,
        match="Expected `mlir-opt` to mark an MLIR pipeline here",
    ):
        list(parse_pipeline("canonicalize[cse]"))


def test_deprecated_args():
    spec = ArgSpec("name", {"hello": ("world",)})
    with pytest.deprecated_call():
        assert spec.args == {"hello": ("world",)}  # pyright: ignore[reportDeprecated]
