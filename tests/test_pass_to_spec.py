from dataclasses import dataclass, field

import pytest

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.utils.parse_pipeline import PipelinePassSpec


@dataclass
class CustomPass(ModulePass):
    name = "custom-pass"

    number: int

    int_list: list[int]

    str_thing: str | None

    list_str: list[str] = field(default_factory=list)

    optional_bool: bool = False

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@dataclass
class EmptyPass(ModulePass):
    name = "empty"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@dataclass
class SimplePass(ModulePass):
    name = "simple"

    a: list[float]
    b: int

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@pytest.mark.parametrize(
    "test_pass, test_spec",
    (
        (
            CustomPass(3, [1, 2], None, ["clown", "season"]),
            PipelinePassSpec(
                "custom-pass",
                {
                    "number": [3],
                    "int_list": [1, 2],
                    "str_thing": [],
                    "list_str": ["clown", "season"],
                    "optional_bool": [False],
                },
            ),
        ),
        (EmptyPass(), PipelinePassSpec("empty", {})),
        (
            SimplePass([3.14, 2.13], 2),
            PipelinePassSpec("simple", {"a": [3.14, 2.13], "b": [2]}),
        ),
    ),
)
def test_pass_to_spec_equality(test_pass: ModulePass, test_spec: PipelinePassSpec):
    assert test_pass.pipeline_pass_spec() == test_spec
