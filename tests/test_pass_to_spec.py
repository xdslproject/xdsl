import dataclasses
from dataclasses import dataclass
from typing import Literal

import pytest

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.utils.parse_pipeline import PipelinePassSpec


@dataclass
class CustomPass(ModulePass):
    name = "custom-pass"

    number: int | float

    int_list: list[int]

    str_thing: str

    nullable_str: str | None

    literal: Literal["yes", "no", "maybe"] = "no"

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

    a: int | float
    b: int

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@pytest.mark.parametrize(
    "test_pass, test_spec",
    (
        (
            CustomPass(3, [1, 2], "hi", "bye"),
            PipelinePassSpec(
                "custom-pass",
                {
                    "number": [3],
                    "int_list": [1, 2],
                    "str_thing": ["hi"],
                    "nullable_str": ["bye"],
                    "literal": ["no"],
                    "optional_bool": [False],
                },
            ),
        ),
        (EmptyPass(), PipelinePassSpec("empty", {})),
        (SimplePass(3.40, 2), PipelinePassSpec("simple", {"a": [3.40], "b": [2]})),
    ),
)
def test_pass_to_spec_equality(test_pass: ModulePass, test_spec: PipelinePassSpec):
    assert test_pass.from_pass_to_spec().name == test_spec.name
    for f in dataclasses.fields(test_pass):
        if len(test_spec.args[f.name]) == 0:
            assert getattr(test_pass, f.name) is None
            assert test_spec.args[f.name] == []
        elif len(test_spec.args[f.name]) == 1:
            assert getattr(test_pass, f.name) == test_spec.args[f.name][0]
        else:
            assert getattr(test_pass, f.name) == test_spec.args[f.name]

    assert str(test_pass.from_pass_to_spec()) == str(test_spec)
