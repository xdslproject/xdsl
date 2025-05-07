from dataclasses import dataclass, field

import pytest

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.utils.parse_pipeline import PipelinePassSpec


@dataclass(frozen=True)
class CustomPass(ModulePass):
    name = "custom-pass"

    number: int

    int_list: tuple[int, ...]

    str_thing: str | None

    list_str: tuple[str, ...] = field(default=())

    optional_bool: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class EmptyPass(ModulePass):
    name = "empty"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class SimplePass(ModulePass):
    name = "simple"

    a: tuple[float, ...]
    b: int

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pass


@pytest.mark.parametrize(
    "test_pass, test_spec",
    [
        (
            CustomPass(3, (1, 2), None, ("clown", "season")),
            PipelinePassSpec(
                "custom-pass",
                {
                    "number": (3,),
                    "int_list": (1, 2),
                    "str_thing": (),
                    "list_str": ("clown", "season"),
                    "optional_bool": (False,),
                },
            ),
        ),
        (EmptyPass(), PipelinePassSpec("empty", {})),
        (
            SimplePass((3.14, 2.13), 2),
            PipelinePassSpec("simple", {"a": (3.14, 2.13), "b": (2,)}),
        ),
    ],
)
def test_pass_to_spec_include_default(
    test_pass: ModulePass,
    test_spec: PipelinePassSpec,
):
    assert test_pass.pipeline_pass_spec(include_default=True) == test_spec


@pytest.mark.parametrize(
    "test_pass, test_spec",
    [
        (
            CustomPass(3, (1, 2), None, ("clown", "season")),
            PipelinePassSpec(
                "custom-pass",
                {
                    "number": (3,),
                    "int_list": (1, 2),
                    "list_str": ("clown", "season"),
                },
            ),
        ),
        (EmptyPass(), PipelinePassSpec("empty", {})),
        (
            SimplePass((3.14, 2.13), 2),
            PipelinePassSpec("simple", {"a": (3.14, 2.13), "b": (2,)}),
        ),
    ],
)
def test_pass_to_spec_exclude_default(
    test_pass: ModulePass, test_spec: PipelinePassSpec
):
    assert test_pass.pipeline_pass_spec(include_default=False) == test_spec
    assert test_pass.pipeline_pass_spec() == test_spec
