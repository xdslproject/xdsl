import re
from dataclasses import dataclass, field
from typing import Literal

import pytest

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.utils.parse_pipeline import PipelinePassSpec


@dataclass(frozen=True)
class CustomPass(ModulePass):
    name = "custom-pass"

    number: int | float

    int_list: tuple[int, ...]

    non_init_thing: int = field(init=False)

    str_thing: str

    nullable_str: str | None

    literal: Literal["yes", "no", "maybe"] = "no"

    optional_bool: bool = False

    annotation: "str" = "default_value"

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

    a: int | float
    b: int | None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pass


def test_pass_instantiation():
    p = CustomPass.from_pass_spec(
        PipelinePassSpec(
            name="custom-pass",
            args={
                "number": (2,),
                "int-list": (1, 2, 3),
                "str-thing": ("hello world",),
                "literal": ("maybe",),
                # "optional" was left out here, as it is optional
                "annotation": ("other_value",),
            },
        )
    )

    assert p.number == 2
    assert p.int_list == (1, 2, 3)
    assert p.str_thing == "hello world"
    assert p.nullable_str is None
    assert p.literal == "maybe"
    assert p.optional_bool is False

    # this should just work
    EmptyPass.from_pass_spec(PipelinePassSpec("empty", dict()))


@pytest.mark.parametrize(
    "spec, error_msg",
    [
        (PipelinePassSpec("wrong", {"a": (1,)}), "Cannot create Pass simple"),
        (PipelinePassSpec("simple", {}), 'requires argument "a"'),
        (
            PipelinePassSpec("simple", {"a": (1,), "no": ()}),
            'Provided arguments ["no"] not found in expected pass arguments ["a", "b"]',
        ),
        (PipelinePassSpec("simple", {"a": ()}), "Argument must contain a value"),
        (PipelinePassSpec("simple", {"a": ("test",)}), "Incompatible types"),
        (
            PipelinePassSpec("simple", {"a": ("test",), "literal": ("definitely",)}),
            "Incompatible types",
        ),
    ],
)
def test_pass_instantiation_error(spec: PipelinePassSpec, error_msg: str):
    """
    Test all possible failure modes in pass instantiation
    """
    with pytest.raises(Exception, match=re.escape(error_msg)):
        SimplePass.from_pass_spec(spec)


def test_required_fields():
    assert CustomPass.required_fields() == {
        "int_list",
        "non_init_thing",
        "number",
        "str_thing",
    }
    assert not EmptyPass.required_fields()
    assert SimplePass.required_fields() == {"a"}
