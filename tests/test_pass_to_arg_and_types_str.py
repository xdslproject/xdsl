from dataclasses import dataclass, field
from typing import Literal

import pytest

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.passes import ModulePass, get_pass_argument_names_and_types


@dataclass(frozen=True)
class CustomPass(ModulePass):
    name = "custom-pass"

    number: int | float

    single_number: int

    int_list: tuple[int, ...]

    non_init_thing: int = field(init=False)

    str_thing: str

    nullable_str: str | None

    literal: Literal["yes", "no", "maybe"] = "no"

    optional_bool: bool = False

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class EmptyPass(ModulePass):
    name = "empty"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@dataclass(frozen=True)
class SimplePass(ModulePass):
    name = "simple"

    a: int | float
    b: int | None
    c: int = 5

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        pass


@pytest.mark.parametrize(
    "str_arg, pass_arg",
    [
        (
            """number=int|float single_number=int int_list=tuple[int, ...] non_init_thing=int str_thing=str nullable_str=str|None literal=no optional_bool=false""",
            CustomPass,
        ),
        ("", EmptyPass),
        ("""a=int|float b=int|None c=5""", SimplePass),
    ],
)
def test_pass_to_arg_and_type_str(str_arg: str, pass_arg: type[ModulePass]):
    assert get_pass_argument_names_and_types(pass_arg) == str_arg
