from dataclasses import dataclass, field
from typing import Literal

import pytest

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass, PassArgOption, get_pass_argument_names_and_types


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
    c: int = 5

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        pass


@pytest.mark.parametrize(
    "arg_options, pass_arg",
    [
        (
            (
                PassArgOption("number", "int|float"),
                PassArgOption("single_number", "int"),
                PassArgOption("int_list", "tuple[int, ...]"),
                PassArgOption("non_init_thing", "int"),
                PassArgOption("str_thing", "str"),
                PassArgOption("nullable_str", "str|None"),
                PassArgOption(
                    "literal",
                    "typing.Literal['yes', 'no', 'maybe']",
                    "no",
                ),
                PassArgOption("optional_bool", "bool", "false"),
            ),
            CustomPass,
        ),
        ((), EmptyPass),
        (
            (
                PassArgOption("a", "int|float"),
                PassArgOption("b", "int|None"),
                PassArgOption("c", "int", "5"),
            ),
            SimplePass,
        ),
    ],
)
def test_pass_to_arg_and_type_str(
    arg_options: tuple[PassArgOption, ...], pass_arg: type[ModulePass]
):
    assert get_pass_argument_names_and_types(pass_arg) == arg_options
