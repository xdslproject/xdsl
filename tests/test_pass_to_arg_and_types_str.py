from dataclasses import dataclass, field
from typing import Literal

import pytest

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass, PassOptionInfo, get_pass_option_infos


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
                PassOptionInfo("number", "int|float"),
                PassOptionInfo("single_number", "int"),
                PassOptionInfo("int_list", "tuple[int, ...]"),
                PassOptionInfo("non_init_thing", "int"),
                PassOptionInfo("str_thing", "str"),
                PassOptionInfo("nullable_str", "str|None"),
                PassOptionInfo(
                    "literal",
                    "typing.Literal['yes', 'no', 'maybe']",
                    "no",
                ),
                PassOptionInfo("optional_bool", "bool", "false"),
            ),
            CustomPass,
        ),
        ((), EmptyPass),
        (
            (
                PassOptionInfo("a", "int|float"),
                PassOptionInfo("b", "int|None"),
                PassOptionInfo("c", "int", "5"),
            ),
            SimplePass,
        ),
    ],
)
def test_pass_to_arg_and_type_str(
    arg_options: tuple[PassOptionInfo, ...], pass_arg: type[ModulePass]
):
    assert get_pass_option_infos(pass_arg) == arg_options
