from typing import Generic, TypeVar

import pytest

from xdsl.utils.hints import get_type_var_mapping

T = TypeVar("T")


class GenThing(Generic[T]):
    pass


class GenThingy(GenThing[T]):
    pass


class IntThing(GenThing[int]):
    pass


class IntThingy(GenThingy[int]):
    pass


@pytest.mark.parametrize(
    "specialized_type, type_var_mapping",
    [
        (IntThing, (GenThing, {T: int})),
        (IntThingy, (GenThingy, {T: int})),
    ],  # pyright: ignore [reportUnknownArgumentType]
)
def test_type_var_mapping(
    specialized_type: type, type_var_mapping: tuple[type, dict[TypeVar, type]]
):
    assert get_type_var_mapping(specialized_type) == type_var_mapping
