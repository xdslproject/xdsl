from typing import Generic, TypeVar

import pytest

from xdsl.utils.hints import get_type_var_mapping

T = TypeVar("T")


class GenericBase(Generic[T]):
    pass


class GenericSubclassOfGenericBase(GenericBase[T]):
    pass


class SpecificSubclassOfGenericBase(GenericBase[int]):
    pass


class SpecificSubclassOfGenericSubclass(GenericSubclassOfGenericBase[int]):
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
