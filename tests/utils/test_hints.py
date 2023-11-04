from typing import Generic, TypeVar

import pytest

from xdsl.utils.hints import get_type_var_mapping

T = TypeVar("T")
R = TypeVar("R")


class GenericBase(Generic[T]):
    pass


class GenericSubclassOfGenericBase(GenericBase[R]):
    pass


class SpecificSubclassOfGenericBase(GenericBase[int]):
    pass


class SpecificSubclassOfGenericSubclass(GenericSubclassOfGenericBase[int]):
    pass


@pytest.mark.parametrize(
    "specialized_type, type_var_mapping",
    [
        (SpecificSubclassOfGenericBase, (GenericBase, {T: int})),
        (SpecificSubclassOfGenericSubclass, (GenericSubclassOfGenericBase, {R: int})),
        (GenericSubclassOfGenericBase[int], (GenericBase, {T: int})),
    ],  # pyright: ignore [reportUnknownArgumentType]
)
def test_type_var_mapping(
    specialized_type: type, type_var_mapping: tuple[type, dict[TypeVar, type]]
):
    assert get_type_var_mapping(specialized_type) == type_var_mapping
