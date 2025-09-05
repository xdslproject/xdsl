from collections.abc import Mapping
from typing import TypeAlias

import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntAttr,
    StringAttr,
    f32,
    i32,
)
from xdsl.dialects.dlti import (
    DataLayoutEntryAttr,
    DataLayoutSpecAttr,
    DLTIEntryMap,
    MapAttr,
    TargetDeviceSpecAttr,
    TargetSystemSpecAttr,
)
from xdsl.ir import Attribute, TypeAttribute
from xdsl.utils.exceptions import VerifyException

DictValueType: TypeAlias = Mapping[
    StringAttr | TypeAttribute | str, "Attribute | str | int | DictValueType"
]


def test_data_layout_entry():
    # Test passing strings for key and value
    entry = DataLayoutEntryAttr("test", "V")
    assert entry.key == StringAttr("test")
    assert entry.value == StringAttr("V")

    # Test passing string for key and int for value
    entry = DataLayoutEntryAttr("test2", 12)
    assert entry.key == StringAttr("test2")
    assert entry.value == IntAttr(12)

    # Test passing string for key and float for value
    entry = DataLayoutEntryAttr("test3", FloatAttr(99.45, f32))
    assert entry.key == StringAttr("test3")
    assert entry.value == FloatAttr(99.45, f32)

    # Test passing type for key and string for value
    entry = DataLayoutEntryAttr(i32, "test")
    assert entry.key == i32
    assert entry.value == StringAttr("test")

    # Test passing StringAttr for key and value
    entry = DataLayoutEntryAttr(StringAttr("k"), StringAttr("v"))
    assert entry.key == StringAttr("k")
    assert entry.value == StringAttr("v")


def test_incorrect_data_layout_entry():
    with pytest.raises(VerifyException):
        DataLayoutEntryAttr("", "V")


@pytest.mark.parametrize(
    "cls", [DataLayoutSpecAttr, TargetDeviceSpecAttr, TargetSystemSpecAttr, MapAttr]
)
@pytest.mark.parametrize(
    "entries",
    [
        [StringAttr("value1"), IntAttr(23), IntAttr(43)],
        [IntAttr(23), FloatAttr(2.4, f32)],
    ],
)
def test_entry_maps(
    cls: type[DLTIEntryMap],
    entries: list[Attribute],
):
    # Comparison
    comparison_attr = cls.new(
        (
            ArrayAttr(
                [
                    DataLayoutEntryAttr("key_" + str(idx), e)
                    for idx, e in enumerate(entries)
                ]
            ),
        )
    )

    # Test initialisation from a dictionary
    attr_one = cls({"key_" + str(idx): e for idx, e in enumerate(entries)})
    assert comparison_attr == attr_one
    # Test initialisation from an array attribute of data layout entries
    attr_two = cls(
        ArrayAttr(
            [DataLayoutEntryAttr("key_" + str(idx), e) for idx, e in enumerate(entries)]
        )
    )
    assert comparison_attr == attr_two


def test_map_attr_embedded_dict():
    first_map = MapAttr({"key": {"key": {"key": "value", i32: 21}}})
    second_map = first_map["key"]
    assert isinstance(second_map, MapAttr)
    third_map = second_map["key"]
    assert isinstance(third_map, MapAttr)
    assert third_map["key"] == StringAttr("value")
    assert third_map[i32] == IntAttr(21)


@pytest.mark.parametrize(
    "cls", [DataLayoutSpecAttr, TargetDeviceSpecAttr, TargetSystemSpecAttr, MapAttr]
)
def test_duplicate_data_layout_map_entries(
    cls: type[DLTIEntryMap],
):
    with pytest.raises(VerifyException):
        cls(ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)]))


@pytest.mark.parametrize(
    "cls", [DataLayoutSpecAttr, TargetDeviceSpecAttr, TargetSystemSpecAttr, MapAttr]
)
def test_not_found_map_entry_lookup(
    cls: type[DLTIEntryMap],
):
    attr = cls(
        ArrayAttr([DataLayoutEntryAttr("key1", "v"), DataLayoutEntryAttr("key2", 12)])
    )
    with pytest.raises(KeyError):
        attr["key3"]
