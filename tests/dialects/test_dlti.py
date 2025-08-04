from collections.abc import Mapping
from typing import TypeAlias

import pytest

from xdsl.dialects.builtin import (
    ArrayAttr,
    Float32Type,
    FloatAttr,
    IntAttr,
    StringAttr,
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
    entry = DataLayoutEntryAttr("test3", FloatAttr(99.45, Float32Type()))
    assert entry.key == StringAttr("test3")
    assert entry.value == FloatAttr(99.45, Float32Type())

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


def generic_specification_test(
    dlti_class: type[DLTIEntryMap],
    comparison_entries: list[Attribute],
):
    # Comparison
    comparison_attr = dlti_class.new(
        (
            ArrayAttr(
                [
                    DataLayoutEntryAttr("key_" + str(idx), e)
                    for idx, e in enumerate(comparison_entries)
                ]
            ),
        )
    )

    # Test initialisation from a dictionary
    attr_one = dlti_class(
        {"key_" + str(idx): e for idx, e in enumerate(comparison_entries)}
    )
    assert comparison_attr == attr_one
    # Test initialisation from an array attribute of data layout entries
    attr_two = dlti_class(
        ArrayAttr(
            [
                DataLayoutEntryAttr("key_" + str(idx), e)
                for idx, e in enumerate(comparison_entries)
            ]
        )
    )
    assert comparison_attr == attr_two


@pytest.mark.parametrize(
    "entries",
    [
        [StringAttr("value1"), IntAttr(23), IntAttr(43)],
        [IntAttr(23), FloatAttr(2.4, Float32Type())],
    ],
)
def test_data_layout_spec(
    entries: list[Attribute],
):
    generic_specification_test(DataLayoutSpecAttr, entries)


@pytest.mark.parametrize(
    "entries",
    [
        [StringAttr("value1"), IntAttr(23), IntAttr(43)],
        [IntAttr(23), FloatAttr(2.4, Float32Type())],
    ],
)
def test_target_device_spec(
    entries: list[Attribute],
):
    generic_specification_test(TargetDeviceSpecAttr, entries)


@pytest.mark.parametrize(
    "entries",
    [
        [StringAttr("value1"), IntAttr(23), IntAttr(43)],
        [IntAttr(23), FloatAttr(2.4, Float32Type())],
    ],
)
def test_target_system_spec(
    entries: list[Attribute],
):
    generic_specification_test(TargetSystemSpecAttr, entries)


@pytest.mark.parametrize(
    "entries",
    [
        [StringAttr("value1"), IntAttr(23), IntAttr(43)],
        [IntAttr(23), FloatAttr(2.4, Float32Type())],
    ],
)
def test_map_attr(
    entries: list[Attribute],
):
    generic_specification_test(MapAttr, entries)


def test_map_attr_embedded_dict():
    m = MapAttr({"key": {"key": {"key": "value"}}})
    assert isinstance(m.entries.data[0].value, MapAttr)
    assert isinstance(m.entries.data[0].value.entries.data[0].value, MapAttr)
    embedded_contents = m.entries.data[0].value.entries.data[0].value
    assert isinstance(embedded_contents.entries.data[0].key, StringAttr)
    assert isinstance(embedded_contents.entries.data[0].value, StringAttr)


def test_duplicate_data_layout_spec_entries():
    with pytest.raises(VerifyException):
        DataLayoutSpecAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )


def test_duplicate_target_device_spec_entries():
    with pytest.raises(VerifyException):
        TargetDeviceSpecAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )


def test_duplicate_system_spec_entries():
    with pytest.raises(VerifyException):
        TargetSystemSpecAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )


def test_duplicate_map_attr_entries():
    with pytest.raises(VerifyException):
        MapAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )
