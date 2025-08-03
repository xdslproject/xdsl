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
from xdsl.utils.hints import isa

DictValueType: TypeAlias = Mapping[
    StringAttr | TypeAttribute | str, "Attribute | str | int | DictValueType"
]


def test_data_layout_entry():
    # Test passing strings for key and value
    entry = DataLayoutEntryAttr("test", "V")
    assert isinstance(entry.key, StringAttr)
    assert isinstance(entry.value, StringAttr)
    assert entry.key.data == "test"
    assert entry.value.data == "V"

    # Test passing string for key and int for value
    entry = DataLayoutEntryAttr("test2", 12)
    assert isinstance(entry.key, StringAttr)
    assert entry.key.data == "test2"
    assert isinstance(entry.value, IntAttr)
    assert entry.value.data == 12

    # Test passing string for key and float for value
    entry = DataLayoutEntryAttr("test3", FloatAttr(99.45, Float32Type()))
    assert isinstance(entry.key, StringAttr)
    assert isinstance(entry.value, FloatAttr)
    assert entry.key.data == "test3"
    assert round(entry.value.value.data, 2) == 99.45  # pyright: ignore

    # Test passing type for key and string for value
    entry = DataLayoutEntryAttr(i32, "test")
    assert isinstance(entry.key, TypeAttribute)
    assert isinstance(entry.value, StringAttr)
    assert entry.key == i32
    assert entry.value.data == "test"

    # Test passing StringAttr for key and value
    entry = DataLayoutEntryAttr(StringAttr("k"), StringAttr("v"))
    assert isinstance(entry.key, StringAttr)
    assert isinstance(entry.value, StringAttr)
    assert entry.key.data == "k"
    assert entry.value.data == "v"


def test_incorrect_data_layout_entry():
    with pytest.raises(VerifyException):
        entry = DataLayoutEntryAttr("", "V")
        entry.verify()


def generic_test_entry_equals_defn(
    dlti_entry: DataLayoutEntryAttr,
    comparison_entry: tuple[
        StringAttr | TypeAttribute | str, Attribute | str | int | DictValueType
    ],
):
    assert isinstance(dlti_entry, DataLayoutEntryAttr)

    comparison_entry_key = comparison_entry[0]
    comparison_entry_value = comparison_entry[1]

    dlti_entry_key = dlti_entry.key
    dlti_entry_value = dlti_entry.value

    if isinstance(comparison_entry_key, str) or isinstance(
        comparison_entry_key, StringAttr
    ):
        assert isinstance(dlti_entry_key, StringAttr)
        assert (
            dlti_entry_key.data == comparison_entry_key.data
            if isinstance(comparison_entry_key, StringAttr)
            else comparison_entry_key
        )
    else:
        assert isinstance(dlti_entry_key, TypeAttribute)
        assert dlti_entry_key == comparison_entry_key

    if isinstance(comparison_entry_value, str) or isinstance(
        comparison_entry_value, StringAttr
    ):
        assert isinstance(dlti_entry_value, StringAttr)
        assert (
            dlti_entry_value.data == dlti_entry_value.data
            if isinstance(comparison_entry_value, StringAttr)
            else comparison_entry_value
        )
    elif isinstance(comparison_entry_value, int) or isinstance(
        comparison_entry_value, IntAttr
    ):
        assert isinstance(dlti_entry_value, IntAttr)
        assert (
            dlti_entry_value.data == dlti_entry_value.data
            if isinstance(comparison_entry_value, IntAttr)
            else comparison_entry_value
        )
    elif isinstance(comparison_entry_value, FloatAttr):
        assert isinstance(dlti_entry_value, FloatAttr)
        assert dlti_entry_value.value.data == dlti_entry_value.value.data
    else:
        pytest.fail("Unknown comparison type")


def generic_specification_test(
    dlti_class: type[DLTIEntryMap],
    contents: ArrayAttr[DataLayoutEntryAttr] | DictValueType,
    comparison_entries: list[
        tuple[StringAttr | TypeAttribute | str, Attribute | str | int]
    ],
):
    spec = dlti_class(contents)
    assert isa(spec.entries, ArrayAttr[DataLayoutEntryAttr])
    assert len(spec.entries) == len(comparison_entries)

    for dlti_entry, comparison in zip(spec.entries.data, comparison_entries):
        generic_test_entry_equals_defn(dlti_entry, comparison)


@pytest.mark.parametrize(
    "entries",
    [
        [("key1", "value1"), ("key2", 23), (i32, 43)],
        [("k", IntAttr(23)), (i32, FloatAttr(2.4, Float32Type()))],
    ],
)
def test_data_layout_spec(
    entries: list[tuple[StringAttr | TypeAttribute | str, Attribute | str | int]],
):
    # Test initialisation from a dictionary
    generic_specification_test(
        DataLayoutSpecAttr, {e[0]: e[1] for e in entries}, entries
    )
    # Test initialisation from an array attribute of data layout entries
    generic_specification_test(
        DataLayoutSpecAttr,
        ArrayAttr([DataLayoutEntryAttr(e[0], e[1]) for e in entries]),
        entries,
    )


@pytest.mark.parametrize(
    "entries",
    [
        [("key1", "value1"), ("key2", 23), (i32, FloatAttr(9.3, Float32Type()))],
        [("k", IntAttr(23)), (i32, FloatAttr(2.4, Float32Type()))],
    ],
)
def test_target_device_spec(
    entries: list[tuple[StringAttr | TypeAttribute | str, Attribute | str | int]],
):
    # Test initialisation from a dictionary
    generic_specification_test(
        TargetDeviceSpecAttr, {e[0]: e[1] for e in entries}, entries
    )
    # Test initialisation from an array attribute of data layout entries
    generic_specification_test(
        TargetDeviceSpecAttr,
        ArrayAttr([DataLayoutEntryAttr(e[0], e[1]) for e in entries]),
        entries,
    )


@pytest.mark.parametrize(
    "entries",
    [
        [("key1", "value1"), ("key2", 23), (i32, "test")],
        [("k", IntAttr(23)), (i32, FloatAttr(2.4, Float32Type()))],
    ],
)
def test_target_system_spec(
    entries: list[tuple[StringAttr | TypeAttribute | str, Attribute | str | int]],
):
    # Test initialisation from a dictionary
    generic_specification_test(
        TargetSystemSpecAttr, {e[0]: e[1] for e in entries}, entries
    )
    # Test initialisation from an array attribute of data layout entries
    generic_specification_test(
        TargetSystemSpecAttr,
        ArrayAttr([DataLayoutEntryAttr(e[0], e[1]) for e in entries]),
        entries,
    )


@pytest.mark.parametrize(
    "entries",
    [
        [("key1", "value1"), ("key2", 23), (i32, 17)],
        [("k", IntAttr(23)), (i32, FloatAttr(2.4, Float32Type()))],
    ],
)
def test_map_attr(
    entries: list[tuple[StringAttr | TypeAttribute | str, Attribute | str | int]],
):
    # Test initialisation from a dictionary
    generic_specification_test(MapAttr, {e[0]: e[1] for e in entries}, entries)
    # Test initialisation from an array attribute of data layout entries
    generic_specification_test(
        MapAttr, ArrayAttr([DataLayoutEntryAttr(e[0], e[1]) for e in entries]), entries
    )


def test_map_attr_embedded_dict():
    m = MapAttr({"key": {"key": {"key": "value"}}})
    m.verify()
    assert isinstance(m.entries.data[0].value, MapAttr)
    assert isinstance(m.entries.data[0].value.entries.data[0].value, MapAttr)
    embedded_contents = m.entries.data[0].value.entries.data[0].value
    assert isinstance(embedded_contents.entries.data[0].key, StringAttr)
    assert isinstance(embedded_contents.entries.data[0].value, StringAttr)


def test_duplicate_data_layout_spec_entries():
    with pytest.raises(VerifyException):
        entry = DataLayoutSpecAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )
        entry.verify()


def test_duplicate_target_device_spec_entries():
    with pytest.raises(VerifyException):
        entry = TargetDeviceSpecAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )
        entry.verify()


def test_duplicate_system_spec_entries():
    with pytest.raises(VerifyException):
        entry = TargetSystemSpecAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )
        entry.verify()


def test_duplicate_map_attr_entries():
    with pytest.raises(VerifyException):
        entry = MapAttr(
            ArrayAttr([DataLayoutEntryAttr("k", "v"), DataLayoutEntryAttr("k", 12)])
        )
        entry.verify()
