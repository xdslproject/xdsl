"""
The Data Layout and Target Information (DLTI) dialect is intended to
hold attributes and other components pertaining to descriptions of
in-memory data layout and compilation targets.

https://mlir.llvm.org/docs/Dialects/DLTIDialect/
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import TypeAlias

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    StringAttr,
)
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

DictValueType: TypeAlias = Mapping[
    StringAttr | TypeAttribute | str, "Attribute | str | int | DictValueType"
]


class DLTITypeConverters:
    """
    Type converter for DLTIEntryMap and DataLayoutEntryAttr parameters
    """

    @staticmethod
    def convert_entry_map_type(
        contents: ArrayAttr[DataLayoutEntryAttr] | DictValueType,
    ) -> ArrayAttr[DataLayoutEntryAttr]:
        if isinstance(contents, Mapping):
            return ArrayAttr([DataLayoutEntryAttr(k, v) for k, v in contents.items()])
        else:
            return contents

    @staticmethod
    def convert_entry_attr_key_type(
        key: StringAttr | TypeAttribute | str,
    ) -> Attribute:
        if isinstance(key, str):
            return StringAttr(key)
        else:
            return key

    @staticmethod
    def convert_entry_attr_value_type(
        value: Attribute | str | int | DictValueType,
    ) -> Attribute:
        if isinstance(value, str):
            return StringAttr(value)
        elif isinstance(value, int):
            return IntAttr(value)
        elif isinstance(value, Mapping):
            return MapAttr(value)
        else:
            return value


@irdl_attr_definition
class DataLayoutEntryAttr(ParametrizedAttribute):
    """
    An attribute to represent an entry of a data layout specification.
    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#datalayoutentryattr
    """

    name = "dlti.dl_entry"

    key: Attribute = param_def(converter=DLTITypeConverters.convert_entry_attr_key_type)
    value: Attribute = param_def(
        converter=DLTITypeConverters.convert_entry_attr_value_type
    )

    def verify(self) -> None:
        if not isinstance(self.key, StringAttr | TypeAttribute):
            raise VerifyException("key must be a string or a type attribute")
        if isinstance(self.key, StringAttr) and not self.key.data:
            raise VerifyException("empty string as DLTI key is not allowed")


class DLTIEntryMap(ParametrizedAttribute, ABC):
    """
    Many DLTI dialect operations contain arrays of DataLayoutEntryInterface,
    with these representing different things such as data layout or information
    about the target hardware. This is the base class that these operations extend.
    """

    # In MLIR, this is a DataLayoutEntryInterface.
    entries: ArrayAttr[DataLayoutEntryAttr] = param_def(
        converter=DLTITypeConverters.convert_entry_map_type
    )

    def __getitem__(self, key: StringAttr | TypeAttribute | str) -> Attribute:
        if isinstance(key, str):
            key = StringAttr(key)
        for entry in self.entries:
            if entry.key == key:
                return entry.value

        raise KeyError

    @classmethod
    def parse_parameters(
        cls, parser: AttrParser
    ) -> tuple[ArrayAttr[DataLayoutEntryAttr]]:
        def parse_entry() -> DataLayoutEntryAttr:
            entry = parser.parse_attribute()
            parser.parse_punctuation("=")
            value = parser.parse_attribute()
            return DataLayoutEntryAttr.new((entry, value))

        entries = parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parse_entry)

        return (ArrayAttr(entries),)

    def print_parameters(self, printer: Printer):
        def print_entry(entry: DataLayoutEntryAttr):
            printer.print_attribute(entry.key)
            printer.print_string(" = ")
            printer.print_attribute(entry.value)

        with printer.in_angle_brackets():
            printer.print_list(self.entries, print_entry)

    def verify(self) -> None:
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise VerifyException("duplicate DLTI entry key")


@irdl_attr_definition(init=False)
class DataLayoutSpecAttr(DLTIEntryMap):
    """
    An attribute to represent a data layout specification.

    A data layout specification is a list of entries that specify (partial) data
    layout information. It is expected to be attached to operations that serve as
    scopes for data layout requests.

    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#datalayoutspecattr
    """

    name = "dlti.dl_spec"


@irdl_attr_definition(init=False)
class TargetDeviceSpecAttr(DLTIEntryMap):
    """
    An attribute to represent target device specification.

    Each device specification describes a single device and its hardware properties.
    Each device specification can contain any number of optional hardware properties
    (e.g., max_vector_op_width below).

    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#targetdevicespecattr
    """

    name = "dlti.target_device_spec"


@irdl_attr_definition(init=False)
class TargetSystemSpecAttr(DLTIEntryMap):
    """
    An attribute to represent target system specification.

    A system specification describes the overall system containing multiple devices,
    with each device having a unique ID (string) and its corresponding
    TargetDeviceSpec object.

    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#targetsystemspecattr
    """

    name = "dlti.target_system_spec"


@irdl_attr_definition(init=False)
class MapAttr(DLTIEntryMap):
    """
    A mapping of DLTI-information by way of key-value pairs

    A Data Layout and Target Information map is a list of entries effectively
    encoding a dictionary, mapping DLTI-related keys to DLTI-related values.

    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#mapattr
    """

    name = "dlti.map"


DLTI = Dialect(
    "dlti",
    [],
    [
        DataLayoutEntryAttr,
        DataLayoutSpecAttr,
        MapAttr,
        TargetDeviceSpecAttr,
        TargetSystemSpecAttr,
    ],
)
