"""
This is a clone of the upstream dlti dialect (https://mlir.llvm.org/docs/Dialects/DLTIDialect/)

This dialect is used to provide data layout and target information.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cache
from typing import Annotated, cast, Sequence

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.ir import Attribute, Dialect
from xdsl.irdl import ParameterDef, ParametrizedAttribute, irdl_attr_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


class DataLayoutEntryInterface(ABC):
    """
    Attribute interface describing an entry in a data layout specification.

    A data layout specification entry is a key-value pair. Its key is either a
    type, when the entry is related to a type or a class of types, or an
    identifier, when it is not. `DataLayoutEntryKey` is an alias allowing one
    to use both key types. Its value is an arbitrary attribute that is
    interpreted either by the type for type keys or by the dialect containing
    the identifier for identifier keys.
    """

    @property
    @abstractmethod
    def key(self) -> Attribute | str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def value(self) -> Attribute:
        raise NotImplementedError()

    def is_type_entry(self) -> bool:
        return isinstance(self.key, Attribute)


class TargetDeviceSpecInterface(ABC):
    """
    Attribute interface describing a target device description specification.

    A target device description specification is a list of device properties (key)
    and their values for a specific device. The device is identified using "device_id"
    (as a key and ui32 value) and "device_type" key which must have a string value.
    Both "device_id" and "device_type" are mandatory keys. As an example, L1 cache
    size could be a device property, and its value would be a device specific size.

    A target device description specification is attached to a module as a module level
    attribute.
    """

    @abstractmethod
    def get_entries(self) -> Iterable[DataLayoutEntryInterface]:
        raise NotImplementedError()

    def get_entries_dict(self) -> dict[str | Attribute, Attribute]:
        return dict((e.key, e.value) for e in self.get_entries())

    def get_spec_for_type(self, type: Attribute) -> Attribute | None:
        return self.get_entries_dict().get(type, None)

    def get_spec_for_identifier(self, id: str) -> Attribute | None:
        return self.get_entries_dict().get(id, None)


class TargetSystemSpecInterface(ABC):
    """
    Attribute interface describing a target system description specification.

    A target system description specification is a list of target device
    specifications, with one device specification for a device in the system. As
    such, a target system description specification allows specifying a heterogeneous
    system, with devices of different types (e.g., CPU, GPU, etc.)

    The only requirement on a valid target system description specification is that
    the "device_id" in every target device description specification needs to be
    unique. This is because, ultimately, this "device_id" is used by the user to
    query a value of a device property.
    """

    @abstractmethod
    def get_entries(self) -> Iterable[tuple[str, TargetDeviceSpecInterface]]:
        raise NotImplementedError()

    def get_entries_dict(self) -> dict[str, TargetDeviceSpecInterface]:
        return dict(self.get_entries())

    def get_device_spec_for_id(self, id: str) -> TargetDeviceSpecInterface | None:
        return self.get_entries_dict().get(id, None)


@irdl_attr_definition
class DlEntryAttr(ParametrizedAttribute, DataLayoutEntryInterface):
    name = "dlti.dl_entry"

    key_: ParameterDef[Attribute]
    value_: ParameterDef[Attribute]

    @property
    def key(self) -> Attribute | str:
        if isinstance(self.key_, StringAttr):
            return self.key_.data
        return self.key_

    @property
    def value(self) -> Attribute:
        return self.value_


@irdl_attr_definition
class TargetDeviceSpec(ParametrizedAttribute, TargetDeviceSpecInterface):
    name = "dlti.target_device_spec"

    entries: ParameterDef[ArrayAttr[Attribute]]

    def get_entries(self) -> Iterable[DataLayoutEntryInterface]:
        return tuple(self.entries)  # pyright: ignore[reportGeneralTypeIssues]

    def _verify(self):
        if not all(isinstance(elem, DataLayoutEntryInterface) for elem in self.entries):
            raise VerifyException("All entries in device spec must implement TargetDeviceSpecInterface")
        if len(set(elem.key for elem in self.entries)) != len(self.entries):
            raise VerifyException("Duplicate keys in device spec!")

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(self.entries, printer.print_attribute)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        return ArrayAttr(parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parser.parse_attribute)),


@irdl_attr_definition
class TargetSystemSpec(ParametrizedAttribute, TargetSystemSpecInterface):

    name = "dlti.target_system_spec"

    entries: ParameterDef[DictionaryAttr]

    def _verify(self):
        for val in self.entries.data.values():
            if not isinstance(val, TargetDeviceSpecInterface):
                raise VerifyException(
                    "Expected all values to implement TargetDeviceSpecInterface"
                )

    def get_entries(self) -> Iterable[tuple[str, TargetDeviceSpecInterface]]:
        return cast(
            Iterable[tuple[str, TargetDeviceSpecInterface]], self.entries.data.items()
        )

    def get_entries_dict(self) -> dict[str, TargetDeviceSpecInterface]:
        return cast(dict[str, TargetDeviceSpecInterface], self.entries.data)

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            print_comma = False
            for key, val in self.entries.data.items():
                if print_comma:
                    printer.print(", ")
                printer.print_string_literal(key)
                printer.print(" : ")
                printer.print(val)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        def parse_kv() -> tuple[str, Attribute]:
            str = parser.parse_str_literal()
            parser.parse_punctuation(":")
            val = parser.parse_attribute()
            return str, val

        items = parser.parse_comma_separated_list(parser.Delimiter.ANGLE, parse_kv)
        return DictionaryAttr(dict(items)),


DLTI = Dialect("dlti", [], [
    DlEntryAttr,
    TargetSystemSpec,
    TargetDeviceSpec,
])
