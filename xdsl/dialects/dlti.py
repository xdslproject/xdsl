"""
The Data Layout and Target Information (DLTI) dialect is intended to
hold attributes and other components pertaining to descriptions of
in-memory data layout and compilation targets.

https://mlir.llvm.org/docs/Dialects/DLTIDialect/
"""

from __future__ import annotations

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class DataLayoutEntryAttr(ParametrizedAttribute):
    """
    An attribute to represent an entry of a data layout specification.
    https://mlir.llvm.org/docs/Dialects/DLTIDialect/#datalayoutentryattr
    """

    name = "dlti.dl_entry"

    key: Attribute
    value: Attribute

    def verify(self) -> None:
        if not isinstance(self.key, StringAttr | TypeAttribute):
            raise VerifyException("key must be a string or a type attribute")


DLTI = Dialect("dlti", [], [DataLayoutEntryAttr])
