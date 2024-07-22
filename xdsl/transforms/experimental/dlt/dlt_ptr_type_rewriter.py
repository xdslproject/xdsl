from dataclasses import dataclass

from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.ir import Attribute
from xdsl.pattern_rewriter import TypeConversionPattern, attr_type_rewrite_pattern


@dataclass
class PtrIdentityTypeRewriter(TypeConversionPattern):

    def __init__(self, ptr: [dlt.PtrType], group: set[StringAttr]):
        self.ptr = ptr
        self.group = group
        super().__init__(recursive=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.PtrType, /) -> Attribute | None:
        if typ.identification in self.group:
            return self.ptr


@dataclass
class PtrIdentityBulkTypeRewriter(TypeConversionPattern):

    def __init__(self, ptr_map: dict[StringAttr, dlt.PtrType]):
        self.ptr_map = ptr_map
        super().__init__(recursive=True)

    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.PtrType, /) -> Attribute | None:
        if typ.identification in self.ptr_map:
            return self.ptr_map[typ.identification]