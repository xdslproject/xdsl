from dataclasses import dataclass

from xdsl.dialects.experimental import dlt
from xdsl.ir import Attribute
from xdsl.pattern_rewriter import TypeConversionPattern, attr_type_rewrite_pattern


@dataclass
class PtrIdentityTypeRewriter(TypeConversionPattern):

    def __init__(self, ptr, group):
        self.ptr = ptr
        self.group = group
        super().__init__(recursive=True)
    @attr_type_rewrite_pattern
    def convert_type(self, typ: dlt.PtrType, /) -> Attribute | None:
        if typ.identification in self.group:
            return self.ptr