import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.ir import SSAValue


@dataclass
class LayoutGraph:
    ident_count: defaultdict[StringAttr, set[SSAValue]] = field(default_factory=lambda: defaultdict(set))

    graph_edges: dict[StringAttr,
    set[tuple[dlt.SetAttr[dlt.MemberAttr], dlt.SetAttr[dlt.DimensionAttr], StringAttr] |
        tuple[None, None, StringAttr]]] \
        = field(default_factory=lambda: dict())

    required_extents: dict[StringAttr, set[dlt.InitDefinedExtentAttr]] = field(default_factory=lambda: dict())

    def types_for(self, ident: StringAttr) -> set[dlt.PtrType]:
        types: set[dlt.PtrType] = set()
        for ssa in self.ident_count[ident]:
            ptr_type = ssa.type
            assert isinstance(ptr_type, dlt.PtrType)
            ptr_type = cast(dlt.PtrType, ptr_type)
            types.add(ptr_type)
        return types

    def get_type_for(self, ident: StringAttr) -> dlt.PtrType:
        types = self.types_for(ident)
        if len(types) != 1:
            raise ValueError()
        for ptr_type in types:
            return ptr_type

    def is_consistent(self):
        pass
        for identity in self.ident_count:
            types = self.types_for(identity)
            if len(types) > 1:
                return False
        for start, members, dims, end in [(s,m,d,e) for s, edges in self.graph_edges.items() for m, d, e in edges]:
            starting_type = self.get_type_for(start)
            ending_type = self.get_type_for(end)
            output = starting_type.contents_type.select_members(members).select_dimensions(dims)
            if output != ending_type.contents_type:
                return False
            for extent in ending_type.filled_extents:
                if extent not in starting_type.filled_extents:
                    return False
            # TODO check layouts are compatible
            # if not layout_can_derive_to(starting_type.layout, ending_type.layout,):
            #     return False

        for identity, extents in self.required_extents.items():
            ptr_type = self.get_type_for(identity)
            for extent in extents:
                if extent not in ptr_type.filled_extents:
                    return False
        return True

@functools.singledispatch
def layout_can_derive_to(starting_layout: dlt.Layout,
                         ending_Layout: dlt.Layout,
                         members: dlt.SetAttr[dlt.MemberAttr],
                         dimensions: dlt.SetAttr[dlt.DimensionAttr]) -> bool:
    pass