import itertools
import typing
from typing import TypeVar

from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.transforms.experimental.dlt.layout_graph import LayoutGraph

T = TypeVar("T", dlt.Layout, list[dlt.Layout])


def _make_dense_layouts(layout: T, layout_map: dict[str, dlt.Layout]) -> T:
    if isinstance(layout, list):
        return [_make_dense_layouts(e, layout_map) for e in layout]
    # elif isinstance(layout, dlt.NamedLayoutAttr):
    #     layout: dlt.NamedLayoutAttr = layout
    #     if layout.abstract_name.data in map:
    #         return map[layout.abstract_name.data]
    #     else:
    #         sub_layout = _make_dense_layouts(layout.child, map)
    #         new_layout = dlt.NamedLayoutAttr(layout.abstract_name, sub_layout)
    #         assert new_layout.abstract_name.data not in map
    #         map[new_layout.abstract_name.data] = new_layout
    #         return new_layout
    elif isinstance(layout, dlt.AbstractLayoutAttr):
        layout: dlt.AbstractLayoutAttr = layout
        sub_layouts = []
        for child in layout.children:
            sub_layout = _make_dense_layouts(child.child, layout_map)
            for dim in list(child.dimensions):
                sub_layout = dlt.DenseLayoutAttr(sub_layout, dim)
            for member in list(child.member_specifiers):
                sub_layout = dlt.MemberLayoutAttr(sub_layout, member)
            sub_layouts.append(sub_layout)
        if len(sub_layouts) == 1:
            sub_layout = sub_layouts[0]
        else:
            assert len(sub_layouts) > 1
            sub_layout = dlt.StructLayoutAttr(sub_layouts)
        return sub_layout
    else:
        children = [
            _make_dense_layouts(child, layout_map) for child in layout.get_children()
        ]
        return layout.from_new_children(children)


def _try_apply_sparse(layout: dlt.Layout):
    if isinstance(layout, list):
        return [_try_apply_sparse(sub_layout) for sub_layout in layout]
    if isinstance(layout, dlt.AbstractLayoutAttr):
        a_layout = typing.cast(dlt.AbstractLayoutAttr, layout)
        if len(dims := a_layout.common_abstract_dimensions()) > 1:
            dims = list(dims)
            dims.sort(
                key=lambda d: (
                    d.extent.value.value.data
                    if isinstance(d.extent, dlt.StaticExtentAttr)
                    else 0
                )
            )
            sparse = [dims.pop()]
            return _make_sparse_layout(a_layout, dims, sparse)
    children = [_try_apply_sparse(child) for child in layout.get_children()]
    return layout.from_new_children(children)


def _make_sparse_layout(
    layout: dlt.AbstractLayoutAttr,
    direct_dims: list[dlt.DimensionAttr],
    sparse_dims: list[dlt.DimensionAttr],
) -> dlt.IndexingLayoutAttr:
    assert layout.common_abstract_dimensions().issuperset(direct_dims + sparse_dims)
    assert all(
        all(dim in child.dimensions for dim in sparse_dims) for child in layout.children
    )
    assert len(set(direct_dims).intersection(set(sparse_dims))) == 0
    direct_node = dlt.AbstractLayoutAttr(
        [([], direct_dims, dlt.PrimitiveLayoutAttr(dlt.IndexRangeType()))]
    )
    abstract_children = [
        dlt.AbstractChildAttr(
            a_child.member_specifiers,
            a_child.dimensions.remove(direct_dims + sparse_dims),
            a_child.child,
        )
        for a_child in layout.children
    ]
    coo_node = dlt.UnpackedCOOLayoutAttr(
        dlt.AbstractLayoutAttr(abstract_children), sparse_dims
    )
    return dlt.IndexingLayoutAttr(direct_node, coo_node)


def _try_apply_index_replace(layout: dlt.Layout):
    if isinstance(layout, list):
        return [_try_apply_index_replace(sub_layout) for sub_layout in layout]
    if isinstance(layout, dlt.AbstractLayoutAttr):
        a_layout = typing.cast(dlt.AbstractLayoutAttr, layout)

        extent_map = {}
        for dim in a_layout.contents_type.all_dimension_attributes():
            extent_map.setdefault(dim.extent, set()).add(dim)
        extents = []
        for e, dims in extent_map.items():
            count = sum(
                1
                for child in a_layout.children
                if len(dims & set(child.dimensions)) >= 1
            )
            extents.append((count, dims))
        extents.sort()

        if len(extents) == 0:
            return layout

        while len(extents) > 1:
            c, dims = extents.pop()
            if c < 2:
                return layout
            failed = False
            picked_list = set()
            ruled_out = set()
            children = sorted(
                [
                    (len(possible := (set(child.dimensions) & dims)), possible, child)
                    for child in a_layout.children
                ]
            )
            for _, possible, child in children:
                possible = possible - ruled_out
                must_use = possible & picked_list
                if len(must_use) == 0:
                    if len(possible) > 0:
                        picked_dim = possible.pop()
                        picked_list |= {picked_dim}
                        ruled_out |= possible
                elif len(must_use) == 1:
                    picked_dim = must_use.pop()
                    picked_list |= {picked_dim}
                    possible -= {picked_dim}
                    ruled_out |= possible
                elif len(must_use) > 1:
                    failed = True
            if failed:
                continue
            if len(picked_list) < 2:
                continue

            return _make_index_replacement(a_layout, list(picked_list))
    children = [_try_apply_index_replace(child) for child in layout.get_children()]
    return layout.from_new_children(children)


def _make_index_replacement(
    layout: dlt.AbstractLayoutAttr, dims: list[dlt.DimensionAttr], dim_name_num=0
) -> dlt.ArithReplaceLayoutAttr:
    extents = {dim.extent for dim in dims}
    assert len(extents) == 1
    assert all(
        len(set(elem.dimensions) & set(dims)) in [0, 1]
        for elem in layout.contents_type.elements
    )
    assert all(
        len(set(child.dimensions) & set(dims)) in [0, 1] for child in layout.children
    )

    extent = extents.pop()

    all_dims_names = [
        d.dimensionName.data for d in layout.contents_type.all_dimension_attributes()
    ]
    all_member_tuples = [
        (m.structName, m.memberName)
        for m in layout.contents_type.all_member_attributes()
    ]

    while (dim_name := f"_D_{dim_name_num}") in all_dims_names:
        dim_name_num += 1

    inner_dim = dlt.DimensionAttr(dim_name, extent)
    outer_dims = []
    for child in layout.children:
        d = set(child.dimensions) & set(dims)
        if len(d) == 0:
            pass
        elif len(d) > 1:
            assert False
        else:
            outer_dims.append(d.pop())

    inner_members = {}
    for outer_dim in outer_dims:
        if outer_dim is None:
            continue
        mem_name_num = 0
        while (
            inner_mem_name := (
                f"_M{dim_name}",
                f"_{outer_dim.dimensionName.data}_{mem_name_num}",
            )
        ) in all_member_tuples:
            mem_name_num += 1
        inner_member = dlt.MemberAttr(inner_mem_name[0], inner_mem_name[1])
        inner_members[outer_dim] = inner_member

    new_children = []
    for child in layout.children:
        d = set(child.dimensions) & set(dims)
        if len(d) == 0:
            continue
        outer_dim = d.pop()
        inner_member = inner_members[outer_dim]
        new_child = dlt.AbstractChildAttr(
            child.member_specifiers.add([inner_member]),
            child.dimensions.remove([outer_dim]).add([inner_dim]),
            child.child,
        )
        new_children.append(new_child)
    abstract_node = dlt.AbstractLayoutAttr(new_children)

    replacements = [
        dlt.ArithReplacementAttr(outer_dim, inner_dim, inner_members[outer_dim])
        for outer_dim in outer_dims
    ]

    index_replace_node = dlt.ArithReplaceLayoutAttr(abstract_node, replacements)
    assert index_replace_node.contents_type == layout.contents_type
    return index_replace_node


class PtrMapping():
    def __init__(self, number:int, ptr_map: dict[StringAttr, dlt.PtrType]):
        self.number = number
        self.keys = tuple(ptr_map.keys())
        self.values = tuple(ptr_map.values())

    def __eq__(self, other) -> bool:
        if not isinstance(other, PtrMapping):
            return False
        return self.keys == other.keys and self.values == other.values

    def __hash__(self):
        return hash((self.keys, self.values))

    def __getitem__(self, key):
        for i, k in enumerate(self.keys):
            if k == key:
                return self.values[i]

    def make_ptr_dict(self) -> dict[StringAttr, dlt.PtrType]:
        return {k:v for k,v in zip(self.keys, self.values)}


class LayoutGenerator:

    def __init__(self, layout_graph: LayoutGraph):
        self.layout_graph = layout_graph
        self.abstract_maps: list[PtrMapping] = [
            PtrMapping(0, layout_graph.get_type_map())
        ]
        self.final_maps: set[PtrMapping] = set()
        self._map_counter = 1

        self._entry_points = self.layout_graph.get_entry_layouts().keys()

    def _next_map_counter(self) -> int:
        num = self._map_counter
        self._map_counter += 1
        return num

    @staticmethod
    def _is_abstract(ptr_map: dict[StringAttr, dlt.PtrType]) -> bool:
        return any(ptr.layout.is_abstract for ptr in ptr_map.values())

    def _abstract_entry_points(
        self, ptr_map: PtrMapping
    ) -> set[StringAttr]:
        return {
            ident for ident in self._entry_points if ptr_map[ident].layout.is_abstract
        }

    def _abstract_points(
        self, ptr_map: PtrMapping
    ) -> set[StringAttr]:
        return {
            ident
            for ident in self.layout_graph.get_idents()
            if ptr_map[ident].layout.is_abstract
        }

    def generate_mappings(self):
        max_size = 0
        while self.abstract_maps:
            max_size = max(max_size, len(self.abstract_maps))
            print(f"Abstract mappings: {len(self.abstract_maps)} ({max_size}), Final mappings: {len(self.final_maps)}")
            parent_mapping = self.abstract_maps.pop()

            print(f"Reifying mapping {parent_mapping.number}")

            idents = self._abstract_entry_points(parent_mapping)
            if not idents:
                print(f"\tNo abstract layout entry points found in {parent_mapping.number}")
                idents = self._abstract_points(parent_mapping)
            if not idents:
                print(f"\tNo abstract layouts found in {parent_mapping.number}")
                self.final_maps.add(parent_mapping)
                continue

            ident = idents.pop()
            print(f"\tReifying {ident.data}")
            ptr = parent_mapping[ident]
            layout = ptr.layout
            new_layouts = self._generate_layouts(layout)
            new_nums = []
            duplicate_nums = []

            for new_layout in new_layouts:
                new_mapping = parent_mapping.make_ptr_dict()
                new_mapping_num = self._next_map_counter()

                # print(f"\t Making new mapping {new_mapping_num}")
                new_nums.append(new_mapping_num)

                new_ptr = ptr.with_new_layout(new_layout, preserve_ident=True)
                new_mapping[ident] = new_ptr
                self.layout_graph.propagate_type(ident, new_mapping)
                changed = [
                    i.data for (i, ptr) in new_mapping.items() if parent_mapping[i] != ptr
                ]

                # print("\t\tPropagated changes to: " + ",".join(changed))

                new_ptr_mapping = PtrMapping(new_mapping_num, new_mapping)
                if new_ptr_mapping in self.abstract_maps:
                    # print(f"\t\tNot adding {new_ptr_mapping.number} as it is a duplicate")
                    duplicate_nums.append(new_mapping_num)
                else:
                    self.abstract_maps.append(new_ptr_mapping)

            print(f"\t\tNew mappings: {new_nums} or which {len(duplicate_nums)} are duplicates")

    def _generate_layouts(self, layout: dlt.Layout) -> list[dlt.Layout]:
        if not layout.is_abstract:
            return []
        layouts = []
        children = layout.get_children()
        for i, child in enumerate(children):
            child_layouts = self._generate_layouts(child)
            for child_layout in child_layouts:
                new_children = children[:i] + [child_layout] + children[i + 1 :]
                layouts.append(layout.from_new_children(new_children))
        if isinstance(layout, dlt.AbstractLayoutAttr):
            layout = typing.cast(dlt.AbstractLayoutAttr, layout)
            layouts.extend(self._reify_layout(layout))

        return layouts

    def _reify_layout(self, abstract_layout: dlt.AbstractLayoutAttr) -> set[dlt.Layout]:
        layouts = set()
        if len(abstract_layout.children) == 1:
            abstract_child = abstract_layout.children.data[0]
            if len(abstract_child.dimensions) == 0 and len(abstract_child.member_specifiers) == 0:
                return {abstract_child.child}

        abstract_children = list(abstract_layout.children)
        for i, abstract_child in enumerate(abstract_children):
            abstract_child_replacements = self._reify_abstract_child(abstract_child)
            for replacement in abstract_child_replacements:
                layouts.add(dlt.AbstractLayoutAttr(abstract_children[:i] + [replacement] + abstract_children[i + 1 :]))

        if len(abstract_children) > 1:
            # for permutation in itertools.permutations(abstract_children):
            #     struct_children = []
            #     for abstract_child in permutation:
            #         struct_children.append(dlt.AbstractLayoutAttr([abstract_child]))
            #     struct_layout = dlt.StructLayoutAttr(struct_children)
            #     layouts.add(struct_layout)

            struct_children = []
            for abstract_child in abstract_children:
                struct_children.append(dlt.AbstractLayoutAttr([abstract_child]))
            struct_layout = dlt.StructLayoutAttr(struct_children)
            layouts.add(struct_layout)


            abstract_children_set = set(abstract_children)
            for subset in itertools.chain.from_iterable(itertools.combinations(abstract_children, r) for r in range(len(abstract_children) + 1)):
                if len(subset) == 0 or len(subset) == len(abstract_children):
                    continue
                other_set = abstract_children_set - set(subset)
                sub_layout = dlt.AbstractLayoutAttr(subset)
                other_layout = dlt.AbstractLayoutAttr(other_set)
                layouts.add(dlt.AbstractLayoutAttr([([],[],sub_layout), ([],[],other_layout)]))

        return layouts


    def _reify_abstract_child(self, abstract_child: dlt.AbstractChildAttr) -> set[dlt.AbstractChildAttr]:
        abstract_child_replacements = set()
        for member in abstract_child.member_specifiers:
            member_layout = dlt.MemberLayoutAttr(abstract_child.child, member)
            abstract_child_replacements.add(dlt.AbstractChildAttr(abstract_child.member_specifiers.without(member), abstract_child.dimensions, member_layout))
        for dimension in abstract_child.dimensions:
            dense_layout = dlt.DenseLayoutAttr(abstract_child.child, dimension)
            abstract_child_replacements.add(
                dlt.AbstractChildAttr(abstract_child.member_specifiers, abstract_child.dimensions.without(dimension),
                                      dense_layout))
        return abstract_child_replacements
