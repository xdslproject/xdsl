import datetime
import dis
import itertools
import time
import typing
from typing import TypeVar

from scripts.visualiseDLT import LayoutPlotter
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
            return _make_sparse_layout(a_layout, dims, sparse, [])
    children = [_try_apply_sparse(child) for child in layout.get_children()]
    return layout.from_new_children(children)


def _make_sparse_layout(
    layout: dlt.AbstractLayoutAttr,
    direct_dims: list[dlt.DimensionAttr],
    sparse_dims: list[dlt.DimensionAttr],
    child_dims: list[dlt.DimensionAttr],
) -> dlt.AbstractLayoutAttr:
    common_dims = layout.common_abstract_dimensions()
    assert common_dims.issuperset(direct_dims + sparse_dims + child_dims)
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
            a_child.dimensions.remove(common_dims).union(child_dims),
            a_child.child,
        )
        for a_child in layout.children
    ]
    coo_node = dlt.UnpackedCOOLayoutAttr(
        dlt.AbstractLayoutAttr(abstract_children), sparse_dims
    )

    indexing_node = dlt.IndexingLayoutAttr(direct_node, coo_node)
    abstract_node= dlt.AbstractLayoutAttr([dlt.AbstractChildAttr([], common_dims.difference(sparse_dims+direct_dims+child_dims), indexing_node)])
    return abstract_node


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
    def __init__(self, number: int, ptr_map: dict[StringAttr, dlt.PtrType]):
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

    def __init__(self, layout_graph: LayoutGraph, plot_dir = None):
        self.layout_graph = layout_graph
        self.abstract_maps: list[tuple[PtrMapping, list[PtrMapping]|None]] = [(
            PtrMapping(0, layout_graph.get_type_map()), None)
        ]
        self._max_size = 0
        self.final_maps: set[PtrMapping] = set()
        self.seen_mappings = set()
        self._map_counter = 1

        self._entry_points = self.layout_graph.get_entry_layouts().keys()

        self._generated_layouts: dict[dlt.AbstractLayoutAttr, typing.Collection[dlt.Layout] | None] = {}

        self.plot_dir = plot_dir

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

    def _print_stats(self):
        print(
            f"{datetime.datetime.now()} : Seen: {len(self.seen_mappings)}, Layout Store: {len(self._generated_layouts)}, Abstract: {len(self.abstract_maps)} ({self._max_size}), Final: {len(self.final_maps)}, Current Abstract Mappings: {["N" if l is None else len(l) for _, l in self.abstract_maps]}")

    def plot_mapping(self, mapping: PtrMapping):
        if self.plot_dir is None:
            return
        mapping_dict = mapping.make_ptr_dict()
        name = f"mapping_{mapping.number}"
        if not LayoutGenerator._is_abstract(mapping_dict):
            name += "_final"
        LayoutPlotter.plot_layout(mapping_dict, name=name, directory=self.plot_dir,
                                  entry_points=set(self._entry_points))


    def generate_mappings(self, take_first = 0):
        start_time = time.time()
        for parent_mapping, _ in self.abstract_maps:
            self.plot_mapping(parent_mapping)

        while self.abstract_maps:
            self._max_size = max(self._max_size, len(self.abstract_maps))
            self._print_stats()
            # parent_mapping, node_children = self.abstract_maps.pop(0)
            parent_mapping, node_children = self.abstract_maps.pop()

            if node_children is None:
                node_children = []
                print(f"Reifying mapping {parent_mapping.number}")

                idents = self._abstract_entry_points(parent_mapping)
                if not idents:
                    print(f"\tNo abstract layout entry points found in {parent_mapping.number}")
                    idents = self._abstract_points(parent_mapping)
                if not idents:
                    print(f"\tNo abstract layouts found in {parent_mapping.number}")

                    if parent_mapping in self.final_maps:
                        print(f"\t {parent_mapping.number} is final but was already found")
                    else:
                        print(f"\t {parent_mapping.number} is final!")
                        self.final_maps.add(parent_mapping)
                        if 0 < take_first <= len(self.final_maps):
                            print(f"{datetime.datetime.now()} : Exiting early with {len(self.final_maps)} mappings")
                            return self.final_maps
                    continue

                ident = idents.pop()
                print(f"\tReifying {ident.data}")
                ptr = parent_mapping[ident]
                layout = ptr.layout
                new_layouts = self._generate_layouts(layout)
                new_nums = []
                duplicate_nums = []
                propagate_times = []

                new_layout_count = len(new_layouts)
                val = f"_/{new_layout_count}"
                print("\tPropagating new layout: "+val, end="")
                chars = len(val)
                for i, new_layout in enumerate(new_layouts):
                    now = time.time()

                    val = f" {i}/{new_layout_count} \t times: {propagate_times}"
                    print(("\b"*chars) + val, end="")
                    chars = len(val)

                    new_mapping = parent_mapping.make_ptr_dict()
                    new_mapping_num = self._next_map_counter()

                    # print(f"\t Making new mapping {new_mapping_num}")
                    new_nums.append(new_mapping_num)

                    new_ptr = ptr.with_new_layout(new_layout, preserve_ident=True)
                    new_mapping[ident] = new_ptr
                    self.layout_graph.propagate_type(ident, new_mapping)
                    # changed = [
                    #     i.data for (i, ptr) in new_mapping.items() if parent_mapping[i] != ptr
                    # ]

                    # print("\t\tPropagated changes to: " + ",".join(changed))

                    new_ptr_mapping = PtrMapping(new_mapping_num, new_mapping)
                    if new_ptr_mapping in self.seen_mappings:
                        # print(f"\t\tNot adding {new_ptr_mapping.number} as it is a duplicate")
                        duplicate_nums.append(new_mapping_num)
                    else:
                        self.plot_mapping(new_ptr_mapping)
                        node_children.append(new_ptr_mapping)
                        self.seen_mappings.add(new_ptr_mapping)
                    propagate_times.append(time.time() - now)

                val = f" {new_layout_count}/{new_layout_count} \t times: {propagate_times}"
                print(("\b" * chars) + val)
                chars = len(val)
                print(f"\t\tNew mappings: {new_nums} of which {len(duplicate_nums)} are duplicates")

            print(f"\t mapping {parent_mapping.number} has {len(node_children)} more children")
            if len(node_children) == 0:
                continue

            new_child = node_children.pop()
            # self.abstract_maps.insert(0, (new_child, None))
            self.abstract_maps.append((parent_mapping, node_children))
            self.abstract_maps.append((new_child, None))

        self._print_stats()
        print(f"Generated {len(self.final_maps)} reified mappings in {time.time()-start_time}s")
        return self.final_maps

    def _generate_layouts(self, layout: dlt.Layout, path=tuple()) -> typing.Collection[dlt.Layout]:
        if not layout.is_abstract:
            return []
        layouts = []
        if isinstance(layout, dlt.AbstractLayoutAttr):
            layout = typing.cast(dlt.AbstractLayoutAttr, layout)
            previous_layouts = self._generated_layouts.get(layout, None)
            if previous_layouts is None:
                print(f"\t\t Generating new layouts at: {path}")
                new_layouts = self._reify_layout(layout)
                layouts.extend(new_layouts)
                self._generated_layouts[layout] = new_layouts
            else:
                print(f"\t\t Using Previously Generated layouts at: {path}")
                layouts.extend(previous_layouts)
            print(f"\t\t\t\t\t\t" + ("###"*(len(path)+1)), end="")
        else:
            children = layout.get_children()
            for i, child in enumerate(children):
                child_layouts = self._generate_layouts(child, path=(*path, i))
                for child_layout in child_layouts:
                    new_children = children[:i] + [child_layout] + children[i + 1 :]
                    layouts.append(layout.from_new_children(new_children))
                if child_layouts:
                    break
            print("\b"*3, end="")

        if len(path) == 0:
            print("\b" * 1, end="")
            list_len = len(layouts)
            if list_len > 1:
                # layouts = set(layouts)
                set_len = len(layouts)
                print_val = f"\t\t\t{list_len-set_len} / {list_len} duplicates generated"
                pass # lets just assume layouts are unique from _reify_layout
            else:
                print_val = f"\t\t\t{list_len} layouts generated"
            print("\b" * 2, end="")
            print("\b"*6, end="")
            # print(print_val)
        return layouts

    def _reify_layout(self, abstract_layout: dlt.AbstractLayoutAttr) -> typing.Collection[dlt.Layout]:
        val = ". "
        print(f"\t\t\treifying abstract layout with {len(abstract_layout.children)} children: "+val, end="")
        chars = len(val)

        # If abstract layout is 'empty' (one child with no abstract dims or members), then remove that abstract layout and return the child
        abstract_children = list(abstract_layout.children)
        if len(abstract_children) == 1:
            abstract_child = abstract_layout.children.data[0]
            if len(abstract_child.dimensions) == 0 and len(abstract_child.member_specifiers) == 0:
                print(("\b"*chars) + f"Single empty abstract layout.")
                return (abstract_child.child,)

        # If abstract layout has any common members, these should ne delt with first as they do not affect physical data layout (even though they may effect ptr reductions with our naive selection statements)
        if len(common_mems := abstract_layout.common_abstract_members()) > 0:
            print(("\b" * chars) + f"Forcing common members to be reified first")
            return (self._reify_members(abstract_layout, common_mems),)

        layouts = list()

        val = f"{len(layouts)} .. "
        print(("\b"*chars) + val, end="")
        chars = len(val)

        # try to split abstract into an abstract of two children
        if len(abstract_children) > 1:
            for subset, other_set in self._subset_pairs(abstract_children):
                if any(c.dimensions or c.member_specifiers for c in subset):
                    sub_layout = dlt.AbstractLayoutAttr(subset)
                    other_layout = dlt.AbstractLayoutAttr(other_set)
                    layouts.append(dlt.AbstractLayoutAttr([([],[],sub_layout), ([],[],other_layout)]))
                print(".", end="")
                chars += 1

        val = f"{len(layouts)} ... "
        print(("\b" * chars) + val, end="")
        chars = len(val)

        # Try to turn an abstract layout into a struct - only when 2 abstract children which will form binary tree like structures which helps to reduce accidently creating equivilent layouts
        if len(abstract_children) == 2:
            layouts.append(dlt.StructLayoutAttr([
                dlt.AbstractLayoutAttr([abstract_children[0]]),
                dlt.AbstractLayoutAttr([abstract_children[1]]),
            ]))
            layouts.append(dlt.StructLayoutAttr([
                dlt.AbstractLayoutAttr([abstract_children[1]]),
                dlt.AbstractLayoutAttr([abstract_children[0]]),
            ]))

        val = f"{len(layouts)} .... "
        print(("\b" * chars) + val, end="")
        chars = len(val)

        val = "..... "
        print(("\b"*chars) + val, end="")
        chars = len(val)

        # Try to insert an index replacement node
        if len(abstract_children) > 1:
            index_swap_layouts = self._get_index_replace_layouts(abstract_layout)
            layouts.extend(index_swap_layouts)

        # Try to insert an indexing node
        if len(abstract_children) == 1:
            layouts.extend(self._try_sparse(abstract_layout))

        if len(abstract_children) == 1:
            layouts.extend(self._try_dense(abstract_layout))

        val = "...... "
        print(("\b"*chars) + val, end="")
        chars = len(val)

        # Try to separate common dims from the abstract node
        if common_dims := list(abstract_layout.common_abstract_dimensions()):
            for subset in self._subsets(common_dims):
                if len(subset) < 1:
                    continue
                non_empty = False
                new_children = []
                for child in abstract_children:
                    non_empty |= len(child.member_specifiers.data) > 0
                    child_dims = child.dimensions.remove(subset)
                    non_empty |= len(child_dims.data) > 0
                    new_children.append(dlt.AbstractChildAttr(
                        child.member_specifiers,
                        child_dims,
                        child.child
                    ))
                if non_empty or len(new_children) > 1:
                    abstract_sub_layout = dlt.AbstractLayoutAttr(new_children)
                    new_layout = dlt.AbstractLayoutAttr([dlt.AbstractChildAttr([], subset, abstract_sub_layout)])
                    layouts.append(new_layout)

        val = "....... "
        print(("\b"*chars) + val, end="")
        chars = len(val)

        val = f"{len(layouts)} "
        print(("\b"*chars) + val, end="")
        chars = len(val)

        print(f"new layouts")
        return layouts

    def _reify_members(self, abstract_layout: dlt.AbstractLayoutAttr, mems: set[dlt.MemberAttr]):
        new_children = [dlt.AbstractChildAttr(c.member_specifiers.remove(mems), c.dimensions, c.child) for c in
                        abstract_layout.children]
        new_layout = dlt.AbstractLayoutAttr(new_children)
        for m in mems:
            new_layout = dlt.MemberLayoutAttr(new_layout, m)
        return new_layout

    def _try_sparse(self, abstract_layout: dlt.AbstractLayoutAttr, must_use: set[dlt.DimensionAttr] = None) -> list[dlt.Layout]:
        layouts = []
        if len(dims := abstract_layout.common_abstract_dimensions()) > 1:
            # print(f"Common Abstract Dims > 1. dims: {[d.dimensionName for d in dims]}")
            for direct in self._subsets(list(dims)):
                rest = dims.difference(direct)
                # print(f"direct: {[d.dimensionName for d in direct]}, rest: {[d.dimensionName for d in rest]}")
                if len(rest) > 0:
                    for sparse in self._subsets(list(rest)):
                        # print(f"sparse: {[d.dimensionName for d in sparse]}")
                        if len(sparse) > 0 and (must_use is None or must_use.issubset(set(sparse)|set(direct))):
                            abstract_dims = rest.difference(sparse)
                            # print(f"abstract_dims: {[d.dimensionName for d in abstract_dims]}")
                            for abstract_child_dims in self._subsets(list(abstract_dims)):
                                # print(f"abstract_child_dims: {[d.dimensionName for d in abstract_child_dims]}")
                                layouts.append(_make_sparse_layout(abstract_layout, list(direct), list(sparse), list(abstract_child_dims)))
        return layouts

    def _try_dense(self, abstract_layout: dlt.AbstractLayoutAttr, must_use: dlt.DimensionAttr = None) -> list[dlt.Layout]:
        layouts = []
        if len(dims := abstract_layout.common_abstract_dimensions()) > 0:
            if must_use is not None:
                assert must_use in dims
                dims = [must_use]

            for dim in dims:
                new_abstract = dlt.AbstractLayoutAttr([dlt.AbstractChildAttr(c.member_specifiers, c.dimensions.without(dim), c.child) for c in abstract_layout.children])
                dense_layout = dlt.DenseLayoutAttr(new_abstract, dim)
                layouts.append(dense_layout)
        return layouts



    # def _reify_abstract_child(self, abstract_child: dlt.AbstractChildAttr) -> set[dlt.AbstractChildAttr]:
    #     abstract_child_replacements = set()
    #     # for member in abstract_child.member_specifiers:
    #     #     member_layout = dlt.MemberLayoutAttr(abstract_child.child, member)
    #     #     abstract_child_replacements.add(dlt.AbstractChildAttr(abstract_child.member_specifiers.without(member), abstract_child.dimensions, member_layout))
    #     if len(abstract_child.member_specifiers) > 0:
    #         member = abstract_child.member_specifiers.take()
    #         member_layout = dlt.MemberLayoutAttr(abstract_child.child, member)
    #         abstract_child_replacements.add(
    #             dlt.AbstractChildAttr(abstract_child.member_specifiers.without(member), abstract_child.dimensions,
    #                                   member_layout))
    #
    #     if len(abstract_child.member_specifiers) == 0:
    #         for dimension in abstract_child.dimensions:
    #             dense_layout = dlt.DenseLayoutAttr(abstract_child.child, dimension)
    #             abstract_child_replacements.add(
    #                 dlt.AbstractChildAttr(abstract_child.member_specifiers, abstract_child.dimensions.without(dimension),
    #                                       dense_layout))
    #     return abstract_child_replacements

    def _get_index_replace_layouts(self, layout: dlt.AbstractLayoutAttr) -> list[dlt.ArithReplaceLayoutAttr]:

        extent_map = {}
        for dim in [d for child in layout.children for d in child.dimensions]:
            extent_map.setdefault(dim.extent, set()).add(dim)
        extents = []
        for e, dims in extent_map.items():
            count = sum(
                1
                for child in layout.children
                if len(dims & set(child.dimensions)) >= 1
            )
            if count > 1:
                extents.append((e, dims))

        if len(extents) == 0:
            return []

        layouts = []

        while len(extents) > 0:
            extent, dims = extents.pop()

            failed = False
            picked_list = set()
            ruled_out = set()
            children = sorted(
                [
                    (set(child.dimensions) & dims, child)
                    for child in layout.children
                ], key=lambda t: len(t[0])
            )

            for possible, child in children:
                possible = possible - ruled_out
                if len(possible) == 0:
                    failed = True
                    break
                must_use = possible & picked_list
                if len(must_use) == 0:
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
                    break
            if (not failed) and (len(picked_list) > 1):
                layouts.extend(self._make_arith_replacement_node(layout, extent, list(picked_list)))

        return layouts

    def _make_arith_replacement_node(self,
            layout: dlt.AbstractLayoutAttr, extent: dlt.Extent, dims: list[dlt.DimensionAttr], dim_name_num=0
    ) -> list[dlt.ArithReplaceLayoutAttr]:
        all_dims_names = [
            d.dimensionName.data for d in layout.contents_type.all_dimension_attributes()
        ]
        all_member_tuples = [
            (m.structName, m.memberName)
            for m in layout.contents_type.all_member_attributes()
        ]

        dim_name_base = "_".join(sorted([d.dimensionName.data for d in dims]))
        while (dim_name := f"_D_{dim_name_base}_{dim_name_num}") in all_dims_names:
            dim_name_num += 1

        inner_dim = dlt.DimensionAttr(dim_name, extent)

        inner_members = {}
        for outer_dim in dims:
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
                assert False
            outer_dim = d.pop()
            inner_member = inner_members[outer_dim]
            new_child = dlt.AbstractChildAttr(
                child.member_specifiers.add([inner_member]),
                child.dimensions.remove([outer_dim]).add([inner_dim]),
                child.child,
            )
            new_children.append(new_child)
        abstract_node = dlt.AbstractLayoutAttr(new_children)

        # Try to insert an indexing node so that we use the inner dim usefully
        new_layouts = []
        new_layouts.extend(self._try_sparse(abstract_node, must_use={inner_dim}))
        new_layouts.extend(self._try_dense(abstract_node, must_use=inner_dim))


        replacements = [
            dlt.ArithReplacementAttr(outer_dim, inner_dim, inner_members[outer_dim])
            for outer_dim in dims
        ]

        layouts = []
        for l in new_layouts:
            index_replace_node = dlt.ArithReplaceLayoutAttr(l, replacements)
            assert index_replace_node.contents_type == layout.contents_type
            layouts.append(index_replace_node)
        return layouts

    def _permutations(self, children: list[dlt.AbstractChildAttr]):
        if len(children) > 3 :
            yield from itertools.permutations(children)
            return
        elif len(children) == 3:
            a, b, c = children[0], children[1], children[2]
            yield a,b,c
            yield a,c,b
            yield b,a,c
            yield b,c,a
            yield c,a,b
            yield c,b,a
            return
        elif len(children) == 2:
            a, b = children[0], children[1]
            yield a,b
            yield b,a
            return
        elif len(children) == 1:
            yield (children[0],)
            return
        else:
            return

    def _subset_pairs(self, children: typing.Sequence):
        if len(children) > 3 :
            children_set = set(children)
            for subset in itertools.chain.from_iterable(itertools.combinations(children, r) for r in range(len(children) + 1)):
                if len(subset) == 0 or len(subset) == len(children):
                    continue
                other_set = children_set - set(subset)
                yield subset, other_set
            return
        elif len(children) == 3:
            a, b, c = children[0], children[1], children[2]
            yield (a,), (b, c)
            yield (b,), (a, c)
            yield (c,), (a, b)
            return
        elif len(children) == 2:
            yield (children[0],), (children[1],)
            return
        else:
            return

    def _subsets(self, children: typing.Sequence):
        if len(children) > 3 :
            yield from itertools.chain.from_iterable(itertools.combinations(children, r) for r in range(len(children) + 1))
            return
        elif len(children) == 3:
            a, b, c = children[0], children[1], children[2]
            yield (a, b, c,)
            yield (a, b,)
            yield (a, c,)
            yield (b, c,)
            yield (a,)
            yield (b,)
            yield (c,)
            yield tuple()
            return
        elif len(children) == 2:
            a, b = children[0], children[1]
            yield (a, b,)
            yield (a,)
            yield (b,)
            yield tuple()
            return
        elif len(children) == 1:
            a = children[0]
            yield (a,)
            yield tuple()
            return
        else:
            yield tuple()
            return


