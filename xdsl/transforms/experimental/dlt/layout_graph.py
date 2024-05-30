import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, cast

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import (
    KnowledgeLayout,
    MemberLayoutAttr,
    AbstractLayoutAttr,
    PrimitiveLayoutAttr,
    Layout,
    DenseLayoutAttr,
    StructLayoutAttr,
)
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.transforms.experimental.dlt_ptr_type_rewriter import PtrIdentityTypeRewriter


@dataclass(frozen=True)
class Edge:
    start: StringAttr
    members: frozenset[dlt.MemberAttr]
    dimensions: frozenset[dlt.DimensionAttr]
    end: StringAttr
    # creation_points: tuple[SSAValue]
    equality: bool = False


@dataclass(frozen=True)
class ExtentConstraint:
    identifier: StringAttr
    extent: dlt.InitDefinedExtentAttr


class ConsistencyError(Exception):
    pass


class LayoutGraph:
    def __init__(self, identifier_uses: dict[StringAttr, set[SSAValue]] = None, edges: Iterable[Edge] = None, extent_constraints: Iterable[ExtentConstraint] = None):
        self.ident_count = {} if identifier_uses is None else identifier_uses
        self.edges = set() if edges is None else set(edges)
        self.extent_constraints = set() if extent_constraints is None else set(extent_constraints)


    def add_ssa_value(self, ssa: SSAValue):
        typ = ssa.type
        if not isinstance(typ, dlt.PtrType):
            raise ValueError("Layout Graph Nodes must have a dlt.PtrType type")
        ptr = cast(dlt.PtrType, typ)
        if not ptr.has_identity:
            raise ValueError("Layout Graph Nodes must have a identified dlt.PtrType")
        ident = ptr.identification
        self.ident_count.setdefault(ident, set()).add(ssa)

    def add_edge(self, start: StringAttr, members: Iterable[dlt.MemberAttr], dimensions: Iterable[dlt.DimensionAttr], end: StringAttr):
        if start not in self.ident_count:
            raise ValueError("Cannot add edge from node that isn't in the graph")
        if end not in self.ident_count:
            raise ValueError("Cannot add edge to node that isn't in the graph")
        self.edges.add(Edge(start, frozenset(members), frozenset(dimensions), end))

    def add_equality_edge(self, start: StringAttr, end: StringAttr):
        if start not in self.ident_count:
            raise ValueError("Cannot add edge from node that isn't in the graph")
        if end not in self.ident_count:
            raise ValueError("Cannot add edge to node that isn't in the graph")
        self.edges.add(Edge(start, frozenset(), frozenset(), end, equality=True))

    def add_extent_constraint(self, ident: StringAttr, extent: dlt.InitDefinedExtentAttr):
        if ident not in self.ident_count:
            raise ValueError("Cannot add extent constraint to node that isn't in the graph")
        self.extent_constraints.add(ExtentConstraint(ident, extent))

    def get_equality_groups(self) -> list[set[StringAttr]]:
        identical_groups = []
        un_seen_idents = set(self.ident_count.keys()).union({edge.start for edge in self.edges}).union({edge.end for edge in self.edges})
        while un_seen_idents:
            ident = un_seen_idents.pop()
            equality_group = {ident} | {edge.end for edge in self.edges if edge.equality and edge.start == ident} | {edge.start for edge in self.edges if edge.equality and edge.end == ident}
            un_seen_idents.difference_update(equality_group)
            new = True
            for group in identical_groups:
                if any(i in group for i in equality_group):
                    group.update(equality_group)
                    new = False
            if new:
                identical_groups.append(equality_group)
        # check there aren't duplicates
        assert sum([len(group) for group in identical_groups]) == len({i for group in identical_groups for i in group})
        return identical_groups

    def get_entry_layouts(self) -> dict[StringAttr, dlt.Layout]:
        return {k: self.get_type_for(k).layout for k in self.ident_count if k not in [e.end for e in self.edges]}

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

    def get_type_map(self) -> dict[StringAttr, dlt.PtrType]:
        return {i:self.get_type_for(i) for i in self.ident_count}

    def consistent_check(self, new_types: dict[StringAttr, dlt.PtrType] = None):
        if new_types is None:
            new_types = {}
            for identity in self.ident_count:
                types = self.types_for(identity)
                if len(types) > 1:
                    return ValueError(f"no new types given, and layoutgraph is already inconsistent as there are multiple types for identity: {identity}")
                new_types[identity] = self.get_type_for(identity)
        else:
            if any(i not in new_types for i in self.ident_count):
                raise ValueError("new_types does not have a type for all the nodes in the graph")

        for edge in self.edges:
            starting_type = new_types[edge.start]
            ending_type = new_types[edge.end]
            output = starting_type.contents_type.select_members(
                edge.members
            ).select_dimensions(edge.dimensions)
            if output != ending_type.contents_type:
                raise ConsistencyError(f"Edge not satisfied - input type with edge selected has different contents type than output.\n"
                                       f"start {edge.start}: {starting_type.contents_type}\n"
                                       f"members: {edge.members}\n"
                                       f"dims: {edge.dimensions}\n"
                                       f"output expected: {output}\n"
                                       f"end {edge.end}: {ending_type.contents_type}")
            for extent in ending_type.filled_extents:
                if extent not in starting_type.filled_extents:
                    raise ConsistencyError(
                        f"Edge not satisfied - output has gained extent from nowhere: {extent}.\n"
                        f"start {edge.start} extents: {starting_type.filled_extents}\n"
                        f"end {edge.end} extents: {ending_type.filled_extents}\n")
            # TODO check layouts are compatible
            if not ptr_can_derive_to(starting_type, ending_type, edge.members, edge.dimensions):
                ptr_can_derive_to(starting_type, ending_type, edge.members, edge.dimensions)
                raise ConsistencyError(
                    f"Edge not satisfied - Cannot derive ending type from starting type\n"
                    f"start {edge.start}: {starting_type}\n"
                    f"end {edge.end} extents: {ending_type}\n")
        for extent_constraint in self.extent_constraints:
            ptr_type = new_types[extent_constraint.identifier]
            if extent_constraint.extent not in ptr_type.filled_extents:
                f"extent constraint not satisfied - {extent_constraint.extent} not found in {extent_constraint.identifier}.\n"
                f"type: {ptr_type}\n"
        return True

    def is_consistent(self, new_types: dict[StringAttr, dlt.PtrType] = None):
        try:
            check = self.consistent_check(new_types)
            return check
        except ConsistencyError as e:
            return False



    def use_types(self, new_types: dict[StringAttr, dlt.PtrType]):
        if any(i not in new_types for i in self.ident_count):
            raise ValueError("new_types does not have a type for all the nodes in the graph")
        type_rewriters = []
        for ident, ptr_type in new_types.items():
            type_rewriters.append(PtrIdentityTypeRewriter(ptr_type, [ident]))

        type_rewriter = PatternRewriteWalker(GreedyRewritePatternApplier(type_rewriters))
        return type_rewriter

    # def propagate_types(self, new_types: dict[StringAttr, dlt.PtrType]):
    #     current_mapping = dict(new_types)
    #     change = True
    #     while change:
    #         change = False
    #         for ident, ptr_type in current_mapping.items():
    #             for edge in [e for e in self.edges if e.start == ident]:
    #                 if edge.end not in current_mapping:
    #                     current_mapping[edge.end] = self.get_type_for(edge.end)
    #                     change = True
    #                 end_type = current_mapping[edge.end]
    #                 new_end_type = self.propagate_ptr(ptr_type, edge, end_type, current_types=)
    #                 if new_end_type != end_type:
    #                     change = True
    #                     current_mapping[edge.end] = new_end_type

    def propagate_type(self, ident: StringAttr, current_types: dict[StringAttr, dlt.PtrType]):
        assert ident in current_types
        assert all(i in current_types for i in self.ident_count)

        for edge in [e for e in self.edges if e.start == ident]:
            start_pointer = current_types[ident]
            end_pointer = current_types[edge.end]
            new_end = self.propagate_ptr(start_pointer, edge, end_pointer, current_types)
            if new_end != end_pointer:
                current_types[edge.end] = new_end
                self.backpropagate_type(edge.end, current_types)
                self.propagate_type(edge.end, current_types)


    def backpropagate_type(self, ident: StringAttr, current_types: dict[StringAttr, dlt.PtrType]):
        assert ident in current_types
        assert all(i in current_types for i in self.ident_count)

        for edge in [e for e in self.edges if e.end == ident]:
            start_pointer = current_types[edge.start]
            end_pointer = current_types[ident]
            new_start = self.backpropagate_ptr(end_pointer, edge, start_pointer, current_types)
            if new_start != start_pointer:
                current_types[edge.start] = new_start
                self.propagate_type(edge.start, current_types)
                self.backpropagate_type(edge.start, current_types)

    def propagate_ptr(self, start: dlt.PtrType, edge: Edge, existing: dlt.PtrType, current_types: dict[StringAttr, dlt.PtrType]) -> dlt.PtrType:
        incoming_edges = [e for e in self.edges if e.end == existing.identification]
        allowable_members = set(start.filled_members) | set(edge.members)
        for e in incoming_edges:
            allowable_members.intersection_update(set(current_types[e.start].filled_members) | set(e.members))
        members_check = set(existing.filled_members).issubset(allowable_members)

        allowable_dimensions = set(start.filled_dimensions) | set(edge.dimensions)
        for e in incoming_edges:
            allowable_dimensions.intersection_update(set(current_types[e.start].filled_dimensions) | set(e.dimensions))
        dimensions_check = set(existing.filled_dimensions).issubset(allowable_dimensions)

        allowable_extents = set(start.filled_extents)
        for e in incoming_edges:
            allowable_extents.intersection_update(set(current_types[e.start].filled_extents))
        extents_check = set(existing.filled_extents).issubset(allowable_extents)

        derive_check = ptr_can_derive_to(start, existing, edge.members, edge.dimensions)
        if members_check and dimensions_check and extents_check and derive_check:
            return existing
        ident = start.identification
        members = set(start.filled_members) | set(edge.members)
        dimensions = set(start.filled_dimensions) | set(edge.dimensions)
        extents = set(start.filled_extents)
        new_layout, members_to_select, dimensions_to_select = minimal_reduce(start.layout, members, dimensions, extents, allowable_members, allowable_dimensions, allowable_extents)
        new_filled_dims = ArrayAttr([d for d in existing.filled_dimensions if d in dimensions_to_select] + list(dimensions_to_select - set(existing.filled_dimensions)))
        new_filled_extents = ArrayAttr([e for e in existing.filled_extents if e in allowable_extents] + list(allowable_extents - set(existing.filled_extents)))
        new_end_type = dlt.PtrType(existing.contents_type, new_layout, dlt.SetAttr(members_to_select), new_filled_dims, new_filled_extents, identity=existing.identification)
        return new_end_type

    def backpropagate_ptr(self, end: dlt.PtrType, edge: Edge, existing: dlt.PtrType, current_types: dict[StringAttr, dlt.PtrType]) -> dlt.PtrType:
        members = (set(existing.filled_members) | edge.members) - set(end.filled_members)
        dimensions = (set(existing.filled_dimensions) | edge.dimensions) - set(end.filled_dimensions)
        new_layout = embed_layout_in(end.layout, existing.layout, members, dimensions, set(existing.filled_extents))
        new_start_type = existing.with_new_layout(new_layout, preserve_ident=True)
        return new_start_type



    def get_transitive_extent_constraints(self, ident: StringAttr, seen: set[StringAttr] = None) -> list[ExtentConstraint]:
        seen = set() if seen is None else seen
        if ident in seen:
            return []
        seen.add(ident)
        base = [c for c in self.extent_constraints if ident == c.identifier]
        children = [c for e in self.edges if ident == e.start for c in self.get_transitive_extent_constraints(e.end, seen)]
        return base + children


class InConsistentLayoutException(Exception):
    pass


def embed_layout_in(new_child_layout: Layout, parent_layout: Layout, members: set[dlt.MemberAttr], dimensions: set[dlt.DimensionAttr], extents: set[dlt.InitDefinedExtentAttr]) -> Layout:
    if members or dimensions:
        match parent_layout:
            case PrimitiveLayoutAttr():
                raise InConsistentLayoutException()
            case KnowledgeLayout():
                kl = cast(KnowledgeLayout, parent_layout)
                return kl.from_new_children([embed_layout_in(new_child_layout, kl.get_child(), members, dimensions, extents)])
            case DenseLayoutAttr():
                dl = cast(DenseLayoutAttr, parent_layout)
                if dl.dimension not in dimensions:
                    raise InConsistentLayoutException()
                return dl.from_new_children([embed_layout_in(new_child_layout, dl.child, members, dimensions-{dl.dimension}, extents)])
            case StructLayoutAttr():
                sl = cast(StructLayoutAttr, parent_layout)
                children = []
                modified_child = None
                for a_child in sl.get_children():
                    if a_child.contents_type.has_selectable(members, dimensions) and \
                            a_child.contents_type.with_selection(members, dimensions) == new_child_layout.contents_type:
                        if modified_child is not None:
                            raise InConsistentLayoutException()
                        modified_child = embed_layout_in(new_child_layout, a_child, members, dimensions, extents)
                        children.append(modified_child)
                    else:
                        children.append(a_child)
                if modified_child is None:
                    raise InConsistentLayoutException()
                return sl.from_new_children(children)
            case MemberLayoutAttr():
                ml = cast(MemberLayoutAttr, parent_layout)
                if ml.member_specifier not in members:
                    raise InConsistentLayoutException()
                return ml.from_new_children([embed_layout_in(new_child_layout, ml.child, members-{ml.member_specifier}, dimensions, extents)])
            case AbstractLayoutAttr():
                al = cast(AbstractLayoutAttr, parent_layout)
                # possible_children = [
                #     child for child in al.children
                #     if child.contents_type.has_selectable(members, dimensions)
                # ]
                # if len(possible_children) > 1 or len(possible_children) == 0:
                #     raise InConsistentLayoutException()

                children = []
                modified_child = None
                for a_child in al.children:
                    if a_child.contents_type.has_selectable(members, dimensions) and \
                            a_child.contents_type.with_selection(members, dimensions) == new_child_layout.contents_type:
                        if modified_child is not None:
                            raise InConsistentLayoutException()
                        abstract_members = set(a_child.member_specifiers) - members
                        abstract_dimensions = set(a_child.dimensions) - dimensions
                        new_child_layout_sub_tree, ms, ds = minimal_reduce(new_child_layout, abstract_members, abstract_dimensions, set(), set(), set(), extents)
                        assert len(ms) == 0 and len(ds) == 0
                        embedded_subtree = embed_layout_in(new_child_layout_sub_tree, a_child.child, set(), set(), extents)
                        if embedded_subtree != new_child_layout_sub_tree:
                            raise InConsistentLayoutException()
                        modified_child = dlt.AbstractChildAttr(abstract_members, abstract_dimensions,
                                                               new_child_layout)
                        children.append(modified_child)
                    else:
                        children.append(a_child)
                if modified_child is None:
                    raise InConsistentLayoutException()
                return AbstractLayoutAttr(children)
            case _:
                raise NotImplementedError()
    else:
        pass
        child_side = new_child_layout
        parent_side = parent_layout
        if child_side.contents_type != parent_side.contents_type:
            raise InConsistentLayoutException()
        if child_side == parent_side:
            return child_side
        elif child_side.node_matches(parent_side):
            assert len(child_side.get_children()) == len(parent_side.get_children())
            return parent_side.from_new_children([embed_layout_in(cs_c, ps_c, set(), set(), extents) for cs_c, ps_c in zip(child_side.get_children(), parent_side.get_children())])
        elif isinstance(parent_side, AbstractLayoutAttr):
            # if the parent side is Abstract then we check that the children layouts of the parent side can have the minimally reduced children of the child side embbed in them directly.
            # Simply this checks that the children of the parent side are correctly possitioned sub-trees of the new layout (but also accounts for more possible abstract layouts further down)
            al = cast(AbstractLayoutAttr, parent_side)
            for a_child in al.children:
                abstract_members = set(a_child.member_specifiers)
                abstract_dimensions = set(a_child.dimensions)

                new_child_layout_sub_tree, ms, ds = minimal_reduce(child_side, abstract_members,
                                                                   abstract_dimensions, set(), set(), set(), extents)
                new_child_layout_sub_tree = structural_reduce_to(new_child_layout_sub_tree, a_child.contents_type)
                assert len(ms) == 0 and len(ds) == 0
                embedded_subtree = embed_layout_in(new_child_layout_sub_tree, a_child.child, set(), set(), extents)
                if embedded_subtree != new_child_layout_sub_tree:
                    raise InConsistentLayoutException()
            if isinstance(child_side, AbstractLayoutAttr):
                return child_side
            else:
                return AbstractLayoutAttr([dlt.AbstractChildAttr([],[],child_side)])
        else:
            raise InConsistentLayoutException()


def structural_reduce_to(layout: dlt.Layout, dlt_type: dlt.TypeType) -> None | dlt.Layout:
    if layout.contents_type == dlt_type:
        return layout
    elif not layout.contents_type.has_selectable_type(dlt_type):
        return None
    else:
        match layout:
            case PrimitiveLayoutAttr():
                return None
            case KnowledgeLayout():
                kl = cast(KnowledgeLayout, layout)
                return structural_reduce_to(kl.get_child(), dlt_type)
            case DenseLayoutAttr():
                return None
            case AbstractLayoutAttr():
                al = cast(AbstractLayoutAttr, layout)
                children = {l
                            for c in al.children
                            if len(c.member_specifiers) == 0 and len(c.dimensions) == 0
                            if (l := structural_reduce_to(c.child, dlt_type)) is not None
                            }
                if children:
                    assert len(children) == 1
                    return children.pop()
                else:
                    return None

            case StructLayoutAttr():
                sl = cast(StructLayoutAttr, layout)
                children = {l for c in sl.children if (l := structural_reduce_to(c, dlt_type)) is not None}
                if children:
                    assert len(children) == 1
                    return children.pop()
                else:
                    return None
            case MemberLayoutAttr():
                return None
            case _:
                raise NotImplementedError()




def has_reduction_to(parent_layout: dlt.Layout, child_layout: dlt.Layout, members: set[dlt.MemberAttr], dimensions: set[dlt.DimensionAttr], extents: set[dlt.InitDefinedExtentAttr]) -> bool:
    if members or dimensions:
        parent_layout = minimal_reduce(parent_layout, members, dimensions, set(), set(), set(), extents)
    if parent_layout == child_layout:
        return True


def minimal_reduce(layout: dlt.Layout,
                   members: set[dlt.MemberAttr], dimensions: set[dlt.DimensionAttr],
                   extents: set[dlt.InitDefinedExtentAttr],
                   allowable_members: set[dlt.MemberAttr], allowable_dimensions: set[dlt.DimensionAttr],
                   allowable_extents: set[dlt.InitDefinedExtentAttr]) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
    possible_extents_check = all(e in allowable_extents for e in layout.get_all_init_base_extents())
    if not possible_extents_check:
        raise InConsistentLayoutException()
    contents_type = layout.contents_type
    must_remove_members = members - allowable_members
    must_remove_dimensions = dimensions - allowable_dimensions
    must_remove_extents = extents - allowable_extents
    if must_remove_members or must_remove_dimensions or must_remove_extents:
        match layout:
            case PrimitiveLayoutAttr():
                raise InConsistentLayoutException()
            case KnowledgeLayout():
                kl = cast(KnowledgeLayout, layout)
                return minimal_reduce(kl.get_child(), members, dimensions, extents, allowable_members, allowable_dimensions, allowable_extents)
            case DenseLayoutAttr():
                dl = cast(DenseLayoutAttr, layout)
                if dl.dimension not in dimensions:
                    raise InConsistentLayoutException()
                return minimal_reduce(dl.child, members, dimensions-{dl.dimension}, extents, allowable_members, allowable_dimensions, allowable_extents)
            case AbstractLayoutAttr():
                al = cast(AbstractLayoutAttr, layout)
                possible_children = [
                    child for child in al.children
                    if child.contents_type.has_selectable(members, dimensions)
                ]
                if len(possible_children) > 1 or len(possible_children) == 0:
                    raise InConsistentLayoutException()
                child = possible_children[0]
                members_to_select = members.difference(child.member_specifiers)
                dimensions_to_select = dimensions.difference(child.dimensions)
                return minimal_reduce(child.child, members_to_select, dimensions_to_select, extents, allowable_members, allowable_dimensions, allowable_extents)
            case StructLayoutAttr():
                sl = cast(StructLayoutAttr, layout)
                possible_children = [
                    child for child in sl.children
                    if child.contents_type.has_selectable(members, dimensions)
                ]
                if len(possible_children) > 1 or len(possible_children) == 0:
                    raise InConsistentLayoutException()
                child = possible_children[0]
                return minimal_reduce(child, members, dimensions, extents, allowable_members, allowable_dimensions, allowable_extents)
            case MemberLayoutAttr():
                ml = cast(MemberLayoutAttr, layout)
                if ml.member_specifier not in members:
                    raise InConsistentLayoutException()
                members_to_select = members.difference({ml.member_specifier})
                child = ml.child
                return minimal_reduce(child, members_to_select, dimensions, extents, allowable_members, allowable_dimensions,
                                      allowable_extents)
            case _:
                raise NotImplementedError()
    else:
        return layout, members, dimensions


def ptr_can_derive_to(
    starting_point: dlt.PtrType,
    end_point: dlt.PtrType,
    members: frozenset[dlt.MemberAttr],
    dimensions: frozenset[dlt.DimensionAttr],
) -> bool:
    starting_point = starting_point.without_identification()
    end_point = end_point.without_identification()
    resulting_contents = starting_point.contents_type.select_members(
        members
    ).select_dimensions(dimensions)
    if resulting_contents != end_point.contents_type:
        raise InConsistentLayoutException(
            "starting_point with members and dimensions selected gives a different contentsType to end_point"
        )
    if starting_point.as_not_base() == end_point:
        # if these ptrs are the same (bar not being a base ptr) all is well
        return True
    members_to_select = set(members).union(starting_point.filled_members)
    dimensions_to_select = set(dimensions).union(starting_point.filled_dimensions)
    extents_to_use = set(starting_point.filled_extents)
    check = can_layout_derive_to(starting_point.layout, starting_point,
                         end_point.layout, end_point,
                         members_to_select, dimensions_to_select,
                         extents_to_use, set())
    # return layout_can_derive_to(
    #     starting_point.layout,
    #     starting_point,
    #     end_point,
    #     members_to_select,
    #     dimensions_to_select,
    #     extents_to_use,
    #     set(),
    # )
    return check

def can_layout_derive_to(
    current_starting_layout: Layout,
    starting_point: dlt.PtrType,
    current_end_layout: Layout,
    end_point: dlt.PtrType,
    members_to_select: set[dlt.MemberAttr],
    dimensions_to_select: set[dlt.DimensionAttr],
    extents_to_use: set[dlt.InitDefinedExtentAttr],
    enclosing_knowledge: set[dlt.KnowledgeLayout],
):
    while current_starting_layout != current_end_layout:
        match current_starting_layout:
            case PrimitiveLayoutAttr():
                return False
            case KnowledgeLayout():
                kl = cast(KnowledgeLayout, current_starting_layout)
                current_starting_layout = kl.get_child()
            case DenseLayoutAttr():
                dl = cast(DenseLayoutAttr, current_starting_layout)
                if dl.dimension not in dimensions_to_select:
                    return False
                dimensions_to_select = dimensions_to_select.difference({dl.dimension})
                current_starting_layout = dl.child
            case AbstractLayoutAttr():
                al = cast(AbstractLayoutAttr, current_starting_layout)
                possible_children = [
                    child for child in al.children
                    if child.contents_type.has_selectable(members_to_select, dimensions_to_select)
                ]
                if len(possible_children) > 1 or len(possible_children) == 0:
                    return False
                child = possible_children[0]
                members_to_select = members_to_select.difference(child.member_specifiers)
                dimensions_to_select = dimensions_to_select.difference(child.dimensions)
                current_starting_layout = child.child
            case StructLayoutAttr():
                sl = cast(StructLayoutAttr, current_starting_layout)
                possible_children = [
                    child for child in sl.children
                    if child.contents_type.has_selectable(members_to_select, dimensions_to_select)
                ]
                if len(possible_children) > 1 or len(possible_children) == 0:
                    return False
                child = possible_children[0]
                current_starting_layout = child
            case MemberLayoutAttr():
                ml = cast(MemberLayoutAttr, current_starting_layout)
                if ml.member_specifier not in members_to_select:
                    return False
                members_to_select = members_to_select.difference({ml.member_specifier})
                current_starting_layout = ml.child
            case _:
                raise NotImplementedError()
    return True

# def end_point_satisfies_constraints(
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
# ) -> bool:
#     if any(e_ext not in extents_to_use for e_ext in end_point.filled_extents):
#         raise InConsistentLayoutException(
#             "endpoint cannot use extents that do not exist in the starting_point or are produced by the layout nodes consumed"
#         )
#     if any(member not in end_point.filled_members for member in members_to_select):
#         raise InConsistentLayoutException(
#             "all members that still need selecting must be selected in end_point"
#         )
#     if any(
#         dimension not in end_point.filled_dimensions
#         for dimension in dimensions_to_select
#     ):
#         raise InConsistentLayoutException(
#             "all dimensions that still need selecting must be in end_point"
#         )
#     return True
#
# def can_layout_derive_to(
#     current_starting_layout: Layout,
#     starting_point: dlt.PtrType,
#     current_end_layout: Layout,
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
# ):
#     if current_starting_layout == current_end_layout:
#         return end_point_satisfies_constraints(
#             end_point, members_to_select, dimensions_to_select, extents_to_use
#         )
#     match current_end_layout:
#         case KnowledgeLayout():
#             knowledge_node = cast(KnowledgeLayout, current_starting_layout)
#             if knowledge_node.can_be_injected(enclosing_knowledge):
#                 new_enclosing_knowledge = enclosing_knowledge.union({knowledge_node})
#                 return can_layout_derive_to(
#                     current_starting_layout,
#                     starting_point,
#                     knowledge_node.get_child(),
#                     end_point,
#                     members_to_select,
#                     dimensions_to_select,
#                     extents_to_use,
#                     new_enclosing_knowledge,
#                 )
#     match current_starting_layout:
#         case KnowledgeLayout():
#             s_knowledge = cast(KnowledgeLayout, current_starting_layout)
#             if isinstance(current_end_layout, KnowledgeLayout):
#                 e_knowledge = cast(KnowledgeLayout, current_end_layout)
#                 if s_knowledge.matches(e_knowledge):
#                     return can_layout_derive_to(
#                         s_knowledge.get_child(),
#                         starting_point,
#                         e_knowledge.get_child(),
#                         end_point,
#                         members_to_select,
#                         dimensions_to_select,
#                         extents_to_use,
#                         enclosing_knowledge.union({e_knowledge}),
#                     )
#             if s_knowledge.can_be_dropped(enclosing_knowledge):
#                 return can_layout_derive_to(
#                     s_knowledge.get_child(),
#                     starting_point,
#                     current_end_layout,
#                     end_point,
#                     members_to_select,
#                     dimensions_to_select,
#                     extents_to_use,
#                     enclosing_knowledge,
#                 )
#             else:
#                 raise InConsistentLayoutException(
#                     "start side knowledge does not match end side, and it cannot be dropped here"
#                 )
#         case DenseLayoutAttr():
#             s_dense = cast(DenseLayoutAttr, current_starting_layout)
#             new_dimensions_to_select = dimensions_to_select.difference(
#                 [s_dense.dimension]
#             )
#             current_starting_layout = s_dense.child
#             if isinstance(current_end_layout, DenseLayoutAttr) and s_dense.dimension == (e_dense := cast(DenseLayoutAttr, current_end_layout)).dimension:
#                 if s_dense.contents_type != e_dense.contents_type:
#                     raise InConsistentLayoutException(
#                         "dense layouts have same dimension but different content types"
#                     )
#                 current_end_layout = e_dense.child
#             else:
#                 if s_dense.dimension not in dimensions_to_select:
#                     raise InConsistentLayoutException(
#                         "starting side dense layout is neither selected or found on the ending side."
#                     )
#             return can_layout_derive_to(
#                 current_starting_layout,
#                 starting_point,
#                 current_end_layout,
#                 end_point,
#                 members_to_select,
#                 new_dimensions_to_select,
#                 extents_to_use,
#                 enclosing_knowledge,
#             )
#         case AbstractLayoutAttr():
#             s_abstract = cast(AbstractLayoutAttr, current_starting_layout)
#             if isinstance(current_end_layout, AbstractLayoutAttr):
#                 e_abstract = cast(AbstractLayoutAttr, current_end_layout)
#                 if len(s_abstract.children) == len(e_abstract.children) and \
#                     all(s_c.child.content_type == e_c.child.content_type and \
#                         s_c.member_specifiers == e_c.member_specifiers and \
#                         s_c.dimension
#                         for s_c, e_c in zip(s_abstract.children, e_abstract.children))
#             for child in s_abstract.children:
#                 child.
#             new_members_to_select = members_to_select.difference(
#                 s_abstract.member_specifiers
#             )
#             new_dimensions_to_select = dimensions_to_select.difference(
#                 s_abstract.dimensions
#             )
#             if isinstance(current_end_layout, AbstractLayoutAttr):
#                 e_abstract = cast(AbstractLayoutAttr, current_end_layout)
#                 if (
#                     s_abstract.dimensions == e_abstract.dimensions
#                     and s_abstract.member_specifiers == e_abstract.member_specifiers
#                     and {s_c.content_type for s_c in s_abstract.children}
#                     == {e_c.content_type for e_c in e_abstract.children}
#                 ):
#                     end_side_children = list(e_abstract.children)
#                     end_side_content_types = [
#                         e_c.content_type for e_c in end_side_children
#                     ]
#                     result = True
#                     for s_c in s_abstract.children:
#                         e_c = end_side_children[
#                             end_side_content_types.index(s_c.content_type)
#                         ]
#                         result &= can_layout_derive_to(
#                             s_c,
#                             starting_point,
#                             e_c,
#                             end_point,
#                             new_members_to_select,
#                             new_dimensions_to_select,
#                             extents_to_use,
#                             enclosing_knowledge,
#                         )
#                     return result
#
#             pass
#
#
# @functools.singledispatch
# def layout_can_derive_to(
#     current_layout: Layout,
#     starting_point: dlt.PtrType,
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
# ) -> bool:
#     raise NotImplementedError()
#
#
# @layout_can_derive_to.register
# def _(
#     current_layout: NamedLayoutAttr,
#     starting_point: dlt.PtrType,
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
# ) -> bool:
#     print("Named Layout Attr")
#     raise NotImplementedError()
#
#
# @layout_can_derive_to.register
# def _(
#     current_layout: AbstractLayoutAttr,
#     starting_point: dlt.PtrType,
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
# ) -> bool:
#     print("Abstract Layout Attr")
#     if current_layout == _skip_knowledge_layout_info(end_point.layout):
#         return end_point_satisfies_constraints(
#             end_point, members_to_select, dimensions_to_select, extents_to_use
#         )
#         # we have found the final layout that needs checking
#     else:
#         if current_layout.dimension not in dimensions_to_select:
#             raise ValueError()
#         child_layout = current_layout.child
#         new_dimensions_to_select = dimensions_to_select.difference(
#             [current_layout.dimension]
#         )
#         return layout_can_derive_to(
#             child_layout,
#             starting_point,
#             end_point,
#             members_to_select,
#             new_dimensions_to_select,
#             extents_to_use,
#             enclosing_knowledge,
#         )
#
#
# @layout_can_derive_to.register
# def _(
#     current_layout: PrimitiveLayoutAttr,
#     starting_point: dlt.PtrType,
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
# ) -> bool:
#     print("Primitive Layout Attr")
#     return False
#
#
# @layout_can_derive_to.register
# def _(
#     current_layout: DenseLayoutAttr,
#     starting_point: dlt.PtrType,
#     end_point: dlt.PtrType,
#     members_to_select: set[dlt.MemberAttr],
#     dimensions_to_select: set[dlt.DimensionAttr],
#     extents_to_use: set[dlt.InitDefinedExtentAttr],
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
# ) -> bool:
#     print("Dense Layout Attr")
#     if current_layout == _skip_knowledge_layout_info(
#         end_point.layout, enclosing_knowledge, that_can_be_injected=True
#     ):
#         return end_point_satisfies_constraints(
#             end_point, members_to_select, dimensions_to_select, extents_to_use
#         )
#         # we have found the final layout that needs checking
#     else:
#         if current_layout.dimension not in dimensions_to_select:
#             raise ValueError()
#         child_layout = current_layout.child
#         new_dimensions_to_select = dimensions_to_select.difference(
#             [current_layout.dimension]
#         )
#         return layout_can_derive_to(
#             child_layout,
#             starting_point,
#             end_point,
#             members_to_select,
#             new_dimensions_to_select,
#             extents_to_use,
#             enclosing_knowledge,
#         )
#
#
# def _skip_knowledge_layout_info(
#     layout: Layout,
#     enclosing_knowledge: set[dlt.KnowledgeLayout],
#     that_can_be_dropped=False,
#     that_can_be_injected=False,
# ) -> Layout:
#     if (
#         that_can_be_dropped
#         and isinstance(layout, KnowledgeLayout)
#         and layout.can_be_dropped(enclosing_knowledge)
#     ):
#         return _skip_knowledge_layout_info(
#             layout.get_child(),
#             enclosing_knowledge,
#             that_can_be_dropped,
#             that_can_be_injected,
#         )
#     elif (
#         that_can_be_injected
#         and isinstance(layout, KnowledgeLayout)
#         and layout.can_be_injected(enclosing_knowledge)
#     ):
#         return _skip_knowledge_layout_info(
#             layout.get_child(),
#             enclosing_knowledge,
#             that_can_be_dropped,
#             that_can_be_injected,
#         )
#     else:
#         return layout
