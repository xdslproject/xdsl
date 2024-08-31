import math
from dataclasses import dataclass
from typing import Iterable, Self, cast

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import IterIdent, PtrIdent
from xdsl.ir import BlockArgument, SSAValue, Use
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.transforms.experimental.dlt.layout_manipulation import (
    InConsistentLayoutException,
    Manipulator,
)
from xdsl.transforms.experimental.dlt.dlt_ptr_type_rewriter import (
    PtrIdentityBulkTypeRewriter,
)


@dataclass(frozen=True)
class Edge:
    start: PtrIdent
    members: frozenset[dlt.MemberAttr]
    dimensions: frozenset[dlt.DimensionAttr]
    end: PtrIdent
    # creation_points: tuple[SSAValue]
    equality: bool = False
    iteration_ident: IterIdent | None = None


@dataclass(frozen=True)
class ExtentConstraint:
    identifier: PtrIdent
    extent: dlt.InitDefinedExtentAttr


class DLTPtrConsistencyError(Exception):
    pass


class LayoutGraph:
    def __init__(
        self,
        identifier_uses: dict[PtrIdent, set[SSAValue]] = None,
        edges: Iterable[Edge] = None,
        extent_constraints: Iterable[ExtentConstraint] = None,
    ):
        self.ident_count = {} if identifier_uses is None else identifier_uses
        self.edges = set() if edges is None else set(edges)
        self.extent_constraints = (
            set() if extent_constraints is None else set(extent_constraints)
        )

    def matches(self, other: Self):
        if not isinstance(other, LayoutGraph):
            return False
        if self.ident_count != other.ident_count:
            if set(self.ident_count.keys()) != set(other.ident_count.keys()):
                return False
            for ident, ssa_set in self.ident_count.items():
                if len(ssa_set) != len(other.ident_count[ident]):
                    return False

        if self.edges != other.edges:
            return False
        if self.extent_constraints != other.extent_constraints:
            return False
        return True

    def add_ssa_value(self, ssa: SSAValue):
        typ = ssa.type
        if not isinstance(typ, dlt.PtrType):
            raise ValueError("Layout Graph Nodes must have a dlt.PtrType type")
        ptr = cast(dlt.PtrType, typ)
        if not ptr.has_identity:
            raise ValueError("Layout Graph Nodes must have a identified dlt.PtrType")
        ident = ptr.identification
        self.ident_count.setdefault(ident, set()).add(ssa)

    def add_edge(
        self,
        start: PtrIdent,
        members: Iterable[dlt.MemberAttr],
        dimensions: Iterable[dlt.DimensionAttr],
        end: PtrIdent,
        iteration_ident: IterIdent | None,
    ):
        if start not in self.ident_count:
            raise ValueError("Cannot add edge from node that isn't in the graph")
        if end not in self.ident_count:
            raise ValueError("Cannot add edge to node that isn't in the graph")
        self.edges.add(Edge(start, frozenset(members), frozenset(dimensions), end, iteration_ident=iteration_ident))

    def add_equality_edge(self, start: PtrIdent, end: PtrIdent):
        if start not in self.ident_count:
            raise ValueError("Cannot add edge from node that isn't in the graph")
        if end not in self.ident_count:
            raise ValueError("Cannot add edge to node that isn't in the graph")
        self.edges.add(Edge(start, frozenset(), frozenset(), end, equality=True))

    def add_extent_constraint(
        self, ident: PtrIdent, extent: dlt.InitDefinedExtentAttr
    ):
        if ident not in self.ident_count:
            raise ValueError(
                "Cannot add extent constraint to node that isn't in the graph"
            )
        self.extent_constraints.add(ExtentConstraint(ident, extent))

    def get_equality_groups(self) -> list[set[PtrIdent]]:
        identical_groups = []
        un_seen_idents = (
            set(self.ident_count.keys())
            .union({edge.start for edge in self.edges})
            .union({edge.end for edge in self.edges})
        )
        while un_seen_idents:
            ident = un_seen_idents.pop()
            equality_group = (
                {ident}
                | {
                    edge.end
                    for edge in self.edges
                    if edge.equality and edge.start == ident
                }
                | {
                    edge.start
                    for edge in self.edges
                    if edge.equality and edge.end == ident
                }
            )
            un_seen_idents.difference_update(equality_group)
            new = True
            for group in identical_groups:
                if any(i in group for i in equality_group):
                    group.update(equality_group)
                    new = False
            if new:
                identical_groups.append(equality_group)
        # check there aren't duplicates
        assert sum([len(group) for group in identical_groups]) == len(
            {i for group in identical_groups for i in group}
        )
        return identical_groups

    def get_entry_layouts(self) -> dict[PtrIdent, dlt.Layout]:
        edge_enders = [e.end for e in self.edges]
        return {
            k: self.get_type_for(k).layout
            for k in self.ident_count
            if k not in edge_enders
        }

    def get_end_points(self) -> set[PtrIdent]:
        edge_starters = [e.start for e in self.edges]
        return {k for k in self.ident_count if k not in edge_starters}

    def types_for(self, ident: PtrIdent) -> set[dlt.PtrType]:
        types: set[dlt.PtrType] = set()
        for ssa in self.ident_count[ident]:
            ptr_type = ssa.type
            assert isinstance(ptr_type, dlt.PtrType)
            ptr_type = cast(dlt.PtrType, ptr_type)
            types.add(ptr_type)
        return types

    def get_type_for(self, ident: PtrIdent) -> dlt.PtrType:
        types = self.types_for(ident)
        if len(types) != 1:
            raise ValueError()
        for ptr_type in types:
            return ptr_type

    def get_type_map(self) -> dict[PtrIdent, dlt.PtrType]:
        return {i: self.get_type_for(i) for i in self.ident_count}

    def get_idents(self) -> Iterable[PtrIdent]:
        return self.ident_count.keys()

    def get_uses(self, ident: PtrIdent) -> set[Use]:
        assert ident in self.ident_count
        ssa_vals = self.ident_count[ident]
        return {u for ssa in ssa_vals for u in ssa.uses}

    def get_transitive_parents(self, ident: PtrIdent) -> set[PtrIdent]:
        idents = {ident}
        changed = True
        active_edges = set(self.edges)
        while changed:
            changed = False
            for edge in active_edges:
                if edge.end in idents:
                    if edge.start not in idents:
                        idents.add(edge.start)
                        changed = True
        return idents

    def get_transitive_closure(self, ident: PtrIdent) -> set[PtrIdent]:
        idents = {ident}
        changed = True
        active_edges = set(self.edges)
        while changed:
            changed = False
            for edge in active_edges:
                if edge.start in idents:
                    if edge.end not in idents:
                        idents.add(edge.end)
                        changed = True
        return idents

    def get_transitive_uses(self, ident: PtrIdent) -> set[Use]:
        idents = self.get_transitive_closure(ident)
        return {u for id in idents for u in self.get_uses(id)}

    def get_base_idents(self, ident: PtrIdent) -> set[tuple[PtrIdent, frozenset[dlt.MemberAttr], frozenset[dlt.DimensionAttr]]]:
        edges = [e for e in self.edges if e.end == ident]
        if len(edges) == 0:
            return {(ident, frozenset(), frozenset())}
        base_idents = set()
        for edge in edges:
            child_bases = self.get_base_idents(edge.start)
            for base, ms, ds in child_bases:
                base_idents.add((base, ms|edge.members, ds|edge.dimensions))
        return base_idents

    def check_consistency(
        self,
        new_types: dict[PtrIdent, dlt.PtrType] = None,
        non_zero_reducible_ptrs: set[PtrIdent] = None,
    ):
        if non_zero_reducible_ptrs is None:
            non_zero_reducible_ptrs = {}
        if new_types is None:
            new_types = {}
            for identity in self.ident_count:
                types = self.types_for(identity)
                if len(types) > 1:
                    return ValueError(
                        f"no new types given, and layoutgraph is already inconsistent as there are multiple types for identity: {identity}"
                    )
                new_types[identity] = self.get_type_for(identity)
        else:
            if any(i not in new_types for i in self.ident_count):
                raise ValueError(
                    "new_types does not have a type for all the nodes in the graph"
                )

        for edge in self.edges:
            starting_type = new_types[edge.start]
            ending_type = new_types[edge.end]
            output = starting_type.contents_type.select_members(
                edge.members
            ).select_dimensions(edge.dimensions)
            if output != ending_type.contents_type:
                raise DLTPtrConsistencyError(
                    f"Edge not satisfied - input type with edge selected has different contents type than output.\n"
                    f"start {edge.start}: {starting_type.contents_type}\n"
                    f"members: {edge.members}\n"
                    f"dims: {edge.dimensions}\n"
                    f"output expected: {output}\n"
                    f"end {edge.end}: {ending_type.contents_type}"
                )
            for extent in ending_type.filled_extents:
                if extent not in starting_type.filled_extents:
                    raise DLTPtrConsistencyError(
                        f"Edge not satisfied - output has gained extent from nowhere: {extent}.\n"
                        f"start {edge.start} extents: {starting_type.filled_extents}\n"
                        f"end {edge.end} extents: {ending_type.filled_extents}\n"
                    )
                        
            if not ptr_can_derive_to(
                starting_type,
                ending_type,
                edge.members,
                edge.dimensions,
                non_zero_reducible_ptrs,
            ):
                false = ptr_can_derive_to(
                    starting_type,
                    ending_type,
                    edge.members,
                    edge.dimensions,
                    non_zero_reducible_ptrs,
                )  # DEBUG
                raise DLTPtrConsistencyError(
                    f"Edge not satisfied - Cannot derive ending type from starting type\n"
                    f"start {edge.start}: {starting_type}\n"
                    f"end {edge.end}: {ending_type}\n"
                    f"edge: {edge.members}, {edge.dimensions}"
                )
        for extent_constraint in self.extent_constraints:
            ptr_type = new_types[extent_constraint.identifier]
            if extent_constraint.extent not in ptr_type.filled_extents:
                f"extent constraint not satisfied - {extent_constraint.extent} not found in {extent_constraint.identifier}.\n"
                f"type: {ptr_type}\n"
        return True

    def is_consistent(
        self,
        new_types: dict[PtrIdent, dlt.PtrType] = None,
        non_zero_reducible_ptrs: set[PtrIdent] = None,
    ):
        try:
            check = self.check_consistency(new_types, non_zero_reducible_ptrs)
            return check
        except DLTPtrConsistencyError as e:
            return False

    def use_types(self, new_types: dict[PtrIdent, dlt.PtrType]):
        if any(i not in new_types for i in self.ident_count):
            raise ValueError(
                "new_types does not have a type for all the nodes in the graph"
            )
        type_rewriter = PatternRewriteWalker(PtrIdentityBulkTypeRewriter(new_types))
        return type_rewriter

    def reduce_types(
        self,
        non_zero_reducible_ptrs: set[PtrIdent],
        current_types: dict[PtrIdent, dlt.PtrType],
    ):
        all_idents = set(self.ident_count.keys())
        entry_point_idents = set(self.get_entry_layouts().keys())
        distances = {i: math.inf for i in all_idents}
        distances.update({i: 0 for i in entry_point_idents})
        open_set = all_idents - entry_point_idents
        while open_set:
            for edge in self.edges:
                distances[edge.end] = min(
                    distances[edge.start] + 1, distances[edge.end]
                )
            min_ident = None
            min_distance = math.inf
            for open in open_set:
                if distances[open] <= min_distance:
                    min_ident = open
            open_set.remove(min_ident)
        distances_tuples = sorted([(-d, i.data, i) for i, d in distances.items()])
        idents = [i for d, i_d, i in distances_tuples]
        # idents = list(end_point_idents) + [i for i in self.ident_count if i not in end_point_idents]

        for ident in idents:
            self.reduce_type(ident, non_zero_reducible_ptrs, current_types)

    def reduce_type(
        self,
        ident: PtrIdent,
        non_zero_reducible_ptrs: set[PtrIdent],
        current_types: dict[PtrIdent, dlt.PtrType],
    ):

        print(f"reducing {ident.data}: ", end="")
        ptr_type = current_types[ident]
        if ptr_type.is_base:
            print(f"stopped - {ident.data} is base ptr")
            return
        available_members = set(ptr_type.filled_members)
        available_dimensions = set(ptr_type.filled_dimensions)

        reductions = []
        reduction = Manipulator.try_reduction(
            ptr_type.layout,
            available_members,
            available_dimensions,
            ident in non_zero_reducible_ptrs,
        )
        while reduction is not None:
            reductions.append(reduction)
            reduction = Manipulator.try_reduction(
                reduction[0], set(reduction[1]), set(reduction[2]), reduction[3]
            )

        for child in reversed(reductions):
            new_layout, left_over_members, left_over_dimensions, non_zero_reducible = child

            filled_dimensions = ArrayAttr(
                [
                    dim
                    for dim in ptr_type.filled_dimensions
                    if dim in left_over_dimensions
                ]
            )
            all_new_extents = new_layout.get_all_extents()
            filled_extents = [
                e for e in ptr_type.filled_extents if e in all_new_extents
            ]
            new_ptr = dlt.PtrType(
                ptr_type.contents_type,
                new_layout,
                dlt.SetAttr(left_over_members),
                filled_dimensions,
                filled_extents,
                ptr_type.is_base,
                ptr_type.identification,
            )
            new_types_map = dict(current_types)
            new_types_map[ident] = new_ptr
            try:
                self.propagate_type(ident, new_types_map, non_zero_reducible_ptrs)
                self.check_consistency(new_types_map, non_zero_reducible_ptrs)
            except (DLTPtrConsistencyError, InConsistentLayoutException) as e:
                print(". ", end="")
            else:
                changed = {
                    i for (i, ptr) in new_types_map.items() if current_types[i] != ptr
                }
                print("changes propagated to: " + ",".join([i.data for i in changed]))
                current_types.update(new_types_map)
                return
        print("stopped - no reductions possible")

    def propagate_type(
        self,
        ident: PtrIdent,
        current_types: dict[PtrIdent, dlt.PtrType],
        non_zero_reducible_ptrs: set[PtrIdent],
    ):
        assert ident in current_types
        assert all(i in current_types for i in self.ident_count)

        for edge in [e for e in self.edges if e.start == ident]:
            start_pointer = current_types[ident]
            end_pointer = current_types[edge.end]
            new_end = self.propagate_ptr(
                start_pointer, edge, end_pointer, current_types, non_zero_reducible_ptrs
            )
            if new_end != end_pointer:
                current_types[edge.end] = new_end
                self.backpropagate_type(edge.end, current_types, non_zero_reducible_ptrs)
                self.propagate_type(edge.end, current_types, non_zero_reducible_ptrs)

    def backpropagate_type(
        self,
        ident: PtrIdent,
        current_types: dict[PtrIdent, dlt.PtrType],
        non_zero_reducible_ptrs: set[PtrIdent],
    ):
        assert ident in current_types
        assert all(i in current_types for i in self.ident_count)

        for edge in [e for e in self.edges if e.end == ident]:
            start_pointer = current_types[edge.start]
            end_pointer = current_types[ident]
            new_start = self.backpropagate_ptr(
                end_pointer, edge, start_pointer, current_types, non_zero_reducible_ptrs
            )
            if new_start != start_pointer:
                current_types[edge.start] = new_start
                self.propagate_type(edge.start, current_types, non_zero_reducible_ptrs)
                self.backpropagate_type(edge.start, current_types, non_zero_reducible_ptrs)

    def propagate_ptr(
        self,
        start: dlt.PtrType,
        edge: Edge,
        existing: dlt.PtrType,
        current_types: dict[PtrIdent, dlt.PtrType],
        non_zero_reducible_ptrs: set[PtrIdent],
    ) -> dlt.PtrType:
        incoming_edges = [e for e in self.edges if e.end == existing.identification]
        allowable_members = set(start.filled_members) | set(edge.members)
        for e in incoming_edges:
            allowable_members.intersection_update(
                set(current_types[e.start].filled_members) | set(e.members)
            )
        members_check = set(existing.filled_members).issubset(allowable_members)

        allowable_dimensions = set(start.filled_dimensions) | set(edge.dimensions)
        for e in incoming_edges:
            allowable_dimensions.intersection_update(
                set(current_types[e.start].filled_dimensions) | set(e.dimensions)
            )
        dimensions_check = set(existing.filled_dimensions).issubset(
            allowable_dimensions
        )

        allowable_extents = set(start.filled_extents)
        for e in incoming_edges:
            allowable_extents.intersection_update(
                set(current_types[e.start].filled_extents)
            )
        extents_check = set(existing.filled_extents).issubset(allowable_extents)

        derive_check = ptr_can_derive_to(
            start,
            existing,
            edge.members,
            edge.dimensions,
            non_zero_reducible_ptrs,
        )
        if members_check and dimensions_check and extents_check and derive_check:
            return existing
        ident = start.identification
        members = set(start.filled_members) | set(edge.members)
        dimensions = set(start.filled_dimensions) | set(edge.dimensions)
        extents = set(start.filled_extents)
        new_layout, members_to_select, dimensions_to_select = (
            Manipulator.minimal_reduction(
                start.layout,
                members,
                dimensions,
                extents,
                allowable_members,
                allowable_dimensions,
                allowable_extents,
                existing.identification in non_zero_reducible_ptrs,
            )
        )
        new_filled_dims = ArrayAttr(
            [d for d in existing.filled_dimensions if d in dimensions_to_select]
            + list(dimensions_to_select - set(existing.filled_dimensions))
        )
        new_filled_extents = ArrayAttr(
            [e for e in existing.filled_extents if e in allowable_extents]
            + list(allowable_extents - set(existing.filled_extents))
        )
        new_end_type = dlt.PtrType(
            existing.contents_type,
            new_layout,
            dlt.SetAttr(members_to_select),
            new_filled_dims,
            new_filled_extents,
            identity=existing.identification,
        )
        return new_end_type

    def backpropagate_ptr(
        self,
        end: dlt.PtrType,
        edge: Edge,
        existing: dlt.PtrType,
        current_types: dict[PtrIdent, dlt.PtrType],
        non_zero_reducible_ptrs: set[PtrIdent],
    ) -> dlt.PtrType:
        members = (set(existing.filled_members) | edge.members) - set(
            end.filled_members
        )
        dimensions = (set(existing.filled_dimensions) | edge.dimensions) - set(
            end.filled_dimensions
        )
        new_layout = Manipulator.embed_layout_in(
            end.layout,
            existing.layout,
            members,
            dimensions,
            set(existing.filled_extents),
        )
        new_start_type = existing.with_new_layout(new_layout, preserve_ident=True)
        return new_start_type

    def get_transitive_extent_constraints(
        self, ident: PtrIdent, seen: set[PtrIdent] = None
    ) -> list[ExtentConstraint]:
        seen = set() if seen is None else seen
        if ident in seen:
            return []
        seen.add(ident)
        base = [c for c in self.extent_constraints if ident == c.identifier]
        children = [
            c
            for e in self.edges
            if ident == e.start
            for c in self.get_transitive_extent_constraints(e.end, seen)
        ]
        return base + children


def ptr_can_derive_to(
    starting_point: dlt.PtrType,
    end_point: dlt.PtrType,
    members: frozenset[dlt.MemberAttr],
    dimensions: frozenset[dlt.DimensionAttr],
    non_zero_reducible_ptrs: set[PtrIdent],
) -> bool:
    starting_point = starting_point.without_identification()
    end_point_ident = end_point.identification
    end_point = end_point.without_identification()
    resulting_contents = starting_point.contents_type.select_members(
        members
    ).select_dimensions(dimensions)
    if resulting_contents != end_point.contents_type:
        raise InConsistentLayoutException(
            "starting_point with members and dimensions selected gives a different contentsType to end_point"
        )
    if (
        starting_point.as_not_base() == end_point
        and len(members) == 0
        and len(dimensions) == 0
    ):
        # if these ptrs are the same (bar not being a base ptr) all is well
        return True
    members_to_select = set(members).union(starting_point.filled_members)
    dimensions_to_select = set(dimensions).union(starting_point.filled_dimensions)
    extents_to_use = set(starting_point.filled_extents)
    check = Manipulator.can_layout_derive_to(
        starting_point.layout,
        starting_point,
        end_point.layout,
        end_point,
        members_to_select,
        dimensions_to_select,
        extents_to_use,
        end_point_ident in non_zero_reducible_ptrs,
    )
    return check
