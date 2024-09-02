import abc
import typing
from enum import member
from typing import Set, Tuple

from tests.test_builtin_traits import test_has_no_terminator_empty_block_with_single_block_region_requires_no_terminator
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import DimensionAttr, Layout, MemberAttr, TypeType

T = typing.TypeVar("T", bound=dlt.Layout)


class InConsistentLayoutException(Exception):
    pass


class ManipulatorMap:
    def __init__(self):
        self.map: dict[typing.Type[dlt.Layout], LayoutNodeManipulator] = {}

    def add(
        self, typ: typing.Type[dlt.Layout], node_manipulator: "LayoutNodeManipulator"
    ):
        self.map[typ] = node_manipulator

    def get(self, layout: dlt.Layout) -> "LayoutNodeManipulator":
        for t, s in reversed(self.map.items()):
            if isinstance(layout, t):
                return s
        raise KeyError(f"Cannot find manipulator for layout: {layout}")

    def minimal_reduction(
        self,
        layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        """
        Minimal reduction should ensure than all members/dimensions that are not allowable are used in finding the subtree layout. If this is not possible an InconsistantLayout exception is raised

        """
        possible_extents_check = all(
            e in allowable_extents for e in layout.get_all_init_base_extents()
        )
        if not possible_extents_check:
            raise InConsistentLayoutException(
                f"layout requires extents that are not made available."
            )

        must_remove_members = members - allowable_members
        must_remove_dimensions = dimensions - allowable_dimensions
        must_remove_extents = extents - allowable_extents
        if must_remove_members or must_remove_dimensions or must_remove_extents:
            result, ms, ds = self.get(layout).minimal_reduction(
                layout,
                members,
                dimensions,
                extents,
                allowable_members,
                allowable_dimensions,
                allowable_extents,
                through_index_reducible,
            )
        else:
            result, ms, ds = layout, members, dimensions

        assert layout.has_sub_layout(result)
        assert ms.issubset(members)
        assert ms.issubset(allowable_members)
        assert ds.issubset(dimensions)
        assert ds.issubset(allowable_dimensions)
        return result, ms, ds

    def maximal_reduction(
        self,
        layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[Layout, set[MemberAttr], set[DimensionAttr], bool]:
        child = layout, members, dimensions, through_index_reducible
        new_child = child
        while new_child is not None:
            child = new_child
            new_child = self.try_reduction(child[0], child[1], child[2], child[3])
        return child

    def try_reduction(
        self,
        layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[Layout, set[MemberAttr], set[DimensionAttr], bool] | None:
        initial_type = layout.contents_type.select_members(members).select_dimensions(dimensions)
        try_result = self.get(layout).try_reduction(layout, set(members), set(dimensions), through_index_reducible)
        if try_result is not None:
            new_layout, new_members, new_dimensions, new_idx_reduce = try_result
            new_type = new_layout.contents_type.select_members(new_members).select_dimensions(new_dimensions)
            assert new_type == initial_type, "try_reduction failed to provide the same resulting type as an answer"
        return try_result
    def structural_reduction(
        self, layout: dlt.Layout, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        # this must either return a child of layout if there is a single child that without consuming a member
        # specifier or dimension can select the dlt_type, or None
        if not layout.contents_type.has_selectable_type(dlt_type):
            raise ValueError(f"Cannot select type {dlt_type} from layout {layout}")
        else:
            result = self.get(layout).structural_reduction(layout, dlt_type)
            assert result is None or result in layout.get_children()
            return result

    def reduce_to_terminal(
        self,
        layout: dlt.Layout,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        assert layout.contents_type.has_selectable(
            members_to_select, dimensions_to_select, base_type
        )

        result = self.get(layout).reduce_to_terminal(
            layout, members_to_select, dimensions_to_select, base_type
        )
        return result

    def can_layout_derive_to(
        self,
        layout: dlt.Layout,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:

        assert starting_point.layout.has_sub_layout(layout)
        assert end_point.layout == end_layout
        assert layout.contents_type.has_selectable(
            selectable_members, selectable_dimensions
        )
        if not layout.contents_type.has_selectable_type(end_layout.contents_type):
            return False
        if not layout.has_sub_layout(end_layout):
            return False
        if layout == end_layout:
            return True
        result = self.get(layout).can_layout_derive_to(
            layout,
            starting_point,
            end_layout,
            end_point,
            selectable_members,
            selectable_dimensions,
            usable_extents,
            through_index_reducible,
        )
        return result

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        if members or dimensions:
            new_layout = self.get(parent_layout).embed_layout_in(
                child_layout, parent_layout, members, dimensions, extents
            )
            return new_layout
        else:
            if child_layout.contents_type != parent_layout.contents_type:
                raise InConsistentLayoutException()
            if child_layout == parent_layout:
                return child_layout
            if child_layout.node_matches(parent_layout):
                assert len(child_layout.get_children()) == len(
                    parent_layout.get_children()
                )
                return parent_layout.from_new_children(
                    [
                        self.embed_layout_in(cs_c, ps_c, set(), set(), extents)
                        for cs_c, ps_c in zip(
                            child_layout.get_children(), parent_layout.get_children()
                        )
                    ]
                )
            if isinstance(parent_layout, dlt.AbstractLayoutAttr):
                # if the parent side is Abstract then we check that the children layouts of the parent side can have
                # the minimally reduced children of the child side embbed in them directly. Simply this checks that
                # the children of the parent side are correctly positioned sub-trees of the new layout (but also
                # accounts for more possible abstract layouts further down)
                al = typing.cast(dlt.AbstractLayoutAttr, parent_layout)
                for a_child in al.children:
                    abstract_members = set(a_child.member_specifiers)
                    abstract_dimensions = set(a_child.dimensions)

                    new_child_layout_sub_tree, ms, ds = self.minimal_reduction(
                        child_layout,
                        abstract_members,
                        abstract_dimensions,
                        set(),
                        set(),
                        set(),
                        extents,
                        False,
                    )
                    assert len(ms) == 0 and len(ds) == 0

                    embedded_subtree = None
                    while embedded_subtree is None:
                        try:
                            embedded_subtree = self.embed_layout_in(
                                new_child_layout_sub_tree,
                                a_child.child,
                                set(),
                                set(),
                                extents,
                            )
                        except InConsistentLayoutException:
                            embedded_subtree = None
                            new_child_layout_sub_tree = self.structural_reduction(
                                new_child_layout_sub_tree, a_child.contents_type
                            )
                            if new_child_layout_sub_tree is None:
                                raise InConsistentLayoutException()

                    if embedded_subtree != new_child_layout_sub_tree:
                        raise InConsistentLayoutException()
                if isinstance(child_layout, dlt.AbstractLayoutAttr):
                    return child_layout
                else:
                    return dlt.AbstractLayoutAttr(
                        [dlt.AbstractChildAttr([], [], child_layout)]
                    )

            raise InConsistentLayoutException()

class LayoutNodeManipulator(abc.ABC, typing.Generic[T]):

    def __init__(self, manipulator: ManipulatorMap):
        self.manipulator = manipulator

    @abc.abstractmethod
    def minimal_reduction(
        self,
        layout: T,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        # If called, then we assume there must be something to reduce (as this case is caught by Manipulator.minimal_reduce)
        raise NotImplementedError()

    @abc.abstractmethod
    def try_reduction(
        self,
        layout: T,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        # simply try to use any the members and dimensions to do a reduction step.
        raise NotImplementedError()

    @abc.abstractmethod
    def structural_reduction(
        self, layout: T, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        raise NotImplementedError

    @abc.abstractmethod
    def reduce_to_terminal(
        self,
        layout: T,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        raise NotImplementedError

    @abc.abstractmethod
    def can_layout_derive_to(
        self,
        layout: T,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: T,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        raise NotImplementedError


class AbstractManipulator(LayoutNodeManipulator[dlt.AbstractLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.AbstractLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        possible_children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(members, dimensions)
        ]
        if len(possible_children) != 1:
            raise InConsistentLayoutException()
        child = possible_children[0]
        return self.manipulator.minimal_reduction(
            child.child,
            members.difference(child.member_specifiers),
            dimensions.difference(child.dimensions),
            extents,
            allowable_members,
            allowable_dimensions,
            allowable_extents,
            through_index_reducible,
        )

    def try_reduction(
        self,
        layout: dlt.AbstractLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        possible_children = [
            child
            for child in layout.children
            if members.issuperset(child.member_specifiers)
            and dimensions.issuperset(child.dimensions)
        ]
        if len(possible_children) != 1:
            return None
        child = possible_children[0]
        new_members = members.difference(child.member_specifiers)
        new_dimensions = dimensions.difference(child.dimensions)
        return child.child, new_members, new_dimensions, through_index_reducible

    def structural_reduction(
        self, layout: dlt.AbstractLayoutAttr, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        children = [
            c.child
            for c in layout.children
            if len(c.member_specifiers) == 0 and len(c.dimensions) == 0
            if c.contents_type.has_selectable_type(dlt_type)
        ]
        if len(children) == 1:
            return children[0]
        else:
            return None

    def reduce_to_terminal(
        self,
        layout: dlt.AbstractLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(
                members_to_select, dimensions_to_select, base_type
            )
        ]
        if len(children) != 1:
            raise InConsistentLayoutException()
        child = children[0]
        return self.manipulator.reduce_to_terminal(
            child.child,
            members_to_select.difference(child.member_specifiers),
            dimensions_to_select.difference(child.dimensions),
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: dlt.AbstractLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        possible_children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(
                selectable_members, selectable_dimensions
            )
        ]
        if len(possible_children) != 1:
            return False
        child = possible_children[0]
        return self.manipulator.can_layout_derive_to(
            child.child,
            starting_point,
            end_layout,
            end_point,
            selectable_members.difference(child.member_specifiers),
            selectable_dimensions.difference(child.dimensions),
            usable_extents,
            through_index_reducible,
        )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.AbstractLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        children = []
        modified_child = None
        for a_child in parent_layout.children:
            if (
                a_child.contents_type.has_selectable(members, dimensions)
                and a_child.contents_type.with_selection(members, dimensions)
                == child_layout.contents_type
            ):
                if modified_child is not None:
                    raise InConsistentLayoutException()
                abstract_members = set(a_child.member_specifiers) - members
                abstract_dimensions = set(a_child.dimensions) - dimensions
                new_child_layout_sub_tree, ms, ds = self.manipulator.minimal_reduction(
                    child_layout,
                    abstract_members,
                    abstract_dimensions,
                    set(),
                    set(),
                    set(),
                    extents,
                    False,
                )
                assert len(ms) == 0 and len(ds) == 0
                embedded_subtree = self.manipulator.embed_layout_in(
                    new_child_layout_sub_tree, a_child.child, set(), set(), extents
                )
                if embedded_subtree != new_child_layout_sub_tree:
                    raise InConsistentLayoutException()
                modified_child = dlt.AbstractChildAttr(
                    abstract_members, abstract_dimensions, child_layout
                )
                children.append(modified_child)
            else:
                children.append(a_child)
        if modified_child is None:
            raise InConsistentLayoutException()
        return dlt.AbstractLayoutAttr(children)


class PrimitiveManipulator(LayoutNodeManipulator[dlt.PrimitiveLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        raise InConsistentLayoutException()

    def try_reduction(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if members or dimensions:
            raise InConsistentLayoutException()
        return None

    def structural_reduction(
        self, layout: dlt.PrimitiveLayoutAttr, dlt_type: dlt.Type
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        if members_to_select or dimensions_to_select:
            raise InConsistentLayoutException()
        elif layout.base_type != base_type:
            raise InConsistentLayoutException()
        else:
            return layout

    def can_layout_derive_to(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        return False

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.PrimitiveLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        raise InConsistentLayoutException()


class ConstantManipulator(LayoutNodeManipulator[dlt.ConstantLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.ConstantLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        raise InConsistentLayoutException()

    def try_reduction(
        self,
        layout: dlt.ConstantLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if members or dimensions:
            raise InConsistentLayoutException()
        return None

    def structural_reduction(
        self, layout: dlt.ConstantLayoutAttr, dlt_type: dlt.Type
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.ConstantLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        if members_to_select or dimensions_to_select:
            raise InConsistentLayoutException()
        elif layout.base_data.type != base_type:
            raise InConsistentLayoutException()
        else:
            return layout

    def can_layout_derive_to(
        self,
        layout: dlt.ConstantLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        return False

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.ConstantLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        raise InConsistentLayoutException()


class MemberManipulator(LayoutNodeManipulator[dlt.MemberLayoutAttr]):
    def minimal_reduction(
        self,
        layout: dlt.MemberLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        if layout.member_specifier not in members:
            raise InConsistentLayoutException()
        child = layout.child
        return self.manipulator.minimal_reduction(
            child,
            members - {layout.member_specifier},
            dimensions,
            extents,
            allowable_members,
            allowable_dimensions,
            allowable_extents,
            through_index_reducible,
        )

    def try_reduction(
        self,
        layout: dlt.MemberLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if layout.member_specifier in members:
            return layout.child, members - {layout.member_specifier}, dimensions, through_index_reducible,
        return None

    def structural_reduction(
        self, layout: dlt.MemberLayoutAttr, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.MemberLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        if layout.member_specifier not in members_to_select:
            raise InConsistentLayoutException()
        return self.manipulator.reduce_to_terminal(
            layout.child,
            members_to_select - {layout.member_specifier},
            dimensions_to_select,
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: dlt.MemberLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        if layout.member_specifier not in selectable_members:
            return False
        return self.manipulator.can_layout_derive_to(
            layout.child,
            starting_point,
            end_layout,
            end_point,
            selectable_members - {layout.member_specifier},
            selectable_dimensions,
            usable_extents,
            through_index_reducible,
        )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.MemberLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        if parent_layout.member_specifier not in members:
            raise InConsistentLayoutException()
        return parent_layout.from_new_children(
            [
                self.manipulator.embed_layout_in(
                    child_layout,
                    parent_layout.child,
                    members - {parent_layout.member_specifier},
                    dimensions,
                    extents,
                )
            ],
        )


class DenseManipulator(LayoutNodeManipulator[dlt.DenseLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.DenseLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        if layout.dimension not in dimensions:
            raise InConsistentLayoutException()
        return self.manipulator.minimal_reduction(
            layout.child,
            members,
            dimensions - {layout.dimension},
            extents,
            allowable_members,
            allowable_dimensions,
            allowable_extents,
            through_index_reducible,
        )

    def try_reduction(
        self,
        layout: dlt.DenseLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if layout.dimension in dimensions:
            return layout.child, members, dimensions - {layout.dimension}, through_index_reducible
        return None

    def structural_reduction(
        self, layout: dlt.DenseLayoutAttr, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.DenseLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        if layout.dimension not in dimensions_to_select:
            raise InConsistentLayoutException()
        return self.manipulator.reduce_to_terminal(
            layout.child,
            members_to_select,
            dimensions_to_select - {layout.dimension},
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: dlt.DenseLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        if layout.dimension not in selectable_dimensions:
            return False
        return self.manipulator.can_layout_derive_to(
            layout.child,
            starting_point,
            end_layout,
            end_point,
            selectable_members,
            selectable_dimensions - {layout.dimension},
            usable_extents,
            through_index_reducible,
        )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.DenseLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        if parent_layout.dimension not in dimensions:
            raise InConsistentLayoutException()
        return parent_layout.from_new_children(
            [
                self.manipulator.embed_layout_in(
                    child_layout,
                    parent_layout.child,
                    members,
                    dimensions - {parent_layout.dimension},
                    extents,
                )
            ]
        )


class StructManipulator(LayoutNodeManipulator[dlt.StructLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        possible_children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(members, dimensions)
        ]
        if len(possible_children) != 1:
            raise InConsistentLayoutException()
        child = possible_children[0]
        return self.manipulator.minimal_reduction(
            child,
            members,
            dimensions,
            extents,
            allowable_members,
            allowable_dimensions,
            allowable_extents,
            through_index_reducible,
        )

    def try_reduction(
        self,
        layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        possible_children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(members, dimensions)
        ]
        if len(possible_children) != 1:
            return None
        child = possible_children[0]
        return child, members, dimensions, through_index_reducible

    def structural_reduction(
        self, layout: dlt.StructLayoutAttr, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        children = [
            c for c in layout.children if c.contents_type.has_selectable_type(dlt_type)
        ]
        if len(children) == 1:
            return children[0]
        else:
            return None

    def reduce_to_terminal(
        self,
        layout: dlt.StructLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(
                members_to_select, dimensions_to_select, base_type
            )
        ]
        if len(children) != 1:
            raise InConsistentLayoutException()
        child = children[0]
        return self.manipulator.reduce_to_terminal(
            child,
            members_to_select,
            dimensions_to_select,
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: dlt.StructLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        possible_children = [
            child
            for child in layout.children
            if child.contents_type.has_selectable(
                selectable_members, selectable_dimensions
            )
        ]
        if len(possible_children) != 1:
            return False
        child = possible_children[0]
        return self.manipulator.can_layout_derive_to(
            child,
            starting_point,
            end_layout,
            end_point,
            selectable_members,
            selectable_dimensions,
            usable_extents,
            through_index_reducible,
        )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        children = []
        modified_child = None
        for child in parent_layout.children:
            if (
                child.contents_type.has_selectable(members, dimensions)
                and child.contents_type.with_selection(members, dimensions)
                == child_layout.contents_type
            ):
                if modified_child is not None:
                    raise InConsistentLayoutException()
                modified_child = self.manipulator.embed_layout_in(
                    child_layout, child, members, dimensions, extents
                )
                children.append(modified_child)
            else:
                children.append(child)
        if modified_child is None:
            raise InConsistentLayoutException()
        return parent_layout.from_new_children(children)


class ArithDropManipulator(LayoutNodeManipulator[dlt.ArithDropLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.ArithDropLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        if layout.dimension not in dimensions:
            raise InConsistentLayoutException()
        return self.manipulator.minimal_reduction(
            layout.child,
            members,
            dimensions - {layout.dimension},
            extents,
            allowable_members,
            allowable_dimensions,
            allowable_extents,
            through_index_reducible,
        )

    def try_reduction(
        self,
        layout: dlt.ArithDropLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if layout.dimension in dimensions:
            return layout.child, members, dimensions - {layout.dimension}, through_index_reducible
        return None

    def structural_reduction(
        self, layout: dlt.ArithDropLayoutAttr, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.ArithDropLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        if layout.dimension not in dimensions_to_select:
            raise InConsistentLayoutException()
        return self.manipulator.reduce_to_terminal(
            layout.child,
            members_to_select,
            dimensions_to_select - {layout.dimension},
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: dlt.ArithDropLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        if layout.dimension not in selectable_dimensions:
            return False
        return self.manipulator.can_layout_derive_to(
            layout.child,
            starting_point,
            end_layout,
            end_point,
            selectable_members,
            selectable_dimensions - {layout.dimension},
            usable_extents,
            through_index_reducible,
        )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.ArithDropLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        if parent_layout.dimension not in dimensions:
            raise InConsistentLayoutException()
        return parent_layout.from_new_children(
            [
                self.manipulator.embed_layout_in(
                    child_layout,
                    parent_layout.child,
                    members,
                    dimensions - {parent_layout.dimension},
                    extents,
                )
            ]
        )


class ArithReplaceManipulator(LayoutNodeManipulator[dlt.ArithReplaceLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.ArithReplaceLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:

        replacements = []
        for dim in dimensions:
            replacement_for = layout.replacement_for(dim)
            if replacement_for is not None:
                replacements.append(replacement_for)

        if len(replacements) != 1:
            raise InConsistentLayoutException()
        replacement = replacements[0]

        new_dimensions = (dimensions - {replacement.outer_dimension}) | {
            replacement.inner_dimension
        }

        if replacement.inner_member in members:
            raise InConsistentLayoutException()

        new_members = members | {replacement.inner_member}

        return self.manipulator.minimal_reduction(
            layout.child,
            new_members,
            new_dimensions,
            extents,
            allowable_members,
            allowable_dimensions,
            allowable_extents,
            through_index_reducible,
        )

    def try_reduction(
        self,
        layout: dlt.ArithReplaceLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:

        replacements = []
        for dim in dimensions:
            replacement_for = layout.replacement_for(dim)
            if replacement_for is not None:
                replacements.append(replacement_for)

        if len(replacements) != 1:
            return None
        replacement = replacements[0]

        new_dimensions = (dimensions - {replacement.outer_dimension}) | {
            replacement.inner_dimension
        }

        if replacement.inner_member in members:
            raise InConsistentLayoutException()

        new_members = members | {replacement.inner_member}

        return layout.child, new_members, new_dimensions, through_index_reducible

    def structural_reduction(
        self, layout: dlt.ArithReplaceLayoutAttr, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.ArithReplaceLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        dims_to_select = layout.outer_dimensions() & dimensions_to_select
        if len(dims_to_select) != 1:
            raise InConsistentLayoutException()
        replacement = layout.replacement_for(dims_to_select.pop())

        return self.manipulator.reduce_to_terminal(
            layout.child,
            members_to_select | {replacement.inner_member},
            (dimensions_to_select - {replacement.outer_dimension})
            | {replacement.inner_dimension},
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: dlt.ArithReplaceLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        selectable_dims = layout.outer_dimensions() & selectable_dimensions
        if len(selectable_dims) != 1:
            return False
        replacement = layout.replacement_for(selectable_dims.pop())

        return self.manipulator.can_layout_derive_to(
            layout.child,
            starting_point,
            end_layout,
            end_point,
            selectable_members | {replacement.inner_member},
            (selectable_dimensions - {replacement.outer_dimension})
            | {replacement.inner_dimension},
            usable_extents,
            through_index_reducible,
        )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.ArithReplaceLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        dims = parent_layout.outer_dimensions() & dimensions
        if len(dims) != 1:
            raise InConsistentLayoutException()
        replacement = parent_layout.replacement_for(dims.pop())
        child_content_type = child_layout.contents_type

        # if the child layout type provides the inner dim and member then we should embed it directly here
        # but if it doesn't then it must have been reduced and so we must embed at those reduced sub-tree positions
        if child_content_type.has_selectable(
            [replacement.inner_member], [replacement.inner_dimension]
        ):
            new_members = members
            new_dimensions = dimensions
        elif child_content_type.has_selectable([replacement.inner_member], []):
            new_members = members
            new_dimensions = dimensions | {replacement.inner_dimension}
        elif child_content_type.has_selectable([], [replacement.inner_dimension]):
            new_members = members | {replacement.inner_member}
            new_dimensions = dimensions
        else:
            new_members = members | {replacement.inner_member}
            new_dimensions = dimensions | {replacement.inner_dimension}

        return parent_layout.from_new_children(
            [
                self.manipulator.embed_layout_in(
                    child_layout,
                    parent_layout.child,
                    new_members,
                    new_dimensions,
                    extents,
                )
            ]
        )


class IndexingManipulator(LayoutNodeManipulator[dlt.IndexingLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.IndexingLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        if not through_index_reducible:
            raise InConsistentLayoutException(
                "Cannot reduce through Indexing node as it is unsafe to hold pointers to values contained inside"
            )
        else:
            dir_mems = (
                    members & layout.directChild.contents_type.all_member_attributes()
            )
            dir_dims = (
                    dimensions & layout.directChild.contents_type.all_dimension_attributes()
            )
            dir_type = TypeType([(dir_mems, dir_dims, layout.indexedChild.indexed_by())])

            selections = layout.directChild.contents_type.has_selectable_type(dir_type)
            if (dlt.SetAttr([]), dlt.SetAttr([])) not in selections:
                raise InConsistentLayoutException()
            child = self.manipulator.get(layout.indexedChild).minimal_reduction(
                layout.indexedChild,
                members - dir_mems,
                dimensions - dir_dims,
                extents,
                allowable_members,
                allowable_dimensions,
                allowable_extents,
                through_index_reducible,
            )
            return child

    def try_reduction(
        self,
        layout: dlt.IndexingLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if not through_index_reducible:
            # if we loop over x,y,z but you tell me that you only care if z is non-zero. then we can skip x and y too.
            return None
        else:
            dir_mems = (
                    members & layout.directChild.contents_type.all_member_attributes()
            )
            dir_dims = (
                    dimensions & layout.directChild.contents_type.all_dimension_attributes()
            )
            dir_type = TypeType([(dir_mems, dir_dims, layout.indexedChild.indexed_by())])
            selections = layout.directChild.contents_type.has_selectable_type(dir_type)
            if (dlt.SetAttr([]), dlt.SetAttr([])) not in selections:
                return None
            child = self.manipulator.get(layout.indexedChild).try_reduction(
                layout.indexedChild,
                members - dir_mems,
                dimensions - dir_dims,
                through_index_reducible,
            )
            return child

    def structural_reduction(
        self, layout: T, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.IndexingLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        direct_type = layout.directChild.contents_type
        direct_members = direct_type.all_member_attributes().intersection(
            members_to_select
        )
        direct_dims = direct_type.all_dimension_attributes().intersection(
            dimensions_to_select
        )
        reduced_direct = self.manipulator.reduce_to_terminal(
            layout.directChild,
            direct_members,
            direct_dims,
            layout.indexedChild.indexed_by(),
        )
        if reduced_direct is None:
            return None

        indexed_members = members_to_select - direct_members
        indexed_dims = dimensions_to_select - direct_dims
        return self.manipulator.reduce_to_terminal(
            layout.indexedChild, indexed_members, indexed_dims, base_type
        )

    def can_layout_derive_to(
        self,
        layout: dlt.IndexingLayoutAttr,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        if not through_index_reducible:
            return False
        else:
            dir_mems = (
                    selectable_members & layout.directChild.contents_type.all_member_attributes()
            )
            dir_dims = (
                    selectable_dimensions & layout.directChild.contents_type.all_dimension_attributes()
            )
            dir_type = TypeType([(dir_mems, dir_dims, layout.indexedChild.indexed_by())])
            selections = layout.directChild.contents_type.has_selectable_type(dir_type)
            if (dlt.SetAttr([]), dlt.SetAttr([])) not in selections:
                return False
            child = self.manipulator.get(layout.indexedChild).can_layout_derive_to(
                layout.indexedChild,
                starting_point,
                end_layout,
                end_point,
                selectable_members - dir_mems,
                selectable_dimensions - dir_dims,
                usable_extents,
                through_index_reducible,
            )
            return child

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.IndexingLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        direct_type = parent_layout.directChild.contents_type
        if (
            direct_type.has_selectable(members, dimensions)
            and direct_type.with_selection(members, dimensions)
            == child_layout.contents_type
        ):
            new_direct_child = self.manipulator.embed_layout_in(
                child_layout, parent_layout.directChild, members, dimensions, extents
            )
            return parent_layout.from_new_children(
                [new_direct_child, parent_layout.indexedChild]
            )
        else:
            index_members = members - direct_type.all_member_attributes()
            index_dimensions = dimensions - direct_type.all_dimension_attributes()
            new_index_child = self.manipulator.embed_layout_in(
                child_layout,
                parent_layout.indexedChild,
                index_members,
                index_dimensions,
                extents,
            )
            return parent_layout.from_new_children(
                [parent_layout.directChild, new_index_child]
            )


class UnpackedCOOManipulator(LayoutNodeManipulator[dlt.UnpackedCOOLayoutAttr]):

    def minimal_reduction(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
        allowable_members: set[dlt.MemberAttr],
        allowable_dimensions: set[dlt.DimensionAttr],
        allowable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr]]:
        if not through_index_reducible:
            raise InConsistentLayoutException(
                "Cannot reduce through Unpacked COO node as it is unsafe to hold pointers to values contained inside"
            )
        else:
            if not all(d in dimensions for d in layout.dimensions):
                raise InConsistentLayoutException()
            return self.manipulator.minimal_reduction(
                layout.child,
                members,
                dimensions - {d for d in layout.dimensions},
                extents,
                allowable_members,
                allowable_dimensions,
                allowable_extents,
                through_index_reducible,
            )

    def try_reduction(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        through_index_reducible: bool,
    ) -> tuple[dlt.Layout, set[dlt.MemberAttr], set[dlt.DimensionAttr], bool] | None:
        if not through_index_reducible:
            return None
        else:
            if not all(d in dimensions for d in layout.dimensions):
                return None
            return layout.child, members, dimensions - {d for d in layout.dimensions}, through_index_reducible

    def structural_reduction(
        self, layout: T, dlt_type: dlt.TypeType
    ) -> None | dlt.Layout:
        return None

    def reduce_to_terminal(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        members_to_select: set[dlt.MemberAttr],
        dimensions_to_select: set[dlt.DimensionAttr],
        base_type: dlt.AcceptedTypes,
    ) -> None | dlt.Layout:
        if any(dim not in dimensions_to_select for dim in layout.dimensions):
            raise InConsistentLayoutException()
        return self.manipulator.reduce_to_terminal(
            layout.child,
            members_to_select,
            dimensions_to_select - set(layout.dimensions),
            base_type,
        )

    def can_layout_derive_to(
        self,
        layout: T,
        starting_point: dlt.PtrType,
        end_layout: dlt.Layout,
        end_point: dlt.PtrType,
        selectable_members: set[dlt.MemberAttr],
        selectable_dimensions: set[dlt.DimensionAttr],
        usable_extents: set[dlt.InitDefinedExtentAttr],
        through_index_reducible: bool,
    ) -> bool:
        if not through_index_reducible:
            return False
        else:
            if not all(d in selectable_dimensions for d in layout.dimensions):
                return False
            return self.manipulator.can_layout_derive_to(
                layout.child,
                starting_point,
                end_layout,
                end_point,
                selectable_members,
                selectable_dimensions - {d for d in layout.dimensions},
                usable_extents,
                through_index_reducible,
            )

    def embed_layout_in(
        self,
        child_layout: dlt.Layout,
        parent_layout: dlt.UnpackedCOOLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extents: set[dlt.InitDefinedExtentAttr],
    ) -> dlt.Layout:
        if any(dim not in dimensions for dim in parent_layout.dimensions):
            raise InConsistentLayoutException()
        return parent_layout.from_new_children(
            [
                self.manipulator.embed_layout_in(
                    child_layout,
                    parent_layout.child,
                    members,
                    dimensions - set(parent_layout.dimensions),
                    extents,
                )
            ]
        )


Manipulator = ManipulatorMap()
Manipulator.add(dlt.AbstractLayoutAttr, AbstractManipulator(Manipulator))
Manipulator.add(dlt.PrimitiveLayoutAttr, PrimitiveManipulator(Manipulator))
Manipulator.add(dlt.ConstantLayoutAttr, ConstantManipulator(Manipulator))
Manipulator.add(dlt.MemberLayoutAttr, MemberManipulator(Manipulator))
Manipulator.add(dlt.DenseLayoutAttr, DenseManipulator(Manipulator))
Manipulator.add(dlt.StructLayoutAttr, StructManipulator(Manipulator))
Manipulator.add(dlt.ArithDropLayoutAttr, ArithDropManipulator(Manipulator))
Manipulator.add(dlt.ArithReplaceLayoutAttr, ArithReplaceManipulator(Manipulator))
Manipulator.add(dlt.IndexingLayoutAttr, IndexingManipulator(Manipulator))
Manipulator.add(dlt.UnpackedCOOLayoutAttr, UnpackedCOOManipulator(Manipulator))
