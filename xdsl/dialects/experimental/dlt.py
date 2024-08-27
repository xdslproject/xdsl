"""
Data Layout Trees 'DLT' is a dialect / DSL for specifying the data-layout of multiple tensors in a unified tree
structure. The objective is to provide access to named tensors by named dimensions such that the actual layout can be
modified at compile time without changing the code that uses the tensors. This then allows us to have complex layouts
that combine structures with members, dense dimensions, and compressed/sparse layouts as well as affine transformations
on dimensions, masking. From a normalised description of the data that needs to exist (and potentially knowledge about
redundancy / structured sparsity) a physically layout can be produced. Then with rewrite rules that preserve soundness
(that all values that must be stored are stored) we can modify the structure automatically to produce a large search
space of different physical layouts to then find more optimal solutions that produce more efficient code.
"""

from __future__ import annotations

import abc
import contextvars
import typing
from abc import ABC
from functools import total_ordering
from typing import Iterable, Iterator, Self, Type

from xdsl.dialects import builtin, func
from xdsl.dialects.builtin import (
    AnyFloat,
    AnyFloatAttr,
    AnyIntegerAttr, ArrayAttr,
    FloatAttr, IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import AttributeCovT, BlockArgument, Dialect, OpResult, TypeAttribute, Use
from xdsl.irdl import *
from xdsl.parser import AttrParser
from xdsl.traits import (
    HasAncestor,
    HasParent,
    IsTerminator,
    NoTerminator,
    SingleBlockImplicitTerminator,
    UseDefChainTrait,
)
from xdsl.utils.exceptions import ComplexVerifyException
from xdsl.utils.hints import isa


@dataclass
class SetOfConstraint(AttrConstraint):
    """
    A constraint that enforces an SetData whose elements all satisfy
    the elem_constr.
    """

    elem_constr: AttrConstraint

    def __init__(self, constr: Attribute | type[Attribute] | AttrConstraint):
        self.elem_constr = attr_constr_coercion(constr)

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, SetAttr):
            raise VerifyException(f"expected SetData attribute, but got {attr}")
        for e in cast(SetAttr[Attribute], attr).data:
            self.elem_constr.verify(e, constraint_vars)


@irdl_attr_definition
class SetAttr(GenericData[frozenset[AttributeCovT, ...]], Iterable[AttributeCovT]):
    """Based on ArrayAttr but for sets. Used for when the order shouldn't matter. By Default putting duplicate values in
    raises an error as it is expected that duplicates should not be in the input for well-formed code.
    This implementation requires that Attributes contained within are hashable which is not necessarily true.
    """

    name = "dlt.set"

    def __init__(self, param: Iterable[AttributeCovT] = tuple()) -> None:
        p = list(param)
        s = frozenset(p)
        if len(s) != len(p):
            raise ValueError(
                f"Cannot form SetAttr with duplicate elements in: {str(p)}"
            )

        super().__init__(frozenset(param))

    @classmethod
    def from_duplicates(cls, param: Iterable[AttributeCovT]) -> SetAttr[AttributeCovT]:
        return SetAttr(set(param))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> frozenset[AttributeCovT, ...]:
        data = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES, parser.parse_attribute
        )
        # the type system can't ensure that the elements are of type _SetAttrT
        result = cast(tuple[AttributeCovT, ...], tuple(data))
        return result

    def print_parameter(self, printer: Printer) -> None:
        """We sort the elements by their str() value just to maintain determinism of output"""
        printer.print_string("{")
        values = list(self.data)
        sorted_values = [
            values[i] for s, i in sorted([(str(s), i) for i, s in enumerate(values)])
        ]
        printer.print_list(sorted_values, printer.print_attribute)
        printer.print_string("}")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        if len(args) == 1:
            return SetOfConstraint(irdl_to_attr_constraint(args[0]))
        if len(args) == 0:
            return SetOfConstraint(AnyAttr())
        raise TypeError(
            f"Attribute SetAttr expects at most 1 type"
            f" parameter, but {len(args)} were given"
        )

    def verify(self) -> None:
        for idx, val in enumerate(
            self.data
        ):  # this check shouldn't be needed? as the Type checking should sort this out
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but {idx} "
                    f"element is of type {type(val)}"
                )

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[AttributeCovT]:
        return iter(self.data)

    def take(self) -> AttributeCovT:
        return next(iter(self.data))

    def without(self, val: AttributeCovT) -> SetAttr[AttributeCovT]:
        if val not in self.data:
            raise ValueError("Cannot remove elements that do not exist in the set. If this is expected, use difference() instead of without()")
        return SetAttr(self.data.difference([val]))

    def difference(self, val: Iterable[AttributeCovT]) -> SetAttr[AttributeCovT]:
        return SetAttr(self.data.difference(val))

    def remove(self, val: Iterable[AttributeCovT]) -> SetAttr[AttributeCovT]:
        if not frozenset(val).issubset(self.data):
            raise ValueError("Cannot remove elements that do not exist in the set. If this is expected, use difference() instead of remove()")
        return SetAttr(self.data.difference(val))

    def union(self, val: Iterable[AttributeCovT]) -> SetAttr[AttributeCovT]:
        return SetAttr(self.data.union(val))

    def add(self, val: Iterable[AttributeCovT]) -> SetAttr[AttributeCovT]:
        if not isinstance(val, list):
            val = list(val)
        return SetAttr(list(self.data) + val)


class DLTCompatibleElementBaseType(abc.ABC):
    @abc.abstractmethod
    def get_size(self) -> tuple[int, int]:
        raise NotImplementedError()


@irdl_attr_definition
class IndexRangeType(
    ParametrizedAttribute, TypeAttribute, DLTCompatibleElementBaseType
):
    name = "dlt.indexRange"

    def get_size(self) -> tuple[int, int]:
        bit_width = builtin.i64.width.data
        bytes = -(bit_width // -8)
        return (bytes, bytes)


AcceptedTypes: TypeAlias = IntegerType | AnyFloat | IndexType | IndexRangeType


@irdl_attr_definition
class MemberAttr(ParametrizedAttribute):
    name = "dlt.member"
    structName: ParameterDef[StringAttr]
    memberName: ParameterDef[StringAttr]

    def __init__(
        self,
        struct_name: StringAttr | str | Iterable[Attribute | str],
        member_name: StringAttr | str = None,
    ):
        if member_name is None:
            assert isinstance(struct_name, Iterable)
            struct_name = tuple(struct_name)
            assert len(struct_name) == 2
            member_name = struct_name[1]
            struct_name = struct_name[0]
        if isinstance(struct_name, str):
            struct_name = StringAttr(struct_name)
        if isinstance(member_name, str):
            member_name = StringAttr(member_name)
        super().__init__((struct_name, member_name))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            result = MemberAttr.internal_parse_parameters(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        sn = StringAttr(parser.parse_identifier())
        parser.parse_punctuation(":")
        mn = StringAttr(parser.parse_identifier())
        return [sn, mn]

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print(self.structName.data)
        printer.print(":")
        printer.print(self.memberName.data)

    @classmethod
    def internal_print_members(
        cls, members: SetAttr[MemberAttr], printer: Printer
    ) -> None:
        printer.print("{")
        m_values = list(members.data)
        sorted_m_values = [
            m_values[i]
            for s, i in sorted([(str(s), i) for i, s in enumerate(m_values)])
        ]
        printer.print_list(
            sorted_m_values, lambda v: v.internal_print_parameters(printer)
        )
        printer.print("}")


@total_ordering
class Stage(Enum):
    STATIC = 1
    SCOPE = 2
    INIT = 3
    DYNAMIC = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Extent(ParametrizedAttribute, abc.ABC):
    value: ParameterDef[StringAttr | IntegerAttr]

    @abstractmethod
    def get_stage(self) -> Stage:
        pass

    def is_static(self) -> bool:
        return self.get_stage() == Stage.STATIC

    def is_scope_time(self) -> bool:
        return self.get_stage() == Stage.SCOPE

    def is_init_time(self) -> bool:
        return self.get_stage() == Stage.INIT

    def is_dynamic(self) -> bool:
        return self.get_stage() == Stage.DYNAMIC

    @abstractmethod
    def base_extents(self) -> list["BaseExtent"]:
        raise NotImplementedError()

    def verify(self) -> None:
        if not any(
            [
                self.is_static(),
                self.is_scope_time(),
                self.is_init_time(),
                self.is_dynamic(),
            ]
        ):
            raise VerifyException(
                "An extent must be at least one of: static, compile-time, init-time, dynamic"
            )

    @classmethod
    def internal_parse_any_extent(cls, parser: AttrParser) -> Extent:
        with parser.in_angle_brackets():
            extent_type = parser.parse_identifier()
            parser.parse_punctuation("|")
            match extent_type:
                case "S":
                    # StaticExtent
                    return StaticExtentAttr(*StaticExtentAttr.internal_parse_extent(parser))
                case "Sc":
                    return ScopeDefinedExtentAttr(*ScopeDefinedExtentAttr.internal_parse_extent(parser))
                case "I":
                    return InitDefinedExtentAttr(*InitDefinedExtentAttr.internal_parse_extent(parser))
                case "D":
                    return DynamicExtentAttr(*DynamicExtentAttr.internal_parse_extent(parser))

    @abstractmethod
    def internal_print_extent(self, printer: Printer) -> None:
        raise NotImplementedError()


class BaseExtent(Extent, ABC):
    def base_extents(self) -> list[Self]:
        return [self]


class AnyExtent(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, Extent):
            raise Exception(f"expected Extent Attribute but got {attr}")


class AnyStaticExtent(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, Extent):
            raise Exception(f"Expected Extent Attribute but got {attr}")
        if not attr.is_static():
            raise Exception(f"Expected Extent Attribute to be static, but got {attr}")


class AnyRunTimeExtent(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, Extent):
            raise Exception(f"Expected Extent Attribute but got {attr}")
        if attr.get_stage() < Stage.INIT:
            raise Exception(
                f"Expected Extent Attribute to be runtime (Init or Dynamic), "
                f"but got {attr.get_stage()} from {attr}"
            )


@irdl_attr_definition
class StaticExtentAttr(BaseExtent):
    name = "dlt.StaticExtent"
    value: ParameterDef[IntegerAttr]

    def __init__(self, value: IntegerAttr | IntAttr | int):
        if isinstance(value, int):
            value = IntegerAttr(value, IndexType())
        if isinstance(value, IntAttr):
            value = IntegerAttr(value, IndexType())
        super().__init__((value,))

    def get_stage(self) -> Stage:
        return Stage.STATIC

    def as_int(self) -> int:
        return self.value.value.data

    def verify(self) -> None:
        if self.value.value.data < 0:
            raise VerifyException(f"Extent cannot be negative: {self}")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_characters("S|")
            result = StaticExtentAttr.internal_parse_extent(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_extent(printer)

    @classmethod
    def internal_parse_extent(cls, parser: AttrParser) -> list[Attribute]:
        i = parser.parse_integer(allow_boolean= False, allow_negative = False)
        return [IntegerAttr(i, IndexType())]

    def internal_print_extent(self, printer: Printer) -> None:
        printer.print("S|")
        printer.print(self.value.value.data)


@irdl_attr_definition
class ScopeDefinedExtentAttr(BaseExtent):
    name = "dlt.ScopeDefinedExtent"

    value: ParameterDef[StringAttr]

    def __init__(self, value: StringAttr | str):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__((value,))

    def get_stage(self) -> Stage:
        return Stage.SCOPE

    def verify(self) -> None:
        pass

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_characters("Sc|")
            result = ScopeDefinedExtentAttr.internal_parse_extent(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_extent(printer)

    @classmethod
    def internal_parse_extent(cls, parser: AttrParser) -> list[Attribute]:
        s = parser.parse_identifier()
        return [StringAttr(s)]

    def internal_print_extent(self, printer: Printer) -> None:
        printer.print("Sc|")
        printer.print(self.value.data)



class RunTimeBaseExtent(BaseExtent, abc.ABC):
    @abc.abstractmethod
    def get_id(self) -> StringAttr:
        raise NotImplementedError()


class AnyRunTimeBaseExtent(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, RunTimeBaseExtent):
            raise Exception(f"Expected RuntimeExtent Attribute but got {attr}")
        if attr.get_stage() < Stage.INIT:
            raise Exception(
                f"Expected Extent Attribute to be runtime (Init or Dynamic), "
                f"but got {attr.get_stage()} from {attr}"
            )


@irdl_attr_definition
class InitDefinedExtentAttr(RunTimeBaseExtent):
    name = "dlt.InitDefinedExtent"

    value: ParameterDef[StringAttr]

    def __init__(self, value: StringAttr | str):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__((value,))

    def get_stage(self) -> Stage:
        return Stage.INIT

    def get_id(self) -> StringAttr:
        return self.value

    def verify(self) -> None:
        pass

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_characters("I|")
            result = InitDefinedExtentAttr.internal_parse_extent(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_extent(printer)

    @classmethod
    def internal_parse_extent(cls, parser: AttrParser) -> list[Attribute]:
        s = parser.parse_identifier()
        return [StringAttr(s)]

    def internal_print_extent(self, printer: Printer) -> None:
        printer.print("I|")
        printer.print(self.value.data)


@irdl_attr_definition
class DynamicExtentAttr(RunTimeBaseExtent):
    name = "dlt.DynamicExtent"

    value: ParameterDef[StringAttr]

    def __init__(self, value: StringAttr | str):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__((value,))

    def get_stage(self) -> Stage:
        return Stage.DYNAMIC

    def get_id(self) -> StringAttr:
        return self.value

    def verify(self) -> None:
        pass

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_characters("D|")
            result = DynamicExtentAttr.internal_parse_extent(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_extent(printer)

    @classmethod
    def internal_parse_extent(cls, parser: AttrParser) -> list[Attribute]:
        s = parser.parse_identifier()
        return [StringAttr(s)]

    def internal_print_extent(self, printer: Printer) -> None:
        printer.print("D|")
        printer.print(self.value.data)


@irdl_attr_definition
class DimensionAttr(ParametrizedAttribute):
    name = "dlt.dimension"
    dimensionName: ParameterDef[StringAttr]
    extent: ParameterDef[Extent]

    def __init__(
        self,
        dimensionName: StringAttr | str | Iterable[Attribute],
        extent: Extent | IntegerAttr | int = None,
    ):
        if extent is None:
            assert isinstance(dimensionName, Iterable)
            t = tuple(dimensionName)
            assert len(t) == 2
            dimensionName = t[0]
            extent = t[1]
        if isinstance(dimensionName, str):
            dimensionName = StringAttr(dimensionName)
        if isinstance(extent, int | IntegerAttr):
            extent = StaticExtentAttr(extent)
        super().__init__((dimensionName, extent))

    def is_static(self) -> bool:
        return self.extent.is_static()

    # extent: ParameterDef[StringAttr|(IntegerAttr[Annotated[IntegerType, i64]])]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            result = DimensionAttr.internal_parse_parameters(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        dn = StringAttr(parser.parse_identifier())
        parser.parse_punctuation(":")
        ext = Extent.internal_parse_any_extent(parser)
        # ext = parser.parse_attribute()
        return [dn, ext]

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print(self.dimensionName.data)
        printer.print(":")
        with printer.in_angle_brackets():
            self.extent.internal_print_extent(printer)
        # printer.print_attribute(self.extent)


@irdl_attr_definition
class ElementAttr(ParametrizedAttribute):
    name = "dlt.element"
    member_specifiers: ParameterDef[SetAttr[MemberAttr]]
    dimensions: ParameterDef[SetAttr[DimensionAttr]]
    base_type: ParameterDef[AcceptedTypes]

    def __init__(
        self,
        member_specifiers: (
            Iterable[MemberAttr | tuple[StringAttr | str, StringAttr | str]]
            | Iterable[Attribute]
        ),
        dimensions: Iterable[
            DimensionAttr | tuple[StringAttr | str, IntegerAttr | int]
        ] = None,
        base_type: AcceptedTypes = None,
    ):
        if base_type is None or dimensions is None:
            assert base_type is None and dimensions is None
            assert isinstance(member_specifiers, Iterable)
            t = tuple(member_specifiers)
            assert len(t) == 3
            member_specifiers = t[0]
            dimensions = t[1]
            base_type = t[2]
        members = []
        for m in member_specifiers:
            if isinstance(m, MemberAttr):
                members.append(m)
            elif isinstance(m, tuple):
                members.append(MemberAttr(m))
            else:
                raise ValueError(f"Unrecognized member_specifier argument: {m}")
        dims = []
        for d in dimensions:
            if isinstance(d, DimensionAttr):
                dims.append(d)
            elif isinstance(d, tuple):
                dims.append(DimensionAttr(d))
            else:
                raise ValueError(f"Unrecognized Dimension argument: {d}")
        super().__init__((SetAttr(members), SetAttr(dims), base_type))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[Attribute, ...]:
        with parser.in_angle_brackets():
            result = ElementAttr.internal_parse_parameters(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> tuple[Attribute, ...]:
        parser.parse_punctuation("(")
        ms = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES,
            lambda: MemberAttr(tuple(MemberAttr.internal_parse_parameters(parser))),
        )
        ms_set = SetAttr(tuple(ms))
        parser.parse_punctuation(",")
        dims = []
        baseType = parser.parse_optional_type()
        while baseType is None:
            dims.append(
                DimensionAttr(tuple(DimensionAttr.internal_parse_parameters(parser)))
            )
            parser.parse_punctuation("->")
            baseType = parser.parse_optional_type()
        parser.parse_punctuation(")")
        dims = SetAttr(tuple(dims))

        return tuple([ms_set, dims, baseType])

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print("(")
        MemberAttr.internal_print_members(self.member_specifiers, printer)
        printer.print(",")

        d_values = list(self.dimensions.data)
        sorted_d_values = [
            d_values[i]
            for s, i in sorted([(str(s), i) for i, s in enumerate(d_values)])
        ]
        for dim in sorted_d_values:
            dim.internal_print_parameters(printer)
            printer.print("->")
        printer.print(self.base_type)
        printer.print(")")

    def verify(self) -> None:
        dim_names = [dim.dimensionName for dim in self.dimensions]
        if len(dim_names) != len(set(dim_names)):
            raise VerifyException(
                "Dimensions in an dlt.element must not have repeated dimension names."
            )

    def get_dimension(self, name: StringAttr):
        for dim in self.dimensions:
            if name == dim.dimensionName:
                return dim
        return None

    def get_dimension_names(self):
        names = []
        for dim in self.dimensions:
            names.append(dim.dimensionName)
        return names

    def select_members(self, members: Iterable[MemberAttr]) -> ElementAttr:
        return ElementAttr(
            self.member_specifiers.difference(members), self.dimensions, self.base_type
        )

    def select_member(self, member: MemberAttr) -> ElementAttr:
        return self.select_members([member])

    def add_members(self, members: Iterable[MemberAttr]) -> ElementAttr:
        return ElementAttr(
            self.member_specifiers.add(members), self.dimensions, self.base_type
        )

    def add_member(self, member: MemberAttr) -> ElementAttr:
        return self.add_members([member])

    def select_dimensions(self, dimensions: Iterable[DimensionAttr]) -> ElementAttr:
        return ElementAttr(
            self.member_specifiers,
            self.dimensions.difference(dimensions),
            self.base_type,
        )

    def select_dimension(self, dimension: DimensionAttr) -> ElementAttr:
        return self.select_dimensions([dimension])

    def add_dimensions(self, dimensions: Iterable[DimensionAttr]) -> ElementAttr:
        return ElementAttr(
            self.member_specifiers, self.dimensions.add(dimensions), self.base_type
        )

    def add_dimension(self, dimension: DimensionAttr) -> ElementAttr:
        return self.add_dimensions([dimension])

    def has_members(self, members: Iterable[MemberAttr]):
        for m in members:
            if m not in self.member_specifiers:
                return False
        return True

    def has_dimensions(self, dimensions: Iterable[DimensionAttr]):
        for d in dimensions:
            if d not in self.dimensions:
                return False
        return True

    def has_selectable(
        self,
        members: Iterable[MemberAttr],
        dimensions: Iterable[DimensionAttr],
        base_type: AcceptedTypes = None,
    ) -> int:
        members = list(members)
        dimensions = list(dimensions)
        return self.has_members(members) and self.has_dimensions(dimensions) and (base_type is None or self.base_type == base_type)

@irdl_attr_definition
class TypeType(ParametrizedAttribute, TypeAttribute):
    name = "dlt.type"
    elements: ParameterDef[SetAttr[ElementAttr]]

    def __init__(
        self,
        elements: (
            Iterable[ElementAttr]
            | Iterable[
                tuple[
                    Iterable[MemberAttr | tuple[StringAttr | str, StringAttr | str]]
                    | Iterable[Attribute],
                    Iterable[
                        DimensionAttr
                        | tuple[StringAttr | str, StringAttr | str | IntegerAttr | int]
                    ],
                    AcceptedTypes,
                ]
            ]
        ),
    ):
        if not isinstance(elements, SetAttr):
            elems = []
            for element in elements:
                if isinstance(element, ElementAttr):
                    elems.append(element)
                else:
                    elems.append(ElementAttr(*element))
            elements = SetAttr(elems)
        super().__init__(tuple([elements]))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[Attribute, ...]:
        return cls.internal_parse_parameters(parser)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> tuple[Attribute, ...]:
        es = parser.parse_comma_separated_list(
            parser.Delimiter.ANGLE,
            lambda: ElementAttr(tuple(ElementAttr.internal_parse_parameters(parser))),
        )

        es_set = SetAttr(tuple(es))
        return tuple([es_set])

    def print_parameters(self, printer: Printer) -> None:
        self.internal_print_parameters(printer)

    def internal_print_parameters(self, printer: Printer) -> None:
        values = list(self.elements)
        sorted_values = [
            values[i]
            for s, i in sorted([(str(s), i) for i, s in enumerate(values)])
        ]
        with printer.in_angle_brackets():
            printer.print_list(
                sorted_values, lambda v: v.internal_print_parameters(printer)
            )

    def verify(self) -> None:
        # sorting not required if we can use SetAttr
        # assert self.elements.data == sorted(self.elements.data),\
        #     "Elements in the type must be sorted"
        if len(self.elements) < 1:
            raise VerifyException("TypeType must have at least one element")
        elems = [
            (elem.member_specifiers.data, elem.dimensions.data)
            for elem in self.elements
        ]
        if len(elems) != len(set(elems)):
            raise VerifyException(
                "Each element in the type must have a unique sets of memberSpecifiers"
            )

    def select_members(self, members: Iterable[MemberAttr]) -> TypeType:
        members = list(members)
        return TypeType(
            [
                e.select_members(members)
                for e in self.elements
                if all(member in e.member_specifiers.data for member in members)
            ]
        )

    def select_member(self, member: MemberAttr) -> TypeType:
        return self.select_members([member])

    def add_members(self, members: Iterable[MemberAttr]) -> TypeType:
        return TypeType([e.add_members(members) for e in self.elements])

    def add_member(self, member: MemberAttr) -> TypeType:
        return self.add_members([member])

    def select_dimensions(self, dimensions: Iterable[DimensionAttr]) -> TypeType:
        dimensions = list(dimensions)
        return TypeType(
            [
                e.select_dimensions(dimensions)
                for e in self.elements
                if all(dimension in e.dimensions.data for dimension in dimensions)
            ]
        )

    def select_dimension(self, dimension: DimensionAttr) -> TypeType:
        return self.select_dimensions([dimension])

    def add_dimensions(self, dimensions: Iterable[DimensionAttr]) -> TypeType:
        return TypeType([e.add_dimensions(dimensions) for e in self.elements])

    def add_dimension(self, dimension: DimensionAttr) -> TypeType:
        return self.add_dimensions([dimension])

    def with_selection(
        self,
        members: Iterable[MemberAttr],
        dimensions: Iterable[DimensionAttr],
        base_type: AcceptedTypes = None,
    ) -> TypeType:
        members = list(members)
        dimensions = list(dimensions)
        return TypeType(
            [
                e
                for e in self.elements
                if e.has_members(members)
                and e.has_dimensions(dimensions)
                and (base_type is None or e.base_type == base_type)
            ]
        )

    def has_selectable(
        self,
        members: Iterable[MemberAttr],
        dimensions: Iterable[DimensionAttr],
        base_type: AcceptedTypes = None,
    ) -> int:
        members = list(members)
        dimensions = list(dimensions)
        return len(
            [
                e
                for e in self.elements
                if e.has_members(members)
                and e.has_dimensions(dimensions)
                and (base_type is None or e.base_type == base_type)
            ]
        )

    def has_selectable_type(self, sub_type: TypeType) -> set[tuple[SetAttr[MemberAttr], SetAttr[DimensionAttr]]]:
        possible_selections = None
        for s_elem in sub_type.elements:
            possible_elems = [
                e for e in self.elements
                if e.has_members(s_elem.member_specifiers)
                and e.has_dimensions(s_elem.dimensions)
                and e.base_type == s_elem.base_type
            ]
            if len(possible_elems) == 0:
                return set()
            possible_elem_selections = [
                (e.member_specifiers.remove(s_elem.member_specifiers),
                 e.dimensions.remove(s_elem.dimensions))
                for e in possible_elems
            ]
            if possible_selections is None:
                possible_selections = {ps:{e:s_elem}
                    for e, ps in zip(possible_elems, possible_elem_selections)
                }
            else:
                possible_selections = {ps: map | {possible_elems[possible_elem_selections.index(ps)]:s_elem}
                                       for ps, map in possible_selections.items()
                                       if ps in possible_elem_selections
                                       and possible_elems[possible_elem_selections.index(ps)] not in map}
        return set(possible_selections.keys())




    def get_single_element(self) -> None | ElementAttr:
        if len(self.elements) == 1:
            return list(self.elements)[0]
        else:
            return None

    def all_member_attributes(self) -> set[MemberAttr]:
        return {m for elem in self.elements for m in elem.member_specifiers}

    def all_dimension_attributes(self) -> set[DimensionAttr]:
        return {d for elem in self.elements for d in elem.dimensions}

    def get_dimension(self, dimension: DimensionAttr) -> DimensionAttr:
        # for elem in self.elements:
        #     for dim in elem.dimensions:
        #         if name == dim.dimensionName:
        #             return dim
        # return None
        select_dim = None
        for elem in self.elements:
            for dim in elem.dimensions:
                if dimension.dimensionName == dim.dimensionName:
                    assert dimension.extent == dim.extent
                    assert dimension == dim
                    assert (
                        select_dim is None or select_dim == dim
                    )  # Embed this check every time we use get dim to ensure all dims of the same name have the same extent
                    select_dim = dim
        return select_dim




class Layout(ParametrizedAttribute, abc.ABC):

    @property
    @abstractmethod
    def contents_type(self) -> TypeType:
        pass

    def walk(self) -> Iterator[Layout]:
        yield self
        for child in self.get_children():
            yield from child.walk()

    @property
    def is_abstract(self) -> bool:
        return any(child.is_abstract for child in self.get_children())

    @abstractmethod
    def node_matches(self, other: Layout) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_children(self) -> list[Layout]:
        raise NotImplementedError()

    @abstractmethod
    def from_new_children(self, children: list[Layout]) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def get_stage(self) -> Stage :
        raise NotImplementedError()

    @abstractmethod
    def get_all_extents(self) -> set[Extent]:
        raise NotImplementedError()

    def get_all_runtime_base_extents(self) -> set[RunTimeBaseExtent]:
        return {e for e in self.get_all_extents() if isinstance(e, RunTimeBaseExtent)}

    def get_all_init_base_extents(self) -> set[RunTimeBaseExtent]:
        return {e for e in self.get_all_extents() if isinstance(e, InitDefinedExtentAttr)}

    def has_sub_layout(self, other: Layout) -> bool:
        for l in self.walk():
            if other == l:
                return True
        return False

    @abstractmethod
    def indexed_by(self) -> None | IndexRangeType | IndexType:
        raise NotImplementedError()

    def verify(self) -> None:
        if self.contents_type is None:
            raise VerifyException("Cannot produce contents type for layout")

    @staticmethod
    def abstract_layout(typetype: TypeType, name: StringAttr | str = None) -> Layout:
        parts = []
        for elem in typetype.elements:
            parts.append((elem.member_specifiers, elem.dimensions, PrimitiveLayoutAttr(elem.base_type)))
        layout = AbstractLayoutAttr(parts)

        # if name is not None:
        #     layout = NamedLayoutAttr(name, layout)

        return layout


class AnyDirectLayout(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, Layout):
            raise Exception(f"expected Layout Attribute but got {attr}")
        layout = cast(Layout, attr)
        if layout.indexed_by() != None:
            raise Exception(f"Layout indexed by {layout.indexed_by()} but None is expected for a Direct Layout.")

DirectLayout = typing.Annotated[Layout, AnyDirectLayout()]

class AnyIndexedLayout(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, Layout):
            raise Exception(f"expected Layout Attribute but got {attr}")
        layout = cast(Layout, attr)
        if layout.indexed_by() not in [IndexType(), IndexRangeType()]:
            raise Exception(f"Layout is not Indexed by Index or IndexRangeType, got {layout.indexed_by()} instead.")

IndexedLayout = typing.Annotated[Layout, AnyIndexedLayout()]


class KnowledgeLayout(Layout, abc.ABC):

    @property
    def is_knowledge(self) -> bool:
        return True

    @property
    def contents_type(self) -> TypeType:
        return self.get_child().contents_type

    def get_stage(self) -> Stage:
        return self.get_child().get_stage()

    def get_all_extents(self) -> set[Extent]:
        return self.get_child().get_all_extents()

    def get_child(self) -> Layout:
        return self.get_children()[0]

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return self.get_child().indexed_by()

    def can_be_dropped(self, dominating_knowledge: set[KnowledgeLayout]) -> bool:
        return False

    def can_be_injected(self, dominating_knowledge: set[KnowledgeLayout]) -> bool:
        return False

    @abstractmethod
    def get_parameters(self) -> tuple[Attribute, ...]:
        raise NotImplementedError()

    def matches(self, other: KnowledgeLayout) -> bool:
        return (
            isinstance(other, type(self))
            and self.get_parameters() == other.get_parameters()
        )

    def node_matches(self, other: Layout) -> bool:
        return  isinstance(other, type(self)) and self.matches(other)


@irdl_attr_definition
class ReadOnlyLayoutAttr(KnowledgeLayout):

    name = "dlt.layout.readOnly"
    child: ParameterDef[DirectLayout]

    def __init__(self, child: Layout):
        super().__init__((child,))

    def can_be_dropped(self, dominating_knowledge: set[KnowledgeLayout]) -> bool:
        return any(self.matches(k) for k in dominating_knowledge)

    def can_be_injected(self, dominating_knowledge: set[KnowledgeLayout]) -> bool:
        return not any(self.matches(k) for k in dominating_knowledge)

    def get_children(self) -> list[Layout]:
        return [self.child]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 1
        return ReadOnlyLayoutAttr(children[0])

    def verify(self) -> None:
        super().verify()

    def get_parameters(self) -> tuple[()]:
        return ()

#
# @irdl_attr_definition
# class NamedLayoutAttr(KnowledgeLayout):
#
#     name = "dlt.layout.named"
#     child: ParameterDef[AnyDirectLayout()]
#     abstract_name: ParameterDef[StringAttr]
#
#     def __init__(self, name: str | StringAttr, child: DirectLayout):
#         if isinstance(name, str):
#             name = StringAttr(name)
#         super().__init__((child, name))
#
#     @property
#     def contents_type(self) -> TypeType:
#         return self.child.contents_type
#
#     @property
#     def is_knowledge(self) -> bool:
#         return True
#
#     def get_name(self) -> str:
#         return self.abstract_name.data
#
#     def get_children(self) -> list[DirectLayout]:
#         return [self.child]
#
#     def from_new_children(self, children: list[DirectLayout]) -> Self:
#         assert len(children) == 1
#         return NamedLayoutAttr(self.abstract_name, children[0])
#
#     def get_stage(self) -> Stage :
#         return self.child.get_stage()
#
#     def get_all_extents(self) -> set[Extent]:
#         return self.child.get_all_extents()
#
#     def verify(self) -> None:
#         super().verify()
#         if self.abstract_name.data == "":
#             raise VerifyException(
#                 f"{self.name}: Named layout cannot have empty string name but found: "
#                 f"{self.abstract_name}"
#             )
#
#     def get_parameters(self) -> tuple[StringAttr]:
#         return (self.abstract_name,)


@irdl_attr_definition
class AbstractChildAttr(ParametrizedAttribute):
    name = "dlt.abstractChild"
    member_specifiers: ParameterDef[SetAttr[MemberAttr]]
    dimensions: ParameterDef[SetAttr[DimensionAttr]]
    child: ParameterDef[DirectLayout]

    def __init__(
        self,
        member_specifiers: Iterable[MemberAttr],
        dimensions: Iterable[DimensionAttr],
        child: Layout,
    ):
        super().__init__((SetAttr(member_specifiers), SetAttr(dimensions), child))

    def verify(self) -> None:
        super().verify()
        child = cast(Layout, self.child)
        try:
            contents_type = child.contents_type.add_members(
                self.member_specifiers
            ).add_dimensions(self.dimensions)
        except ValueError as e:
            raise VerifyException(
                "Unable to form contents type - probably because member or dimension appears in both the child and this attribute"
            ) from e

    @property
    def contents_type(self) -> TypeType:
        return self.child.contents_type.add_members(self.member_specifiers).add_dimensions(self.dimensions)

    def __lt__(self, other):
        if not isinstance(other, AbstractChildAttr):
            return NotImplemented
        if len(self.member_specifiers) < len(other.member_specifiers):
            return True
        if len(self.member_specifiers) > len(other.member_specifiers):
            return False

        self_members = sorted([(member.structName.data,member.memberName.data) for member in self.member_specifiers])
        other_members = sorted([(member.structName.data, member.memberName.data) for member in other.member_specifiers])
        if self_members < other_members:
            return True
        if self_members > other_members:
            return False

        if len(self.dimensions) < len(other.dimensions):
            return True
        if len(self.dimensions) > len(other.dimensions):
            return False

        self_dims = sorted([(dim.dimensionName.data, str(dim.extent)) for dim in self.dimensions])
        other_dims = sorted([(dim.dimensionName.data, str(dim.extent)) for dim in other.dimensions])
        if self_dims < other_dims:
            return True
        if self_dims > other_dims:
            return False

        return str(self.child) < str(other.child)


@irdl_attr_definition
class AbstractLayoutAttr(Layout):

    name = "dlt.layout.abstract"

    children: ParameterDef[ArrayAttr[AbstractChildAttr]]

    def __init__(
        self,
        children: Iterable[AbstractChildAttr | tuple[Iterable[MemberAttr], Iterable[DimensionAttr], Layout]],
    ):
        cs = []
        for child in children:
            if isinstance(child, AbstractChildAttr):
                cs.append(child)
            elif isinstance(child, tuple) and len(child) == 3:
                cs.append(AbstractChildAttr(*child))
            else:
                raise ValueError("Cannot produce Abstract Layout Attribute from child that is malformed")
        cs.sort()
        children = ArrayAttr(cs)
        if len(children) != len({c.contents_type for c in children}):
            raise ValueError(
                "Abstract Layout's children must have unique contents types"
            )
        super().__init__((children,))

    @property
    def contents_type(self) -> TypeType:
        return TypeType([e for child in self.children for e in child.contents_type.elements])

    @property
    def is_abstract(self) -> bool:
        return True

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None

    def verify(self) -> None:
        super().verify()
        if len(self.children) == 0:
            raise VerifyException(f"{self.name} must have at least 1 child layout")
        try:
            t = self.contents_type
        except ValueError | VerifyException as e:
            raise VerifyException("Abstract Layout's content Type cannot be formed") from e

    def get_children(self) -> list[Layout]:
        return [child.child for child in self.children]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == len(self.children)
        return AbstractLayoutAttr([AbstractChildAttr(child.member_specifiers, child.dimensions, new_child) for child, new_child in zip(self.children, children)])

    def get_stage(self) -> Stage:
        return max([dim.extent.get_stage() for child in self.children for dim in child.dimensions] + [child.child.get_stage() for child in self.children])

    def get_all_extents(self) -> set[Extent]:
        return {e for child in self.children for e in child.child.get_all_extents()} | {dim.extent for child in self.children for dim in child.dimensions}

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, AbstractLayoutAttr) and \
            len(self.children) == len(other.children) and \
            [c.member_specifers == o.member_specifiers and
             c.dimensions == o.dimensions
             for c, o in zip(self.children, other.children)]

    def common_abstract_dimensions(self) -> set[DimensionAttr]:
        return set.intersection(*[set(c.dimensions) for c in self.children])

    def common_abstract_members(self) -> set[MemberAttr]:
        return set.intersection(*[set(c.member_specifiers) for c in self.children])

@irdl_attr_definition
class PrimitiveLayoutAttr(Layout):
    name = "dlt.layout.primitive"
    base_type: ParameterDef[AcceptedTypes]

    def __init__(self, base_type: AcceptedTypes):
        super().__init__((base_type,))

    @property
    def contents_type(self) -> TypeType:
        return TypeType([ElementAttr([], [], self.base_type)])

    def get_children(self) -> list[Layout]:
        return []

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 0
        return PrimitiveLayoutAttr(self.base_type)

    def get_stage(self) -> Stage:
        return Stage.STATIC

    def get_all_extents(self) -> set[Extent]:
        return set()

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, PrimitiveLayoutAttr) and self.base_type == other.base_type

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None


@irdl_attr_definition
class ConstantLayoutAttr(Layout):
    name = "dlt.layout.constant"
    base_data: ParameterDef[AnyIntegerAttr | AnyFloatAttr]
    # base_type: ParameterDef[AcceptedTypes]

    def __init__(self, base_data: AnyIntegerAttr | AnyFloatAttr | int | float):
        if isinstance(base_data, int):
            base_data = IntegerAttr(base_data, IndexType())
        elif isinstance(base_data, float):
            base_data = FloatAttr(base_data, builtin.f32)
        super().__init__((base_data,))

    @property
    def contents_type(self) -> TypeType:
        return TypeType([ElementAttr([], [], self.base_data.type)])

    def get_children(self) -> list[Layout]:
        return []

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 0
        return ConstantLayoutAttr(self.base_data)

    def get_stage(self) -> Stage:
        return Stage.STATIC

    def get_all_extents(self) -> set[Extent]:
        return set()

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, ConstantLayoutAttr) and self.base_data == other.base_data

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None

    def verify(self) -> None:
        super().verify()
        if not isinstance(self.base_data.type, AcceptedTypes):
            raise VerifyException("Constant Layout cannot use a Constant type outside of AcceptedTypes")


@irdl_attr_definition
class DenseLayoutAttr(Layout):
    name = "dlt.layout.dense"
    child: ParameterDef[DirectLayout]
    dimension: ParameterDef[DimensionAttr]

    def __init__(
        self,
        child: Layout,
        dim: StringAttr | str | DimensionAttr,
        extent: StringAttr | str | IntegerAttr | int = None,
    ):
        if extent is None:
            assert isinstance(dim, DimensionAttr)
        else:
            dim = DimensionAttr(dim, extent)
        super().__init__((child, dim))

    @property
    def contents_type(self) -> TypeType:
        child_type: TypeType = self.child.contents_type
        elems = []
        for e in child_type.elements:
            elems.append(
                ElementAttr(
                    e.member_specifiers,
                    list(e.dimensions) + [self.dimension],
                    e.base_type,
                )
            )
        return TypeType(elems)

    def get_children(self) -> list[Layout]:
        return [self.child]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 1
        return DenseLayoutAttr(children[0], self.dimension)

    def get_stage(self) -> Stage:
        return max(self.dimension.extent.get_stage(), self.child.get_stage())

    def get_all_extents(self) -> set[Extent]:
        return self.child.get_all_extents() | {self.dimension.extent}

    def verify(self) -> None:
        super().verify()
        if self.dimension.extent.get_stage() > Stage.INIT:
            raise VerifyException(
                f"{self.name} does not support Dimensions with extent: {self.dimension.extent} "
                f"- it has an incompatible Stage."
            )

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, DenseLayoutAttr) and self.dimension == other.dimension

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None


@irdl_attr_definition
class MemberLayoutAttr(Layout):
    name = "dlt.layout.member"
    child: ParameterDef[DirectLayout]
    member_specifier: ParameterDef[MemberAttr]

    def __init__(
        self,
        child: Layout,
        member: StringAttr | str | MemberAttr,
        member_name: StringAttr | str = None,
    ):
        if member_name is None:
            assert isinstance(member, MemberAttr)
        else:
            member = MemberAttr(member, member_name)
        super().__init__((child, member))

    @property
    def contents_type(self) -> TypeType:
        child_type: TypeType = self.child.contents_type
        elems = []
        for e in child_type.elements:
            elems.append(
                ElementAttr(
                    list(e.member_specifiers) + [self.member_specifier],
                    e.dimensions,
                    e.base_type,
                )
            )
        return TypeType(elems)

    def get_children(self) -> list[Layout]:
        return [self.child]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 1
        return MemberLayoutAttr(children[0], self.member_specifier)

    def get_stage(self) -> Stage :
        return self.child.get_stage()

    def get_all_extents(self) -> set[Extent]:
        return self.child.get_all_extents()

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, MemberLayoutAttr) and self.member_specifier == other.member_specifier

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return self.child.indexed_by()


@irdl_attr_definition
class StructLayoutAttr(Layout):
    name = "dlt.layout.struct"
    children: ParameterDef[ArrayAttr[DirectLayout]]

    def __init__(self, children: Iterable[Layout]):
        if not isinstance(children, ArrayAttr):
            children = ArrayAttr(children)
        super().__init__((children,))
        assert self.contents_type

    @property
    def contents_type(self) -> TypeType:
        return TypeType([e for c in self.children for e in c.contents_type.elements])

    def get_children(self) -> list[Layout]:
        return list(self.children)

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == len(self.children)
        return StructLayoutAttr(children)

    def get_stage(self) -> Stage:
        return max([child.get_stage() for child in self.children])

    def get_all_extents(self) -> set[Extent]:
        return {e for child in self.children for e in child.get_all_extents()}

    def verify(self):
        super().verify()
        if self.get_stage() > Stage.INIT:
            raise VerifyException(
                f"{self.name} does not support the Dynamic Staged extents of the sub-layouts: "
                f"{[child for child in self.children if child.get_stage() > Stage.INIT]}"
            )

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, StructLayoutAttr) and len(self.children) == len(other.children)

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None


@irdl_attr_definition
class UnpackedCOOLayoutAttr(Layout):
    name = "dlt.layout.indexed.unpackedCOO"
    child: ParameterDef[DirectLayout]
    dimensions: ParameterDef[ArrayAttr[DimensionAttr]]
    buffer_scaler: ParameterDef[IntAttr]

    def __init__(self, child: Layout, dimensions: Iterable[DimensionAttr], buffer_scaler: int | IntAttr = 0):
        if not isinstance(dimensions, ArrayAttr):
            dimensions = ArrayAttr(dimensions)
        if not isinstance(buffer_scaler, IntAttr):
            buffer_scaler = IntAttr(buffer_scaler)

        super().__init__((child, dimensions, buffer_scaler))

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return IndexRangeType()

    @property
    def contents_type(self) -> TypeType:
        return self.child.contents_type.add_dimensions(self.dimensions)

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, UnpackedCOOLayoutAttr) and self.dimensions == other.dimensions

    def get_children(self) -> list[Layout]:
        return [self.child]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 1
        return UnpackedCOOLayoutAttr(children[0], self.dimensions, self.buffer_scaler)

    def get_stage(self) -> Stage :
        child_stage = self.child.get_stage()
        dimension_stages = [d.extent.get_stage() for d in self.dimensions]
        return max(*dimension_stages, child_stage)

    def get_all_extents(self) -> set[Extent]:
        extents = {d.extent for d in self.dimensions}
        return self.child.get_all_extents() | extents

    def is_buffered(self) -> bool:
        return self.buffer_scaler.data != 0


@irdl_attr_definition
class IndexingLayoutAttr(Layout):
    name = "dlt.layout.indexing"
    directChild: ParameterDef[DirectLayout]
    indexedChild: ParameterDef[IndexedLayout]

    def __init__(self, directChild: Layout, indexedChild: Layout):
        super().__init__((directChild, indexedChild))

    @property
    def contents_type(self) -> TypeType:
        idx_elems: SetAttr[ElementAttr] = self.indexedChild.contents_type.elements
        dir_elms: SetAttr[ElementAttr] = self.directChild.contents_type.elements
        new_elems = []
        for i_e in idx_elems:
            for d_e in dir_elms:
                new_elems.append(i_e.add_members(d_e.member_specifiers).add_dimensions(d_e.dimensions))
        # cross product of elems in the direct and indexed child. This allows for 
        return TypeType(new_elems)

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, IndexingLayoutAttr)

    def get_children(self) -> list[Layout]:
        return [self.directChild, self.indexedChild]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 2
        return IndexingLayoutAttr(children[0], children[1])

    def get_stage(self) -> Stage :
        return max([child.get_stage() for child in self.get_children()])

    def get_all_extents(self) -> set[Extent]:
        return {e for child in self.get_children() for e in child.get_all_extents()}

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None

    def verify(self) -> None:
        super().verify()
        if self.contents_type is None:
            raise VerifyException("Cannot produce contents type for layout")
        index_type = self.indexedChild.indexed_by()
        if index_type == None:
            raise VerifyException("Index child is indexed by None - this is not allow.")
        direct_child_contents = self.directChild.contents_type
        if direct_child_contents.get_single_element() is None:
            raise VerifyException("Indexed node does not support multi-tensor structures as the direct_child")
        direct_base_types = [e.base_type for e in direct_child_contents.elements]
        if not all(index_type == bt for bt in direct_base_types):
            raise VerifyException(f"Index child calls for index type; {index_type} but direct child has type "
                                  f"{direct_child_contents} with base types: {direct_base_types}")


@irdl_attr_definition
class ArithDropLayoutAttr(Layout):
    name = "dlt.layout.arith.drop"
    child: ParameterDef[DirectLayout]
    dimension: ParameterDef[DimensionAttr]

    def __init__(self, child: Layout, dim: DimensionAttr):
        super().__init__((child, dim))

    @property
    def contents_type(self) -> TypeType:
        return self.child.contents_type.add_dimension(self.dimension)

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, ArithDropLayoutAttr) and self.dimension == other.dimension

    def get_children(self) -> list[Layout]:
        return [self.child]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 1
        return ArithDropLayoutAttr(children[0], self.dimension)

    def get_stage(self) -> Stage :
        return max(self.child.get_stage(), self.dimension.extent.get_stage())

    def get_all_extents(self) -> set[Extent]:
        return self.child.get_all_extents() | {self.dimension.extent}

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None

@irdl_attr_definition
class ArithReplacementAttr(ParametrizedAttribute):
    name= "dlt.ArithReplacement"
    outer_dimension: ParameterDef[DimensionAttr]
    inner_dimension: ParameterDef[DimensionAttr]
    inner_member: ParameterDef[MemberAttr]

    def __init__(self, outer: DimensionAttr, inner: DimensionAttr, inner_member: MemberAttr):

        super().__init__((outer, inner, inner_member))

    def verify(self) -> None:
        super().verify()
        if self.outer_dimension.extent != self.inner_dimension.extent:
            raise VerifyException("ArithReplacement inner and outer dimension extents must be the same.")



@irdl_attr_definition
class ArithReplaceLayoutAttr(Layout):
    name = "dlt.layout.arith.replace"
    child: ParameterDef[DirectLayout]
    replacements: ParameterDef[SetAttr[ArithReplacementAttr]]

    def __init__(self, child: Layout, replacements: Iterable[ArithReplacementAttr]):
        super().__init__((child, SetAttr(replacements)))

    def verify(self) -> None:
        super().verify()
        inner_dims = {r.inner_dimension for r in self.replacements}
        if len(inner_dims) != 1:
            raise VerifyException("All replacements within a ArithReplacementLayout node must have the same inner dimension.")

        outer_dims = {r.outer_dimension for r in self.replacements}
        if len(outer_dims) != len(self.replacements):
            raise VerifyException(
                "All replacements within a ArithReplacementLayout node must have a unique outer dimension.")

        inner_members = {r.inner_member for r in self.replacements}
        if len(inner_members) != len(self.replacements):
            raise VerifyException(
                "All replacements within a ArithReplacementLayout node must have a unique inner member.")

        extents = {r.inner_dimension.extent for r in self.replacements} | {r.outer_dimension.extent for r in self.replacements}
        if len(extents) != 1:
            raise VerifyException("All dimensions used in ArithReplaceLayout must have the same extent")

    @property
    def contents_type(self) -> TypeType:
        new_elems = []
        child_elems = self.child.contents_type.elements
        for elem in child_elems:
            new_elem = elem
            for r in self.replacements:
                if elem.has_selectable([r.inner_member],[r.inner_dimension]):
                    new_elem = elem.select_member(r.inner_member).select_dimension(r.inner_dimension).add_dimension(r.outer_dimension)
                    break
            new_elems.append(new_elem)
        return TypeType(new_elems)

    def node_matches(self, other: Layout) -> bool:
        return isinstance(other, ArithReplaceLayoutAttr) and self.replacements == other.replacements

    def get_children(self) -> list[Layout]:
        return [self.child]

    def from_new_children(self, children: list[Layout]) -> Self:
        assert len(children) == 1
        return ArithReplaceLayoutAttr(children[0], self.replacements)

    def get_stage(self) -> Stage :
        return self.child.get_stage() # we don't change any extents so we can pass this call through

    def get_all_extents(self) -> set[Extent]:
        return self.child.get_all_extents() # we don't change any extents so we can pass this call through

    def indexed_by(self) -> None | IndexRangeType | IndexType:
        return None

    def replacement_for(self, dimension: DimensionAttr) -> None | ArithReplacementAttr:
        replacements = [r for r in self.replacements if r.outer_dimension == dimension]
        assert len(replacements) <= 1
        if replacements:
            return replacements.pop()
        else:
            return None

    def outer_dimensions(self) -> set[DimensionAttr]:
        return {r.outer_dimension for r in self.replacements}

    def inner_dimension(self) -> DimensionAttr:
        dims = {r.inner_dimension for r in self.replacements}
        assert len(dims) == 1
        return dims.pop()


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute):
    name = "dlt.ptr"
    contents_type: ParameterDef[TypeType]
    layout: ParameterDef[DirectLayout]
    filled_members: ParameterDef[SetAttr[MemberAttr]]
    filled_dimensions: ParameterDef[ArrayAttr[DimensionAttr]]
    filled_extents: ParameterDef[ArrayAttr[AnyRunTimeBaseExtent()]]
    base: ParameterDef[StringAttr]
    identification: ParameterDef[StringAttr]

    def __init__(
        self,
        type_type: TypeType,
        layout: Layout = None,
        members: SetAttr[MemberAttr] = SetAttr([]),
        dimensions: ArrayAttr[DimensionAttr] = ArrayAttr([]),
        extents: (
            Sequence[InitDefinedExtentAttr] | ArrayAttr[InitDefinedExtentAttr]
        ) = ArrayAttr([]),
        base: bool = False,
        identity: StringAttr | str = "",
    ):
        base = StringAttr("Y") if base else StringAttr("N")
        identity = StringAttr(identity) if isinstance(identity, str) else identity
        if layout is None:
            layout = Layout.abstract_layout(type_type)
        extents = ArrayAttr(extents)
        super().__init__(
            tuple([type_type, layout, members, dimensions, extents, base, identity])
        )

    @property
    def is_base(self):
        return self.base.data == "Y"

    def as_base(self) -> PtrType:
        return PtrType(
            self.contents_type,
            self.layout,
            self.filled_members,
            self.filled_dimensions,
            self.filled_extents,
            identity=self.identification,
            base=True,
        )

    def as_not_base(self) -> PtrType:
        return PtrType(
            self.contents_type,
            self.layout,
            self.filled_members,
            self.filled_dimensions,
            self.filled_extents,
            identity=self.identification,
            base=False,
        )

    @property
    def has_identity(self):
        return self.identification.data != ""

    def with_identification(self, identity: str | StringAttr) -> PtrType:
        return PtrType(
            self.contents_type,
            self.layout,
            self.filled_members,
            self.filled_dimensions,
            self.filled_extents,
            base=self.is_base,
            identity=identity,
        )

    def without_identification(self) -> PtrType:
        return PtrType(
            self.contents_type,
            self.layout,
            self.filled_members,
            self.filled_dimensions,
            self.filled_extents,
            base=self.is_base,
        )

    # def with_layout_name(self, name: str | StringAttr, preserve_ident=False) -> PtrType:
    #     return PtrType(
    #         self.contents_type,
    #         NamedLayoutAttr(name, self.layout),
    #         self.filled_members,
    #         self.filled_dimensions,
    #         self.filled_extents,
    #         base=self.is_base,
    #         identity=self.identification if preserve_ident else "",
    #     )

    def with_new_layout(
        self, layout: Layout, remove_bloat=False, preserve_ident=False
    ) -> PtrType:
        filled_members = set(self.filled_members)
        filled_dimensions = list(self.filled_dimensions)
        filled_extents = list(self.filled_extents)
        if remove_bloat:
            layout_contents_type = layout.contents_type
            all_members = layout_contents_type.all_member_attributes()
            filled_members = filled_members.intersection(all_members)
            contents_type = layout_contents_type.select_members(filled_members)
            useful_dims = contents_type.all_dimension_attributes()
            filled_dimensions = [d for d in filled_dimensions if d in useful_dims]
            contents_type = contents_type.select_dimensions(filled_dimensions)
            useful_extents = {e for e in layout.get_all_runtime_base_extents() if isinstance(e, InitDefinedExtentAttr)}
            filled_extents = [e for e in filled_extents if e in useful_extents]
        return PtrType(
            self.contents_type,
            layout,
            SetAttr(filled_members),
            ArrayAttr(filled_dimensions),
            ArrayAttr(filled_extents),
            base=self.is_base,
            identity=self.identification if preserve_ident else "",
        )

    def verify(self) -> None:
        layout_type: TypeType = self.layout.contents_type
        ptr_type = layout_type.select_members(self.filled_members)
        ptr_type = ptr_type.select_dimensions(self.filled_dimensions)
        if ptr_type != self.contents_type:
            raise VerifyException(
                f"{self.name}: layout does not provide the expected contents type"
            )

        extents = {e for e in self.layout.get_all_runtime_base_extents() if isinstance(e, InitDefinedExtentAttr)}
        for e in extents:
            if e not in self.filled_extents:
                raise VerifyException(
                    f"{self.name}: filled base extents does not have expected extent: {e}"
                )

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[Attribute, ...]:
        with parser.in_angle_brackets:
            return cls.internal_parse_parameters(parser)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> tuple[Attribute, ...]:
        ident = StringAttr(parser.parse_str_literal())
        parser.parse_punctuation(",")
        base = StringAttr("Y") if parser.parse_boolean() else StringAttr("N")
        parser.parse_punctuation(",")
        content_type = TypeType(*TypeType.internal_parse_parameters(parser))
        parser.parse_punctuation(",")
        filled_members = SetAttr(parser.parse_comma_separated_list(parser.Delimiter.BRACES,
                                                           lambda: MemberAttr(MemberAttr.internal_parse_parameters(parser))))
        parser.parse_punctuation(",")
        filled_dimensions = SetAttr(parser.parse_comma_separated_list(parser.Delimiter.SQUARE,
                                                                   lambda: DimensionAttr(
                                                                       DimensionAttr.internal_parse_parameters(parser))))
        parser.parse_punctuation(",")
        filled_extents = SetAttr(parser.parse_comma_separated_list(parser.Delimiter.SQUARE,
                                                                      lambda: InitDefinedExtentAttr(
                                                                          *InitDefinedExtentAttr.internal_parse_extent(
                                                                              parser))))
        parser.parse_punctuation(",")
        layout = parser.parse_attribute()
        return tuple([content_type, layout, filled_members, filled_dimensions, filled_extents, base, ident])

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print("\"")
        printer.print(self.identification.data)
        printer.print("\"")
        printer.print(", ")
        if self.is_base:
            printer.print("true")
        else:
            printer.print("false")
        printer.print_string(", ")
        self.contents_type.internal_print_parameters(printer)
        printer.print_string(", ")

        values = list(self.filled_members)
        sorted_values = [
            values[i]
            for s, i in sorted([(str(m), i) for i, m in enumerate(values)])
        ]
        printer.print("{")
        printer.print_list(
            sorted_values, lambda m: m.internal_print_parameters(printer)
        )
        printer.print("}")
        printer.print(", ")

        values = list(self.filled_dimensions)
        sorted_values = [
            values[i]
            for s, i in sorted([(str(d), i) for i, d in enumerate(values)])
        ]
        printer.print("[")
        printer.print_list(
            sorted_values, lambda d: d.internal_print_parameters(printer)
        )
        printer.print("]")
        printer.print(", ")

        values = list(self.filled_extents)
        sorted_values = [
            values[i]
            for s, i in sorted([(str(e), i) for i, e in enumerate(values)])
        ]
        printer.print("[")
        printer.print_list(
            sorted_values, lambda e: e.internal_print_extent(printer)
        )
        printer.print("]")
        printer.print(", ")

        printer.print_attribute(self.layout)


# @irdl_attr_definition
# class PtrBaseType(ParametrizedAttribute, TypeAttribute):
#     name = "dlt.ptrBase"
#     ptr_type: ParameterDef[PtrType]
#
#     def __init__(self, ptr_type: PtrType):
#         super().__init__(tuple([ptr_type]))
#
#     @property
#     def contents_type(self) -> TypeType:
#         return self.ptr_type.contents_type


#
#
#
# @irdl_op_definition
# class StructOp(IRDLOperation):
#     name = "dlt.struct"
#
#     res: OpResult = result_def(TypeType)
#     region: Region = region_def("single_block")
#
#     def verify_(self) -> None:
#         isa(self.region.block.last_op, StructYieldOp)
#         if self.res.type != self.region.block.last_op.output_type():
#             raise VerifyException("Struct result type must be the dlt.type corrosponding to the elements in the yield op")
#         pass
#
# @irdl_op_definition
# class StructYieldOp(IRDLOperation):
#     name = "dlt.structYield"
#     arguments: VarOperand = var_operand_def(TypeType)
#
#     traits = traits_def(
#         lambda: frozenset([IsTerminator(), HasParent(StructOp)])
#     )
#
#     def verify_(self) -> None:
#         elements = [elem for arg in self.arguments if isinstance(arg.type, TypeType) for elem in arg.type.elements]
#         TypeType(elements) # check that making this type doesn't cause an error
#
#     def output_type(self):
#         elements = [elem for arg in self.arguments if isinstance(arg.type, TypeType) for elem in arg.type.elements]
#         type = TypeType(elements)
#         return type
#
# #TODO
#
# @irdl_op_definition
# class IndexingOp(IRDLOperation):
#     name = "dlt.indexing"
# #TODO
#
# @irdl_op_definition
# class MemberOp(IRDLOperation):
#     name = "dlt.member"
# #TODO
#
#
#
# @irdl_op_definition
# class PrimitiveOp(IRDLOperation):
#     name = "dlt.primitive"
#     of: AcceptedTypes = attr_def(AcceptedTypes)
#     res: OpResult = result_def(TypeType)
#
#     def __init__(self, of: AcceptedTypes):
#         type = TypeType([ElementAttr(tuple([SetAttr([]), SetAttr([]), of]))])
#         super().__init__(attributes={"of":of}, result_types=[type])
# #TODO
#
# @irdl_op_definition
# class ConstOp(IRDLOperation):
#     name = "dlt.const"
#     value: AnyFloatAttr | IntegerAttr = attr_def(AnyFloatAttr | IntegerAttr)
#     res: OpResult = result_def(TypeType)
#
# #TODO
#
# @irdl_op_definition
# class DenseOp(IRDLOperation):
#     name = "dlt.dense"
#     child: OperandDef = operand_def(TypeType)
#     dimension: DimensionAttr = attr_def(DimensionAttr)
#     res: OpResult = result_def(TypeType)
#
#     def verify_(self) -> None:
#         elements = [elem for elem in self.child.type.elements]
#         new_elements = []
#         for elem in elements:
#             dims = list(elem.dimensions)
#             dims.append(self.dimension)
#             new_dims = SetAttr(dims)
#             new_elem = ElementAttr(tuple([elem.member_specifiers, new_dims, elem.base_type]))
#             new_elements.append(new_elem)
#         new_type = TypeType(new_elements)
#         res_type = self.res.type
#         if new_type != res_type:
#             raise VerifyException("Result type does not match input type with added dimension")
#
# #TODO
#
# @irdl_op_definition
# class UnpackedCoordinateFormatOp(IRDLOperation):
#     name = "dlt.upcoo"
# #TODO
#
# @irdl_op_definition
# class IndexAffineOp(IRDLOperation):
#     name = "dlt.indexAffine"
# #TODO


@irdl_op_definition
class LayoutScopeOp(IRDLOperation):
    name = "dlt.layoutScope"  # Be the point that all internal dlt operations use a reference and source for layout information
    extent_names: ArrayAttr[ScopeDefinedExtentAttr] = attr_def(
        ArrayAttr[ScopeDefinedExtentAttr]
    )
    extent_values: ArrayAttr[IntegerAttr] = attr_def(ArrayAttr[IntegerAttr])
    body: Region = region_def("single_block")
    traits = frozenset(
        [
            NoTerminator(),
        ]
    )

    def __init__(
        self,
        scope_extents: Sequence[
            tuple[ScopeDefinedExtentAttr | StringAttr | str, IntegerAttr | int]
        ],
        ops: list[Operation] | Block | Region,
    ):
        extent_names = []
        extent_values = []
        for e, v in scope_extents:
            if isinstance(e, StringAttr | str):
                e = ScopeDefinedExtentAttr(e)
            if isinstance(v, int):
                v = IntegerAttr(v, IndexType())
            extent_names.append(e)
            extent_values.append(v)
        extent_names = ArrayAttr(extent_names)
        extent_values = ArrayAttr(extent_values)

        if isinstance(ops, Region):
            region = ops
        elif isinstance(ops, Block):
            region = Region(ops)
        else:
            region = Region(Block(ops))

        super().__init__(
            regions=[region],
            attributes={"extent_names": extent_names, "extent_values": extent_values},
        )

    def get_function_map(self) -> dict[StringAttr, tuple[func.FuncOp, list[func.Call]]]:
        funcs: dict[StringAttr, tuple[func.FuncOp, list[func.Call]]] = {}
        for op in self.walk():
            if isinstance(op, func.FuncOp):
                op: func.FuncOp = op
                if op.sym_name in funcs:
                    raise ComplexVerifyException(
                        "Duplicate function name found. "
                        "FuncOp names must be unique within a scope",
                        {
                            funcs[op.sym_name][
                                0
                            ]: f"This func was found first, with name: "
                            f"{funcs[op.sym_name][0].sym_name}.",
                            op: f"This function found later with an equal name: {op.sym_name}.",
                        },
                    )
                funcs[op.sym_name] = (op, [])
        for op in self.walk():
            if isinstance(op, func.Call):
                op: func.Call = op
                func_name = op.callee.root_reference
                if func_name not in funcs:
                    raise ComplexVerifyException(
                        "func.call's callee cannot be found within the dlt Scope.",
                        {
                            op: f"This func.Call calls {func_name} which has not been found."
                        },
                    )
                assert func_name in funcs
                func_op, calls = funcs[func_name]
                calls.append(op)
        return funcs

    def get_call_map(self) -> dict[func.Call, set[func.FuncOp]]:
        call_map: dict[func.Call, func.FuncOp] = {}
        for op in self.walk():
            if isinstance(op, func.Call):
                call_op: func.Call = op
                parent = op.parent_op()
                while not isinstance(parent, func.FuncOp | LayoutScopeOp):
                    parent = parent.parent_op()
                if isinstance(parent, func.FuncOp):
                    call_map[call_op] = parent
        return call_map

    def verify_(self) -> None:
        function_map = self.get_function_map()
        call_map = self.get_call_map()

        violations = {}
        for function, calls in function_map.values():
            caller_functions: set[func.FuncOp] = set()
            callers = {call for call in calls}

            caller_functions_changed = True
            while caller_functions_changed:
                caller_functions_changed = False
                for call in callers:
                    if call in call_map and call_map[call] not in caller_functions:
                        caller_functions_changed = True
                        caller_functions.add(call_map[call])
                for caller_function in caller_functions:
                    op, op_calls = function_map[caller_function.sym_name]
                    for op_call in op_calls:
                        callers.add(op_call)

            if function in caller_functions:
                for i, type in enumerate(function.function_type.inputs):
                    if isinstance(type, PtrType):
                        if not type.has_identity:
                            violations.setdefault(function, []).append(
                                f"This function's input {i} must have an dlt.ptr identitification"
                            )
                for i, type in enumerate(function.function_type.outputs):
                    if isinstance(type, PtrType):
                        if not type.has_identity:
                            violations.setdefault(function, []).append(
                                f"This function's output {i} must have an dlt.ptr identitification"
                            )
        if violations:
            raise ComplexVerifyException(
                "Recursive Functions must name the layouts of their dlt.PtrType "
                "inputs and outputs to maintain correctness during layout generation",
                violations,
            )

        if len(self.extent_values) != len(self.extent_names):
            raise VerifyException(
                f"{self.name}: lengths of extent names and extent values must match"
            )
        if len(self.body.block.args) != 0:
            raise VerifyException(
                f"{self.name}: lengths of body block arguments must be 0"
            )


class DTLLayoutScopedOp(IRDLOperation):
    traits = traits_def(lambda: frozenset([HasAncestor(LayoutScopeOp)]))

    def get_scope(self) -> LayoutScopeOp:
        parent = self.parent_op()
        while not isinstance(parent, LayoutScopeOp):
            parent = parent.parent_op()
        return parent

    @staticmethod
    def get_scope_for(op: Operation) -> LayoutScopeOp | None:
        parent = op.parent_op()
        while not isinstance(parent, LayoutScopeOp):
            parent = parent.parent_op()
        return parent


@irdl_op_definition
class SelectOp(DTLLayoutScopedOp):
    name = "dlt.select"  # take a ptrType and constrain a member field or a dimension value.
    tree: Operand = operand_def(PtrType)
    members: SetAttr[MemberAttr] = attr_def(SetAttr[MemberAttr])
    dimensions: ArrayAttr[DimensionAttr] = attr_def(ArrayAttr[DimensionAttr])
    values: VarOperand = var_operand_def(IndexType)

    res: OpResult = result_def(PtrType)

    def __init__(
        self,
        tree: SSAValue | Operation,
        members: SetAttr[MemberAttr] | Iterable[MemberAttr],
        dimensions: ArrayAttr[DimensionAttr] | Iterable[DimensionAttr],
        values: Sequence[SSAValue | Operation],
        result_type: Attribute | None = None,
    ):
        if not isinstance(tree, SSAValue):
            tree = SSAValue.get(tree)
        if not isinstance(members, SetAttr):
            members = SetAttr(members)
        if not isinstance(dimensions, ArrayAttr):
            dimensions = ArrayAttr(dimensions)
        if not isinstance(values, tuple):
            values = tuple(values)
        if result_type is None:
            assert isinstance(tree.type, PtrType)
            result_type = SelectOp.calculateResultType(tree.type, members, dimensions)
            result_type = result_type.as_not_base()
        super().__init__(
            operands=[tree, values],
            result_types=[result_type],
            attributes={"members": members, "dimensions": dimensions},
        )

    def verify_(self) -> None:

        tree_type: PtrType = cast(PtrType, self.tree.type)
        res_type: PtrType = cast(PtrType, self.res.type)
        try:
            derived_output_contents_type = tree_type.contents_type.select_members(
                self.members
            ).select_dimensions(self.dimensions)
        except VerifyException as e:
            raise VerifyException(
                f"{self.name}. Problem deriving contents type from input type given. It is probably "
                f"because the selection of members and dimensions produces an empty type."
            ) from e
        if derived_output_contents_type != res_type.contents_type:
            raise VerifyException(
                f"{self.name} contents type mis-match: input has content: {tree_type.contents_type}, "
                f"which given members: {self.members}, and dimensions: {self.dimensions}, "
                f"should result in {derived_output_contents_type}, but result type given has: "
                f"{res_type.contents_type}"
            )
        if res_type.is_base:
            raise VerifyException(
                "Once selected, a Ptr-Type cannot be a memory handle base"
            )
        if not tree_type.layout.has_sub_layout(res_type.layout):
            raise VerifyException(
                "The resulting layout of a Select Op should be a sub-tree of the starting ptr layout."
            )
        # maybe we should calculate if the layouts are valid here too, but this requires some infrastructure for finding
        # sub-layouts from the layout given the selected members and dimensions - and there is not a unique answer as
        # we don't *need* to enforce that the layout is minimal at all steps given the filled_members and
        # filled_dimensions of the tree and res types.

    @classmethod
    def parse(cls: type[SelectOp], parser: Parser) -> SelectOp:
        # dlt.select{root:e, node:x}(A:0, B:10) from %1
        ms = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES,
            lambda: MemberAttr(tuple(MemberAttr.internal_parse_parameters(parser))),
        )
        members = SetAttr(ms)

        def parseDim() -> tuple[DimensionAttr, SSAValue]:
            # ident = parser.parse_identifier()
            dim = DimensionAttr(tuple(DimensionAttr.internal_parse_parameters(parser)))
            parser.parse_punctuation(":")
            operand = parser.parse_operand()
            return (dim, operand)

        dims = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parseDim)
        dim_attrs, dim_operands = zip(*dims)
        dimensions = ArrayAttr(dim_attrs)
        parser.parse_keyword("from")
        tree = parser.parse_operand()

        if parser.parse_optional_punctuation(":"):
            parser.parse_punctuation("(")
            tree_type = parser.parse_type()
            if tree.type != tree_type:
                parser.raise_error(
                    f"Type given: {tree_type} does not match expected: {tree.type}"
                )
            parser.parse_punctuation(")")
            parser.parse_punctuation("->")
            res_type = parser.parse_type()
            res_type_clac = SelectOp.calculateResultType(tree.type, members, dimensions)
            if res_type != res_type_clac:
                parser.raise_error(
                    f"parsed type {res_type} does not match expected type f{res_type_clac}"
                )
        else:
            res_type = SelectOp.calculateResultType(tree.type, members, dimensions)

        selectOp = SelectOp(tree, members, dimensions, dim_operands, res_type)
        return selectOp

    @staticmethod
    def calculateResultType(
        input_type: PtrType,
        members: Iterable[MemberAttr],
        dimensions: Iterable[DimensionAttr],
    ) -> PtrType:
        new_type = input_type.contents_type.select_members(members).select_dimensions(
            dimensions
        )

        current_filled_members = input_type.filled_members.add(members)
        current_filled_dimensions = input_type.filled_dimensions
        assert all(d not in current_filled_dimensions for d in dimensions)
        current_filled_dimensions = ArrayAttr(
            list(current_filled_dimensions.data) + list(dimensions)
        )
        new_content_type: TypeType = input_type.layout.contents_type
        new_content_type = new_content_type.select_members(
            current_filled_members
        ).select_dimensions(current_filled_dimensions)
        assert new_content_type == new_type

        return PtrType(
            new_type,
            input_type.layout,
            current_filled_members,
            current_filled_dimensions,
            input_type.filled_extents,
            input_type.is_base,
        )

    def print(self, printer: Printer):
        MemberAttr.internal_print_members(self.members, printer)

        def print_d_v(dv: tuple[DimensionAttr, SSAValue]):
            d, v = dv
            d.internal_print_parameters(printer)
            printer.print(":")
            printer.print(v)

        printer.print("(")
        printer.print_list(zip(self.dimensions.data, self.values), print_d_v)
        printer.print(")")
        printer.print(" from ")
        printer.print(self.tree)
        printer.print(" : ")
        printer.print("(")
        printer.print(self.tree.type)
        printer.print(")")
        printer.print(" -> ")
        printer.print(self.res.type)


@irdl_op_definition
class GetOp(DTLLayoutScopedOp):
    name = "dlt.get"  # take a PtrType that points only to primitives (no member fields or dimensions) and get the value
    tree: Operand = operand_def(PtrType)
    get_type: AcceptedTypes = attr_def(AcceptedTypes)
    res: OpResult = result_def(AcceptedTypes)

    def __init__(self, tree: SSAValue, get_type: AcceptedTypes):
        super().__init__(
            operands=[tree], result_types=[get_type], attributes={"get_type": get_type}
        )

    def verify_(self) -> None:
        ptr_type = cast(PtrType, self.tree.type)
        elem = ptr_type.contents_type.get_single_element()
        if elem is None:
            raise VerifyException(
                f"{self.name} cannot get into ptr that has more than one element"
            )
        if len(elem.dimensions) > 0:
            raise VerifyException(
                f"{self.name} cannot get into ptr that has dimensions"
            )
        if len(elem.member_specifiers) > 0:
            raise VerifyException(
                f"{self.name} cannot get into ptr that has unspecified members"
            )
        if self.get_type != elem.base_type:
            raise VerifyException(f"{self.name} cannot get {self.get_type} from {elem}")

@irdl_op_definition
class GetSOp(DTLLayoutScopedOp):
    name = "dlt.gets"  # take a PtrType that points only to primitives (no member fields or dimensions) and get the value
    tree: Operand = operand_def(PtrType)
    get_type: AcceptedTypes = attr_def(AcceptedTypes)
    res: OpResult = result_def(AcceptedTypes)
    found: OpResult = result_def(IntegerType(1))

    def __init__(self, tree: SSAValue, get_type: AcceptedTypes):
        super().__init__(
            operands=[tree], result_types=[get_type, IntegerType(1)], attributes={"get_type": get_type}
        )

    def verify_(self) -> None:
        ptr_type = cast(PtrType, self.tree.type)
        elem = ptr_type.contents_type.get_single_element()
        if elem is None:
            raise VerifyException(
                f"{self.name} cannot get into ptr that has more than one element"
            )
        if len(elem.dimensions) > 0:
            raise VerifyException(
                f"{self.name} cannot get into ptr that has dimensions"
            )
        if len(elem.member_specifiers) > 0:
            raise VerifyException(
                f"{self.name} cannot get into ptr that has unspecified members"
            )
        if self.get_type != elem.base_type:
            raise VerifyException(f"{self.name} cannot get {self.get_type} from {elem}")

@irdl_op_definition
class SetOp(DTLLayoutScopedOp):
    name = "dlt.set"  # take a PtrType that points only to primitives (no member fields or dimensions) and set the value
    tree: Operand = operand_def(PtrType)
    value: Operand = operand_def(AcceptedTypes)
    set_type: AcceptedTypes = attr_def(AcceptedTypes)

    def __init__(
        self, tree: SSAValue, set_type: AcceptedTypes, value: SSAValue | Operation
    ):
        super().__init__(operands=[tree, value], attributes={"set_type": set_type})

    def verify_(self) -> None:
        ptr_type = cast(PtrType, self.tree.type)
        elem = ptr_type.contents_type.get_single_element()
        if elem is None:
            raise VerifyException(
                f"{self.name} cannot set into ptr that has more than one element"
            )
        if len(elem.dimensions) > 0:
            raise VerifyException(
                f"{self.name} cannot set into ptr that has dimensions"
            )
        if len(elem.member_specifiers) > 0:
            raise VerifyException(
                f"{self.name} cannot set into ptr that has unspecified members"
            )
        if self.set_type != elem.base_type:
            raise VerifyException(f"{self.name} cannot set {self.set_type} into {elem}")


@irdl_op_definition
class ExtractExtentOp(DTLLayoutScopedOp):
    name = "dlt.extractExtent"
    tree: Operand = operand_def(PtrType)
    extent: Extent = attr_def(Extent)
    res: OpResult = result_def(IndexType())

    def __init__(self, tree: SSAValue, extent: Extent):
        super().__init__(
            operands=[tree], result_types=[IndexType()], attributes={"extent": extent}
        )

    def verify_(self) -> None:
        AnyExtent().verify(self.extent, {})
        ptr_type = cast(PtrType, self.tree.type)
        if self.extent not in ptr_type.filled_extents:
            raise VerifyException(
                "Cannot extract Extent from ptr that does not contain that extent"
            )


@irdl_op_definition
class CopyOp(DTLLayoutScopedOp):
    name = "dlt.copy"  # take src and dst Ptrtypes and copy all values of the copy_type primitive from one to the other.
    src: Operand = operand_def(PtrType)
    dst: Operand = operand_def(PtrType)
    src_dimensions: ArrayAttr[DimensionAttr] = attr_def(ArrayAttr[DimensionAttr])
    dst_dimensions: ArrayAttr[DimensionAttr] = attr_def(ArrayAttr[DimensionAttr])
    copy_type: AcceptedTypes = attr_def(AcceptedTypes)

    def __init__(
        self,
        src: SSAValue,
        src_dims: Iterable[DimensionAttr],
        dst: SSAValue,
        dst_dims: Iterable[DimensionAttr],
        copy_type: AcceptedTypes,
    ):
        if not isinstance(src_dims, ArrayAttr):
            src_dims = ArrayAttr(src_dims)
        if not isinstance(dst_dims, ArrayAttr):
            dst_dims = ArrayAttr(dst_dims)

        super().__init__(
            operands=[src, dst],
            attributes={
                "src_dimensions": src_dims,
                "dst_dimensions": dst_dims,
                "copy_type": copy_type,
            },
        )
        self.check_inputs()

    def verify_(self) -> None:
        self.check_inputs()

    def check_inputs(self) -> None:
        for name, operand, dims in [
            ("src", self.src, self.src_dimensions),
            ("dst", self.dst, self.dst_dimensions),
        ]:
            assert isinstance(operand.type, PtrType)
            ptr_type = cast(PtrType, operand.type)
            try:
                inner_type = ptr_type.contents_type.select_dimensions(dims)
            except VerifyException as e:
                raise VerifyException(
                    f"{self.name}: Cannot construct copy from {name} {operand} with dims: {dims}"
                )
            if inner_type.get_single_element() is None:
                raise VerifyException(
                    f"{self.name}: Cannot construct copy from {name} {operand} with dims: {dims} as result does not have single element."
                )

    # TODO Verify the tree layout types match perfectly


@irdl_op_definition
class ClearOp(DTLLayoutScopedOp):
    name = "dlt.clear"  # take a Ptrtype and set all the values of clear_type to 0 - possibly changing the runtime sparsity
    tree: Operand = operand_def(PtrType)
    # dimensions: AttributeDef = attr_def(SetAttr[DimensionAttr])
    clear_type: AttributeDef = attr_def(AcceptedTypes)


@irdl_op_definition
class IterateYieldOp(AbstractYieldOperation[Attribute], DTLLayoutScopedOp):
    name = "dlt.iterateYield"

    traits = traits_def(lambda: frozenset([IsTerminator(), HasParent(IterateOp)]))


class IterateUseDefChainTrait(UseDefChainTrait):

    @classmethod
    def get_defs_following_from_operand_for(
        cls, op: Operation, index: int
    ) -> set[SSAValue]:
        if isinstance(op, IterateOp):
            other_args = len(op.extent_args) + len(op.tensors)
            if index >= other_args:
                arg_index = index - other_args
                other_block_args = len(op.extents) + len(op.tensors)
                return {
                    op.body.block.args[other_block_args + arg_index],
                    op.res[arg_index],
                }
        elif isinstance(op, IterateYieldOp):
            parent_op = op.parent_op()
            if isinstance(parent_op, IterateOp):
                return {parent_op.body.block.args[index + 1], parent_op.res[index]}
        return set()

    @classmethod
    def get_operands_leading_to_op_result_for(
        cls, op: Operation, index: int
    ) -> set[Use]:
        if isinstance(op, IterateOp):
            other_args = len(op.extent_args) + len(op.tensors)
            uses = {Use(op, other_args + index)}
            yield_op = op.body.block.last_op
            if isinstance(yield_op, IterateYieldOp):
                uses.add(Use(yield_op, index))
            return uses
        return set()

    @classmethod
    def get_operands_leading_to_block_argument_for(
        cls, op: Operation, arg: BlockArgument
    ) -> set[Use]:
        if isinstance(op, IterateOp):
            other_block_args = len(op.extents) + len(op.tensors)
            if arg.index >= other_block_args:
                other_args = len(op.extent_args) + len(op.tensors)
                return {Use(op, other_args + (arg.index - other_block_args))}
        return set()


class IterationOrderBase(ParametrizedAttribute, abc.ABC):

    @abc.abstractmethod
    def get_extent_indices(self) -> set[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_tensor_indices(self) -> set[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_children(self) -> list[IterationOrder]:
        raise NotImplementedError

    @abc.abstractmethod
    def from_children(self, children: list[IterationOrder]) -> Self:
        raise NotImplementedError

    def is_abstract(self) -> bool:
        return any(child.is_abstract for child in self.get_children())


class AnyIterationOrder(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, IterationOrderBase):
            raise Exception(f"expected Iteration Order Attribute but got {attr}")


IterationOrder = typing.Annotated[IterationOrderBase, AnyIterationOrder()]


@irdl_attr_definition
class AbstractIterationOrderAttr(IterationOrderBase):
    name = "dlt.iteration.abstract"

    extent_indices: ParameterDef[SetAttr[IntAttr]]
    child: ParameterDef[IterationOrder]

    def __init__(self, extent_indices: Iterable[int | IntAttr], child: IterationOrder):
        if not isinstance(extent_indices, SetAttr):
            idxs = set()
            for i in extent_indices:
                if not isinstance(i, IntAttr):
                    i = IntAttr(i)
                idxs.add(i)
            extent_indices = SetAttr(idxs)
        if len(extent_indices) == 0:
            raise ValueError("Cannot of abstract order with no extents")
        super().__init__((extent_indices, child,))

    def get_extent_indices(self) -> set[int]:
        child = self.child.get_extent_indices()
        return {i.data for i in self.extent_indices} | child

    def get_tensor_indices(self) -> set[int]:
        return self.child.get_tensor_indices()

    def get_children(self) -> list[IterationOrder]:
        return [self.child]

    def from_children(self, children: list[IterationOrder]) -> Self:
        assert len(children) == 1
        return AbstractIterationOrderAttr(self.extent_indices, children[0])

    def is_abstract(self) -> bool:
        return True

    def verify(self) -> None:
        super().verify()
        if duplicates := (set(self.extent_indices) & self.child.get_extent_indices()):
            raise VerifyException(f"Child produces extent indices {duplicates} that are also present in this parent {self.name}")

    @staticmethod
    def generate_for(extents_indices: list[int]) -> IterationOrder:
        base = BodyIterationOrderAttr()
        if len(extents_indices) == 0:
            return base
        else:
            return AbstractIterationOrderAttr(extents_indices, base)


@irdl_attr_definition
class BodyIterationOrderAttr(IterationOrderBase):
    name = "dlt.iteration.body"

    def get_extent_indices(self) -> set[int]:
        return set()

    def get_tensor_indices(self) -> set[int]:
        return set()

    def get_children(self) -> list[IterationOrder]:
        return []

    def from_children(self, children: list[IterationOrder]) -> Self:
        assert len(children) == 0
        return BodyIterationOrderAttr()


@irdl_attr_definition
class NestedIterationOrderAttr(IterationOrderBase):
    name = "dlt.iteration.nested"

    extent_index: ParameterDef[IntAttr]
    child: ParameterDef[IterationOrder]

    def __init__(self, extent_index: int | IntAttr, child: IterationOrder):
        if isinstance(extent_index, int):
            extent_index = IntAttr(extent_index)
        super().__init__((extent_index, child,))

    def get_extent_indices(self) -> set[int]:
        child = self.child.get_extent_indices()
        if self.extent_index.data in child:
            raise ValueError()
        return {self.extent_index.data} | child

    def get_tensor_indices(self) -> set[int]:
        return self.child.get_tensor_indices()

    def get_children(self) -> list[IterationOrder]:
        return [self.child]

    def from_children(self, children: list[IterationOrder]) -> Self:
        assert len(children) == 1
        return NestedIterationOrderAttr(self.extent_index, children[0])

    def verify(self) -> None:
        super().verify()
        if self.extent_index in self.child.get_extent_indices():
            raise VerifyException(f"Child produces extent index {self.extent_index} that is also present in this parent {self.name}")

    @staticmethod
    def generate_for(num_extents: list[int] | list[IntAttr], body: IterationOrder = None) -> IterationOrder:
        if len(num_extents) == 0:
            return BodyIterationOrderAttr() if body is None else body
        idx = num_extents.pop(0)
        return NestedIterationOrderAttr(idx, NestedIterationOrderAttr.generate_for(num_extents, body=body))


@irdl_attr_definition
class NonZeroIterationOrderAttr(IterationOrderBase):
    name = "dlt.iteration.nonzero"

    extent_index: ParameterDef[IntAttr]
    tensor_index: ParameterDef[IntAttr]
    child: ParameterDef[IterationOrder]

    def __init__(self, extent_index: int | IntAttr, tensor_index: int | IntAttr, child: IterationOrder):
        if isinstance(extent_index, int):
            extent_index = IntAttr(extent_index)
        if isinstance(tensor_index, int):
            tensor_index = IntAttr(tensor_index)
        super().__init__((extent_index, tensor_index, child,))

    def get_extent_indices(self) -> set[int]:
        child = self.child.get_extent_indices()
        if self.extent_index.data in child:
            raise ValueError()
        return {self.extent_index.data} | child

    def get_tensor_indices(self) -> set[int]:
        child = self.child.get_tensor_indices()
        if self.tensor_index.data in child:
            raise ValueError()
        return {self.tensor_index.data} | child

    def get_children(self) -> list[IterationOrder]:
        return [self.child]

    def from_children(self, children: list[IterationOrder]) -> Self:
        assert len(children) == 1
        return NonZeroIterationOrderAttr(self.extent_index, self.tensor_index, children[0])

    def verify(self) -> None:
        super().verify()
        if self.extent_index in self.child.get_extent_indices():
            raise VerifyException(f"Child produces extent index {self.extent_index} that is also present in this parent {self.name}")
        if self.tensor_index in self.child.get_tensor_indices():
            raise VerifyException(f"Child produces tensor index {self.tensor_index} that is also present in this parent {self.name}")


@irdl_op_definition
class IterateOp(DTLLayoutScopedOp):
    name = "dlt.iterate"  # Iterate over a multiple dimension-extent pairs, given some context tensors that might be used inside.
    # TODO attribute for type of itteration - Non-zero vs whole space
    extents: ArrayAttr[Extent] = attr_def(ArrayAttr[Extent])  # extents to iterate over.
    extent_args: VarOperand = var_operand_def(
        IndexType
    )  # len(extent_args) == len([e in extents if !e.is_static()])
    dimensions: ArrayAttr[ArrayAttr[SetAttr[DimensionAttr]]] = attr_def(
        ArrayAttr[ArrayAttr[SetAttr[DimensionAttr]]]
    )
    # for each tensor operand, store [{set of dimension where dimension.extent == e} for e in extents]
    tensors: VarOperand = var_operand_def(PtrType)

    order: IterationOrder = attr_def(IterationOrder)
    identification: StringAttr = attr_def(StringAttr)

    iter_args: VarOperand = var_operand_def(
        AnyAttr()
    )  # any other args for the loop that will be returned

    res: VarOpResult = var_result_def(AnyAttr())

    body: Region = region_def("single_block")
    # block args:    [induction_arg: IndexType() for e in extents]
    #              + [tensor: PtrType<t.type.select(dims)> for (t, dims) in zip(tensors, dimension_names)]
    #              + [iter_arg for iter_arg in iter_args]

    traits = frozenset(
        [SingleBlockImplicitTerminator(IterateYieldOp), HasAncestor(LayoutScopeOp)]
    )
    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        extents: Sequence[Extent],
        extent_args: Sequence[SSAValue | Operation],
        dimensions: Iterable[Iterable[Iterable[DimensionAttr]]],
        tensors: Sequence[SSAValue],
        iter_args: Sequence[SSAValue | Operation],
        order: IterationOrder = None,
        identification: StringAttr | str = None,
        body: Region | Sequence[Operation] | Sequence[Block] | Block = None,
    ):
        if body is None:
            body = Block()
            arg_idx = 0
            for e in extents:
                body.insert_arg(IndexType(), arg_idx)
                arg_idx += 1
            for t, t_ds in zip(tensors, dimensions):
                t_dimensions = [d for ds in t_ds for d in ds]
                ptr_type = SelectOp.calculateResultType(t.type, [], t_dimensions)
                body.insert_arg(ptr_type, arg_idx)
                arg_idx += 1
            body_iter_args = []
            for arg in iter_args:
                body_iter_args.append(body.insert_arg(SSAValue.get(arg).type, arg_idx))
                arg_idx += 1
            body.add_op(IterateYieldOp(*body_iter_args))

        if isinstance(body, Block):
            body = [body]

        if order is None:
            order = AbstractIterationOrderAttr.generate_for(list(range(len(extents))))
        if identification is None:
            identification = ""
        if isinstance(identification, str):
            identification = StringAttr(identification)

        dimensions = ArrayAttr(
            [
                ArrayAttr([SetAttr([dim for dim in dim_set]) for dim_set in t_dims])
                for t_dims in dimensions
            ]
        )
        extents = ArrayAttr([e for e in extents])

        super().__init__(
            operands=[extent_args, tensors, tuple(iter_args)],
            result_types=[[SSAValue.get(a).type for a in iter_args]],
            regions=[body],
            attributes={"extents": extents, "dimensions": dimensions, "order": order, "identification": identification},
        )

    def verify_(self):
        if (len(self.extents) + len(self.tensors) + len(self.iter_args)) != len(
            self.body.block.args
        ):
            raise VerifyException(
                f"Wrong number of block arguments, expected"
                f" {(len(self.extents) + len(self.tensors) + len(self.iter_args))}, got {len(self.body.block.args)}. "
                f"The body must have:"
                f"the induction variables,"
                f"selected tensors, and "
                f"loop-carried variables as arguments."
            )
        if self.body.block.args:
            induction_vars = [
                (i, self.body.block.args[i], extent)
                for i, extent in enumerate(self.extents)
            ]
            extent_arg_idx = 0
            for i, induction_var, extent in induction_vars:
                if induction_var.type != IndexType():
                    raise VerifyException(
                        f"The first {len(self.extents)} block arguments are expected to be of type "
                        f"IndexType(), but {induction_var} was found at index {i}."
                    )
                if extent.get_stage() >= Stage.INIT:
                    if len(self.extent_args) <= extent_arg_idx:
                        raise VerifyException(
                            f"There are not enough extent_args for the extents provided. extent {extent} "
                            f"expects to use extent_arg at index {i} but this is out of bounds."
                        )
                    extent_arg_idx += 1

            if extent_arg_idx != len(self.extent_args):
                raise VerifyException(f"Extent args expected {extent_arg_idx} but {len(self.extent_args)} were supplied")

            if len(self.tensors) != len(self.dimensions):
                raise VerifyException(f"Expected number of tensors and length of outer array of dimensions to be equal. Found {len(self.tensors)} Tensors but {len(self.dimensions)} tensot dimension specifications")
            tensor_vars = [
                (i, i - len(self.extents), self.body.block.args[i])
                for i in range(len(self.extents), len(self.extents) + len(self.tensors))
            ]
            for i, t, tensor_var in tensor_vars:
                tensor_dim_sets = self.dimensions.data[t]
                if len(tensor_dim_sets) != len(self.extents):
                    raise VerifyException(
                        f"Dimension sets for self.tensor[{t}] does not match number of extents. "
                        f"Dimensions gotten: {tensor_dim_sets}, extents: {self.extents}."
                    )
                all_dims = []
                for dim_set, extent in zip(tensor_dim_sets, self.extents):
                    for dim in dim_set:
                        all_dims.append(dim)
                        if dim.extent != extent:
                            raise VerifyException(
                                f"Cannot use extent {extent} in dimension {dim} for tensor "
                                f"{tensor_var} as the extents do no match."
                            )
                if len(all_dims) != len(set(all_dims)):
                    raise VerifyException(
                        f"Cannot select with {tensor_dim_sets} as it contains duplicate dimensions."
                    )
                tensor_arg = self.tensors[t]
                tensor_arg_type = cast(PtrType, tensor_arg.type)
                if (
                    tensor_arg_type.contents_type.select_dimensions(all_dims)
                    != tensor_var.type.contents_type
                ):
                    raise VerifyException(
                        f"Block argument with wrong contents_type, expected "
                        f"{tensor_arg_type.contents_type.select_dimensions(all_dims)} for got "
                        f"{tensor_var.type.contents_type}"
                    )
                if not tensor_arg_type.layout.has_sub_layout(tensor_var.type.layout):
                    raise VerifyException(
                        f"Block argument with inconsistent layout, input layout is "
                        f"{tensor_arg_type.layout}, but block argument has layout: "
                        f"{tensor_var.type.layout}"
                    )

            iter_arg_vars = [
                (i, i - len(self.extents) - len(self.tensors), self.body.block.args[i])
                for i in range(
                    len(self.extents) + len(self.tensors),
                    len(self.extents) + len(self.tensors) + len(self.iter_args),
                )
            ]
            for i, iter, iter_arg_var in iter_arg_vars:
                if iter_arg_var.type != self.iter_args[iter].type:
                    raise VerifyException(
                        f"Block arguments with wrong type, expected {self.iter_args[iter].type}, got "
                        f"{iter_arg_var.type}. At block arg index {i}"
                    )

        if len(self.body.ops) > 0 and isinstance(
            self.body.block.last_op, IterateYieldOp
        ):
            yieldop = self.body.block.last_op
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The dlt.iterate must yield its carried variables."
                )
            for idx, arg in enumerate(yieldop.arguments):
                if self.iter_args[idx].type != arg.type:
                    raise VerifyException(
                        f"Expected {self.iter_args[idx].type}, got {arg.type}. The "
                        f"dlt.iterate's dlt.iterateYield must match carried variables types."
                    )

        self.verify_order()

    def verify_order(self, order: IterationOrder = None, e: type[Exception] = VerifyException):
        if order is None:
            order = self.order
        if not set(range(len(self.tensors))).issuperset(order.get_tensor_indices()):
            raise e(f"Order {order} refers to tensors indices: {order.get_tensor_indices()} but iterate op only has {len(self.tensors)} tensor args")
        if not set(range(len(self.extents))) == (order.get_extent_indices()):
            raise e(f"Expected to find extent indices: [0,{len(self.extents)}) but got {order.get_extent_indices()}")

    @property
    def has_identity(self):
        return self.identification.data != ""

    def set_identity(self, ident: str|StringAttr):
        if isinstance(ident, str):
            ident = StringAttr(ident)
        self.identification = ident

    def get_input_arg_for_block_arg(self, block_arg: BlockArgument) -> SSAValue:
        assert block_arg.block.parent_op() is self
        idx = block_arg.index
        idx -= len(self.extents)
        if idx - len(self.tensors) < 0:
            return self.tensors[idx]
        idx -= len(self.tensors)
        if idx - len(self.iter_args) < 0:
            return self.iter_args[idx]
        raise ValueError(f"Cannot find input for block arg: {block_arg}")

    def get_yield_arg_for_result(self, result: OpResult) -> SSAValue:
        assert result in self.res
        for i, r in enumerate(self.res):
            if result == r:
                yield_op = self.body.block.last_op
                assert isinstance(yield_op, IterateYieldOp)
                return yield_op.arguments[i]
        assert False

    def get_result_for_yield_use(self, use: Use):
        assert use.operation == self.get_yield_op()
        return self.results[use.index]

    def get_yield_op(self) -> IterateYieldOp:
        op = self.body.block.last_op
        assert isinstance(op, IterateYieldOp)
        return op

    def get_block_arg_and_dims_for_input_arg(
        self, use: Use
    ) -> tuple[BlockArgument, ArrayAttr[SetAttr[DimensionAttr]] | None]:
        assert use.operation == self
        idx = use.index
        idx -= len(self.extent_args)
        assert idx >= 0
        if idx < len(self.tensors):
            assert use in self.tensors[idx].uses
            return (
                self.body.block.args[len(self.extents) + idx],
                self.dimensions.data[idx],
            )
        idx -= len(self.tensors)
        assert 0 <= idx < len(self.iter_args)
        assert use in self.iter_args[idx].uses
        return self.body.block.args[len(self.extents) + len(self.tensors) + idx], None

    def get_block_arg_for_input_arg(self, use: Use) -> BlockArgument:
        assert use.operation == self
        idx = use.index
        idx -= len(self.extent_args)
        assert idx >= 0
        if idx < len(self.tensors):
            assert use in self.tensors[idx].uses
            return self.body.block.args[len(self.extents) + idx]
        idx -= len(self.tensors)
        assert 0 <= idx < len(self.iter_args)
        assert use in self.iter_args[idx].uses
        return self.body.block.args[len(self.extents) + len(self.tensors) + idx]

    def get_block_arg_for_iter_arg_idx(self, index: int) -> BlockArgument:
        assert 0 <= index < len(self.iter_args)
        return self.body.block.args[len(self.extents) + len(self.tensors) + index]

    def get_result_for_iter_arg_idx(self, index: int) -> OpResult:
        assert 0 <= index < len(self.iter_args)
        return self.res[index]

    def get_block_args_for_extent_args(self) -> list[BlockArgument]:
        return list(self.body.block.args[0:len(self.extents)])

    def get_block_args_for_tensor_args(self) -> list[BlockArgument]:
        return list(self.body.block.args[len(self.extents):len(self.extents)+len(self.tensors)])

    def get_block_arg_for_tensor_arg_idx(self, index: int) -> BlockArgument:
        index += len(self.extents)
        assert index < len(self.extents) + len(self.tensors)
        return self.body.block.args[index]

    def get_result_for_input_arg(self, use: Use) -> tuple[OpResult, int]:
        assert use.operation == self
        idx = use.index
        idx -= len(self.extent_args) + len(self.tensors)
        assert 0 <= idx < len(self.iter_args)
        assert use in self.iter_args[idx].uses
        return self.res[idx], idx

    # def print(self, printer: Printer):
    #     block = self.body.block
    #     indices = [block.args[i] for i in range(len(self.dimensions))]
    #     iter_args = [block.args[i] for i in range(len(self.dimensions), len(block.args))]
    #     printer.print_string(" ")
    #     printer.print_list(label =
    #         zip(indices, self.iter_args),
    #         lambda pair: print_assignment(printer, *pair),
    #     )
    #     printer.print_ssa_value(index)
    #     printer.print_string(" = ")
    #     printer.print_ssa_value(self.lb)
    #     printer.print_string(" to ")
    #     printer.print_ssa_value(self.ub)
    #     printer.print_string(" step ")
    #     printer.print_ssa_value(self.step)
    #     printer.print_string(" ")
    #     if iter_args:
    #         printer.print_string("iter_args(")
    #         printer.print_list(
    #             zip(iter_args, self.iter_args),
    #             lambda pair: print_assignment(printer, *pair),
    #         )
    #         printer.print_string(") -> (")
    #         printer.print_list((a.type for a in iter_args), printer.print_attribute)
    #         printer.print_string(") ")
    #     printer.print_region(
    #         self.body,
    #         print_entry_block_args=False,
    #         print_empty_block=False,
    #         print_block_terminators=bool(iter_args),
    #     )


# TODO


@irdl_op_definition
class AllocOp(DTLLayoutScopedOp):
    name = "dlt.alloc"  # do the memory allocations etc to form a ptrType

    # layout: OperandDef = operand_def(TypeType)
    initialValues: VarOperand = var_operand_def(PtrType)
    init_extent_sizes: VarOperand = var_operand_def(
        IndexType
    )  # these do not mean unknown sizes that are static per dlt layout scope, but Init in that they can change within the layout scope.
    init_extents: ArrayAttr[InitDefinedExtentAttr] = attr_def(
        ArrayAttr[InitDefinedExtentAttr]
    )

    res: OpResult = result_def(PtrType)

    irdl_options = [AttrSizedOperandSegments()]

    def __init__(
        self,
        ptr: PtrType | TypeType,
        extents: dict[InitDefinedExtentAttr, SSAValue],
        initial_values: Sequence[SSAValue] = None,
    ) -> None:
        if len(extents) > 0:
            extents, extent_vars = zip(*extents.items())
        else:
            extents, extent_vars = [], []
        if isinstance(ptr, TypeType):
            ptr = PtrType(ptr).as_base()
        if initial_values is None:
            initial_values = []

        super().__init__(
            operands=[initial_values, extent_vars],
            attributes={"init_extents": builtin.ArrayAttr(extents)},
            result_types=[ptr],
        )

    def init_extent_mapping(self):
        return {e: s for e, s in zip(self.init_extents, self.init_extent_sizes)}

    def verify_(self) -> None:
        res_type = cast(PtrType, self.res.type)
        if not res_type.is_base:
            raise VerifyException(
                "PtrType result of AllocOp must have 'base' attribute set to \"Y\""
            )
        if len(res_type.filled_dimensions.data) != 0:
            raise VerifyException(
                f"{self.name}: result of Alloc cannot have prefilled dimension selectors"
            )

        extents = [
            e
            for e in cast(PtrType, self.res.type).layout.get_all_extents()
            if cast(Extent, e).is_init_time()
        ]
        if not all(e in self.init_extents for e in extents):
            raise VerifyException(
                "When allocating a layout with InitDefinedExtents, these must be provided"
            )


@irdl_op_definition
class DeallocOp(DTLLayoutScopedOp):
    name = "dlt.dealloc"  # take a dlt layout as a TypeType and do the memory allocations etc to form a ptrType

    tree: Operand = operand_def(PtrType)

    def __init__(
        self,
        ptr: SSAValue,
    ) -> None:
        super().__init__(
            operands=[ptr],
            attributes={},
            result_types=[],
        )

    def verify_(self) -> None:
        if not isinstance(self.tree.type, PtrType):
            raise VerifyException("DeallocOp can only deallocate a dlt.ptr operand")
        if not self.tree.type.is_base:
            raise VerifyException(
                "PtrType argument of DeallocOp must have 'base' attribute set to \"Y\""
            )


# TODO


@irdl_op_definition
class AssertLayoutOp(DTLLayoutScopedOp):
    name = "dlt.assert"  # take a dlt layout as a TypeType and assert that a given memref has the layout to form a ptrType


# TODO


DLT = Dialect(
    "DLT",
    [  # ops
        # StructOp,
        # StructYieldOp,
        # IndexingOp,
        # MemberOp,
        # PrimitiveOp,
        # ConstOp,
        # DenseOp,
        # UnpackedCoordinateFormatOp,
        # IndexAffineOp,
        LayoutScopeOp,
        SelectOp,
        GetOp,
        GetSOp,
        SetOp,
        CopyOp,
        ClearOp,
        IterateYieldOp,
        IterateOp,
        AllocOp,
        DeallocOp,
    ],
    [  # attrs
        SetAttr,
        MemberAttr,
        StaticExtentAttr,
        ScopeDefinedExtentAttr,
        DimensionAttr,
        ElementAttr,
        TypeType,
        IndexRangeType,
        PrimitiveLayoutAttr,
        ConstantLayoutAttr,
        # NamedLayoutAttr,
        AbstractLayoutAttr,
        StructLayoutAttr,
        DenseLayoutAttr,
        IndexingLayoutAttr,
        UnpackedCOOLayoutAttr,
        ArithDropLayoutAttr,
        PtrType,
    ],
)
