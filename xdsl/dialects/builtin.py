from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import Self, deprecated

from xdsl.ir import (
    Attribute,
    AttributeCovT,
    Block,
    BlockOps,
    Data,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
    TypedAttribute,
)
from xdsl.ir.affine import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineMap,
    AffineSet,
    AffineSymExpr,
)
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    GenericAttrConstraint,
    GenericData,
    IRDLOperation,
    MessageConstraint,
    ParamAttrConstraint,
    ParameterDef,
    attr_constr_coercion,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    opt_attr_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    IsolatedFromAbove,
    NoMemoryEffect,
    NoTerminator,
    OptionalSymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import DiagnosticException, VerifyException
from xdsl.utils.isattr import isattr

if TYPE_CHECKING:
    from xdsl.parser import AttrParser, Parser
    from xdsl.printer import Printer

DYNAMIC_INDEX = -1
"""
A constant value denoting a dynamic index in a shape.
"""


class ShapedType(ABC):
    @abstractmethod
    def get_num_dims(self) -> int: ...

    @abstractmethod
    def get_shape(self) -> tuple[int, ...]: ...

    def element_count(self) -> int:
        return prod(self.get_shape())

    @staticmethod
    def strides_for_shape(shape: Sequence[int], factor: int = 1) -> tuple[int, ...]:
        import operator
        from itertools import accumulate

        return tuple(accumulate(reversed(shape), operator.mul, initial=factor))[-2::-1]


class AnyShapedType(AttrConstraint):
    def verify(
        self, attr: Attribute, constraint_context: ConstraintContext | None = None
    ) -> None:
        if not isinstance(attr, ShapedType):
            raise Exception(f"expected type ShapedType but got {attr}")


_ContainerElementTypeT = TypeVar(
    "_ContainerElementTypeT", bound=Attribute | None, covariant=True
)


class ContainerType(Generic[_ContainerElementTypeT], ABC):
    @abstractmethod
    def get_element_type(self) -> _ContainerElementTypeT:
        pass


@irdl_attr_definition
class NoneAttr(ParametrizedAttribute):
    """An attribute representing the absence of an attribute."""

    name = "none"


@dataclass(frozen=True)
class ArrayOfConstraint(AttrConstraint):
    """
    A constraint that enforces an ArrayData whose elements all satisfy
    the elem_constr.
    """

    elem_constr: AttrConstraint

    def __init__(self, constr: Attribute | type[Attribute] | AttrConstraint):
        object.__setattr__(self, "elem_constr", attr_constr_coercion(constr))

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if not isinstance(attr, ArrayAttr):
            raise VerifyException(f"expected ArrayData attribute, but got {attr}")
        for e in cast(ArrayAttr[Attribute], attr).data:
            self.elem_constr.verify(e, constraint_context)


@irdl_attr_definition
class ArrayAttr(GenericData[tuple[AttributeCovT, ...]], Iterable[AttributeCovT]):
    name = "array"

    def __init__(self, param: Iterable[AttributeCovT]) -> None:
        super().__init__(tuple(param))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[AttributeCovT, ...]:
        with parser.in_angle_brackets():
            data = parser.parse_comma_separated_list(
                parser.Delimiter.SQUARE, parser.parse_attribute
            )
            # the type system can't ensure that the elements are of type _ArrayAttrT
            result = cast(tuple[AttributeCovT, ...], tuple(data))
            return result

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        assert len(args) == 1
        return ArrayOfConstraint(irdl_to_attr_constraint(args[0]))

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[AttributeCovT]:
        return iter(self.data)


AnyArrayAttr: TypeAlias = ArrayAttr[Attribute]


@irdl_attr_definition
class StringAttr(Data[str]):
    name = "string"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')


@irdl_attr_definition
class BytesAttr(Data[bytes]):
    name = "bytes"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> bytes:
        with parser.in_angle_brackets():
            return parser.parse_bytes_literal()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')


@irdl_attr_definition
class SymbolNameAttr(ParametrizedAttribute):
    name = "symbol_name"
    data: ParameterDef[StringAttr]

    def __init__(self, data: str | StringAttr) -> None:
        if isinstance(data, str):
            data = StringAttr(data)
        super().__init__([data])


@irdl_attr_definition
class SymbolRefAttr(ParametrizedAttribute):
    name = "symbol_ref"
    root_reference: ParameterDef[StringAttr]
    nested_references: ParameterDef[ArrayAttr[StringAttr]]

    def __init__(
        self,
        root: str | StringAttr,
        nested: Sequence[str] | Sequence[StringAttr] | ArrayAttr[StringAttr] = [],
    ) -> None:
        if isinstance(root, str):
            root = StringAttr(root)
        if not isinstance(nested, ArrayAttr):
            nested = ArrayAttr(
                [StringAttr(x) if isinstance(x, str) else x for x in nested]
            )
        super().__init__([root, nested])

    def string_value(self):
        root = self.root_reference.data
        for ref in self.nested_references.data:
            root += "." + ref.data
        return root


class EmptyArrayAttrConstraint(AttrConstraint):
    """
    Constrain attribute to be empty ArrayData
    """

    def verify(
        self, attr: Attribute, constraint_context: ConstraintContext | None = None
    ) -> None:
        if not isinstance(attr, ArrayAttr):
            raise VerifyException(f"expected ArrayData attribute, but got {attr}")
        attr = cast(ArrayAttr[Attribute], attr)
        if attr.data:
            raise VerifyException(f"expected empty array, but got {attr}")


FlatSymbolRefAttrConstraint = MessageConstraint(
    ParamAttrConstraint(SymbolRefAttr, [AnyAttr(), EmptyArrayAttrConstraint()]),
    "Unexpected nested symbols in FlatSymbolRefAttr.",
)
"""Constrain SymbolRef to be FlatSymbolRef"""

FlatSymbolRefAttr = Annotated[SymbolRefAttr, FlatSymbolRefAttrConstraint]
"""SymbolRef constrained to have an empty `nested_references` property."""

FlatSymbolRefAttrConstr = AllOf((base(SymbolRefAttr), FlatSymbolRefAttrConstraint))


@irdl_attr_definition
class IntAttr(Data[int]):
    name = "builtin.int"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        with parser.in_angle_brackets():
            data = parser.parse_integer()
            return data

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(f"{self.data}")


class Signedness(Enum):
    "Signedness semantics for integer"

    SIGNLESS = 0
    "No signedness semantics"

    SIGNED = 1
    UNSIGNED = 2

    def value_range(self, bitwidth: int) -> tuple[int, int]:
        """
        For a given bitwidth, returns (min, max+1), where min and max are the smallest and
        largest representable values.

        Signless integers are bit patterns, so the representable range is the union of the
        signed and unsigned representable ranges.
        """
        match self:
            case Signedness.SIGNLESS:
                min_value = -((1 << bitwidth) >> 1)
                max_value = 1 << bitwidth
            case Signedness.SIGNED:
                min_value = -((1 << bitwidth) >> 1)
                max_value = 1 << (bitwidth - 1)
            case Signedness.UNSIGNED:
                min_value = 0
                max_value = 1 << bitwidth

        return min_value, max_value


@irdl_attr_definition
class SignednessAttr(Data[Signedness]):
    name = "builtin.signedness"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> Signedness:
        with parser.in_angle_brackets():
            if parser.parse_optional_keyword("signless") is not None:
                return Signedness.SIGNLESS
            if parser.parse_optional_keyword("signed") is not None:
                return Signedness.SIGNED
            if parser.parse_optional_keyword("unsigned") is not None:
                return Signedness.UNSIGNED
            parser.raise_error("`signless`, `signed`, or `unsigned` expected")

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            data = self.data
            if data == Signedness.SIGNLESS:
                printer.print_string("signless")
            elif data == Signedness.SIGNED:
                printer.print_string("signed")
            elif data == Signedness.UNSIGNED:
                printer.print_string("unsigned")
            else:
                raise ValueError(f"Invalid signedness {data}")


class FixedBitwidthType(TypeAttribute, ABC):
    """
    A type attribute with a defined bitwidth
    """

    name = "abstract.bitwidth_type"

    @property
    @abstractmethod
    def bitwidth(self) -> int:
        """
        Contiguous memory footprint in bits
        """
        raise NotImplementedError()

    @property
    def size(self) -> int:
        """
        Contiguous memory footprint in bytes, defaults to `ceil(bitwidth / 8)`
        """
        return self.bitwidth >> 3 + bool(self.bitwidth % 8)


@irdl_attr_definition
class IntegerType(ParametrizedAttribute, FixedBitwidthType):
    name = "integer_type"
    width: ParameterDef[IntAttr]
    signedness: ParameterDef[SignednessAttr]

    def __init__(
        self,
        data: int | IntAttr,
        signedness: Signedness | SignednessAttr = Signedness.SIGNLESS,
    ) -> None:
        if isinstance(data, int):
            data = IntAttr(data)
        if isinstance(signedness, Signedness):
            signedness = SignednessAttr(signedness)
        super().__init__([data, signedness])

    def value_range(self) -> tuple[int, int]:
        return self.signedness.data.value_range(self.width.data)

    @property
    def bitwidth(self) -> int:
        return self.width.data


i64 = IntegerType(64)
i32 = IntegerType(32)
i16 = IntegerType(16)
i8 = IntegerType(8)
i1 = IntegerType(1)
I64 = Annotated[IntegerType, i64]
I32 = Annotated[IntegerType, i32]
I16 = Annotated[IntegerType, i16]
I8 = Annotated[IntegerType, i8]
I1 = Annotated[IntegerType, i1]


SignlessIntegerConstraint = ParamAttrConstraint(
    IntegerType, [IntAttr, SignednessAttr(Signedness.SIGNLESS)]
)
"""Type constraint for signless IntegerType."""

AnySignlessIntegerType: TypeAlias = Annotated[IntegerType, SignlessIntegerConstraint]
"""Type alias constrained to signless IntegerType."""


@irdl_attr_definition
class UnitAttr(ParametrizedAttribute):
    name = "unit"


@irdl_attr_definition
class LocationAttr(ParametrizedAttribute):
    """
    An attribute representing source code location.
    Only supports unknown locations for now.
    """

    name = "loc"


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "index"


IndexTypeConstr = BaseAttr(IndexType)

_IntegerAttrType = TypeVar(
    "_IntegerAttrType", bound=IntegerType | IndexType, covariant=True
)
IntegerAttrTypeConstr = IndexTypeConstr | BaseAttr(IntegerType)
AnySignlessIntegerOrIndexType: TypeAlias = Annotated[
    Attribute, AnyOf([IndexType, SignlessIntegerConstraint])
]
"""Type alias constrained to IndexType or signless IntegerType."""


@irdl_attr_definition
class IntegerAttr(
    Generic[_IntegerAttrType],
    TypedAttribute[_IntegerAttrType],
):
    name = "integer"
    value: ParameterDef[IntAttr]
    type: ParameterDef[_IntegerAttrType]

    @overload
    def __init__(
        self,
        value: int | IntAttr,
        value_type: _IntegerAttrType,
    ) -> None: ...

    @overload
    def __init__(
        self: IntegerAttr[IntegerType], value: int | IntAttr, value_type: int
    ) -> None: ...

    def __init__(
        self, value: int | IntAttr, value_type: int | IntegerType | IndexType
    ) -> None:
        if isinstance(value, int):
            value = IntAttr(value)
        if isinstance(value_type, int):
            value_type = IntegerType(value_type)
        super().__init__([value, value_type])

    @staticmethod
    def from_int_and_width(value: int, width: int) -> IntegerAttr[IntegerType]:
        return IntegerAttr(value, width)

    @staticmethod
    def from_index_int_value(value: int) -> IntegerAttr[IndexType]:
        return IntegerAttr(value, IndexType())

    def verify(self) -> None:
        if isinstance(int_type := self.type, IndexType):
            return

        min_value, max_value = int_type.value_range()

        if not (min_value <= self.value.data < max_value):
            raise VerifyException(
                f"Integer value {self.value.data} is out of range for "
                f"type {self.type} which supports values in the "
                f"range [{min_value}, {max_value})"
            )

    @classmethod
    def parse_with_type(
        cls: type[IntegerAttr[_IntegerAttrType]],
        parser: AttrParser,
        type: Attribute,
    ) -> IntegerAttr[_IntegerAttrType]:
        assert isinstance(type, IntegerType) or isinstance(type, IndexType)
        return cast(
            IntegerAttr[_IntegerAttrType],
            IntegerAttr(parser.parse_integer(allow_boolean=(type == i1)), type),
        )

    def print_without_type(self, printer: Printer):
        return printer.print(self.value.data)

    @classmethod
    def constr(
        cls,
        *,
        value: AttrConstraint | None = None,
        type: GenericAttrConstraint[_IntegerAttrType] = IntegerAttrTypeConstr,
    ) -> GenericAttrConstraint[IntegerAttr[_IntegerAttrType]]:
        if value is None and type == AnyAttr():
            return BaseAttr[IntegerAttr[_IntegerAttrType]](IntegerAttr)
        return ParamAttrConstraint[IntegerAttr[_IntegerAttrType]](
            IntegerAttr,
            (
                value,
                type,
            ),
        )


AnyIntegerAttr: TypeAlias = IntegerAttr[IntegerType | IndexType]
AnyIntegerAttrConstr: BaseAttr[AnyIntegerAttr] = BaseAttr(IntegerAttr)
BoolAttr: TypeAlias = IntegerAttr[Annotated[IntegerType, IntegerType(1)]]


class _FloatType(ABC):
    @property
    @abstractmethod
    def bitwidth(self) -> int:
        raise NotImplementedError()


@irdl_attr_definition
class BFloat16Type(ParametrizedAttribute, FixedBitwidthType, _FloatType):
    name = "bf16"

    @property
    def bitwidth(self) -> int:
        return 16


@irdl_attr_definition
class Float16Type(ParametrizedAttribute, FixedBitwidthType, _FloatType):
    name = "f16"

    @property
    def bitwidth(self) -> int:
        return 16


@irdl_attr_definition
class Float32Type(ParametrizedAttribute, FixedBitwidthType, _FloatType):
    name = "f32"

    @property
    def bitwidth(self) -> int:
        return 32


@irdl_attr_definition
class Float64Type(ParametrizedAttribute, FixedBitwidthType, _FloatType):
    name = "f64"

    @property
    def bitwidth(self) -> int:
        return 64


@irdl_attr_definition
class Float80Type(ParametrizedAttribute, FixedBitwidthType, _FloatType):
    name = "f80"

    @property
    def bitwidth(self) -> int:
        return 80


@irdl_attr_definition
class Float128Type(ParametrizedAttribute, FixedBitwidthType, _FloatType):
    name = "f128"

    @property
    def bitwidth(self) -> int:
        return 128


AnyFloat: TypeAlias = (
    BFloat16Type | Float16Type | Float32Type | Float64Type | Float80Type | Float128Type
)
AnyFloatConstr = (
    BaseAttr(BFloat16Type)
    | BaseAttr(Float16Type)
    | BaseAttr(Float32Type)
    | BaseAttr(Float64Type)
    | BaseAttr(Float80Type)
    | BaseAttr(Float128Type)
)


@irdl_attr_definition
class FloatData(Data[float]):
    name = "float_data"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> float:
        with parser.in_angle_brackets():
            return float(parser.parse_number())

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")


_FloatAttrType = TypeVar("_FloatAttrType", bound=AnyFloat, covariant=True)


@irdl_attr_definition
class FloatAttr(Generic[_FloatAttrType], ParametrizedAttribute):
    name = "float"

    value: ParameterDef[FloatData]
    type: ParameterDef[_FloatAttrType]

    @overload
    def __init__(self, data: float | FloatData, type: _FloatAttrType) -> None: ...

    @overload
    def __init__(self, data: float | FloatData, type: int) -> None: ...

    def __init__(
        self, data: float | FloatData, type: int | _FloatAttrType | AnyFloat
    ) -> None:
        if isinstance(data, FloatData):
            data_attr = data
        else:
            data_attr = FloatData(data)
        if isinstance(type, int):
            if type == 16:
                type = Float16Type()
            elif type == 32:
                type = Float32Type()
            elif type == 64:
                type = Float64Type()
            elif type == 80:
                type = Float80Type()
            elif type == 128:
                type = Float128Type()
            else:
                raise ValueError(f"Invalid bitwidth: {type}")
        super().__init__([data_attr, type])


AnyFloatAttr: TypeAlias = FloatAttr[AnyFloat]
AnyFloatAttrConstr: BaseAttr[AnyFloatAttr] = BaseAttr(FloatAttr)


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, TypeAttribute):
    name = "complex"
    element_type: ParameterDef[IntegerType | AnyFloat]

    def __init__(self, element_type: IntegerType | AnyFloat) -> None:
        ParametrizedAttribute.__init__(self, [element_type])


@irdl_attr_definition
class DictionaryAttr(GenericData[dict[str, Attribute]]):
    name = "dictionary"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> dict[str, Attribute]:
        return parser.parse_optional_dictionary_attr_dict()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_attr_dict(self.data)

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        raise Exception(f"Unsupported operation on {DictionaryAttr.name}")

    def verify(self) -> None:
        return super().verify()


@irdl_attr_definition
class TupleType(ParametrizedAttribute):
    name = "tuple"

    types: ParameterDef[ArrayAttr[Attribute]]

    def __init__(self, types: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(types, list):
            types = ArrayAttr(types)
        super().__init__([types])


@irdl_attr_definition
class VectorType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[AttributeCovT],
):
    name = "vector"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[AttributeCovT]
    num_scalable_dims: ParameterDef[IntAttr]

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int | IntAttr],
        num_scalable_dims: int | IntAttr = 0,
    ) -> None:
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        if isinstance(num_scalable_dims, int):
            num_scalable_dims = IntAttr(num_scalable_dims)
        super().__init__([shape, element_type, num_scalable_dims])

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_num_scalable_dims(self) -> int:
        return self.num_scalable_dims.data

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

    def verify(self):
        if self.get_num_scalable_dims() < 0:
            raise VerifyException(
                f"Number of scalable dimensions {self.get_num_dims()} cannot"
                " be negative"
            )
        if self.get_num_scalable_dims() > self.get_num_dims():
            raise VerifyException(
                f"Number of scalable dimensions {self.get_num_scalable_dims()}"
                " cannot be larger than number of dimensions"
                f" {self.get_num_dims()}"
            )


AnyVectorType: TypeAlias = VectorType[Attribute]


@irdl_attr_definition
class TensorType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[AttributeCovT],
):
    name = "tensor"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[AttributeCovT]
    encoding: ParameterDef[Attribute]

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int | IntAttr],
        encoding: Attribute = NoneAttr(),
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__([shape, element_type, encoding])

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type


AnyTensorType: TypeAlias = TensorType[Attribute]
AnyTensorTypeConstr = BaseAttr[TensorType[Attribute]](TensorType)


@irdl_attr_definition
class UnrankedTensorType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[AttributeCovT],
):
    name = "unranked_tensor"

    element_type: ParameterDef[AttributeCovT]

    def __init__(self, element_type: AttributeCovT) -> None:
        super().__init__([element_type])

    def get_element_type(self) -> AttributeCovT:
        return self.element_type


AnyUnrankedTensorType: TypeAlias = UnrankedTensorType[Attribute]


@dataclass(frozen=True, init=False)
class ContainerOf(AttrConstraint):
    """A type constraint that can be nested once in a vector or a tensor."""

    elem_constr: AttrConstraint

    def __init__(
        self, elem_constr: Attribute | type[Attribute] | AttrConstraint
    ) -> None:
        object.__setattr__(self, "elem_constr", attr_constr_coercion(elem_constr))

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if isinstance(attr, VectorType) or isinstance(attr, TensorType):
            attr = cast(VectorType[Attribute] | TensorType[Attribute], attr)
            self.elem_constr.verify(attr.element_type, constraint_context)
        else:
            self.elem_constr.verify(attr, constraint_context)


VectorOrTensorOf: TypeAlias = (
    VectorType[AttributeCovT]
    | TensorType[AttributeCovT]
    | UnrankedTensorType[AttributeCovT]
)


@dataclass(frozen=True)
class VectorRankConstraint(AttrConstraint):
    """
    Constrain a vector to be of a given rank.
    """

    expected_rank: int
    """The expected vector rank."""

    def verify(
        self, attr: Attribute, constraint_context: ConstraintContext | None = None
    ) -> None:
        if not isinstance(attr, VectorType):
            raise VerifyException(f"{attr} should be of type VectorType.")
        if attr.get_num_dims() != self.expected_rank:
            raise VerifyException(
                f"Expected vector rank to be {self.expected_rank}, got {attr.get_num_dims()}."
            )


@dataclass(frozen=True)
class VectorBaseTypeConstraint(AttrConstraint):
    """
    Constrain a vector to be of a given base type.
    """

    expected_type: Attribute
    """The expected vector base type."""

    def verify(
        self, attr: Attribute, constraint_context: ConstraintContext | None = None
    ) -> None:
        if not isinstance(attr, VectorType):
            raise VerifyException(f"{attr} should be of type VectorType.")
        attr = cast(VectorType[Attribute], attr)
        if attr.element_type != self.expected_type:
            raise VerifyException(
                f"Expected vector type to be {self.expected_type}, got {attr.element_type}."
            )


@dataclass(frozen=True)
class VectorBaseTypeAndRankConstraint(AttrConstraint):
    """
    Constrain a vector to be of a given rank and base type.
    """

    expected_type: Attribute
    """The expected vector base type."""

    expected_rank: int
    """The expected vector rank."""

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        constraint = AllOf(
            (
                VectorBaseTypeConstraint(self.expected_type),
                VectorRankConstraint(self.expected_rank),
            )
        )
        constraint.verify(attr, constraint_context)


@irdl_attr_definition
class DenseResourceAttr(ParametrizedAttribute):
    name = "dense_resource"

    resource_handle: ParameterDef[StringAttr]

    # Should be a ShapedType, but this is not defined yet in xDSL
    type: ParameterDef[Attribute]

    @staticmethod
    def from_params(handle: str | StringAttr, type: Attribute) -> DenseResourceAttr:
        if isinstance(handle, str):
            handle = StringAttr(handle)
        return DenseResourceAttr([handle, type])


@irdl_attr_definition
class DenseArrayBase(ParametrizedAttribute):
    name = "array"

    elt_type: ParameterDef[IntegerType | IndexType | AnyFloat]
    data: ParameterDef[ArrayAttr[IntAttr] | ArrayAttr[FloatData]]

    def verify(self):
        if isinstance(self.elt_type, IntegerType | IndexType):
            for d in self.data.data:
                if isinstance(d, FloatData):
                    raise VerifyException(
                        "dense array of integer or index element type "
                        "should only contain integers"
                    )
        else:
            for d in self.data.data:
                if isinstance(d, IntAttr):
                    raise VerifyException(
                        "dense array of float element type "
                        "should only contain floats"
                    )

    @deprecated("Please use `create_dense_int` instead.")
    @staticmethod
    def create_dense_int_or_index(
        data_type: IntegerType | IndexType, data: Sequence[int] | Sequence[IntAttr]
    ) -> DenseArrayBase:
        assert not isinstance(data_type, IndexType), "Index type is not supported"
        return DenseArrayBase.create_dense_int(data_type, data)

    @staticmethod
    def create_dense_int(
        data_type: IntegerType, data: Sequence[int] | Sequence[IntAttr]
    ) -> DenseArrayBase:
        if len(data) and isinstance(data[0], int):
            attr_list = [IntAttr(d) for d in cast(Sequence[int], data)]
        else:
            attr_list = cast(Sequence[IntAttr], data)

        return DenseArrayBase([data_type, ArrayAttr(attr_list)])

    @staticmethod
    def create_dense_float(
        data_type: AnyFloat, data: Sequence[int | float] | Sequence[FloatData]
    ) -> DenseArrayBase:
        if len(data) and isinstance(data[0], int | float):
            attr_list = [FloatData(float(d)) for d in cast(Sequence[int | float], data)]
        else:
            attr_list = cast(Sequence[FloatData], data)

        return DenseArrayBase([data_type, ArrayAttr(attr_list)])

    @overload
    @staticmethod
    def from_list(
        data_type: IntegerType | IndexType, data: Sequence[int] | Sequence[IntAttr]
    ) -> DenseArrayBase: ...

    @overload
    @staticmethod
    def from_list(
        data_type: Attribute, data: Sequence[int | float] | Sequence[FloatData]
    ) -> DenseArrayBase: ...

    @staticmethod
    def from_list(
        data_type: Attribute,
        data: (
            Sequence[int]
            | Sequence[int | float]
            | Sequence[IntAttr]
            | Sequence[FloatData]
        ),
    ) -> DenseArrayBase:
        if isinstance(data_type, IntegerType):
            _data = cast(Sequence[int] | Sequence[IntAttr], data)
            return DenseArrayBase.create_dense_int(data_type, _data)
        elif isattr(data_type, AnyFloatConstr):
            _data = cast(Sequence[int | float] | Sequence[FloatData], data)
            return DenseArrayBase.create_dense_float(data_type, _data)
        else:
            raise TypeError(f"Unsupported element type {data_type}")

    def as_tuple(self) -> tuple[int, ...] | tuple[float, ...]:
        """
        Get the "raw" data out as a tuple. This will not
        apply the datatype restrictions that the array element
        type would suggest!

        e.g. given a dense<i8: 99999999, 255, 256>, as_tuple()
        would return 1234567, 255, 256 and not 135, 255, 0 (mod 256)
        """
        return tuple(x.data for x in self.data.data)


@irdl_attr_definition
class FunctionType(ParametrizedAttribute, TypeAttribute):
    name = "fun"

    inputs: ParameterDef[ArrayAttr[Attribute]]
    outputs: ParameterDef[ArrayAttr[Attribute]]

    @staticmethod
    def from_lists(
        inputs: Sequence[Attribute], outputs: Sequence[Attribute]
    ) -> FunctionType:
        return FunctionType([ArrayAttr(inputs), ArrayAttr(outputs)])

    @staticmethod
    def from_attrs(
        inputs: ArrayAttr[Attribute], outputs: ArrayAttr[Attribute]
    ) -> FunctionType:
        return FunctionType([inputs, outputs])


@irdl_attr_definition
class OpaqueAttr(ParametrizedAttribute):
    name = "opaque"

    ident: ParameterDef[StringAttr]
    value: ParameterDef[StringAttr]
    type: ParameterDef[Attribute]

    @staticmethod
    def from_strings(name: str, value: str, type: Attribute = NoneAttr()) -> OpaqueAttr:
        return OpaqueAttr([StringAttr(name), StringAttr(value), type])


class MemrefLayoutAttr(Attribute, ABC):
    """
    Interface for any attribute acceptable as a memref layout.
    """

    name = "abstract.memref_layout_att"

    @abstractmethod
    def get_affine_map(self) -> AffineMap:
        """
        Return the affine mapping from the iteration space of this
        layout to the element offset in linear memory. The resulting
        affine map thus has only one result.
        """
        raise NotImplementedError()

    def get_strides(self) -> Sequence[int | None] | None:
        """
        (optional) Return the list of strides, representing the element offset
        in linear memory for every dimension in the iteration space of
        this memref layout attribute.

        Note: The dimension of the iteration space may differ from the dimension
        of the data it represents. For instance, this can occur in a tiled layout.

        This is only applicable to hyper-rectangular layouts.
        If this is not applicable for a given layout, returns None
        """
        return None


@irdl_attr_definition
class StridedLayoutAttr(MemrefLayoutAttr, ParametrizedAttribute):
    """
    An attribute representing a strided layout of a shaped type.
    See https://mlir.llvm.org/docs/Dialects/Builtin/#stridedlayoutattr

    Contrary to MLIR, we represent dynamic offsets and strides with
    `NoneAttr`, and we do not restrict offsets and strides to 64-bits
    integers.
    """

    name = "strided"

    strides: ParameterDef[ArrayAttr[IntAttr | NoneAttr]]
    offset: ParameterDef[IntAttr | NoneAttr]

    def __init__(
        self,
        strides: (
            ArrayAttr[IntAttr | NoneAttr] | Sequence[int | None | IntAttr | NoneAttr]
        ),
        offset: int | None | IntAttr | NoneAttr = 0,
    ) -> None:
        if not isinstance(strides, ArrayAttr):
            strides_values: list[IntAttr | NoneAttr] = []
            for stride in strides:
                if isinstance(stride, int):
                    strides_values.append(IntAttr(stride))
                elif stride is None:
                    strides_values.append(NoneAttr())
                else:
                    strides_values.append(stride)
            strides = ArrayAttr(strides_values)

        if isinstance(offset, int):
            offset = IntAttr(offset)
        if offset is None:
            offset = NoneAttr()

        super().__init__([strides, offset])

    def get_strides(self) -> Sequence[int | None]:
        return tuple(
            None if isinstance(stride, NoneAttr) else stride.data
            for stride in self.strides
        )

    def get_offset(self) -> int | None:
        if isinstance(self.offset, NoneAttr):
            return None
        else:
            return self.offset.data

    def get_affine_map(self) -> AffineMap:
        """
        Return the affine mapping from the iteration space of this
        layout to the element offset in linear memory. The resulting
        affine map thus has only one result.

        For dynamic strides, this results in an affinemap with a number
        of symbols, ordered in the following manner:
            (1) Symbol for the dynamic offset of the layout
            (2) Symbols for every dynamic stride of the layout
        """

        # keep track of number of symbols
        nb_symbols = 0

        result = AffineConstantExpr(0)

        # add offset
        if isinstance(self.offset, IntAttr):
            result += AffineConstantExpr(self.offset.data)
        else:  # NoneAttr
            result += AffineSymExpr(nb_symbols)
            nb_symbols += 1

        for dim, stride in enumerate(self.strides.data):
            if isinstance(stride, IntAttr):
                stride_expr = AffineConstantExpr(stride.data)
            else:  # NoneAttr
                stride_expr = AffineSymExpr(nb_symbols)
                nb_symbols += 1
            result += AffineDimExpr(dim) * stride_expr

        return AffineMap(len(self.strides), nb_symbols, (result,))


@irdl_attr_definition
class AffineMapAttr(MemrefLayoutAttr, Data[AffineMap]):
    """An Attribute containing an AffineMap object."""

    name = "affine_map"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> AffineMap:
        with parser.in_angle_brackets():
            data = parser.parse_affine_map()
            return data

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")

    @staticmethod
    def constant_map(value: int) -> AffineMapAttr:
        return AffineMapAttr(AffineMap.constant_map(value))

    def get_affine_map(self) -> AffineMap:
        return self.data


@irdl_attr_definition
class AffineSetAttr(Data[AffineSet]):
    """An attribute containing an AffineSet object."""

    name = "affine_set"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> AffineSet:
        return parser.parse_affine_set()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")


@irdl_op_definition
class UnrealizedConversionCastOp(IRDLOperation):
    name = "builtin.unrealized_conversion_cast"

    inputs = var_operand_def()
    outputs = var_result_def()

    traits = frozenset([NoMemoryEffect()])

    @staticmethod
    def get(inputs: Sequence[SSAValue | Operation], result_type: Sequence[Attribute]):
        return UnrealizedConversionCastOp.build(
            operands=[inputs],
            result_types=[result_type],
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        if parser.parse_optional_characters("to") is None:
            args = parser.parse_comma_separated_list(
                parser.Delimiter.NONE,
                parser.parse_unresolved_operand,
            )
            parser.parse_punctuation(":")
            input_types = parser.parse_comma_separated_list(
                parser.Delimiter.NONE,
                parser.parse_type,
            )
            parser.parse_characters("to")
            inputs = parser.resolve_operands(args, input_types, parser.pos)
        else:
            inputs = list[SSAValue]()
        output_types = parser.parse_comma_separated_list(
            parser.Delimiter.NONE,
            parser.parse_type,
        )
        attributes = parser.parse_optional_attr_dict()
        return cls(
            operands=[inputs], result_types=[output_types], attributes=attributes
        )

    def print(self, printer: Printer):
        def print_fn(operand: SSAValue) -> None:
            return printer.print_attribute(operand.type)

        if self.inputs:
            printer.print(" ")
            printer.print_list(self.inputs, printer.print_operand)
            printer.print_string(" : ")
            printer.print_list(self.inputs, print_fn)
        printer.print_string(" to ")
        printer.print_list(self.outputs, print_fn)
        printer.print_op_attributes(self.attributes)


class UnregisteredOp(Operation, ABC):
    """
    An unregistered operation.

    Each unregistered op is registered as a subclass of `UnregisteredOp`,
    and op with different names have distinct subclasses.
    """

    name = "builtin.unregistered"
    traits = frozenset()

    @property
    def op_name(self) -> StringAttr:
        if "op_name__" not in self.attributes:
            raise ValueError("missing 'op_name__' attribute")
        op_name = self.attributes["op_name__"]
        if not isinstance(op_name, StringAttr):
            raise ValueError(
                f"'op_name__' is expected to have 'StringAttr' type, got {op_name}"
            )
        return op_name

    @classmethod
    def with_name(cls, name: str) -> type[Operation]:
        """
        Return a new unregistered operation type given a name.
        This function should not be called directly. Use methods from
        `MLContext` to get an `UnregisteredOp` type.
        """

        class UnregisteredOpWithName(UnregisteredOp):
            @classmethod
            def create(
                cls,
                *,
                operands: Sequence[SSAValue] = (),
                result_types: Sequence[Attribute] = (),
                properties: Mapping[str, Attribute] = {},
                attributes: Mapping[str, Attribute] = {},
                successors: Sequence[Block] = (),
                regions: Sequence[Region] = (),
            ):
                op = super().create(
                    operands=operands,
                    result_types=result_types,
                    properties=properties,
                    attributes=attributes,
                    successors=successors,
                    regions=regions,
                )
                op.attributes["op_name__"] = StringAttr(name)
                return op

        return UnregisteredOpWithName


class UnregisteredAttr(ParametrizedAttribute, ABC):
    """
    An unregistered attribute or type.

    Each unregistered attribute is registered as a subclass of
    `UnregisteredAttr`, and attribute with different names have
    distinct subclasses.

    Since attributes do not have a generic format, unregistered
    attributes represent their original parameters as a string,
    which is exactly the content parsed from the textual
    representation.
    """

    name = "builtin.unregistered"

    attr_name: ParameterDef[StringAttr]
    is_type: ParameterDef[IntAttr]
    is_opaque: ParameterDef[IntAttr]
    value: ParameterDef[StringAttr]
    """
    This parameter is non-null is the attribute is a type, and null otherwise.
    """

    def __init__(
        self,
        attr_name: str | StringAttr,
        is_type: bool | IntAttr,
        is_opaque: bool | IntAttr,
        value: str | StringAttr,
    ):
        if isinstance(attr_name, str):
            attr_name = StringAttr(attr_name)
        if isinstance(is_type, bool):
            is_type = IntAttr(int(is_type))
        if isinstance(is_opaque, bool):
            is_opaque = IntAttr(int(is_opaque))
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__([attr_name, is_type, is_opaque, value])

    @classmethod
    def with_name_and_type(cls, name: str, is_type: bool) -> type[UnregisteredAttr]:
        """
        Return a new unregistered attribute type given a name and a
        boolean indicating if the attribute can be a type.
        This function should not be called directly. Use methods from
        `MLContext` to get an `UnregisteredAttr` type.
        """

        @irdl_attr_definition
        class UnregisteredAttrWithName(UnregisteredAttr):
            def verify(self):
                if self.attr_name.data != name:
                    raise VerifyException("Unregistered attribute name mismatch")
                if self.is_type.data != int(is_type):
                    raise VerifyException("Unregistered attribute is_type mismatch")

        @irdl_attr_definition
        class UnregisteredAttrTypeWithName(UnregisteredAttr, TypeAttribute):
            def verify(self):
                if self.attr_name.data != name:
                    raise VerifyException("Unregistered attribute name mismatch")
                if self.is_type.data != int(is_type):
                    raise VerifyException("Unregistered attribute is_type mismatch")

        if is_type:
            return UnregisteredAttrWithName
        else:
            return UnregisteredAttrTypeWithName


@irdl_op_definition
class ModuleOp(IRDLOperation):
    name = "builtin.module"

    sym_name = opt_attr_def(StringAttr)

    body = region_def("single_block")

    traits = frozenset(
        [
            IsolatedFromAbove(),
            NoTerminator(),
            OptionalSymbolOpInterface(),
            SymbolTable(),
        ]
    )

    def __init__(
        self,
        ops: list[Operation] | Region,
        attributes: Mapping[str, Attribute] | None = None,
    ):
        if attributes is None:
            attributes = {}
        if isinstance(ops, Region):
            region = ops
        else:
            region = Region(Block(ops))
        super().__init__(regions=[region], attributes=attributes)

    @property
    def ops(self) -> BlockOps:
        return self.body.ops

    @classmethod
    def parse(cls, parser: Parser) -> ModuleOp:
        attributes = parser.parse_optional_attr_dict_with_keyword()
        if attributes is not None:
            attributes = attributes.data
        region = parser.parse_region()

        # Add a block if the region is empty
        if not region.blocks:
            region.add_block(Block())

        return ModuleOp(region, attributes)

    def print(self, printer: Printer) -> None:
        if len(self.attributes) != 0:
            printer.print(" attributes ")
            printer.print_op_attributes(self.attributes)

        if not self.body.block.ops:
            # Do not print the entry block if the region has an empty block
            printer.print(" {\n")
            printer.print("}")
        else:
            printer.print(" ", self.body)


# FloatXXType shortcuts
bf16 = BFloat16Type()
f16 = Float16Type()
f32 = Float32Type()
f64 = Float64Type()
f80 = Float80Type()
f128 = Float128Type()


_MemRefTypeElement = TypeVar("_MemRefTypeElement", bound=Attribute, covariant=True)
_UnrankedMemrefTypeElems = TypeVar(
    "_UnrankedMemrefTypeElems", bound=Attribute, covariant=True
)
_UnrankedMemrefTypeElemsInit = TypeVar("_UnrankedMemrefTypeElemsInit", bound=Attribute)


@irdl_attr_definition
class NoneType(ParametrizedAttribute, TypeAttribute):
    name = "none_type"


@irdl_attr_definition
class MemRefType(
    Generic[_MemRefTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[_MemRefTypeElement],
):
    name = "memref"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[_MemRefTypeElement]
    layout: ParameterDef[MemrefLayoutAttr | NoneAttr]
    memory_space: ParameterDef[Attribute]

    def __init__(
        self,
        element_type: _MemRefTypeElement,
        shape: ArrayAttr[IntAttr] | Iterable[int | IntAttr],
        layout: MemrefLayoutAttr | NoneAttr = NoneAttr(),
        memory_space: Attribute = NoneAttr(),
    ):
        s: ArrayAttr[IntAttr]
        if isinstance(shape, ArrayAttr):
            # Temporary cast until Pyright is fixed to not infer ArrayAttr[int] as a
            # possible value for shape
            s = cast(ArrayAttr[IntAttr], shape)
        else:
            s = ArrayAttr(
                [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
            )
        super().__init__(
            (
                s,
                element_type,
                layout,
                memory_space,
            )
        )

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> _MemRefTypeElement:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_punctuation("<", " in memref attribute")
        shape = parser.parse_attribute()
        parser.parse_punctuation(",", " between shape and element type parameters")
        type = parser.parse_attribute()
        # If we have a layout or a memory space, parse both of them.
        if parser.parse_optional_punctuation(",") is None:
            parser.parse_punctuation(">", " at end of memref attribute")
            return [shape, type, NoneAttr(), NoneAttr()]
        layout = parser.parse_attribute()
        parser.parse_punctuation(",", " between layout and memory space")
        memory_space = parser.parse_attribute()
        parser.parse_punctuation(">", " at end of memref attribute")

        return [shape, type, layout, memory_space]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.shape, ", ", self.element_type)
        if self.layout != NoneAttr() or self.memory_space != NoneAttr():
            printer.print(", ", self.layout, ", ", self.memory_space)
        printer.print(">")

    def get_affine_map(self) -> AffineMap:
        """
        Return the affine mapping from the iteration space of this
        memref's layout to the element offset in linear memory.
        """
        if isinstance(self.layout, NoneAttr):
            # empty shape not supported
            if self.get_shape() == ():
                raise DiagnosticException(
                    f"Unsupported empty shape in memref of type {self}"
                )

            strides = self.strides_for_shape(self.get_shape())
            map = StridedLayoutAttr(strides).get_affine_map()
        else:
            map = self.layout.get_affine_map()

        return map

    def get_affine_map_in_bytes(self) -> AffineMap:
        """
        Return the affine mapping from the iteration space of this
        memref's layout to the byte offset in linear memory.

        Unlike the get_affine_map, this function accounts for element width.
        """

        map = self.get_affine_map()

        # account for element width
        assert isinstance(self.element_type, FixedBitwidthType)

        return AffineMap(
            map.num_dims,
            map.num_symbols,
            tuple(result * self.element_type.size for result in map.results),
        )

    def get_strides(self) -> Sequence[int | None] | None:
        """
        Yields the strides of the memref for each dimension.
        The stride of a dimension is the number of elements that are skipped when
        incrementing the corresponding index by one.
        """
        match self.layout:
            case NoneAttr():
                return ShapedType.strides_for_shape(self.get_shape())
            case _:
                return self.layout.get_strides()

    @classmethod
    def constr(
        cls,
        *,
        shape: GenericAttrConstraint[Attribute] | None = None,
        element_type: GenericAttrConstraint[_MemRefTypeElement] = AnyAttr(),
        layout: GenericAttrConstraint[Attribute] | None = None,
        memory_space: GenericAttrConstraint[Attribute] | None = None,
    ) -> GenericAttrConstraint[MemRefType[_MemRefTypeElement]]:
        if (
            shape is None
            and element_type == AnyAttr()
            and layout is None
            and memory_space is None
        ):
            return BaseAttr[MemRefType[_MemRefTypeElement]](MemRefType)
        return ParamAttrConstraint[MemRefType[_MemRefTypeElement]](
            MemRefType, (shape, element_type, layout, memory_space)
        )


AnyMemRefType: TypeAlias = MemRefType[Attribute]
AnyMemRefTypeConstr = BaseAttr[MemRefType[Attribute]](MemRefType)


@irdl_attr_definition
class UnrankedMemrefType(
    Generic[_UnrankedMemrefTypeElems],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_UnrankedMemrefTypeElems],
):
    name = "unranked_memref"

    element_type: ParameterDef[_UnrankedMemrefTypeElems]
    memory_space: ParameterDef[Attribute]

    @staticmethod
    def from_type(
        referenced_type: _UnrankedMemrefTypeElemsInit,
        memory_space: Attribute = NoneAttr(),
    ) -> UnrankedMemrefType[_UnrankedMemrefTypeElemsInit]:
        return UnrankedMemrefType([referenced_type, memory_space])

    def get_element_type(self) -> _UnrankedMemrefTypeElems:
        return self.element_type


AnyUnrankedMemrefType: TypeAlias = UnrankedMemrefType[Attribute]

RankedStructure: TypeAlias = (
    VectorType[AttributeCovT] | TensorType[AttributeCovT] | MemRefType[AttributeCovT]
)


@irdl_attr_definition
class DenseIntOrFPElementsAttr(
    ParametrizedAttribute, ContainerType[IntegerType | IndexType | AnyFloat]
):
    name = "dense"
    type: ParameterDef[
        RankedStructure[IntegerType]
        | RankedStructure[IndexType]
        | RankedStructure[AnyFloat]
    ]
    data: ParameterDef[ArrayAttr[AnyIntegerAttr] | ArrayAttr[AnyFloatAttr]]

    # The type stores the shape data
    def get_shape(self) -> tuple[int, ...] | None:
        if isinstance(self.type, UnrankedTensorType):
            return None
        return self.type.get_shape()

    def get_element_type(self) -> IntegerType | IndexType | AnyFloat:
        return self.type.get_element_type()

    @property
    def shape_is_complete(self) -> bool:
        shape = self.get_shape()
        if shape is None or not len(shape):
            return False

        n = 1
        for dim in shape:
            if dim < 1:
                # Dimensions need to be greater or equal to one
                return False
            n *= dim

        # Product of dimensions needs to equal length
        return n == len(self.data.data)

    @staticmethod
    def create_dense_index(
        type: RankedStructure[IndexType],
        data: Sequence[int] | Sequence[IntegerAttr[IndexType]],
    ) -> DenseIntOrFPElementsAttr:
        if len(data) and isinstance(data[0], int):
            attr_list = [
                IntegerAttr.from_index_int_value(d) for d in cast(Sequence[int], data)
            ]
        else:
            attr_list = cast(Sequence[IntegerAttr[IndexType]], data)

        return DenseIntOrFPElementsAttr([type, ArrayAttr(attr_list)])

    @staticmethod
    def create_dense_int(
        type: RankedStructure[IntegerType],
        data: Sequence[int] | Sequence[IntegerAttr[IntegerType]],
    ) -> DenseIntOrFPElementsAttr:
        if len(data) and isinstance(data[0], int):
            attr_list = [
                IntegerAttr[IntegerType](d, type.element_type)
                for d in cast(Sequence[int], data)
            ]
        else:
            attr_list = cast(Sequence[IntegerAttr[IntegerType]], data)

        return DenseIntOrFPElementsAttr([type, ArrayAttr(attr_list)])

    @staticmethod
    def create_dense_float(
        type: RankedStructure[AnyFloat],
        data: Sequence[int | float] | Sequence[AnyFloatAttr],
    ) -> DenseIntOrFPElementsAttr:
        if len(data) and isinstance(data[0], int | float):
            attr_list = [
                FloatAttr(float(d), type.element_type)
                for d in cast(Sequence[int | float], data)
            ]
        else:
            attr_list = cast(Sequence[AnyFloatAttr], data)

        return DenseIntOrFPElementsAttr([type, ArrayAttr(attr_list)])

    @overload
    @staticmethod
    def from_list(
        type: (
            RankedStructure[AnyFloat | IntegerType | IndexType]
            | RankedStructure[AnyFloat]
            | RankedStructure[IntegerType]
            | RankedStructure[IndexType]
        ),
        data: (
            Sequence[int]
            | Sequence[IntegerAttr[IndexType]]
            | Sequence[IntegerAttr[IntegerType]]
        ),
    ) -> DenseIntOrFPElementsAttr: ...

    @overload
    @staticmethod
    def from_list(
        type: (
            RankedStructure[AnyFloat | IntegerType | IndexType]
            | RankedStructure[AnyFloat]
            | RankedStructure[IntegerType]
            | RankedStructure[IndexType]
        ),
        data: Sequence[int | float] | Sequence[AnyFloatAttr],
    ) -> DenseIntOrFPElementsAttr: ...

    @staticmethod
    def from_list(
        type: (
            RankedStructure[AnyFloat | IntegerType | IndexType]
            | RankedStructure[AnyFloat]
            | RankedStructure[IntegerType]
            | RankedStructure[IndexType]
        ),
        data: Sequence[int | float] | Sequence[AnyIntegerAttr] | Sequence[AnyFloatAttr],
    ) -> DenseIntOrFPElementsAttr:
        if isinstance(type.element_type, AnyFloat):
            new_type = cast(RankedStructure[AnyFloat], type)
            new_data = cast(Sequence[int | float] | Sequence[FloatAttr[AnyFloat]], data)
            return DenseIntOrFPElementsAttr.create_dense_float(new_type, new_data)
        elif isinstance(type.element_type, IntegerType):
            new_type = cast(RankedStructure[IntegerType], type)
            new_data = cast(Sequence[int] | Sequence[IntegerAttr[IntegerType]], data)
            return DenseIntOrFPElementsAttr.create_dense_int(new_type, new_data)
        else:
            new_type = cast(RankedStructure[IndexType], type)
            new_data = cast(Sequence[int] | Sequence[IntegerAttr[IndexType]], data)
            return DenseIntOrFPElementsAttr.create_dense_index(new_type, new_data)

    @staticmethod
    def vector_from_list(
        data: Sequence[int] | Sequence[float],
        data_type: IntegerType | IndexType | AnyFloat,
    ) -> DenseIntOrFPElementsAttr:
        t = VectorType(data_type, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)

    @staticmethod
    def tensor_from_list(
        data: (
            Sequence[int]
            | Sequence[float]
            | Sequence[IntegerAttr[IndexType]]
            | Sequence[IntegerAttr[IntegerType]]
            | Sequence[AnyFloatAttr]
        ),
        data_type: IntegerType | IndexType | AnyFloat,
        shape: Sequence[int],
    ) -> DenseIntOrFPElementsAttr:
        t = TensorType(data_type, shape)
        return DenseIntOrFPElementsAttr.from_list(t, data)


Builtin = Dialect(
    "builtin",
    [
        ModuleOp,
        UnregisteredOp,
        UnrealizedConversionCastOp,
    ],
    [
        UnregisteredAttr,
        # Attributes
        StringAttr,
        SymbolRefAttr,
        SymbolNameAttr,
        IntAttr,
        IntegerAttr,
        ArrayAttr,
        DictionaryAttr,
        DenseIntOrFPElementsAttr,
        DenseResourceAttr,
        UnitAttr,
        FloatData,
        LocationAttr,
        NoneAttr,
        OpaqueAttr,
        # Types
        ComplexType,
        FunctionType,
        BFloat16Type,
        Float16Type,
        Float32Type,
        Float64Type,
        Float80Type,
        Float128Type,
        FloatAttr,
        SignednessAttr,
        TupleType,
        IntegerType,
        IndexType,
        NoneType,
        VectorType,
        TensorType,
        UnrankedTensorType,
        AffineMapAttr,
        AffineSetAttr,
        MemRefType,
        UnrankedMemrefType,
    ],
)
