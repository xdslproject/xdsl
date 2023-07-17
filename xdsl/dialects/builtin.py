from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from xdsl.ir import (
    Attribute,
    AttributeCovT,
    AttributeInvT,
    Block,
    BlockOps,
    Data,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AttrConstraint,
    GenericData,
    IRDLOperation,
    ParameterDef,
    VarOperand,
    VarOpResult,
    VarRegion,
    attr_constr_coercion,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    region_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.traits import IsolatedFromAbove, NoTerminator
from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.parser import AttrParser, Parser
    from xdsl.printer import Printer


class ShapedType(ABC):
    @abstractmethod
    def get_num_dims(self) -> int:
        ...

    @abstractmethod
    def get_shape(self) -> tuple[int, ...]:
        ...

    def element_count(self) -> int:
        return prod(self.get_shape())


class AnyShapedType(AttrConstraint):
    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
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


@dataclass
class ArrayOfConstraint(AttrConstraint):
    """
    A constraint that enforces an ArrayData whose elements all satisfy
    the elem_constr.
    """

    elem_constr: AttrConstraint

    def __init__(self, constr: Attribute | type[Attribute] | AttrConstraint):
        self.elem_constr = attr_constr_coercion(constr)

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, Data):
            raise Exception(f"expected data ArrayData but got {attr}")
        for e in cast(ArrayAttr[Attribute], attr).data:
            self.elem_constr.verify(e, constraint_vars)


@irdl_attr_definition
class ArrayAttr(GenericData[tuple[AttributeCovT, ...]], Iterable[AttributeCovT]):
    name = "array"

    def __init__(self, param: Iterable[AttributeCovT]) -> None:
        super().__init__(tuple(param))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[AttributeCovT]:
        data = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_attribute
        )
        # the type system can't ensure that the elements are of type _ArrayAttrT
        result = cast(tuple[AttributeCovT], tuple(data))
        return result

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        if len(args) == 1:
            return ArrayOfConstraint(irdl_to_attr_constraint(args[0]))
        if len(args) == 0:
            return ArrayOfConstraint(AnyAttr())
        raise TypeError(
            f"Attribute ArrayAttr expects at most type"
            f" parameter, but {len(args)} were given"
        )

    def verify(self) -> None:
        for idx, val in enumerate(self.data):
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but {idx} "
                    f"element is of type {type(val)}"
                )

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
        return parser.parse_str_literal()

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
        nested: list[str] | list[StringAttr] | ArrayAttr[StringAttr] = [],
    ) -> None:
        if isinstance(root, str):
            root = StringAttr(root)
        if isinstance(nested, list):
            nested = ArrayAttr(
                [StringAttr(x) if isinstance(x, str) else x for x in nested]
            )
        super().__init__([root, nested])

    def string_value(self):
        root = self.root_reference.data
        for ref in self.nested_references.data:
            root += "." + ref.data
        return root


@irdl_attr_definition
class IntAttr(Data[int]):
    name = "int"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> int:
        data = parser.parse_integer()
        return data

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")


class Signedness(Enum):
    "Signedness semantics for integer"

    SIGNLESS = 0
    "No signedness semantics"

    SIGNED = 1
    UNSIGNED = 2


@irdl_attr_definition
class SignednessAttr(Data[Signedness]):
    name = "signedness"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> Signedness:
        if parser.parse_optional_keyword("signless") is not None:
            return Signedness.SIGNLESS
        if parser.parse_optional_keyword("signed") is not None:
            return Signedness.SIGNED
        if parser.parse_optional_keyword("unsigned") is not None:
            return Signedness.UNSIGNED
        parser.raise_error("`signless`, `signed`, or `unsigned` expected")

    def print_parameter(self, printer: Printer) -> None:
        data = self.data
        if data == Signedness.SIGNLESS:
            printer.print_string("signless")
        elif data == Signedness.SIGNED:
            printer.print_string("signed")
        elif data == Signedness.UNSIGNED:
            printer.print_string("unsigned")
        else:
            raise ValueError(f"Invalid signedness {data}")


@irdl_attr_definition
class IntegerType(ParametrizedAttribute, TypeAttribute):
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


i64 = IntegerType(64)
i32 = IntegerType(32)
i1 = IntegerType(1)


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


_IntegerAttrType = TypeVar(
    "_IntegerAttrType", bound=IntegerType | IndexType, covariant=True
)
_IntegerAttrTypeInv = TypeVar("_IntegerAttrTypeInv", bound=IntegerType | IndexType)


@irdl_attr_definition
class IntegerAttr(Generic[_IntegerAttrType], ParametrizedAttribute):
    name = "integer"
    value: ParameterDef[IntAttr]
    type: ParameterDef[_IntegerAttrType]

    @overload
    def __init__(
        self: IntegerAttr[_IntegerAttrType],
        value: int | IntAttr,
        value_type: _IntegerAttrType,
    ) -> None:
        ...

    @overload
    def __init__(
        self: IntegerAttr[IntegerType], value: int | IntAttr, value_type: int
    ) -> None:
        ...

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
        if isinstance(self.type, IntegerType):
            match self.type.signedness.data:
                case Signedness.SIGNLESS:
                    min_value = -(1 << self.type.width.data)
                    max_value = 1 << self.type.width.data
                case Signedness.SIGNED:
                    min_value = -(1 << (self.type.width.data - 1))
                    max_value = (1 << (self.type.width.data - 1)) - 1
                case Signedness.UNSIGNED:
                    min_value = 0
                    max_value = (1 << self.type.width.data) - 1
                case _:
                    assert False, "unreachable"

            if not (min_value <= self.value.data <= max_value):
                raise VerifyException(
                    f"Integer value {self.value.data} is out of range for "
                    f"type {self.type} which supports values in the "
                    f"range [{min_value}, {max_value}]"
                )


AnyIntegerAttr: TypeAlias = IntegerAttr[IntegerType | IndexType]


@irdl_attr_definition
class BFloat16Type(ParametrizedAttribute, TypeAttribute):
    name = "bf16"


@irdl_attr_definition
class Float16Type(ParametrizedAttribute, TypeAttribute):
    name = "f16"


@irdl_attr_definition
class Float32Type(ParametrizedAttribute, TypeAttribute):
    name = "f32"


@irdl_attr_definition
class Float64Type(ParametrizedAttribute, TypeAttribute):
    name = "f64"


@irdl_attr_definition
class Float80Type(ParametrizedAttribute, TypeAttribute):
    name = "f80"


@irdl_attr_definition
class Float128Type(ParametrizedAttribute, TypeAttribute):
    name = "f128"


AnyFloat: TypeAlias = (
    BFloat16Type | Float16Type | Float32Type | Float64Type | Float80Type | Float128Type
)


@irdl_attr_definition
class FloatData(Data[float]):
    name = "float_data"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> float:
        return float(parser.parse_number())

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")


_FloatAttrType = TypeVar("_FloatAttrType", bound=AnyFloat, covariant=True)

_FloatAttrTypeInv = TypeVar("_FloatAttrTypeInv", bound=AnyFloat)


@irdl_attr_definition
class FloatAttr(Generic[_FloatAttrType], ParametrizedAttribute):
    name = "float"

    value: ParameterDef[FloatData]
    type: ParameterDef[_FloatAttrType]

    @overload
    def __init__(self, data: float | FloatData, type: _FloatAttrType) -> None:
        ...

    @overload
    def __init__(self, data: float | FloatData, type: int) -> None:
        ...

    def __init__(
        self, data: float | FloatData, type: int | _FloatAttrType | AnyFloat
    ) -> None:
        if isinstance(data, float):
            data = FloatData(data)
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
        super().__init__([data, type])


AnyFloatAttr: TypeAlias = FloatAttr[AnyFloat]


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
        printer.print_string("{")
        printer.print_dictionary(
            self.data, printer.print_string_literal, printer.print_attribute
        )
        printer.print_string("}")

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

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[AttributeCovT]
    num_scalable_dims: ParameterDef[IntAttr]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_num_scalable_dims(self) -> int:
        return self.num_scalable_dims.data

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.value.data for i in self.shape.data)

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

    @staticmethod
    def from_element_type_and_shape(
        referenced_type: AttributeInvT,
        shape: Iterable[int | IntegerAttr[IndexType]],
        num_scalable_dims: int | IntAttr = 0,
    ) -> VectorType[AttributeInvT]:
        if isinstance(num_scalable_dims, int):
            num_scalable_dims = IntAttr(num_scalable_dims)
        return VectorType(
            [
                ArrayAttr(
                    [
                        IntegerAttr[IntegerType].from_index_int_value(d)
                        if isinstance(d, int)
                        else d
                        for d in shape
                    ]
                ),
                referenced_type,
                num_scalable_dims,
            ]
        )

    @staticmethod
    def from_params(
        referenced_type: AttributeInvT,
        shape: ArrayAttr[IntegerAttr[IntegerType]] = ArrayAttr(
            [IntegerAttr.from_int_and_width(1, 64)]
        ),
        num_scalable_dims: IntAttr = IntAttr(0),
    ) -> VectorType[AttributeInvT]:
        return VectorType([shape, referenced_type, num_scalable_dims])


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

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[AttributeCovT]
    encoding: ParameterDef[Attribute]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.value.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

    @staticmethod
    def from_type_and_list(
        referenced_type: AttributeInvT,
        shape: Iterable[int | IntegerAttr[IndexType]] | None = None,
        encoding: Attribute = NoneAttr(),
    ) -> TensorType[AttributeInvT]:
        if shape is None:
            shape = [1]
        return TensorType(
            [
                ArrayAttr(
                    [
                        IntegerAttr[IntegerType].from_index_int_value(d)
                        if isinstance(d, int)
                        else d
                        for d in shape
                    ]
                ),
                referenced_type,
                encoding,
            ]
        )

    @staticmethod
    def from_params(
        referenced_type: AttributeInvT,
        shape: AnyArrayAttr = AnyArrayAttr([IntegerAttr.from_int_and_width(1, 64)]),
        encoding: Attribute = NoneAttr(),
    ) -> TensorType[AttributeInvT]:
        return TensorType([shape, referenced_type, encoding])


AnyTensorType: TypeAlias = TensorType[Attribute]


@irdl_attr_definition
class UnrankedTensorType(Generic[AttributeCovT], ParametrizedAttribute, TypeAttribute):
    name = "unranked_tensor"

    element_type: ParameterDef[AttributeCovT]

    @staticmethod
    def from_type(referenced_type: AttributeInvT) -> UnrankedTensorType[AttributeInvT]:
        return UnrankedTensorType([referenced_type])


AnyUnrankedTensorType: TypeAlias = UnrankedTensorType[Attribute]


@dataclass(init=False)
class ContainerOf(AttrConstraint):
    """A type constraint that can be nested once in a vector or a tensor."""

    elem_constr: AttrConstraint

    def __init__(
        self, elem_constr: Attribute | type[Attribute] | AttrConstraint
    ) -> None:
        self.elem_constr = attr_constr_coercion(elem_constr)

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if isinstance(attr, VectorType) or isinstance(attr, TensorType):
            self.elem_constr.verify(attr.element_type, constraint_vars)  # type: ignore
        else:
            self.elem_constr.verify(attr, constraint_vars)


VectorOrTensorOf: TypeAlias = (
    VectorType[AttributeCovT]
    | TensorType[AttributeCovT]
    | UnrankedTensorType[AttributeCovT]
)

RankedVectorOrTensorOf: TypeAlias = (
    VectorType[AttributeCovT] | TensorType[AttributeCovT]
)


@dataclass
class VectorRankConstraint(AttrConstraint):
    """
    Constrain a vector to be of a given rank.
    """

    expected_rank: int
    """The expected vector rank."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, VectorType):
            raise VerifyException(f"{attr} should be of type VectorType.")
        if attr.get_num_dims() != self.expected_rank:
            raise VerifyException(
                f"Expected vector rank to be {self.expected_rank}, got {attr.get_num_dims()}."
            )


@dataclass
class VectorBaseTypeConstraint(AttrConstraint):
    """
    Constrain a vector to be of a given base type.
    """

    expected_type: Attribute
    """The expected vector base type."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, VectorType):
            raise VerifyException(f"{attr} should be of type VectorType.")
        if attr.element_type != self.expected_type:  # type: ignore
            raise VerifyException(
                f"Expected vector type to be {self.expected_type}, got {attr.element_type}."  # type: ignore
            )


@dataclass
class VectorBaseTypeAndRankConstraint(AttrConstraint):
    """
    Constrain a vector to be of a given rank and base type.
    """

    expected_type: Attribute
    """The expected vector base type."""

    expected_rank: int
    """The expected vector rank."""

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        constraint = AllOf(
            [
                VectorBaseTypeConstraint(self.expected_type),
                VectorRankConstraint(self.expected_rank),
            ]
        )
        constraint.verify(attr, constraint_vars)


@irdl_attr_definition
class DenseIntOrFPElementsAttr(
    ParametrizedAttribute, ContainerType[IntegerType | IndexType | AnyFloat]
):
    name = "dense"
    type: ParameterDef[
        RankedVectorOrTensorOf[IntegerType]
        | RankedVectorOrTensorOf[IndexType]
        | RankedVectorOrTensorOf[AnyFloat]
    ]
    data: ParameterDef[ArrayAttr[AnyIntegerAttr] | ArrayAttr[AnyFloatAttr]]

    # The type stores the shape data
    def get_shape(self) -> tuple[int] | None:
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
        type: RankedVectorOrTensorOf[IndexType],
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
        type: RankedVectorOrTensorOf[IntegerType],
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
        type: RankedVectorOrTensorOf[AnyFloat],
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
        type: RankedVectorOrTensorOf[AnyFloat | IntegerType | IndexType],
        data: Sequence[int]
        | Sequence[IntegerAttr[IndexType]]
        | Sequence[IntegerAttr[IntegerType]],
    ) -> DenseIntOrFPElementsAttr:
        ...

    @overload
    @staticmethod
    def from_list(
        type: RankedVectorOrTensorOf[AnyFloat | IntegerType | IndexType],
        data: Sequence[int | float] | Sequence[AnyFloatAttr],
    ) -> DenseIntOrFPElementsAttr:
        ...

    @staticmethod
    def from_list(
        type: RankedVectorOrTensorOf[AnyFloat | IntegerType | IndexType],
        data: Sequence[int | float] | Sequence[AnyIntegerAttr] | Sequence[AnyFloatAttr],
    ) -> DenseIntOrFPElementsAttr:
        if isinstance(type.element_type, AnyFloat):
            new_type = cast(RankedVectorOrTensorOf[AnyFloat], type)
            new_data = cast(Sequence[int | float] | Sequence[FloatAttr[AnyFloat]], data)
            return DenseIntOrFPElementsAttr.create_dense_float(new_type, new_data)

        match type.element_type:
            case IntegerType():
                new_type = cast(RankedVectorOrTensorOf[IntegerType], type)
                new_data = cast(
                    Sequence[int] | Sequence[IntegerAttr[IntegerType]], data
                )
                return DenseIntOrFPElementsAttr.create_dense_int(new_type, new_data)
            case IndexType():
                new_type = cast(RankedVectorOrTensorOf[IndexType], type)
                new_data = cast(Sequence[int] | Sequence[IntegerAttr[IndexType]], data)
                return DenseIntOrFPElementsAttr.create_dense_index(new_type, new_data)

    @staticmethod
    def vector_from_list(
        data: Sequence[int] | Sequence[float],
        data_type: IntegerType | IndexType | AnyFloat,
    ) -> DenseIntOrFPElementsAttr:
        t = VectorType.from_element_type_and_shape(data_type, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)

    @staticmethod
    def tensor_from_list(
        data: Sequence[int]
        | Sequence[float]
        | Sequence[IntegerAttr[IndexType]]
        | Sequence[IntegerAttr[IntegerType]]
        | Sequence[AnyFloatAttr],
        data_type: IntegerType | IndexType | AnyFloat,
        shape: Sequence[int],
    ) -> DenseIntOrFPElementsAttr:
        t = AnyTensorType.from_type_and_list(data_type, shape)
        return DenseIntOrFPElementsAttr.from_list(t, data)


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

    elt_type: ParameterDef[IntegerType | AnyFloat]
    data: ParameterDef[ArrayAttr[IntAttr] | ArrayAttr[FloatData]]

    def verify(self):
        if isinstance(self.elt_type, IntegerType):
            for d in self.data.data:
                if isinstance(d, FloatData):
                    raise VerifyException(
                        "dense array of integer element type "
                        "should only contain integers"
                    )
        else:
            for d in self.data.data:
                if isinstance(d, IntAttr):
                    raise VerifyException(
                        "dense array of float element type "
                        "should only contain floats"
                    )

    @staticmethod
    def create_dense_int_or_index(
        data_type: IntegerType | IndexType, data: Sequence[int] | Sequence[IntAttr]
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
    ) -> DenseArrayBase:
        ...

    @overload
    @staticmethod
    def from_list(
        data_type: Attribute, data: Sequence[int | float] | Sequence[FloatData]
    ) -> DenseArrayBase:
        ...

    @staticmethod
    def from_list(
        data_type: Attribute,
        data: Sequence[int]
        | Sequence[int | float]
        | Sequence[IntAttr]
        | Sequence[FloatData],
    ) -> DenseArrayBase:
        if isinstance(data_type, IndexType | IntegerType):
            _data = cast(Sequence[int] | Sequence[IntAttr], data)
            return DenseArrayBase.create_dense_int_or_index(data_type, _data)
        elif isinstance(data_type, AnyFloat):
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


@irdl_attr_definition
class StridedLayoutAttr(ParametrizedAttribute):
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
        strides: ArrayAttr[IntAttr | NoneAttr]
        | Sequence[int | None | IntAttr | NoneAttr],
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


@irdl_attr_definition
class AffineMapAttr(Data[AffineMap]):
    """An Attribute containing an AffineMap object."""

    name = "affine_map"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> AffineMap:
        data = parser.parse_affine_map()
        return data

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")

    @staticmethod
    def constant_map(value: int) -> AffineMapAttr:
        return AffineMapAttr(AffineMap.constant_map(value))


@irdl_op_definition
class UnrealizedConversionCastOp(IRDLOperation):
    name = "builtin.unrealized_conversion_cast"

    inputs: VarOperand = var_operand_def()
    outputs: VarOpResult = var_result_def()

    @staticmethod
    def get(inputs: Sequence[SSAValue | Operation], result_type: Sequence[Attribute]):
        return UnrealizedConversionCastOp.build(
            operands=[inputs],
            result_types=[result_type],
        )


class UnregisteredOp(IRDLOperation, ABC):
    """
    An unregistered operation.

    Each unregistered op is registered as a subclass of `UnregisteredOp`,
    and op with different names have distinct subclasses.
    """

    name = "builtin.unregistered"

    op_name: StringAttr = attr_def(StringAttr, attr_name="op_name__")
    args: VarOperand = var_operand_def()
    res: VarOpResult = var_result_def()
    regs: VarRegion = var_region_def()

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
                operands: Sequence[SSAValue] = (),
                result_types: Sequence[Attribute] = (),
                attributes: Mapping[str, Attribute] = {},
                successors: Sequence[Block] = (),
                regions: Sequence[Region] = (),
            ):
                op = super().create(
                    operands, result_types, attributes, successors, regions
                )
                op.attributes["op_name__"] = StringAttr(name)
                return op

        return irdl_op_definition(UnregisteredOpWithName)


@irdl_attr_definition
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
    value: ParameterDef[StringAttr]
    """
    This parameter is non-null is the attribute is a type, and null otherwise.
    """

    def __init__(
        self,
        attr_name: str | StringAttr,
        is_type: bool | IntAttr,
        value: str | StringAttr,
    ):
        if isinstance(attr_name, str):
            attr_name = StringAttr(attr_name)
        if isinstance(is_type, bool):
            is_type = IntAttr(int(is_type))
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__([attr_name, is_type, value])

    @classmethod
    def with_name_and_type(cls, name: str, is_type: bool) -> type[UnregisteredAttr]:
        """
        Return a new unregistered attribute type given a name and a
        boolean indicating if the attribute can be a type.
        This function should not be called directly. Use methods from
        `MLContext` to get an `UnregisteredAttr` type.
        """

        class UnregisteredAttrWithName(UnregisteredAttr):
            def verify(self):
                if self.attr_name.data != name:
                    raise VerifyException("Unregistered attribute name mismatch")
                if self.is_type.data != int(is_type):
                    raise VerifyException("Unregistered attribute is_type mismatch")

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

    body: Region = region_def("single_block")

    traits = frozenset([IsolatedFromAbove(), NoTerminator()])

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
        if len(region.blocks) == 0:
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
f80 = Float64Type()
f128 = Float64Type()


Builtin = Dialect(
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
        VectorType,
        TensorType,
        UnrankedTensorType,
        AffineMapAttr,
    ],
)
