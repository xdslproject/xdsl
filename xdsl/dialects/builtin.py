from __future__ import annotations
from abc import ABC

from dataclasses import dataclass
from enum import Enum
from typing import (
    Iterable,
    TypeAlias,
    List,
    cast,
    Type,
    Sequence,
    TYPE_CHECKING,
    Any,
    TypeVar,
    overload,
    Iterator,
)

from xdsl.ir import (
    Block,
    BlockOps,
    Data,
    TypeAttribute,
    ParametrizedAttribute,
    Operation,
    Region,
    Attribute,
    Dialect,
    SSAValue,
    AttributeCovT,
    AttributeInvT,
)

from xdsl.irdl import (
    AllOf,
    OpAttr,
    VarOpResult,
    VarOperand,
    VarRegion,
    irdl_attr_definition,
    attr_constr_coercion,
    irdl_data_definition,
    irdl_to_attr_constraint,
    irdl_op_definition,
    ParameterDef,
    SingleBlockRegion,
    Generic,
    GenericData,
    AttrConstraint,
    AnyAttr,
    IRDLOperation,
)
from xdsl.utils.deprecation import deprecated_constructor
from xdsl.utils.exceptions import VerifyException

if TYPE_CHECKING:
    from xdsl.parser import Parser
    from utils.exceptions import ParseError
    from xdsl.printer import Printer


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

    def __init__(self, constr: Attribute | Type[Attribute] | AttrConstraint):
        self.elem_constr = attr_constr_coercion(constr)

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, Data):
            raise Exception(f"expected data ArrayData but got {attr}")
        for e in cast(ArrayAttr[Attribute], attr).data:
            self.elem_constr.verify(e)


@irdl_attr_definition
class ArrayAttr(GenericData[tuple[AttributeCovT, ...]], Iterable[AttributeCovT]):
    name = "array"

    def __init__(self, param: Iterable[AttributeCovT]) -> None:
        super().__init__(tuple(param))

    @staticmethod
    def parse_parameter(parser: Parser) -> tuple[AttributeCovT]:
        parser.parse_char("[")
        data = parser.parse_list_of(parser.try_parse_attribute, "Expected attribute")
        parser.parse_char("]")
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
        if not isinstance(self.data, tuple):
            raise VerifyException(
                f"Wrong type given to attribute {self.name}: got"
                f" {type(self.data)}, but expected list of"
                " attributes"
            )
        for idx, val in enumerate(self.data):
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but {idx} "
                    f"element is of type {type(val)}"
                )

    @staticmethod
    @deprecated_constructor
    def from_list(data: List[AttributeCovT]) -> ArrayAttr[AttributeCovT]:
        return ArrayAttr[AttributeCovT](data)

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[AttributeCovT]:
        return iter(self.data)


AnyArrayAttr: TypeAlias = ArrayAttr[Attribute]


@irdl_attr_definition
class StringAttr(Data[str]):
    name = "string"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        data = parser.parse_str_literal()
        return data

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')

    @staticmethod
    @deprecated_constructor
    def from_str(data: str) -> StringAttr:
        return StringAttr(data)

    @staticmethod
    @deprecated_constructor
    def from_int(data: int) -> StringAttr:
        return StringAttr(str(data))


@irdl_attr_definition
class SymbolNameAttr(ParametrizedAttribute):
    name = "symbol_name"
    data: ParameterDef[StringAttr]

    def __init__(self, data: str | StringAttr) -> None:
        if isinstance(data, str):
            data = StringAttr(data)
        super().__init__([data])

    @staticmethod
    @deprecated_constructor
    def from_str(data: str) -> SymbolNameAttr:
        return SymbolNameAttr(data)

    @staticmethod
    @deprecated_constructor
    def from_string_attr(data: StringAttr) -> SymbolNameAttr:
        return SymbolNameAttr(data)


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

    @staticmethod
    @deprecated_constructor
    def from_str(root: str, nested: List[str] = []) -> SymbolRefAttr:
        return SymbolRefAttr(root, nested)

    @staticmethod
    @deprecated_constructor
    def from_string_attr(
        root: StringAttr, nested: List[StringAttr] = []
    ) -> SymbolRefAttr:
        return SymbolRefAttr(root, nested)

    def string_value(self):
        root = self.root_reference.data
        for ref in self.nested_references.data:
            root += "." + ref.data
        return root


@irdl_attr_definition
class IntAttr(Data[int]):
    name = "int"

    @staticmethod
    def parse_parameter(parser: Parser) -> int:
        data = parser.parse_integer()
        return data

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")

    @staticmethod
    @deprecated_constructor
    def from_int(data: int) -> IntAttr:
        return IntAttr(data)


class Signedness(Enum):
    "Signedness semantics for integer"

    SIGNLESS = 0
    "No signedness semantics"

    SIGNED = 1
    UNSIGNED = 2


@irdl_data_definition
class SignednessAttr(Data[Signedness]):
    name = "signedness"

    @staticmethod
    def parse_parameter(parser: Parser) -> Signedness:
        value = parser.expect(
            parser.try_parse_bare_id, "Expected `signless`, `signed`, or `unsigned`."
        )
        if value.text == "signless":
            return Signedness.SIGNLESS
        elif value.text == "signed":
            return Signedness.SIGNED
        elif value.text == "unsigned":
            return Signedness.UNSIGNED
        raise ParseError(value, "Expected signedness")

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

    @staticmethod
    @deprecated_constructor
    def from_enum(signedness: Signedness) -> SignednessAttr:
        return SignednessAttr(signedness)


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

    @staticmethod
    @deprecated_constructor
    def from_width(
        width: int, signedness: Signedness = Signedness.SIGNLESS
    ) -> IntegerType:
        return IntegerType(width, signedness)


i64 = IntegerType(64)
i32 = IntegerType(32)
i1 = IntegerType(1)


@irdl_attr_definition
class UnitAttr(ParametrizedAttribute):
    name = "unit"


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "index"


_IntegerAttrTyp = TypeVar(
    "_IntegerAttrTyp", bound=IntegerType | IndexType, covariant=True
)
_IntegerAttrTypInv = TypeVar("_IntegerAttrTypInv", bound=IntegerType | IndexType)


@irdl_attr_definition
class IntegerAttr(Generic[_IntegerAttrTyp], ParametrizedAttribute):
    name = "integer"
    value: ParameterDef[IntAttr]
    typ: ParameterDef[_IntegerAttrTyp]

    @overload
    def __init__(
        self: IntegerAttr[_IntegerAttrTyp], value: int | IntAttr, typ: _IntegerAttrTyp
    ) -> None:
        ...

    @overload
    def __init__(
        self: IntegerAttr[IntegerType], value: int | IntAttr, typ: int
    ) -> None:
        ...

    def __init__(
        self, value: int | IntAttr, typ: int | IntegerType | IndexType
    ) -> None:
        if isinstance(value, int):
            value = IntAttr(value)
        if isinstance(typ, int):
            typ = IntegerType(typ)
        super().__init__([value, typ])

    @staticmethod
    def from_int_and_width(value: int, width: int) -> IntegerAttr[IntegerType]:
        return IntegerAttr(value, width)

    @staticmethod
    def from_index_int_value(value: int) -> IntegerAttr[IndexType]:
        return IntegerAttr(value, IndexType())

    @staticmethod
    def from_params(
        value: int | IntAttr, typ: int | IntegerType | IndexType
    ) -> IntegerAttr[IntegerType | IndexType]:
        return IntegerAttr(value, typ)


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

    @staticmethod
    def parse_parameter(parser: Parser) -> float:
        span = parser.expect(parser.try_parse_float_literal, "Expect float literal")
        return float(span.text)

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"{self.data}")

    @staticmethod
    @deprecated_constructor
    def from_float(data: float) -> FloatData:
        return FloatData(data)


_FloatAttrTyp = TypeVar("_FloatAttrTyp", bound=AnyFloat, covariant=True)

_FloatAttrTypInv = TypeVar("_FloatAttrTypInv", bound=AnyFloat)


@irdl_attr_definition
class FloatAttr(Generic[_FloatAttrTyp], ParametrizedAttribute):
    name = "float"

    value: ParameterDef[FloatData]
    type: ParameterDef[_FloatAttrTyp]

    @overload
    def __init__(self, data: float | FloatData, type: _FloatAttrTyp) -> None:
        ...

    @overload
    def __init__(self, data: float | FloatData, type: int) -> None:
        ...

    def __init__(
        self, data: float | FloatData, type: int | _FloatAttrTyp | AnyFloat
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

    @staticmethod
    @deprecated_constructor
    def from_value(value: float, type: _FloatAttrTypInv) -> FloatAttr[_FloatAttrTypInv]:
        return FloatAttr(FloatData.from_float(value), type)

    @staticmethod
    @deprecated_constructor
    def from_float_and_width(value: float, width: int) -> FloatAttr[AnyFloat]:
        return FloatAttr(value, width)


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

    @staticmethod
    def parse_parameter(parser: Parser) -> dict[str, Attribute]:
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
        if not isinstance(self.data, dict):
            raise VerifyException(
                f"Wrong type given to attribute {self.name}: got"
                f" {type(self.data)}, but expected dictionary of"
                " attributes"
            )
        for key, val in self.data.items():
            if not isinstance(key, str):
                raise VerifyException(
                    f"{self.name} key expects str, but {key} "
                    f"element is of type {type(key)}"
                )
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} key expects attribute, but {val} "
                    f"element is of type {type(val)}"
                )

    @staticmethod
    @deprecated_constructor
    def from_dict(data: dict[str | StringAttr, Attribute]) -> DictionaryAttr:
        to_add_data: dict[str, Attribute] = {}
        for k, v in data.items():
            # try to coerce keys into StringAttr
            if isinstance(k, StringAttr):
                k = k.data
            # if coercion fails, raise KeyError!
            if not isinstance(k, str):
                raise TypeError(
                    f"DictionaryAttr.from_dict expects keys to"
                    f" be of type str or StringAttr, but {type(k)} provided"
                )
            to_add_data[k] = v
        return DictionaryAttr(to_add_data)


@irdl_attr_definition
class TupleType(ParametrizedAttribute):
    name = "tuple"

    types: ParameterDef[ArrayAttr[Attribute]]

    def __init__(self, types: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(types, list):
            types = ArrayAttr(types)
        super().__init__([types])

    @staticmethod
    @deprecated_constructor
    def from_type_list(types: List[Attribute]) -> TupleType:
        return TupleType(types)


@irdl_attr_definition
class VectorType(Generic[AttributeCovT], ParametrizedAttribute, TypeAttribute):
    name = "vector"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[AttributeCovT]
    num_scalable_dims: ParameterDef[IntAttr]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_num_scalable_dims(self) -> int:
        return self.num_scalable_dims.data

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

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
class TensorType(Generic[AttributeCovT], ParametrizedAttribute, TypeAttribute):
    name = "tensor"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[AttributeCovT]
    encoding: ParameterDef[Attribute]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

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

    def verify(self, attr: Attribute) -> None:
        if isinstance(attr, VectorType) or isinstance(attr, TensorType):
            self.elem_constr.verify(attr.element_type)  # type: ignore
        else:
            self.elem_constr.verify(attr)


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

    def verify(self, attr: Attribute) -> None:
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

    def verify(self, attr: Attribute) -> None:
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

    def verify(self, attr: Attribute) -> None:
        constraint = AllOf(
            [
                VectorBaseTypeConstraint(self.expected_type),
                VectorRankConstraint(self.expected_rank),
            ]
        )
        constraint.verify(attr)


@irdl_attr_definition
class DenseIntOrFPElementsAttr(ParametrizedAttribute):
    name = "dense"
    type: ParameterDef[
        RankedVectorOrTensorOf[IntegerType]
        | RankedVectorOrTensorOf[IndexType]
        | RankedVectorOrTensorOf[AnyFloat]
    ]
    data: ParameterDef[ArrayAttr[AnyIntegerAttr] | ArrayAttr[AnyFloatAttr]]

    # The type stores the shape data
    @property
    def shape(self) -> List[int] | None:
        if isinstance(self.type, UnrankedTensorType):
            return None
        return self.type.get_shape()

    @property
    def shape_is_complete(self) -> bool:
        shape = self.shape
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
        if isinstance(type.element_type, IntegerType):
            new_type = cast(RankedVectorOrTensorOf[IntegerType], type)
            new_data = cast(Sequence[int] | Sequence[IntegerAttr[IntegerType]], data)
            return DenseIntOrFPElementsAttr.create_dense_int(new_type, new_data)
        elif isinstance(type.element_type, IndexType):
            new_type = cast(RankedVectorOrTensorOf[IndexType], type)
            new_data = cast(Sequence[int] | Sequence[IntegerAttr[IndexType]], data)
            return DenseIntOrFPElementsAttr.create_dense_index(new_type, new_data)
        elif isinstance(type.element_type, AnyFloat):
            new_type = cast(RankedVectorOrTensorOf[AnyFloat], type)
            new_data = cast(Sequence[int | float] | Sequence[FloatAttr[AnyFloat]], data)
            return DenseIntOrFPElementsAttr.create_dense_float(new_type, new_data)
        else:
            raise TypeError(f"Unsupported element type {type.element_type}")

    @staticmethod
    def vector_from_list(
        data: Sequence[int] | Sequence[float], typ: IntegerType | IndexType | AnyFloat
    ) -> DenseIntOrFPElementsAttr:
        t = VectorType.from_element_type_and_shape(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)

    @staticmethod
    def tensor_from_list(
        data: Sequence[int]
        | Sequence[float]
        | Sequence[IntegerAttr[IndexType]]
        | Sequence[IntegerAttr[IntegerType]]
        | Sequence[AnyFloatAttr],
        typ: IntegerType | IndexType | AnyFloat,
        shape: Sequence[int],
    ) -> DenseIntOrFPElementsAttr:
        t = AnyTensorType.from_type_and_list(typ, shape)
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
        typ: IntegerType | IndexType, data: Sequence[int] | Sequence[IntAttr]
    ) -> DenseArrayBase:
        if len(data) and isinstance(data[0], int):
            attr_list = [IntAttr(d) for d in cast(Sequence[int], data)]
        else:
            attr_list = cast(Sequence[IntAttr], data)

        return DenseArrayBase([typ, ArrayAttr(attr_list)])

    @staticmethod
    def create_dense_float(
        typ: AnyFloat, data: Sequence[int | float] | Sequence[FloatData]
    ) -> DenseArrayBase:
        if len(data) and isinstance(data[0], int | float):
            attr_list = [FloatData(float(d)) for d in cast(Sequence[int | float], data)]
        else:
            attr_list = cast(Sequence[FloatData], data)

        return DenseArrayBase([typ, ArrayAttr(attr_list)])

    @overload
    @staticmethod
    def from_list(
        type: IntegerType | IndexType, data: Sequence[int] | Sequence[IntAttr]
    ) -> DenseArrayBase:
        ...

    @overload
    @staticmethod
    def from_list(
        type: Attribute, data: Sequence[int | float] | Sequence[FloatData]
    ) -> DenseArrayBase:
        ...

    @staticmethod
    def from_list(
        type: Attribute,
        data: Sequence[int]
        | Sequence[int | float]
        | Sequence[IntAttr]
        | Sequence[FloatData],
    ) -> DenseArrayBase:
        if isinstance(type, IndexType | IntegerType):
            _data = cast(Sequence[int] | Sequence[IntAttr], data)
            return DenseArrayBase.create_dense_int_or_index(type, _data)
        elif isinstance(type, AnyFloat):
            _data = cast(Sequence[int | float] | Sequence[FloatData], data)
            return DenseArrayBase.create_dense_float(type, _data)
        else:
            raise TypeError(f"Unsupported element type {type}")

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


@irdl_op_definition
class UnrealizedConversionCastOp(IRDLOperation):
    name = "builtin.unrealized_conversion_cast"

    inputs: VarOperand
    outputs: VarOpResult

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

    op_name__: OpAttr[StringAttr]
    args: VarOperand
    res: VarOpResult
    regs: VarRegion

    @property
    def op_name(self) -> StringAttr:
        return self.op_name__  # type: ignore

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
                operands: Sequence[SSAValue] | None = None,
                result_types: Sequence[Attribute] | None = None,
                attributes: dict[str, Attribute] | None = None,
                successors: Sequence[Block] | None = None,
                regions: Sequence[Region] | None = None,
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

    body: SingleBlockRegion

    def __init__(self, ops: List[Operation] | Region):
        if isinstance(ops, Region):
            region = ops
        else:
            region = Region(Block(ops))
        super().__init__(regions=[region])

    @property
    def ops(self) -> BlockOps:
        return self.body.ops


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
    ],
)
