from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, List, cast, Type, Sequence, Optional

from xdsl.ir import (MLContext, TYPE_CHECKING, Data, MLIRType,
                     ParametrizedAttribute, Operation)
from xdsl.irdl import (irdl_attr_definition, attr_constr_coercion,
                       irdl_to_attr_constraint, irdl_op_definition, builder,
                       ParameterDef, SingleBlockRegionDef, TypeVar, Generic,
                       GenericData, AttrConstraint, Any, Attribute, Region,
                       VerifyException, AnyAttr)

if TYPE_CHECKING:
    from xdsl.parser import Parser
    from xdsl.printer import Printer


@dataclass
class Builtin:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(StringAttr)
        self.ctx.register_attr(FlatSymbolRefAttr)
        self.ctx.register_attr(SymbolNameAttr)
        self.ctx.register_attr(IntAttr)
        self.ctx.register_attr(IntegerAttr)
        self.ctx.register_attr(ArrayAttr)
        self.ctx.register_attr(VectorType)
        self.ctx.register_attr(TensorType)
        self.ctx.register_attr(DenseIntOrFPElementsAttr)
        self.ctx.register_attr(UnitAttr)
        self.ctx.register_attr(TupleType)

        self.ctx.register_attr(FunctionType)
        self.ctx.register_attr(Float16Type)
        self.ctx.register_attr(Float32Type)
        self.ctx.register_attr(Float64Type)
        self.ctx.register_attr(FloatData)
        self.ctx.register_attr(FloatAttr)
        self.ctx.register_attr(IntegerType)
        self.ctx.register_attr(IndexType)

        self.ctx.register_attr(NoneAttr)
        self.ctx.register_attr(OpaqueAttr)

        self.ctx.register_op(ModuleOp)


@irdl_attr_definition
class StringAttr(Data[str]):
    name = "string"

    @staticmethod
    def parse_parameter(parser: Parser) -> str:
        data = parser.parse_str_literal()
        return data

    @staticmethod
    def print_parameter(data: str, printer: Printer) -> None:
        printer.print_string(f'"{data}"')

    @staticmethod
    @builder
    def from_str(data: str) -> StringAttr:
        return StringAttr(data)


@irdl_attr_definition
class SymbolNameAttr(ParametrizedAttribute):
    name = "symbol_name"
    data: ParameterDef[StringAttr]

    @staticmethod
    @builder
    def from_str(data: str) -> SymbolNameAttr:
        return SymbolNameAttr([StringAttr.from_str(data)])

    @staticmethod
    @builder
    def from_string_attr(data: StringAttr) -> SymbolNameAttr:
        return SymbolNameAttr([data])


@irdl_attr_definition
class FlatSymbolRefAttr(ParametrizedAttribute):
    name = "flat_symbol_ref"
    data: ParameterDef[StringAttr]

    @staticmethod
    @builder
    def from_str(data: str) -> FlatSymbolRefAttr:
        return FlatSymbolRefAttr([StringAttr(data)])

    @staticmethod
    @builder
    def from_string_attr(data: StringAttr) -> FlatSymbolRefAttr:
        return FlatSymbolRefAttr([data])


@irdl_attr_definition
class IntAttr(Data[int]):
    name = "int"

    @staticmethod
    def parse_parameter(parser: Parser) -> int:
        data = parser.parse_int_literal()
        return data

    @staticmethod
    def print_parameter(data: int, printer: Printer) -> None:
        printer.print_string(f'{data}')

    @staticmethod
    @builder
    def from_int(data: int) -> IntAttr:
        return IntAttr(data)


@irdl_attr_definition
class IntegerType(ParametrizedAttribute):
    name = "integer_type"
    width: ParameterDef[IntAttr]

    @staticmethod
    @builder
    def from_width(width: int) -> IntegerType:
        return IntegerType([IntAttr.from_int(width)])


i64 = IntegerType.from_width(64)
i32 = IntegerType.from_width(32)
i1 = IntegerType.from_width(1)


@irdl_attr_definition
class UnitAttr(ParametrizedAttribute):
    name = "unit"


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "index"


_IntegerAttrTyp = TypeVar("_IntegerAttrTyp",
                          bound=IntegerType | IndexType,
                          covariant=True)


@irdl_attr_definition
class IntegerAttr(Generic[_IntegerAttrTyp], ParametrizedAttribute):
    name = "integer"
    value: ParameterDef[IntAttr]
    typ: ParameterDef[_IntegerAttrTyp]

    @staticmethod
    @builder
    def from_int_and_width(value: int, width: int) -> IntegerAttr[IntegerType]:
        return IntegerAttr(
            [IntAttr.from_int(value),
             IntegerType.from_width(width)])

    @staticmethod
    @builder
    def from_index_int_value(value: int) -> IntegerAttr[IndexType]:
        return IntegerAttr([IntAttr.from_int(value), IndexType()])

    @staticmethod
    @builder
    def from_params(
            value: int | IntAttr,
            typ: int | Attribute) -> IntegerAttr[IntegerType | IndexType]:
        value = IntAttr.build(value)
        if not isinstance(typ, IndexType):
            typ = IntegerType.build(typ)
        return IntegerAttr([value, typ])


AnyIntegerAttr: TypeAlias = IntegerAttr[IntegerType | IndexType]


class Float16Type(ParametrizedAttribute, MLIRType):
    name = "f16"


f16 = Float16Type()


@irdl_attr_definition
class Float32Type(ParametrizedAttribute, MLIRType):
    name = "f32"


f32 = Float32Type()


class Float64Type(ParametrizedAttribute, MLIRType):
    name = "f64"


f64 = Float64Type()

AnyFloat: TypeAlias = Float16Type | Float32Type | Float64Type


@irdl_attr_definition
class FloatData(Data[float]):
    name = "float_data"

    @staticmethod
    def parse_parameter(parser: Parser) -> float:
        return parser.parse_float_literal()

    @staticmethod
    def print_parameter(data: float, printer: Printer) -> None:
        printer.print_string(f'{data}')

    @staticmethod
    @builder
    def from_float(data: float) -> FloatData:
        return FloatData(data)


_FloatAttrTyp = TypeVar("_FloatAttrTyp", bound=AnyFloat, covariant=True)

_FloatAttrTypContr = TypeVar("_FloatAttrTypContr", bound=AnyFloat)


@irdl_attr_definition
class FloatAttr(Generic[_FloatAttrTyp], ParametrizedAttribute):
    name = "float"

    value: ParameterDef[FloatData]
    type: ParameterDef[_FloatAttrTyp]

    @staticmethod
    @builder
    def from_value(
        value: float, type: _FloatAttrTypContr = Float32Type()
    ) -> FloatAttr[_FloatAttrTypContr]:
        return FloatAttr([FloatData.from_float(value), type])

    @staticmethod
    @builder
    def from_float_and_width(value: float, width: int) -> FloatAttr[AnyFloat]:
        if width == 16:
            return FloatAttr([FloatData.from_float(value), Float16Type()])
        if width == 32:
            return FloatAttr([FloatData.from_float(value), Float32Type()])
        if width == 64:
            return FloatAttr([FloatData.from_float(value), Float64Type()])
        raise ValueError(f"Invalid bitwidth: {width}")


AnyFloatAttr: TypeAlias = FloatAttr[AnyFloat]


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


_ArrayAttrT = TypeVar("_ArrayAttrT", bound=Attribute, covariant=True)


@irdl_attr_definition
class ArrayAttr(GenericData[List[_ArrayAttrT]]):
    name = "array"

    @staticmethod
    def parse_parameter(parser: Parser) -> List[_ArrayAttrT]:
        parser.parse_char("[")
        data = parser.parse_list(parser.parse_optional_attribute)
        parser.parse_char("]")
        # the type system can't ensure that the elements are of type A
        # and not just of type Attribute, therefore, the following cast
        return cast(List[_ArrayAttrT], data)

    @staticmethod
    def print_parameter(data: List[_ArrayAttrT], printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        if len(args) == 1:
            return ArrayOfConstraint(irdl_to_attr_constraint(args[0]))
        if len(args) == 0:
            return ArrayOfConstraint(AnyAttr())
        raise TypeError(f"Attribute ArrayAttr expects at most type"
                        f" parameter, but {len(args)} were given")

    def verify(self) -> None:
        if not isinstance(self.data, list):
            raise VerifyException(
                f"Wrong type given to attribute {self.name}: got"
                f" {type(self.data)}, but expected list of"
                " attributes")
        for idx, val in enumerate(self.data):
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but {idx} "
                    f"element is of type {type(val)}")

    @staticmethod
    @builder
    def from_list(data: List[_ArrayAttrT]) -> ArrayAttr[_ArrayAttrT]:
        return ArrayAttr(data)


AnyArrayAttr: TypeAlias = ArrayAttr[Attribute]


@irdl_attr_definition
class TupleType(ParametrizedAttribute):
    name = "tuple"

    types: ParameterDef[ArrayAttr[Attribute]]

    @staticmethod
    @builder
    def from_type_list(types: List[Attribute]) -> TupleType:
        return TupleType([ArrayAttr.from_list(types)])


_VectorTypeElems = TypeVar("_VectorTypeElems", bound=Attribute)


@irdl_attr_definition
class VectorType(Generic[_VectorTypeElems], ParametrizedAttribute, MLIRType):
    name = "vector"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_VectorTypeElems]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
        referenced_type: _VectorTypeElems,
        shape: Optional[List[int | IntegerAttr[IndexType]]] = None
    ) -> VectorType[_VectorTypeElems]:
        if shape is None:
            shape = [1]
        return VectorType([
            ArrayAttr.from_list(
                [IntegerAttr[IntegerType].build(d) for d in shape]),
            referenced_type
        ])

    @staticmethod
    @builder
    def from_params(
        referenced_type: _VectorTypeElems,
        shape: ArrayAttr[IntegerAttr[IntegerType]] = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> VectorType[_VectorTypeElems]:
        return VectorType([shape, referenced_type])


AnyVectorType: TypeAlias = VectorType[Attribute]

_TensorTypeElems = TypeVar("_TensorTypeElems", bound=Attribute, covariant=True)


@irdl_attr_definition
class TensorType(Generic[_TensorTypeElems], ParametrizedAttribute, MLIRType):
    name = "tensor"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_TensorTypeElems]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
        referenced_type: _TensorTypeElems,
        shape: Optional[Sequence[int | IntegerAttr[IndexType]]] = None
    ) -> TensorType[_TensorTypeElems]:
        if shape is None:
            shape = [1]
        return TensorType([
            ArrayAttr.from_list(
                [IntegerAttr[IndexType].build(d) for d in shape]),
            referenced_type
        ])

    @staticmethod
    @builder
    def from_params(
        referenced_type: _VectorTypeElems,
        shape: AnyArrayAttr = AnyArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> TensorType[_VectorTypeElems]:
        return TensorType([shape, referenced_type])


AnyTensorType: TypeAlias = TensorType[Attribute]


@dataclass(init=False)
class ContainerOf(AttrConstraint):
    """A type constraint that can be nested once in a vector or a tensor."""
    elem_constr: AttrConstraint

    def __init__(
            self,
            elem_constr: Attribute | type[Attribute] | AttrConstraint) -> None:
        self.elem_constr = attr_constr_coercion(elem_constr)

    def verify(self, attr: Attribute) -> None:
        if isinstance(attr, VectorType) or isinstance(attr, TensorType):
            self.elem_constr.verify(attr.element_type)  # type: ignore
        else:
            self.elem_constr.verify(attr)


@irdl_attr_definition
class DenseIntOrFPElementsAttr(ParametrizedAttribute):
    name = "dense"
    type: ParameterDef[VectorType[IntegerType] | VectorType[IndexType]
                       | VectorType[AnyFloat] | TensorType[IndexType]
                       | TensorType[IntegerType] | TensorType[AnyFloat]]
    # TODO add support for multi-dimensional data
    data: ParameterDef[ArrayAttr[AnyIntegerAttr] | ArrayAttr[AnyFloatAttr]]

    @staticmethod
    @builder
    def from_index_list(
            type: VectorType[IndexType] | TensorType[IndexType],
            data: List[int | IntegerAttr[IndexType]]
    ) -> DenseIntOrFPElementsAttr:
        attr_list = [
            IntegerAttr.from_index_int_value(d) if isinstance(d, int) else d
            for d in data
        ]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(attr_list)])

    @staticmethod
    @builder
    def from_int_list(
        type: VectorType[IntegerType] | TensorType[IntegerType],
        data: List[int | IntegerAttr[IntegerType]]
    ) -> DenseIntOrFPElementsAttr:
        attr_list = [
            IntegerAttr.from_params(d, type.element_type) if isinstance(
                d, int) else d for d in data
        ]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(attr_list)])

    @staticmethod
    @builder
    def from_float_list(
            type: VectorType[AnyFloat] | TensorType[AnyFloat],
            data: List[float | AnyFloatAttr]) -> DenseIntOrFPElementsAttr:
        data_attr = [
            FloatAttr.from_value(d, type.element_type)
            if isinstance(d, float) else d for d in data
        ]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(data_attr)])

    @staticmethod
    @builder
    def from_list(
        type: AnyVectorType | AnyTensorType,
        data: List[int | IntegerAttr[IntegerType]]
        | List[int | IntegerAttr[IndexType]] | List[float | AnyFloatAttr]
    ) -> DenseIntOrFPElementsAttr:
        if isinstance(type.element_type, IntegerType):
            return DenseIntOrFPElementsAttr.from_int_list(type, data)
        elif isinstance(type.element_type, IndexType):
            return DenseIntOrFPElementsAttr.from_index_list(type, data)
        elif isinstance(type.element_type, AnyFloat):
            return DenseIntOrFPElementsAttr.from_float_list(type, data)
        else:
            raise TypeError(f"Unsupported element type {type.element_type}")

    @staticmethod
    @builder
    def vector_from_list(
            data: List[int] | List[float],
            typ: IntegerType | IndexType | AnyFloat
    ) -> DenseIntOrFPElementsAttr:
        t = AnyVectorType.from_type_and_list(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)

    @staticmethod
    @builder
    def tensor_from_list(
            data: List[int] | List[float],
            typ: IntegerType | IndexType | AnyFloat
    ) -> DenseIntOrFPElementsAttr:
        t = AnyTensorType.from_type_and_list(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)


class Float16Type(ParametrizedAttribute, MLIRType):
    name = "f16"


f16 = Float16Type()


@irdl_attr_definition
class UnitAttr(ParametrizedAttribute):
    name: str = "unit"


@irdl_attr_definition
class FunctionType(ParametrizedAttribute, MLIRType):
    name = "fun"

    inputs: ParameterDef[ArrayAttr[Attribute]]
    outputs: ParameterDef[ArrayAttr[Attribute]]

    @staticmethod
    @builder
    def from_lists(inputs: List[Attribute],
                   outputs: List[Attribute]) -> Attribute:
        return FunctionType(
            [ArrayAttr.from_list(inputs),
             ArrayAttr.from_list(outputs)])

    @staticmethod
    @builder
    def from_attrs(inputs: ArrayAttr[Attribute],
                   outputs: ArrayAttr[Attribute]) -> Attribute:
        return FunctionType([inputs, outputs])


@irdl_attr_definition
class NoneAttr(ParametrizedAttribute):
    """An attribute representing the absence of an attribute."""
    name: str = "none"


@irdl_attr_definition
class OpaqueAttr(ParametrizedAttribute):
    name: str = "opaque"

    ident: ParameterDef[StringAttr]
    value: ParameterDef[StringAttr]
    type: ParameterDef[Attribute]

    @staticmethod
    def from_strings(name: str, value: str,
                     type: Attribute = NoneAttr()) -> OpaqueAttr:
        return OpaqueAttr(
            [StringAttr.from_str(name),
             StringAttr.from_str(value), type])


@irdl_op_definition
class ModuleOp(Operation):
    name: str = "builtin.module"

    body = SingleBlockRegionDef()

    @property
    def ops(self) -> List[Operation]:
        return self.regions[0].blocks[0].ops

    @staticmethod
    def from_region_or_ops(ops: List[Operation] | Region) -> ModuleOp:
        if isinstance(ops, list):
            region = Region.from_operation_list(ops)
        elif isinstance(ops, Region):
            region = ops
        else:
            raise TypeError(
                f"Expected region or operation list in ModuleOp.get, but got '{ops}'"
            )
        op = ModuleOp.create([], [], regions=[region])
        return op
