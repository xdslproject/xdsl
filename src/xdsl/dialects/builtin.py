from __future__ import annotations
from dataclasses import dataclass

from xdsl.irdl import *
from xdsl.ir import *

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
        self.ctx.register_attr(Float32Type)
        self.ctx.register_attr(IntegerType)
        self.ctx.register_attr(IndexType)

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
    def from_width(width: int) -> Attribute:
        return IntegerType([IntAttr.from_int(width)])


i64 = IntegerType.from_width(64)
i32 = IntegerType.from_width(32)
i1 = IntegerType.from_width(1)


@irdl_attr_definition
class IndexType(ParametrizedAttribute):
    name = "index"


@irdl_attr_definition
class IntegerAttr(ParametrizedAttribute):
    name = "integer"
    value: ParameterDef[IntAttr]
    typ: ParameterDef[IntegerType | IndexType]

    @staticmethod
    @builder
    def from_int_and_width(value: int, width: int) -> IntegerAttr:
        return IntegerAttr(
            [IntAttr.from_int(value),
             IntegerType.from_width(width)])

    @staticmethod
    @builder
    def from_index_int_value(value: int) -> IntegerAttr:
        return IntegerAttr([IntAttr.from_int(value), IndexType()])

    @staticmethod
    @builder
    def from_params(value: int | IntAttr, typ: int | Attribute) -> IntegerAttr:
        value = IntAttr.build(value)
        if not isinstance(typ, IndexType):
            typ = IntegerType.build(typ)
        return IntegerAttr([value, typ])


@irdl_attr_definition
class ArrayAttr(Data[List[A]]):
    name = "array"

    @staticmethod
    def parse_parameter(parser: Parser) -> List[A]:
        parser.parse_char("[")
        data = parser.parse_list(parser.parse_optional_attribute)
        parser.parse_char("]")
        # the type system can't ensure that the elements are of type A
        # and not just of type Attribute, therefore, the following cast
        return cast(List[A], data)

    @staticmethod
    def print_parameter(data: List[A], printer: Printer) -> None:
        printer.print_string("[")
        printer.print_list(data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    @builder
    def from_list(data: List[A]) -> ArrayAttr[A]:
        return ArrayAttr(data)


@irdl_attr_definition
class TupleType(ParametrizedAttribute):
    name = "tuple"

    types = ParameterDef(ArrayOfConstraint(Attribute))

    @staticmethod
    @builder
    def from_type_list(types: List[Attribute]) -> TupleType:
        return TupleType([ArrayAttr.from_list(types)])  #type: ignore


@irdl_attr_definition
class VectorType(ParametrizedAttribute):
    name = "vector"

    shape: ParameterDef[ArrayAttr[IntegerAttr]]
    element_type: ParameterDef[Attribute]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
            referenced_type: Attribute,
            shape: Optional[List[int | IntegerAttr]] = None) -> VectorType:
        if shape is None:
            shape = [1]
        return VectorType([
            ArrayAttr.from_list([IntegerAttr.build(d) for d in shape]),
            referenced_type
        ])

    @staticmethod
    @builder
    def from_params(
        referenced_type: Attribute,
        shape: ArrayAttr[IntegerAttr] = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> VectorType:
        return VectorType([shape, referenced_type])


@irdl_attr_definition
class TensorType(ParametrizedAttribute):
    name = "tensor"

    shape: ParameterDef[ArrayAttr[IntegerAttr]]
    element_type: ParameterDef[Attribute]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
            referenced_type: Attribute,
            shape: Optional[Sequence[int | IntegerAttr]] = None) -> TensorType:
        if shape is None:
            shape = [1]
        return TensorType([
            ArrayAttr.from_list([IntegerAttr.build(d) for d in shape]),
            referenced_type
        ])

    @staticmethod
    @builder
    def from_params(
        referenced_type: Attribute,
        shape: ArrayAttr[IntegerAttr] = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> TensorType:
        return TensorType([shape, referenced_type])


@irdl_attr_definition
class DenseIntOrFPElementsAttr(ParametrizedAttribute):
    name = "dense"
    # TODO add support for FPElements
    type: ParameterDef[VectorType | TensorType]
    # TODO add support for multi-dimensional data
    data: ParameterDef[ArrayAttr[IntegerAttr]]

    @staticmethod
    @builder
    def from_int_list(type: VectorType | TensorType, data: List[int],
                      bitwidth: int) -> DenseIntOrFPElementsAttr:
        data_attr = [IntegerAttr.from_int_and_width(d, bitwidth) for d in data]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(data_attr)])

    @staticmethod
    @builder
    def from_list(
            type: VectorType | TensorType,
            data: List[int] | List[IntegerAttr]) -> DenseIntOrFPElementsAttr:
        element_type = type.element_type
        # Only use the element_type if the passed data is an int, o/w use the IntegerAttr
        data_attr = [(IntegerAttr.from_params(d, element_type) if isinstance(
            d, int) else d) for d in data]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(data_attr)])

    @staticmethod
    @builder
    def vector_from_list(
            data: List[int],
            typ: IntegerType | IndexType) -> DenseIntOrFPElementsAttr:
        t = VectorType.from_type_and_list(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)

    @staticmethod
    @builder
    def tensor_from_list(
            data: List[int],
            typ: IntegerType | IndexType) -> DenseIntOrFPElementsAttr:
        t = TensorType.from_type_and_list(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)


@irdl_attr_definition
class Float32Type(ParametrizedAttribute):
    name = "f32"


f32 = Float32Type()


@irdl_attr_definition
class UnitAttr(ParametrizedAttribute):
    name: str = "unit"


@irdl_attr_definition
class FunctionType(ParametrizedAttribute):
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


@irdl_op_definition
class ModuleOp(Operation):
    name: str = "module"

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
