from __future__ import annotations
from dataclasses import dataclass

from xdsl.irdl import *
from xdsl.ir import *
from typing import overload

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

        self.ctx.register_attr(FunctionType)
        self.ctx.register_attr(Float32Type)
        self.ctx.register_attr(IntegerType)
        self.ctx.register_attr(IndexType)

        self.ctx.register_op(ModuleOp)
        self.ctx.register_op(FuncOp)


@irdl_attr_definition
class StringAttr(Data):
    name = "string"
    data: str

    @staticmethod
    def parse(parser: Parser) -> StringAttr:
        data = parser.parse_str_literal()
        return StringAttr(data)

    def print(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')

    @staticmethod
    @builder
    def from_str(data: str) -> StringAttr:
        return StringAttr(data)


@irdl_attr_definition
class SymbolNameAttr(ParametrizedAttribute):
    name = "symbol_name"
    data = ParameterDef(StringAttr)

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
    data = ParameterDef(StringAttr)

    @staticmethod
    @builder
    def from_str(data: str) -> FlatSymbolRefAttr:
        return FlatSymbolRefAttr([StringAttr(data)])

    @staticmethod
    @builder
    def from_string_attr(data: StringAttr) -> FlatSymbolRefAttr:
        return FlatSymbolRefAttr([data])


@irdl_attr_definition
class IntAttr(Data):
    name = "int"
    data: int

    @staticmethod
    def parse(parser: Parser) -> IntAttr:
        data = parser.parse_int_literal()
        return IntAttr(data)

    def print(self, printer: Printer) -> None:
        printer.print_string(f'{self.data}')

    @staticmethod
    @builder
    def from_int(data: int) -> IntAttr:
        return IntAttr(data)


@irdl_attr_definition
class IntegerType(ParametrizedAttribute):
    name = "integer_type"
    width = ParameterDef(IntAttr)

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
    value = ParameterDef(IntAttr)
    typ = ParameterDef(AnyOf([IntegerType, IndexType]))

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
    def from_params(value: Union[int, IntAttr],
                    typ: Union[int, Attribute]) -> IntegerAttr:
        value = IntAttr.build(value)
        if not isinstance(typ, IndexType):
            typ = IntegerType.build(typ)
        return IntegerAttr([value, typ])


@irdl_attr_definition
class ArrayAttr(Data):
    name = "array"
    data: List[Attribute]

    @staticmethod
    def parse(parser) -> ArrayAttr:
        parser.parse_char("[")
        data = parser.parse_list(parser.parse_optional_attribute)
        parser.parse_char("]")
        return ArrayAttr.get(data)

    def print(self, printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    @builder
    def from_list(data: List[Attribute]) -> ArrayAttr:
        return ArrayAttr(data)


@dataclass
class ArrayOfConstraint(AttrConstraint):
    """
    A constraint that enforces an ArrayData whose elements all satisfy
    the elem_constr.
    """
    elem_constr: AttrConstraint

    def __init__(self, constr: Union[Attribute, typing.Type[Attribute],
                                     AttrConstraint]):
        self.elem_constr = attr_constr_coercion(constr)

    def verify(self, data: Data) -> None:
        if not isinstance(data, ArrayAttr):
            raise Exception(f"expected data ArrayData but got {data}")

        for e in data.data:
            self.elem_constr.verify(e)


@irdl_attr_definition
class VectorType(ParametrizedAttribute):
    name = "vector"

    shape = ParameterDef(ArrayOfConstraint(IntegerAttr))
    element_type = ParameterDef(AnyAttr())

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_shape(self) -> List[int]:
        return [i.parameters[0].data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
            referenced_type: Attribute,
            shape: List[Union[int, IntegerAttr]] = None) -> VectorType:
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
        shape: ArrayAttr = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> VectorType:
        return VectorType([shape, referenced_type])


@irdl_attr_definition
class TensorType(ParametrizedAttribute):
    name = "tensor"

    shape = ParameterDef(ArrayOfConstraint(IntegerAttr))
    element_type = ParameterDef(AnyAttr())

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_shape(self) -> List[int]:
        return [i.parameters[0].data for i in self.shape.data]

    @staticmethod
    @builder
    def from_type_and_list(
            referenced_type: Attribute,
            shape: List[Union[int, IntegerAttr]] = None) -> TensorType:
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
        shape: ArrayAttr = ArrayAttr.from_list(
            [IntegerAttr.from_int_and_width(1, 64)])
    ) -> TensorType:
        return TensorType([shape, referenced_type])


@irdl_attr_definition
class DenseIntOrFPElementsAttr(ParametrizedAttribute):
    name = "dense"
    # TODO add support for FPElements
    type = ParameterDef(AnyOf([VectorType, TensorType]))
    # TODO add support for multi-dimensional data
    data = ParameterDef(ArrayOfConstraint(IntegerAttr))

    @staticmethod
    @builder
    def from_int_list(type: Union[VectorType, TensorType], data: List[int],
                      bitwidth) -> DenseIntOrFPElementsAttr:
        data_attr = [IntegerAttr.from_int_and_width(d, bitwidth) for d in data]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(data_attr)])

    @staticmethod
    @builder
    def from_list(
            type: Union[VectorType, TensorType],
            data: List[Union[int, IntegerAttr]]) -> DenseIntOrFPElementsAttr:
        element_type = type.element_type
        # Only use the element_type if the passed data is an int, o/w use the IntegerAttr
        data_attr = [(IntegerAttr.from_params(d, element_type) if isinstance(
            d, int) else d) for d in data]
        return DenseIntOrFPElementsAttr([type, ArrayAttr.from_list(data_attr)])

    @staticmethod
    @builder
    def vector_from_list(
            data: List[int],
            typ: Union[IntegerType, IndexType]) -> DenseIntOrFPElementsAttr:
        t = VectorType.from_type_and_list(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)

    @staticmethod
    @builder
    def tensor_from_list(
            data: List[int],
            typ: Union[IntegerType, IndexType]) -> DenseIntOrFPElementsAttr:
        t = TensorType.from_type_and_list(typ, [len(data)])
        return DenseIntOrFPElementsAttr.from_list(t, data)


@irdl_attr_definition
class Float32Type(ParametrizedAttribute):
    name = "f32"


f32 = Float32Type()


@irdl_attr_definition
class FunctionType(ParametrizedAttribute):
    name = "fun"

    inputs = ParameterDef(ArrayOfConstraint(AnyAttr()))
    outputs = ParameterDef(ArrayOfConstraint(AnyAttr()))

    @staticmethod
    @builder
    def from_lists(inputs: List[Attribute],
                   outputs: List[Attribute]) -> Attribute:
        return FunctionType(
            [ArrayAttr.from_list(inputs),
             ArrayAttr.from_list(outputs)])

    @staticmethod
    @builder
    def from_attrs(inputs: ArrayAttr, outputs: ArrayAttr) -> Attribute:
        return FunctionType([inputs, outputs])


@irdl_op_definition
class FuncOp(Operation):
    name: str = "builtin.func"

    body = RegionDef()
    sym_name = AttributeDef(StringAttr)
    type = AttributeDef(FunctionType)
    sym_visibility = AttributeDef(StringAttr)

    @staticmethod
    def from_callable(
            name: str, input_types: List[Attribute],
            return_types: List[Attribute],
            func: Callable[[BlockArgument, ...], List[Operation]]) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        op = FuncOp.build(
            attributes={
                "sym_name": name,
                "type": type_attr,
                "sym_visibility": "private"
            },
            regions=[Region([Block.from_callable(input_types, func)])])
        return op

    @staticmethod
    def from_region(name: str, input_types: List[Attribute],
                    return_types: List[Attribute], region: Region) -> FuncOp:
        type_attr = FunctionType.from_lists(input_types, return_types)
        op = FuncOp.build(attributes={
            "sym_name": name,
            "type": type_attr,
            "sym_visibility": "private"
        },
                          regions=[region])
        return op


@irdl_op_definition
class ModuleOp(Operation):
    name: str = "module"

    body = SingleBlockRegionDef()

    @property
    def ops(self) -> List[Operation]:
        return self.regions[0].blocks[0].ops

    @staticmethod
    def from_region_or_ops(ops: Union[List[Operation], Region]) -> ModuleOp:
        if isinstance(ops, list):
            region = Region([Block([], ops)])
        elif isinstance(ops, Region):
            region = ops
        else:
            raise TypeError(
                f"Expected region or operation list in ModuleOp.get, but got '{ops}'"
            )
        op = ModuleOp.create([], [], regions=[region])
        return op
