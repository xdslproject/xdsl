from __future__ import annotations
from dataclasses import dataclass

from xdsl.util import block
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
        self.ctx.register_attr(VectorAttr)

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
    def from_int_and_width(value: int, width: int = 64) -> IntegerAttr:
        return IntegerAttr(
            [IntAttr.from_int(value),
             IntegerType.from_width(width)])

    @staticmethod
    def from_index_int_value(value: int) -> IntegerAttr:
        return IntegerAttr([IntAttr.from_int(value), IndexType()])

    @staticmethod
    @builder
    def from_params(value: Union[int, IntAttr],
                    typ: Union[int, Attribute] = 64) -> IntegerAttr:
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
class VectorAttr(ParametrizedAttribute):
    name = "vector"
    data = ParameterDef(ArrayOfConstraint(IntegerAttr))

    @staticmethod
    def from_int_list(data: List[int]) -> VectorAttr:
        data_attr = [IntegerAttr.from_int_and_width(d) for d in data]
        return VectorAttr([ArrayAttr.from_list(data_attr)])

    @staticmethod
    @builder
    def from_list(data: List[Union[int, IntegerAttr]]) -> VectorAttr:
        data_attr = [IntegerAttr.build(d) for d in data]
        return VectorAttr([ArrayAttr.from_list(data_attr)])


@irdl_attr_definition
class Float32Type(ParametrizedAttribute):
    name = "f32"


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
    type = AttributeDef(Attribute)
    sym_visibility = AttributeDef(StringAttr)


@irdl_op_definition
class ModuleOp(Operation):
    name: str = "module"

    body = SingleBlockRegionDef()

    @property
    def ops(self) -> List[Operation]:
        return self.regions[0].blocks[0].ops


@overload
def func(name: str, type: Attribute, return_type: Attribute,
         f: Callable[[BlockArgument], List[Operation]]) -> Operation:
    ...


@overload
def func(
        name: str, types: List[Attribute], return_type: Attribute,
        f: Callable[[BlockArgument, BlockArgument],
                    List[Operation]]) -> Operation:
    ...


@overload
def func(
    name: str, types: List[Attribute], return_type: Attribute,
    f: Callable[[BlockArgument, BlockArgument, BlockArgument], List[Operation]]
) -> Operation:
    ...


def func(name, input_types, return_types, f) -> Operation:
    type_attr = FunctionType.build(input_types, return_types)
    op = FuncOp.create(
        [], [],
        attributes={
            "sym_name": StringAttr.build(name),
            "type": type_attr,
            "sym_visibility": StringAttr.from_str("private")
        },
        regions=[Region([block(input_types, f)])])
    return op


# This function is easier to use while parsing, but makes the
# inline definitions complicated
def func2(name, input_types, return_types, region: Region) -> Operation:
    type_attr = FunctionType.get(input_types, return_types)
    op = FuncOp.create(
        [], [],
        attributes={
            "sym_name": StringAttr.build(name),
            "type": type_attr,
            "sym_visibility": StringAttr.from_str("private")
        },
        regions=[region])
    return op


def module(ops: Union[List[Operation], Region]) -> Operation:
    if isinstance(ops, List):
        region = Region([Block([], ops)])
    else:
        region = ops
    op = ModuleOp.create([], [], regions=[region])
    return op
