from dataclasses import dataclass
from xdsl.irdl import *
from xdsl.ir import *


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


@dataclass(frozen=True)
class StringAttr(Data):
    name = "string"
    data: str

    # parser should be of type parser.Parser
    @staticmethod
    def parse(parser) -> 'StringAttr':
        data = parser.parse_str_literal()
        return StringAttr(data)

    def print(self, printer) -> None:
        printer.print_string(f'"{self.data}"')

    @staticmethod
    def get(data: str) -> 'StringAttr':
        return StringAttr(data)


@irdl_attr_definition
class SymbolNameAttr(ParametrizedAttribute):
    name = "symbol_name"
    data = ParameterDef(StringAttr)

    @staticmethod
    def get(data: str) -> 'SymbolNameAttr':
        return SymbolNameAttr([StringAttr.get(data)])


@irdl_attr_definition
class FlatSymbolRefAttr(ParametrizedAttribute):
    name = "flat_symbol_ref"
    data = ParameterDef(StringAttr)

    @staticmethod
    def get(data: str) -> 'FlatSymbolRefAttr':
        return FlatSymbolRefAttr([StringAttr(data)])


@dataclass(frozen=True)
class IntAttr(Data):
    name = "int"
    data: int

    # parser should be of type parser.Parser
    @staticmethod
    def parse(parser: Any) -> 'IntAttr':
        data = parser.parse_int_literal()
        return IntAttr(data)

    # printer should be of type printer.Printer
    def print(self, printer) -> None:
        printer.print_string(f'{self.data}')

    @staticmethod
    def get(data: int) -> 'IntAttr':
        return IntAttr(data)


@irdl_attr_definition
class IntegerType:
    name = "integer_type"

    width = ParameterDef(IntAttr)

    @staticmethod
    def get(width: int) -> Attribute:
        return IntegerType([IntAttr.get(width)])


@irdl_attr_definition
class IndexType:
    name = "index"


@irdl_attr_definition
class IntegerAttr(ParametrizedAttribute):
    name = "integer"

    value = ParameterDef(IntAttr)
    typ = ParameterDef(AnyOf([IntegerType, IndexType]))

    @staticmethod
    def get(
        value: int, type: Attribute = IntegerType.get(64)) -> 'IntegerAttr':
        return IntegerAttr([IntAttr.get(value), type])


@dataclass(frozen=True)
class ArrayAttr(Data):
    name = "array"
    data: List[Attribute]

    @staticmethod
    def parse(parser) -> 'ArrayAttr':
        parser.parse_char("[")
        data = parser.parse_list(parser.parse_optional_attribute)
        parser.parse_char("]")
        return ArrayAttr.get(data)

    def print(self, printer) -> None:
        printer.print_string("[")
        printer.print_list(self.data, printer.print_attribute)
        printer.print_string("]")

    @staticmethod
    def get(data: List[Attribute]) -> 'ArrayAttr':
        return ArrayAttr(data)


# A constraint that enforces an ArrayData whose elements all satisfy
# the elem_constr
@dataclass
class ArrayOfConstraint(AttrConstraint):
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
    def get(data: List[int]) -> 'VectorAttr':
        data_attr = [IntegerAttr.get(d) for d in data]
        return VectorAttr([ArrayAttr.get(data_attr)])


@irdl_attr_definition
class Float32Type:
    name = "f32"

    @staticmethod
    def get() -> Attribute:
        return Float32Type()


@irdl_attr_definition
class FunctionType:
    name = "fun"

    inputs = ParameterDef(ArrayOfConstraint(AnyAttr()))
    outputs = ParameterDef(ArrayOfConstraint(AnyAttr()))

    @staticmethod
    def get(inputs: List[Attribute], outputs: List[Attribute]) -> Attribute:
        return FunctionType([ArrayAttr.get(inputs), ArrayAttr.get(outputs)])


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
