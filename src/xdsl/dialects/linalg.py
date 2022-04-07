from __future__ import annotations
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.memref import *
from xdsl.dialects.builtin import IntegerType, Float32Type, IntegerAttr, FlatSymbolRefAttr
from xdsl.parser import Parser


@dataclass
class Linalg:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Generic)
        self.ctx.register_op(Yield)
        self.ctx.register_op(Fill)
        self.ctx.register_op(Index)


@irdl_op_definition
class Generic(Operation):
    name: str = "linalg.generic"
    inputs = VarOperandDef(AnyAttr)
    outputs = VarOperandDef(AnyOf([MemRefType, TensorType]))
    indexing_maps = AttributeDef(ArrayAttr)
    iterator_types = AttributeDef(ArrayAttr)
    doc = AttributeDef(StringAttr)
    library_call = AttributeDef(StringAttr)

    region = RegionDef()
    output = VarResultDef(TensorType)

    @staticmethod
    def parse(parser: Parser) -> Generic:

        attributes = parser.parse_op_attributes()
        parser.parse_string("ins")
        operands = parser.parse_operands()
        parser.parse_string("outs")
        operands.extend(parser.parse_operands())

        op = Generic.create(operands, [], attributes, [])
        return op

    def verify(self) -> None:
        return


@irdl_op_definition
class Yield(Operation):
    name: str = "linalg.yield"
    values = VarOperandDef(AnyAttr)


@irdl_op_definition
class Fill(Operation):
    name: str = "linalg.fill"
    value = OperandDef(AnyOf([FloatType, IntegerType, VectorType]))
    output = OperandDef(AnyOf([MemRefType, TensorType]))  # should be AnyShaped
    result = OptResultDef(TensorType)


@irdl_op_definition
class Index(Operation):
    name: str = "linalg.index"
    dim = AttributeDef(i64)
    result = ResultDef(IndexType)