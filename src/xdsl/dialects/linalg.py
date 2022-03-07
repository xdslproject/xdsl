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

@irdl_op_definition
class Generic(Operation):
    name: str = "linalg.generic"
    inputs = VarOperandDef(AnyAttr)
    # should be VarOperandDef(AnyShaped)
    outputs = VarOperandDef(AnyAttr)
    indexing_maps = AttributeDef(ArrayAttr)
    iterator_types = AttributeDef(ArrayAttr)
    doc = AttributeDef(StringAttr)
    library_call = AttributeDef(StringAttr)

    region = RegionDef()
    # output type should be VarResultDef(AnyRankedTensor())
    # if this operates on memrefs, no output result is produced, 
    # on tensors the result is of tensor type
    output = VarResultDef(AnyAttr())

    @staticmethod
    def parse(parser: Parser) -> Generic:

        attributes = parser.parse_op_attributes()
        parser.parse_string("ins")
        operands = parser.parse_operands()
        parser.parse_string("outs")
        operands.extend(parser.parse_operands())

        attributes | parser.parse_op_attributes()

        op = Generic.create(operands,[],attributes,[])
        return op

    def verify(self) -> None:
        return

class Yield(Operation):
    name: str = "linalg.yield"
    values = VarOperandDef(AnyAttr)