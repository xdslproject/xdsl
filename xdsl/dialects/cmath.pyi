from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
)
from xdsl.ir import (
    Dialect,
    Operand,
    OpResult,
    ParametrizedAttribute,
)
from xdsl.irdl import IRDLOperation

class complex(TypeAttribute, ParametrizedAttribute):
    p0: Float32Type | Float64Type

class norm(IRDLOperation):
    o0: Operand
    r0: OpResult

class mul(IRDLOperation):
    o0: Operand
    o1: Operand
    r0: OpResult

Cmath: Dialect
