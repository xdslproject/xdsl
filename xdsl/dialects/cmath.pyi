from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
)
from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
)
from xdsl.irdl import IRDLOperation

class complex(ParametrizedAttribute):
    p0: Float32Type | Float64Type

class norm(IRDLOperation):
    o0: Operand
    r0: Result

class mul(IRDLOperation):
    o0: Operand
    o1: Operand
    r0: Result

Cmath: Dialect
