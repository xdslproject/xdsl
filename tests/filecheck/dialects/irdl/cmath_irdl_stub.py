# RUN: python %s | filecheck %s

from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.dialects.irdl.irdl import DialectOp
from xdsl.interpreters.irdl import make_dialect
from xdsl.parser import Parser
from xdsl.traits import SymbolTable
from xdsl.utils.dialect_stub import DialectStubGenerator

# Test dynamically generating Python type stubs for an irdl dialect.

if __name__ == "__main__":

    # Register all dialects for lazy-loading
    ctx = MLContext()
    for n, f in get_all_dialects().items():
        # Except cmath to avoid conflict with the one we're going to load from its IRDL description
        if n == "cmath":
            continue
        ctx.register_dialect(n, f)

    # Open the IRDL description of cmath, parse it
    f = open("tests/filecheck/dialects/irdl/cmath.irdl.mlir")
    parser = Parser(ctx, f.read())
    module = parser.parse_module()

    # Make it a PyRDL Dialect
    dialect_op = SymbolTable.lookup_symbol(module, "cmath")
    assert isinstance(dialect_op, DialectOp)
    dialect = make_dialect(dialect_op)

    # Generate and print type stubs!
    stub = DialectStubGenerator(dialect)
    print(stub.generate_dialect_stubs())

# CHECK:       from xdsl.dialects.builtin import (
# CHECK-NEXT:      Float32Type,
# CHECK-NEXT:      Float64Type,
# CHECK-NEXT:  )
# CHECK-NEXT:  from xdsl.ir import (
# CHECK-NEXT:      Dialect,
# CHECK-NEXT:      OpResult,
# CHECK-NEXT:      ParametrizedAttribute,
# CHECK-NEXT:      TypeAttribute,
# CHECK-NEXT:  )
# CHECK-NEXT:  from xdsl.irdl import (
# CHECK-NEXT:      IRDLOperation,
# CHECK-NEXT:      Operand,
# CHECK-NEXT:  )
# CHECK-EMPTY:
# CHECK-NEXT:  class complex(TypeAttribute, ParametrizedAttribute):
# CHECK-NEXT:      p0 : "Float32Type | Float64Type"
# CHECK-EMPTY:
# CHECK-EMPTY:
# CHECK-NEXT:  class norm(IRDLOperation):
# CHECK-NEXT:      o0 : Operand
# CHECK-NEXT:      r0 : OpResult
# CHECK-EMPTY:
# CHECK-EMPTY:
# CHECK-NEXT:  class mul(IRDLOperation):
# CHECK-NEXT:      o0 : Operand
# CHECK-NEXT:      o1 : Operand
# CHECK-NEXT:      r0 : OpResult
# CHECK-EMPTY:
# CHECK-EMPTY:
# CHECK-NEXT:  Cmath : Dialect
# CHECK-EMPTY:
