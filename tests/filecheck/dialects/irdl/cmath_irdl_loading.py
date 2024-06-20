# RUN: python %s | filecheck %s

from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.dialects.irdl.irdl import DialectOp
from xdsl.interpreters.irdl import make_dialect
from xdsl.parser import Parser
from xdsl.traits import SymbolTable

# Test dynamically registering a dialect from an IRDL file

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

    # Register it for lazy loading
    ctx.register_dialect("cmath", lambda: dialect)

    # Roundtrip a cmath file!
    f = open("tests/filecheck/dialects/cmath/cmath_ops.mlir")
    parser = Parser(ctx, f.read())
    module = parser.parse_module()
    module.verify()
    print(module)

# CHECK:       builtin.module {
# CHECK-NEXT:    func.func @conorm(%p : !cmath.complex<f32>, %q : !cmath.complex<f32>) -> f32 {
# CHECK-NEXT:      %norm_p = "cmath.norm"(%p) : (!cmath.complex<f32>) -> f32
# CHECK-NEXT:      %norm_q = "cmath.norm"(%q) : (!cmath.complex<f32>) -> f32
# CHECK-NEXT:      %pq = arith.mulf %norm_p, %norm_q : f32
# CHECK-NEXT:      func.return %pq : f32
# CHECK-NEXT:    }
# CHECK-NEXT:    func.func @conorm2(%a : !cmath.complex<f32>, %b : !cmath.complex<f32>) -> f32 {
# CHECK-NEXT:      %ab = "cmath.mul"(%a, %b) : (!cmath.complex<f32>, !cmath.complex<f32>) -> !cmath.complex<f32>
# CHECK-NEXT:      %conorm = "cmath.norm"(%ab) : (!cmath.complex<f32>) -> f32
# CHECK-NEXT:      func.return %conorm : f32
# CHECK-NEXT:    }
# CHECK-NEXT:  }
