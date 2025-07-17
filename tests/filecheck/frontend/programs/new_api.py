# RUN: python %s | filecheck %s

from xdsl.dialects.arith import AddfOp, MulfOp
from xdsl.dialects.builtin import f64
from xdsl.frontend.pyast.context import PyASTContext

# `FrontendContext` encapsulates the mapping from Python to xDSL constructs
ctx = PyASTContext()
ctx.register_type(float, f64)
ctx.register_function(float.__add__, AddfOp)
ctx.register_function(float.__mul__, MulfOp)


# Functions can be parsed in a context to yield a lazy wrapper for the IR
# representation of that function
@ctx.parse_program
def test_arith(x: float, y: float, z: float) -> float:
    return x + y * z


# The lazy wrapper can still be called as if it were a native Python function
print(test_arith(1.0, 2.0, 3.0))
# CHECK: 7.0

# But the wrapper also provides a property to get an IR representation
print(test_arith.module)
# CHECK:       builtin.module {
# CHECK-NEXT:  func.func @test_arith(%x : f64, %y : f64, %z : f64) -> f64 {
# CHECK-NEXT:    %0 = arith.mulf %y, %z : f64
# CHECK-NEXT:    %1 = arith.addf %x, %0 : f64
# CHECK-NEXT:    func.return %1 : f64
# CHECK-NEXT:  }
# CHECK-NEXT:}


# The `ctx.parse_program` decorator can also be invoked with arguments
@ctx.parse_program(desymref=False)
def test_add(x: float, y: float) -> float:
    return x + y


# And the extracted module is built only once, then cached
module = test_add.module
assert module is test_add.module
print(module)
# CHECK:       builtin.module {
# CHECK-NEXT:    func.func @test_add(%x : f64, %y : f64) -> f64 {
# CHECK-NEXT:      symref.declare "x"
# CHECK-NEXT:      symref.update @x = %x : f64
# CHECK-NEXT:      symref.declare "y"
# CHECK-NEXT:      symref.update @y = %y : f64
# CHECK-NEXT:      %0 = symref.fetch @y : f64
# CHECK-NEXT:      %1 = symref.fetch @x : f64
# CHECK-NEXT:      %2 = arith.addf %1, %0 : f64
# CHECK-NEXT:      func.return %2 : f64
# CHECK-NEXT:    }
# CHECK-NEXT:  }
