# RUN: python %s | filecheck %s

from ctypes import c_size_t

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

ctx = PyASTContext(post_transforms=[])
ctx.register_type(bool, builtin.i1)
ctx.register_type(int, bigint.bigint)
ctx.register_type(c_size_t, builtin.IndexType())
ctx.register_type(float, builtin.f64)


# CHECK:      %{{.*}} = scf.if %{{.*}} -> (!bigint.bigint) {
# CHECK-NEXT:   %{{.*}} = symref.fetch @x : !bigint.bigint
# CHECK-NEXT:   scf.yield %{{.*}} : !bigint.bigint
# CHECK-NEXT: } else {
# CHECK-NEXT:   %{{.*}} = symref.fetch @y : !bigint.bigint
# CHECK-NEXT:   scf.yield %{{.*}} : !bigint.bigint
# CHECK-NEXT: }
@ctx.parse_program
def test_if_expr(cond: bool, x: int, y: int) -> int:
    return x if cond else y


print(test_if_expr.module)


# CHECK:      scf.if %{{.*}} {
# CHECK-NEXT: }
@ctx.parse_program
def test_if_I(cond: bool):
    if cond:
        pass
    else:
        pass
    return


print(test_if_I.module)


# CHECK:      %{{.*}} = symref.fetch @a : i1
# CHECK-NEXT: scf.if %{{.*}} {
# CHECK-NEXT: } else {
# CHECK-NEXT:   %{{.*}} = symref.fetch @b : i1
# CHECK-NEXT:   scf.if %{{.*}} {
# CHECK-NEXT:   } else {
# CHECK-NEXT:     %{{.*}} = symref.fetch @c : i1
# CHECK-NEXT:     scf.if %{{.*}} {
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }
@ctx.parse_program
def test_if_II(a: bool, b: bool, c: bool):
    if a:
        pass
    elif b:
        pass
    elif c:
        pass
    return


print(test_if_II.module)


# CHECK:      %{{.*}} = symref.fetch @cond : i1
# CHECK-NEXT: scf.if %{{.*}} {
# CHECK-NEXT: }
@ctx.parse_program
def test_if_III(cond: bool):
    if cond:
        pass
    return


print(test_if_III.module)


# CHECK: Expected the same types for if expression, but got !bigint.bigint and f64.
@ctx.parse_program
def test_type_mismatch_in_if_expr(cond: bool, x: int, y: float) -> int:
    return x if cond else y  # pyright: ignore[reportReturnType]


try:
    test_type_mismatch_in_if_expr.module
except CodeGenerationException as e:
    print(e.msg)
