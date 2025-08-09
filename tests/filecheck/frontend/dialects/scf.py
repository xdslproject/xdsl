# RUN: python %s | filecheck %s

from ctypes import c_size_t

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

p = FrontendProgram()
p.register_type(bool, builtin.i1)
p.register_type(int, bigint.bigint)
p.register_type(c_size_t, builtin.IndexType())
p.register_type(float, builtin.f64)


with CodeContext(p):
    # CHECK:      %{{.*}} = scf.if %{{.*}} -> (!bigint.bigint) {
    # CHECK-NEXT:   %{{.*}} = symref.fetch @x : !bigint.bigint
    # CHECK-NEXT:   scf.yield %{{.*}} : !bigint.bigint
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %{{.*}} = symref.fetch @y : !bigint.bigint
    # CHECK-NEXT:   scf.yield %{{.*}} : !bigint.bigint
    # CHECK-NEXT: }
    def test_if_expr(cond: bool, x: int, y: int) -> int:
        return x if cond else y

    # CHECK:      scf.if %{{.*}} {
    # CHECK-NEXT: }
    def test_if_I(cond: bool):
        if cond:
            pass
        else:
            pass
        return

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
    def test_if_II(a: bool, b: bool, c: bool):
        if a:
            pass
        elif b:
            pass
        elif c:
            pass
        return

    # CHECK:      %{{.*}} = symref.fetch @cond : i1
    # CHECK-NEXT: scf.if %{{.*}} {
    # CHECK-NEXT: }
    def test_if_III(cond: bool):
        if cond:
            pass
        return


p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK: Expected the same types for if expression, but got !bigint.bigint and f64.
        def test_type_mismatch_in_if_expr(cond: bool, x: int, y: float) -> int:
            return x if cond else y  # pyright: ignore[reportReturnType]

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
