# RUN: python %s | filecheck %s

from ctypes import c_size_t

from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(bool, builtin.i1)
p.register_type(int, bigint.bigint)
p.register_type(c_size_t, builtin.IndexType())
p.register_type(float, builtin.f64)
with CodeContext(p):
    # CHECK:      func.func @test_for_I(%{{.*}} : index) {
    # CHECK:        %{{.*}} = arith.constant 0 : index
    # CHECK-NEXT:   %{{.*}} = symref.fetch @end : index
    # CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:   }
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }

    def test_for_I(end: c_size_t):
        for _ in range(
            end  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
        ):
            pass
        return

    # CHECK:      func.func @test_for_II(%{{.*}} : index, %{{.*}} : index) {
    # CHECK:        %{{.*}} = symref.fetch @start : index
    # CHECK-NEXT:   %{{.*}} = symref.fetch @end : index
    # CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:   }
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def test_for_II(start: c_size_t, end: c_size_t):
        for _ in range(
            start,  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
            end,  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
        ):
            pass
        return

    # CHECK:      func.func @test_for_III(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
    # CHECK:        %{{.*}} = symref.fetch @start : index
    # CHECK-NEXT:   %{{.*}} = symref.fetch @end : index
    # CHECK-NEXT:   %{{.*}} = symref.fetch @step : index
    # CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:   }
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def test_for_III(start: c_size_t, end: c_size_t, step: c_size_t):
        for _ in range(
            start,  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
            end,  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
            step,  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
        ):
            pass
        return

    # CHECK:        func.func @test_for_IV(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index) {
    # CHECK:          %{{.*}} = arith.constant 0 : index
    # CHECK-NEXT:     %{{.*}} = symref.fetch @a : index
    # CHECK-NEXT:     %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:       %{{.*}} = arith.constant 0 : index
    # CHECK-NEXT:       %{{.*}} = symref.fetch @b : index
    # CHECK-NEXT:       %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:         %{{.*}} = arith.constant 0 : index
    # CHECK-NEXT:         %{{.*}} = symref.fetch @c : index
    # CHECK-NEXT:         %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:         }
    # CHECK-NEXT:       }
    # CHECK-NEXT:     }
    # CHECK-NEXT:   func.return
    # CHECK-NEXT:   }
    def test_for_IV(a: c_size_t, b: c_size_t, c: c_size_t):
        for _ in range(
            a  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
        ):
            for _ in range(
                b  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
            ):
                for _ in range(
                    c  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
                ):
                    pass
        return


p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop end, got '!bigint.bigint'.
        def test_not_supported_loop_I(end: int):
            for _ in range(end):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop start, got 'f64'.
        def test_not_supported_loop_II(start: float, end: c_size_t):
            for _ in range(start, end):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop step, got 'f64'.
        def test_not_supported_loop_III(start: c_size_t, end: c_size_t, step: float):
            for _ in range(start, end, step):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

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
            return x if cond else y

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
