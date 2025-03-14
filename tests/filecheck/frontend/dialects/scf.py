# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.dialects.builtin import f32, i1, i32, index
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      func.func @test_for_I(%{{.*}} : index) {
    # CHECK:        %{{.*}} = arith.constant 0 : index
    # CHECK-NEXT:   %{{.*}} = symref.fetch @end : index
    # CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:   }
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }

    def test_for_I(end: index):
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
    def test_for_II(start: index, end: index):
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
    def test_for_III(start: index, end: index, step: index):
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
    def test_for_IV(a: index, b: index, c: index):
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
        # CHECK: Expected 'index' type for loop end, got 'i32'.
        def test_not_supported_loop_I(end: i32):
            for _ in range(end):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop start, got 'f32'.
        def test_not_supported_loop_II(start: f32, end: index):
            for _ in range(start, end):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop step, got 'f32'.
        def test_not_supported_loop_III(start: index, end: index, step: f32):
            for _ in range(start, end, step):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      %{{.*}} = scf.if %{{.*}} -> (i32) {
    # CHECK-NEXT:   %{{.*}} = symref.fetch @x : i32
    # CHECK-NEXT:   scf.yield %{{.*}} : i32
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %{{.*}} = symref.fetch @y : i32
    # CHECK-NEXT:   scf.yield %{{.*}} : i32
    # CHECK-NEXT: }
    def test_if_expr(cond: i1, x: i32, y: i32) -> i32:
        return x if cond else y

    # CHECK:      scf.if %{{.*}} {
    # CHECK-NEXT: }
    def test_if_I(cond: i1):
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
    def test_if_II(a: i1, b: i1, c: i1):
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
    def test_if_III(cond: i1):
        if cond:
            pass
        return


p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK: Expected the same types for if expression, but got i32 and f32.
        def test_type_mismatch_in_if_expr(cond: i1, x: i32, y: f32) -> i32:
            return x if cond else y

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
