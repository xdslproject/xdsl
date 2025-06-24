# RUN: python %s | filecheck %s

from xdsl.dialects.builtin import (
    I1,
    I32,
    Float32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    f32,
    i1,
    i32,
)
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(IntegerAttr[I1], i1)
p.register_type(IntegerAttr[I32], i32)
p.register_type(FloatAttr[Float32Type], f32)
p.register_type(IndexType, IndexType())
with CodeContext(p):
    # CHECK:      func.func @test_for_I(%{{.*}} : index) {
    # CHECK:        %{{.*}} = arith.constant 0 : index
    # CHECK-NEXT:   %{{.*}} = symref.fetch @end : index
    # CHECK-NEXT:   %{{.*}} = arith.constant 1 : index
    # CHECK-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    # CHECK-NEXT:   }
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }

    def test_for_I(end: IndexType):
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
    def test_for_II(start: IndexType, end: IndexType):
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
    def test_for_III(start: IndexType, end: IndexType, step: IndexType):
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
    def test_for_IV(a: IndexType, b: IndexType, c: IndexType):
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
        def test_not_supported_loop_I(end: IntegerAttr[I32]):
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
        def test_not_supported_loop_II(start: FloatAttr[Float32Type], end: IndexType):
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
        def test_not_supported_loop_III(
            start: IndexType, end: IndexType, step: FloatAttr[Float32Type]
        ):
            for _ in range(start, end, step):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

with CodeContext(p):
    # CHECK:      %{{.*}} = scf.if %{{.*}} -> (i32) {
    # CHECK-NEXT:   %{{.*}} = symref.fetch @x : i32
    # CHECK-NEXT:   scf.yield %{{.*}} : i32
    # CHECK-NEXT: } else {
    # CHECK-NEXT:   %{{.*}} = symref.fetch @y : i32
    # CHECK-NEXT:   scf.yield %{{.*}} : i32
    # CHECK-NEXT: }
    def test_if_expr(
        cond: IntegerAttr[I1], x: IntegerAttr[I32], y: IntegerAttr[I32]
    ) -> IntegerAttr[I32]:
        return x if cond else y

    # CHECK:      scf.if %{{.*}} {
    # CHECK-NEXT: }
    def test_if_I(cond: IntegerAttr[I1]):
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
    def test_if_II(a: IntegerAttr[I1], b: IntegerAttr[I1], c: IntegerAttr[I1]):
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
    def test_if_III(cond: IntegerAttr[I1]):
        if cond:
            pass
        return


p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK: Expected the same types for if expression, but got i32 and f32.
        def test_type_mismatch_in_if_expr(
            cond: IntegerAttr[I1], x: IntegerAttr[I32], y: FloatAttr[Float32Type]
        ) -> IntegerAttr[I32]:
            return x if cond else y

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
