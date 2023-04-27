# RUN: python %s | filecheck %s

from xdsl.frontend.context import CodeContext
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.dialects.builtin import index, i32, f32


p = FrontendProgram()
with CodeContext(p):
    #      CHECK: func.func() ["sym_name" = "test_for_I"
    #      CHECK: %{{.*}} : !index = arith.constant() ["value" = 0 : !index]
    # CHECK-NEXT: %{{.*}} : !index = symref.fetch() ["symbol" = @end]
    # CHECK-NEXT: %{{.*}} : !index = arith.constant() ["value" = 1 : !index]
    # CHECK-NEXT: scf.for(%{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index) {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_for_I(end: index):
        for _ in range(end):  # type: ignore
            pass

    #      CHECK: func.func() ["sym_name" = "test_for_II"
    #      CHECK: %{{.*}} : !index = symref.fetch() ["symbol" = @start]
    # CHECK-NEXT: %{{.*}} : !index = symref.fetch() ["symbol" = @end]
    # CHECK-NEXT: %{{.*}} : !index = arith.constant() ["value" = 1 : !index]
    # CHECK-NEXT: scf.for(%{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index) {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_for_II(start: index, end: index):
        for _ in range(start, end):  # type: ignore
            pass

    #      CHECK: func.func() ["sym_name" = "test_for_III"
    #      CHECK: %{{.*}} : !index = symref.fetch() ["symbol" = @start]
    # CHECK-NEXT: %{{.*}} : !index = symref.fetch() ["symbol" = @end]
    # CHECK-NEXT: %{{.*}} : !index = symref.fetch() ["symbol" = @step]
    # CHECK-NEXT: scf.for(%{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index) {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_for_III(start: index, end: index, step: index):
        for _ in range(start, end, step):  # type: ignore
            pass

    #      CHECK: func.func() ["sym_name" = "test_for_IV"
    #      CHECK: %{{.*}} : !index = arith.constant() ["value" = 0 : !index]
    # CHECK-NEXT: %{{.*}} : !index = symref.fetch() ["symbol" = @a]
    # CHECK-NEXT: %{{.*}} : !index = arith.constant() ["value" = 1 : !index]
    # CHECK-NEXT: scf.for(%{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index) {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   %{{.*}} : !index = arith.constant() ["value" = 0 : !index]
    # CHECK-NEXT:   %{{.*}} : !index = symref.fetch() ["symbol" = @b]
    # CHECK-NEXT:   %{{.*}} : !index = arith.constant() ["value" = 1 : !index]
    # CHECK-NEXT:   scf.for(%{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index) {
    # CHECK-NEXT:   ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:     %{{.*}} : !index = arith.constant() ["value" = 0 : !index]
    # CHECK-NEXT:     %{{.*}} : !index = symref.fetch() ["symbol" = @c]
    # CHECK-NEXT:     %{{.*}} : !index = arith.constant() ["value" = 1 : !index]
    # CHECK-NEXT:     scf.for(%{{.*}} : !index, %{{.*}} : !index, %{{.*}} : !index) {
    # CHECK-NEXT:     ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:       scf.yield()
    # CHECK-NEXT:     }
    # CHECK-NEXT:     scf.yield()
    # CHECK-NEXT:   }
    # CHECK-NEXT:   scf.yield()
    # CHECK-NEXT: }
    def test_for_IV(a: index, b: index, c: index):
        for _ in range(a):  # type: ignore
            for _ in range(b):  # type: ignore
                for _ in range(c):  # type: ignore
                    pass


p.compile(desymref=False)
print(p.xdsl())


try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop end, got 'i32'.
        def test_not_supported_loop_I(end: i32):
            for _ in range(end):  # type: ignore
                pass

    p.compile(desymref=False)
    print(p.xdsl())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop start, got 'f32'.
        def test_not_supported_loop_II(start: f32, end: index):
            for _ in range(start, end):  # type: ignore
                pass

    p.compile(desymref=False)
    print(p.xdsl())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop step, got 'f32'.
        def test_not_supported_loop_III(start: index, end: index, step: f32):
            for _ in range(start, end, step):  # type: ignore
                pass

    p.compile(desymref=False)
    print(p.xdsl())
except CodeGenerationException as e:
    print(e.msg)
