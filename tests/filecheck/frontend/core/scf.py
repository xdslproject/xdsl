# RUN: python %s | filecheck %s

from xdsl.frontend.context import CodeContext
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.program import FrontendProgram

# We need *some* types to test with, but the translation is fixed across all frontends.
from xdsl.frontend.default.builtin import index, i1, i32, f32


p = FrontendProgram()
with CodeContext(p):
    # CHECK:      "func.func"() ({
    # CHECK-NEXT: ^0(%{{.*}} : index):
    # CHECK:        %{{.*}} = "arith.constant"() {"value" = 0 : index} : () -> index
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @end} : () -> index
    # CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 1 : index} : () -> index
    # CHECK-NEXT:   "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
    # CHECK-NEXT:   ^1(%{{.*}} : index):
    # CHECK-NEXT:     "scf.yield"() : () -> ()
    # CHECK-NEXT:   }) : (index, index, index) -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_for_I", "function_type" = (index) -> (), "sym_visibility" = "private"} : () -> ()

    def test_for_I(end: index):
        for _ in range(end):  # type: ignore
            pass

    # CHECK:      "func.func"() ({
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : index, %{{.*}} : index):
    # CHECK:        %{{.*}} = "symref.fetch"() {"symbol" = @start} : () -> index
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @end} : () -> index
    # CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 1 : index} : () -> index
    # CHECK-NEXT:   "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
    # CHECK-NEXT:   ^3(%{{.*}} : index):
    # CHECK-NEXT:     "scf.yield"() : () -> ()
    # CHECK-NEXT:   }) : (index, index, index) -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_for_II", "function_type" = (index, index) -> (), "sym_visibility" = "private"} : () -> ()
    def test_for_II(start: index, end: index):
        for _ in range(start, end):  # type: ignore
            pass

    # CHECK:      "func.func"() ({
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
    # CHECK:        %{{.*}} = "symref.fetch"() {"symbol" = @start} : () -> index
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @end} : () -> index
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @step} : () -> index
    # CHECK-NEXT:   "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
    # CHECK-NEXT:   ^5(%{{.*}} : index):
    # CHECK-NEXT:     "scf.yield"() : () -> ()
    # CHECK-NEXT:   }) : (index, index, index) -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_for_III", "function_type" = (index, index, index) -> (), "sym_visibility" = "private"} : () -> ()
    def test_for_III(start: index, end: index, step: index):
        for _ in range(start, end, step):  # type: ignore
            pass

    # CHECK:        "func.func"() ({
    # CHECK-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : index, %{{.*}} : index):
    # CHECK:          %{{.*}} = "arith.constant"() {"value" = 0 : index} : () -> index
    # CHECK-NEXT:     %{{.*}} = "symref.fetch"() {"symbol" = @a} : () -> index
    # CHECK-NEXT:     %{{.*}} = "arith.constant"() {"value" = 1 : index} : () -> index
    # CHECK-NEXT:     "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
    # CHECK-NEXT:     ^{{.*}}(%{{.*}} : index):
    # CHECK-NEXT:       %{{.*}} = "arith.constant"() {"value" = 0 : index} : () -> index
    # CHECK-NEXT:       %{{.*}} = "symref.fetch"() {"symbol" = @b} : () -> index
    # CHECK-NEXT:       %{{.*}} = "arith.constant"() {"value" = 1 : index} : () -> index
    # CHECK-NEXT:       "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
    # CHECK-NEXT:       ^{{.*}}(%{{.*}} : index):
    # CHECK-NEXT:         %{{.*}} = "arith.constant"() {"value" = 0 : index} : () -> index
    # CHECK-NEXT:         %{{.*}} = "symref.fetch"() {"symbol" = @c} : () -> index
    # CHECK-NEXT:         %{{.*}} = "arith.constant"() {"value" = 1 : index} : () -> index
    # CHECK-NEXT:         "scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
    # CHECK-NEXT:         ^{{.*}}(%{{.*}} : index):
    # CHECK-NEXT:           "scf.yield"() : () -> ()
    # CHECK-NEXT:         }) : (index, index, index) -> ()
    # CHECK-NEXT:         "scf.yield"() : () -> ()
    # CHECK-NEXT:       }) : (index, index, index) -> ()
    # CHECK-NEXT:       "scf.yield"() : () -> ()
    # CHECK-NEXT:     }) : (index, index, index) -> ()
    # CHECK-NEXT:   }) {"sym_name" = "test_for_IV", "function_type" = (index, index, index) -> (), "sym_visibility" = "private"} : () -> ()
    # CHECK-NEXT: }) : () -> ()
    def test_for_IV(a: index, b: index, c: index):
        for _ in range(a):  # type: ignore
            for _ in range(b):  # type: ignore
                for _ in range(c):  # type: ignore
                    pass


p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop end, got 'i32'.
        def test_not_supported_loop_I(end: i32):
            for _ in range(end):  # type: ignore
                pass

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop start, got 'f32'.
        def test_not_supported_loop_II(start: f32, end: index):
            for _ in range(start, end):  # type: ignore
                pass

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected 'index' type for loop step, got 'f32'.
        def test_not_supported_loop_III(start: index, end: index, step: f32):
            for _ in range(start, end, step):  # type: ignore
                pass

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      %{{.*}} = "scf.if"(%{{.*}}) ({
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @x} : () -> i32
    # CHECK-NEXT:   "scf.yield"(%{{.*}}) : (i32) -> ()
    # CHECK-NEXT: }, {
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @y} : () -> i32
    # CHECK-NEXT:   "scf.yield"(%{{.*}}) : (i32) -> ()
    # CHECK-NEXT: }) : (i1) -> i32
    def test_if_expr(cond: i1, x: i32, y: i32) -> i32:
        return x if cond else y

    # CHECK:      "scf.if"(%{{.*}}) ({
    # CHECK-NEXT:   "scf.yield"() : () -> ()
    # CHECK-NEXT: }, {
    # CHECK-NEXT:   "scf.yield"() : () -> ()
    # CHECK-NEXT: }) : (i1) -> ()
    def test_if_I(cond: i1):
        if cond:
            pass
        else:
            pass

    # CHECK:      %{{.*}} = "symref.fetch"() {"symbol" = @a} : () -> i1
    # CHECK-NEXT: "scf.if"(%{{.*}}) ({
    # CHECK-NEXT:   "scf.yield"() : () -> ()
    # CHECK-NEXT: }, {
    # CHECK-NEXT:   %{{.*}} = "symref.fetch"() {"symbol" = @b} : () -> i1
    # CHECK-NEXT:   "scf.if"(%{{.*}}) ({
    # CHECK-NEXT:     "scf.yield"() : () -> ()
    # CHECK-NEXT:   }, {
    # CHECK-NEXT:     %{{.*}} = "symref.fetch"() {"symbol" = @c} : () -> i1
    # CHECK-NEXT:     "scf.if"(%{{.*}}) ({
    # CHECK-NEXT:       "scf.yield"() : () -> ()
    # CHECK-NEXT:     }, {
    # CHECK-NEXT:       "scf.yield"() : () -> ()
    # CHECK-NEXT:     }) : (i1) -> ()
    # CHECK-NEXT:     "scf.yield"() : () -> ()
    # CHECK-NEXT:   }) : (i1) -> ()
    # CHECK-NEXT:   "scf.yield"() : () -> ()
    # CHECK-NEXT: }) : (i1) -> ()
    def test_if_II(a: i1, b: i1, c: i1):
        if a:
            pass
        elif b:
            pass
        elif c:
            pass

    # CHECK:      %{{.*}} = "symref.fetch"() {"symbol" = @cond} : () -> i1
    # CHECK-NEXT: "scf.if"(%{{.*}}) ({
    # CHECK-NEXT:   "scf.yield"() : () -> ()
    # CHECK-NEXT: }, {
    # CHECK-NEXT:   "scf.yield"() : () -> ()
    # CHECK-NEXT: }) : (i1) -> ()
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
            return x if cond else y  # type: ignore

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
