# RUN: python %s | filecheck %s

from xdsl.frontend.context import CodeContext
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.program import FrontendProgram


p = FrontendProgram()
with CodeContext(p):
    # CHECK:      "func.func"() ({
    # CHECK-NEXT:   "affine.for"() ({
    # CHECK-NEXT:   ^0(%0 : index):
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) {"lower_bound" = 0 : index, "upper_bound" = 100 : index, "step" = 1 : index} : () -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_affine_for_I", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()

    def test_affine_for_I():
        for _ in range(100):
            pass
        return

    # CHECK:      "func.func"() ({
    # CHECK-NEXT:   "affine.for"() ({
    # CHECK-NEXT:   ^1(%1 : index):
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) {"lower_bound" = 10 : index, "upper_bound" = 30 : index, "step" = 1 : index} : () -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_affine_for_II", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    def test_affine_for_II():
        for _ in range(10, 30):
            pass
        return

    # CHECK:      "func.func"() ({
    # CHECK-NEXT:   "affine.for"() ({
    # CHECK-NEXT:   ^2(%2 : index):
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) {"lower_bound" = 1 : index, "upper_bound" = 20 : index, "step" = 5 : index} : () -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_affine_for_III", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    def test_affine_for_III():
        for _ in range(1, 20, 5):
            pass
        return

    # CHECK:      "func.func"() ({
    # CHECK-NEXT:   "affine.for"() ({
    # CHECK-NEXT:   ^3(%3 : index):
    # CHECK-NEXT:     "affine.for"() ({
    # CHECK-NEXT:     ^4(%4 : index):
    # CHECK-NEXT:       "affine.for"() ({
    # CHECK-NEXT:       ^5(%5 : index):
    # CHECK-NEXT:         "affine.yield"() : () -> ()
    # CHECK-NEXT:       }) {"lower_bound" = 0 : index, "upper_bound" = 30 : index, "step" = 1 : index} : () -> ()
    # CHECK-NEXT:       "affine.yield"() : () -> ()
    # CHECK-NEXT:     }) {"lower_bound" = 0 : index, "upper_bound" = 20 : index, "step" = 1 : index} : () -> ()
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) {"lower_bound" = 0 : index, "upper_bound" = 10 : index, "step" = 1 : index} : () -> ()
    # CHECK-NEXT: }) {"sym_name" = "test_affine_for_IV", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
    def test_affine_for_IV():
        for _ in range(10):
            for _ in range(20):
                for _ in range(30):
                    pass
        return


p.compile(desymref=False)
print(p.textual_format())


try:
    with CodeContext(p):
        # CHECK: Expected integer constant for loop end, got 'float'.
        def test_not_supported_affine_loop_I():
            for _ in range(12.0):  # type: ignore
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected integer constant for loop start, got 'str'.
        def test_not_supported_affine_loop_II():
            for _ in range("boom", 100):  # type: ignore
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)

try:
    with CodeContext(p):
        # CHECK: Expected integer constant for loop step, got 'float'.
        def test_not_supported_affine_loop_III():
            for _ in range(0, 100, 1.0):  # type: ignore
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
