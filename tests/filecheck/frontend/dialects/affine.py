# RUN: python %s | filecheck %s

from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK:      func.func @test_affine_for_I() {
    # CHECK-NEXT:   "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (100)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    # CHECK-NEXT:   ^0(%0 : index):
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) : () -> ()
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }

    def test_affine_for_I():
        for _ in range(100):
            pass
        return

    # CHECK:      func.func @test_affine_for_II() {
    # CHECK-NEXT:   "affine.for"() <{lowerBoundMap = affine_map<() -> (10)>, upperBoundMap = affine_map<() -> (30)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    # CHECK-NEXT:   ^0(%0 : index):
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) : () -> ()
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def test_affine_for_II():
        for _ in range(10, 30):
            pass
        return

    # CHECK:      func.func @test_affine_for_III() {
    # CHECK-NEXT:   "affine.for"() <{lowerBoundMap = affine_map<() -> (1)>, upperBoundMap = affine_map<() -> (20)>, step = 5 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    # CHECK-NEXT:   ^0(%0 : index):
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) : () -> ()
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
    def test_affine_for_III():
        for _ in range(1, 20, 5):
            pass
        return

    # CHECK:      func.func @test_affine_for_IV() {
    # CHECK-NEXT:   "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (10)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    # CHECK-NEXT:   ^0(%0 : index):
    # CHECK-NEXT:     "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (20)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    # CHECK-NEXT:     ^1(%1 : index):
    # CHECK-NEXT:       "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, upperBoundMap = affine_map<() -> (30)>, step = 1 : index, operandSegmentSizes = array<i32: 0, 0, 0>}> ({
    # CHECK-NEXT:       ^2(%2 : index):
    # CHECK-NEXT:         "affine.yield"() : () -> ()
    # CHECK-NEXT:       }) : () -> ()
    # CHECK-NEXT:       "affine.yield"() : () -> ()
    # CHECK-NEXT:     }) : () -> ()
    # CHECK-NEXT:     "affine.yield"() : () -> ()
    # CHECK-NEXT:   }) : () -> ()
    # CHECK-NEXT:   func.return
    # CHECK-NEXT: }
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
            for _ in range(
                12.0  # pyright: ignore[reportArgumentType]
            ):
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
            for _ in range(
                "boom",  # pyright: ignore[reportArgumentType]
                100,
            ):
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
            for _ in range(
                0,
                100,
                1.0,  # pyright: ignore[reportArgumentType]
            ):
                pass
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
