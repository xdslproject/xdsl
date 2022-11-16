# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext

p = FrontendProgram()
with CodeContext(p):
    #      CHECK: affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 10 : !index, "step" = 1 : !index] {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   affine.yield()
    # CHECK-NEXT: }
    # CHECK-NEXT: affine.for() ["lower_bound" = 50 : !index, "upper_bound" = 100 : !index, "step" = 1 : !index] {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   affine.yield()
    # CHECK-NEXT: }
    # CHECK-NEXT: affine.for() ["lower_bound" = 50 : !index, "upper_bound" = 100 : !index, "step" = 2 : !index] {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   affine.yield()
    # CHECK-NEXT: }
    def affine_for_loops():
        for i in range(10):
            pass
        for i in range(50, 100):
            pass
        for i in range(50, 100, 2):
            pass
    
    #      CHECK: affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 10 : !index, "step" = 1 : !index] {
    # CHECK-NEXT: ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:   affine.for() ["lower_bound" = 2 : !index, "upper_bound" = 100 : !index, "step" = 1 : !index] {
    # CHECK-NEXT:   ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:     affine.for() ["lower_bound" = 30 : !index, "upper_bound" = 1000 : !index, "step" = 10 : !index] {
    # CHECK-NEXT:     ^{{.*}}(%{{.*}} : !index):
    # CHECK-NEXT:       affine.yield()
    # CHECK-NEXT:     }
    # CHECK-NEXT:     affine.yield()
    # CHECK-NEXT:   }
    # CHECK-NEXT:   affine.yield()
    # CHECK-NEXT: }
    def nested_affine_for_loops():
        for i in range(10):
            for j in range(2, 100):
                for k in range(30, 1000, 10):
                    pass

p.compile()
print(p)
