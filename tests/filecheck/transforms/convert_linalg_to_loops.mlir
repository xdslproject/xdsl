// RUN: xdsl-opt -p convert-linalg-to-loops %s | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
%D, %E, %F = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
%G, %H, %I = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)

linalg.generic {
    indexing_maps = [
        affine_map<() -> ()>,
        affine_map<() -> ()>,
        affine_map<() -> ()>
    ],
    iterator_types = []
} ins(%A, %B : memref<f64>, memref<f64>) outs(%C : memref<f64>) {
^bb0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}
// CHECK-NEXT:    %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:    %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:    %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    memref.store %{{.*}}, %{{.*}}[] : memref<f64>


linalg.generic {
    indexing_maps = [
        affine_map<(i, j, k) -> (i, j)>,
        affine_map<(i, j, k) -> (j, k)>,
        affine_map<(i, j, k) -> (i, k)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%D, %E : memref<2x3xf64>, memref<3x4xf64>) outs(%F : memref<2x4xf64>) {
^bb0(%d : f64, %e : f64, %acc_old : f64):
    %prod = arith.mulf %d, %e : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    %0 = arith.constant 2 : index
// CHECK-NEXT:    %1 = arith.constant 3 : index
// CHECK-NEXT:    %2 = arith.constant 4 : index
// CHECK-NEXT:    %3 = arith.constant 0 : index
// CHECK-NEXT:    %4 = arith.constant 1 : index
// CHECK-NEXT:    scf.for %5 = %3 to %0 step %4 {
// CHECK-NEXT:      scf.for %6 = %3 to %1 step %4 {
// CHECK-NEXT:        scf.for %7 = %3 to %2 step %4 {
// CHECK-NEXT:          %d = memref.load %D[%5, %6] : memref<2x3xf64>
// CHECK-NEXT:          %e = memref.load %E[%6, %7] : memref<3x4xf64>
// CHECK-NEXT:          %acc_old_1 = memref.load %F[%5, %7] : memref<2x4xf64>
// CHECK-NEXT:          %prod_1 = arith.mulf %d, %e : f64
// CHECK-NEXT:          %acc_new_1 = arith.addf %acc_old_1, %prod_1 : f64
// CHECK-NEXT:          memref.store %acc_new_1, %F[%5, %7] : memref<2x4xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }


linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i + j)>,
        affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
} ins(%G, %H : memref<4xf64>, memref<2xf64>) outs(%I : memref<3xf64>) {
^bb0(%g : f64, %h : f64, %acc_old : f64):
    %prod = arith.mulf %g, %h : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = affine.apply affine_map<(d0, d1) -> ((d0 + d1))>
// CHECK-NEXT:        %{{.*}} = memref.load %G[%{{.*}}] : memref<4xf64>
// CHECK-NEXT:        %{{.*}} = memref.load %H[%{{.*}}] : memref<2xf64>
// CHECK-NEXT:        %{{.*}} = memref.load %I[%{{.*}}] : memref<3xf64>
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        memref.store %{{.*}}, %I[%{{.*}}] : memref<3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Scalar argument
%zero = arith.constant 0.0 : f64
linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%zero : f64) outs(%D : memref<2x3xf64>) {
^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
}

// CHECK-NEXT:    %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// CHECK-NEXT:  }
